# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .freq_addons import BandFiLM2d


class MBA(nn.Module):
    """
    MBA: stronger adapter with multi-branch residuals and dynamic routing
    - Branch A: LoRA-style 1x1 conv bottleneck (pointwise convs) -> ΔA
    - Branch B: Local depthwise conv (3x3 + optional dilated 5x5) -> ΔB
    - Branch C: Frequency-aware FiLM (BandFiLM2d) residual -> ΔC
    - Branch D (optional): Lightweight row/column SSM via separable 1D depthwise convs -> ΔD
    Δ = softmax([wA,wB,wC,wD]) · [ΔA,ΔB,ΔC,ΔD];  out = x + Δ

    Inputs/Outputs are NHWC to match existing ViT block usage.
    """

    def __init__(
        self,
        dim: int,
        red: Optional[int] = None,
        lora_rank: int = 16,
        lora_alpha: float = 1.0,
        use_freq: bool = True,
        use_ssm: bool = True,
        enable: bool = True,
    ) -> None:
        super().__init__()
        self.enable = bool(enable)
        c = int(dim)
        r = int(lora_rank)
        cr = int((c // 3) if red is None else red)
        cr = max(cr, 32)

        # A) LoRA-style bottleneck on NCHW via pointwise convs
        self.lora = nn.Sequential(
            nn.Conv2d(c, r, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(r, c, kernel_size=1, bias=False),
        )
        self.lora_scale = nn.Parameter(torch.tensor(float(lora_alpha)))

        # B) Local depthwise conv (3x3 + dilated 5x5) with projection
        self.local_reduce = nn.Conv2d(c, cr, kernel_size=1, bias=False)
        self.dw3 = nn.Conv2d(cr, cr, kernel_size=3, padding=1, groups=cr, bias=False)
        self.dw5 = nn.Conv2d(cr, cr, kernel_size=5, padding=2, groups=cr, bias=False)
        self.local_proj = nn.Conv2d(cr, c, kernel_size=1, bias=False)

        # C) Frequency-aware FiLM residual
        self.use_freq = use_freq
        self.band_film = BandFiLM2d(in_channels=c, high_ratio=0.5, smooth=True, enable=use_freq)

        # D) Lightweight SSM via separable 1D depthwise convs (row/column mixers)
        self.use_ssm = use_ssm
        self.ssm_dw_row = nn.Conv2d(c, c, kernel_size=(7, 1), padding=(3, 0), groups=c, bias=False)
        self.ssm_dw_col = nn.Conv2d(c, c, kernel_size=(1, 7), padding=(0, 3), groups=c, bias=False)

        # Routing weights from global pooled feature
        branches = 3 + (1 if use_ssm else 0)
        self.router = nn.Sequential(
            nn.Conv2d(c, max(c // 4, 64), 1), nn.GELU(), nn.Conv2d(max(c // 4, 64), branches, 1)
        )

        self.norm = nn.BatchNorm2d(c)

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return x_nhwc
        # NHWC -> NCHW
        x = x_nhwc.permute(0, 3, 1, 2).contiguous()

        # Routing logits
        g = F.adaptive_avg_pool2d(x, output_size=1)
        w_logits = self.router(g)  # (B,branches,1,1)
        w = torch.softmax(w_logits.flatten(1), dim=1)  # (B,branches)

        # Branch A: LoRA
        dA = self.lora(x) * self.lora_scale

        # Branch B: Local depthwise convs
        y = self.local_reduce(x)
        y = self.dw3(y) + self.dw5(y)
        dB = self.local_proj(F.gelu(y))

        # Branch C: Frequency FiLM residual
        if self.use_freq:
            mod = self.band_film(x)
            dC = mod - x
        else:
            dC = torch.zeros_like(x)

        # Branch D: row/column mixers
        if self.use_ssm:
            z = self.ssm_dw_row(x)
            z = self.ssm_dw_col(z)
            dD = self.norm(z)
        else:
            dD = None

        # Combine with routing weights
        deltas = [dA, dB, dC] + ([dD] if dD is not None else [])
        # Stack as (B,branches,C,H,W)
        deltas_stack = torch.stack(deltas, dim=1)
        w_b = w.view(w.shape[0], -1, 1, 1, 1)
        d = (deltas_stack * w_b).sum(dim=1)

        out = x + d
        out = out.permute(0, 2, 3, 1).contiguous()
        return out
