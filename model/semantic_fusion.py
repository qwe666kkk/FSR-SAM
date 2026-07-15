# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .block import ModifyPPM


class SFEM(nn.Module):
    """
    Semantic Fusion & Enhancement Module
    - Per-scale semantic extraction via light PPM
    - Cross-scale self-calibration (weighting) and fusion
    - Output refined semantic feature and a single-channel guidance map

    Expect four scales with 256 channels each.
    """

    def __init__(self, in_ch: int = 256, bins: Tuple[int, ...] = (3, 6, 9, 12)):
        super().__init__()
        red = max(in_ch // 4, 64)
        self.ppms = nn.ModuleList([ModifyPPM(in_ch, red, list(bins)) for _ in range(4)])
        # After ModifyPPM, channels = in_ch + len(bins) * red
        self.ppm_out_ch = in_ch + len(bins) * red
        self.proj = nn.ModuleList([nn.Conv2d(self.ppm_out_ch, in_ch, 1) for _ in range(4)])

        # Scale weights by global pooling -> 1x1 conv -> softmax over 4 scales
        self.w_conv = nn.Conv2d(in_ch, 1, 1)

        # Fuse and refine
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
        )
        self.out_proj = nn.Conv2d(in_ch, in_ch, 1)

        # Semantic guidance head
        self.sem_head = nn.Sequential(nn.Conv2d(in_ch, 1, 1), nn.Sigmoid())

    def forward(self, img_feature: torch.Tensor, feats: List[torch.Tensor]):
        # feats expected length 4, each (B,256,Hk,Wk). Use the highest-res feature as target size of fusion.
        if not isinstance(feats, (list, tuple)):
            feats = list(feats)
        feats = feats[-4:]
        # choose base spatial size from img_feature
        Ht, Wt = img_feature.shape[-2:]
        sems = []
        for i in range(4):
            s = self.ppms[i](feats[i])
            s = self.proj[i](s)
            if s.shape[-2:] != (Ht, Wt):
                s = F.interpolate(s, size=(Ht, Wt), mode='bilinear', align_corners=False)
            sems.append(s)

        sem_stack = torch.stack(sems, dim=1)  # (B,4,C,H,W)
        # Weighting: global spatial pooling per scale, then 1x1 conv to scalar, softmax across scales
        pooled = [F.adaptive_avg_pool2d(s, output_size=1) for s in sems]  # each (B,C,1,1)
        logits = [self.w_conv(p) for p in pooled]  # each (B,1,1,1)
        w_logits = torch.stack(logits, dim=1)  # (B,4,1,1)
        w = torch.softmax(w_logits, dim=1)  # (B,4,1,1)
        # w has shape (B,4,1,1,1); broadcast across (C,H,W)
        fused = (sem_stack * w).sum(dim=1)  # (B,C,H,W)

        fused = self.fuse_conv(fused)
        out = img_feature + self.out_proj(fused)
        sem_g = self.sem_head(out)
        return out, sem_g
