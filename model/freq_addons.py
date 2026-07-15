# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


VALID_FR_MODES = ("off", "fft_identity", "amplitude", "full")


# --------- Basic FFT utilities ----------
def _radial_mask(h: int, w: int, high_ratio: float = 0.5, device=None, inverse: bool = False):
    """
    Generate a radial mask; values beyond radius r are kept (high-frequency by default).
    """
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r0 = high_ratio * min(h, w) * 0.5
    mask = (rr >= r0).float()
    if inverse:
        mask = 1.0 - mask
    return mask  # [H,W], 1 = high freq (default)


def _fft2_energy(x: torch.Tensor, high_ratio: float = 0.5) -> torch.Tensor:
    """
    x: [B,C,H,W], return normalized high-frequency energy map [B,1,H,W]
    """
    B, C, H, W = x.shape
    orig_dtype = x.dtype
    needs_cast = orig_dtype in (torch.float16, torch.bfloat16)
    if needs_cast:
        x = x.to(torch.float32)
    X = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
    mask = _radial_mask(H, W, high_ratio, device=x.device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    en = (X.abs() * mask).sum(dim=1, keepdim=True).real
    en = en / (en.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    en = en.clamp_(0, 1)
    if needs_cast:
        en = en.to(orig_dtype)
    return en


def _fft2_highpass(x: torch.Tensor, high_ratio: float = 0.5) -> torch.Tensor:
    """
    Extract high-frequency component and inverse transform back to spatial domain.
    """
    B, C, H, W = x.shape
    orig_dtype = x.dtype
    needs_cast = orig_dtype in (torch.float16, torch.bfloat16)
    if needs_cast:
        x = x.to(torch.float32)
    X = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
    mask = _radial_mask(H, W, high_ratio, device=x.device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    X_hp = X * mask
    x_hp = torch.fft.ifft2(torch.fft.ifftshift(X_hp), norm="ortho").real
    if needs_cast:
        x_hp = x_hp.to(orig_dtype)
    return x_hp


def _fft2_amp_phase(x: torch.Tensor):
    """
    Compute shifted FFT along with amplitude and phase for later reconstruction.
    """
    X = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
    amp = X.abs()
    phase = torch.angle(X)
    return amp, phase


def spectral_power(
    x: torch.Tensor,
    remove_spatial_mean: bool = True,
) -> torch.Tensor:
    """Return the shifted 2-D power spectrum for a BCHW feature tensor.

    The computation is always performed in float32 so that the analysis is
    stable under AMP. Removing the per-channel spatial mean prevents the DC
    component from dominating comparisons between feature stages.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW/CHW input, got shape {tuple(x.shape)}")
    x = x.float()
    if remove_spatial_mean:
        x = x - x.mean(dim=(-2, -1), keepdim=True)
    spec = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
    return spec.abs().square()


def high_frequency_energy_ratio(
    x: torch.Tensor,
    cutoff: float = 0.5,
    remove_spatial_mean: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute E_high for each sample in a BCHW/CHW feature tensor.

    ``cutoff`` is the radial distance from the shifted spectrum center,
    normalized by the maximum corner radius. The returned tensor is ``[B]``.
    """
    if not 0.0 <= float(cutoff) <= 1.0:
        raise ValueError(f"cutoff must be in [0, 1], got {cutoff}")
    power = spectral_power(x, remove_spatial_mean=remove_spatial_mean)
    h, w = power.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=power.device, dtype=torch.float32),
        torch.arange(w, device=power.device, dtype=torch.float32),
        indexing="ij",
    )
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = torch.sqrt((yy - cy).square() + (xx - cx).square())
    radius = radius / radius.max().clamp_min(1.0)
    mask = radius >= float(cutoff)
    high = power[..., mask].sum(dim=(-1, -2))
    total = power.sum(dim=(-1, -2, -3))
    return high / total.clamp_min(eps)


def radial_power_profile(
    x: torch.Tensor,
    bins: int = 64,
    remove_spatial_mean: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalized radial power profiles.

    Returns ``(centers, profile)`` where centers has shape ``[bins]`` and the
    profile has shape ``[B, bins]``. Each profile integrates to one.
    """
    if bins < 2:
        raise ValueError("bins must be >= 2")
    power = spectral_power(x, remove_spatial_mean=remove_spatial_mean).sum(dim=1)
    b, h, w = power.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=power.device, dtype=torch.float32),
        torch.arange(w, device=power.device, dtype=torch.float32),
        indexing="ij",
    )
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = torch.sqrt((yy - cy).square() + (xx - cx).square())
    radius = radius / radius.max().clamp_min(1.0)
    bin_index = torch.clamp((radius * bins).long(), min=0, max=bins - 1).reshape(-1)
    profile = torch.zeros((b, bins), device=power.device, dtype=power.dtype)
    profile.scatter_add_(1, bin_index.unsqueeze(0).expand(b, -1), power.reshape(b, -1))
    profile = profile / profile.sum(dim=1, keepdim=True).clamp_min(1e-12)
    centers = (torch.arange(bins, device=power.device, dtype=power.dtype) + 0.5) / bins
    return centers, profile


# --------- A) Band-FiLM ----------
class BandFiLM2d(nn.Module):
    """
    FiLM modulation driven by high-frequency energy map: X = X * (1 + gamma*gate) + beta*gate.
    """

    def __init__(
        self, in_channels: int, hidden: int = 0, high_ratio: float = 0.5, smooth: bool = True, enable: bool = True
    ):
        super().__init__()
        self.enable = enable
        self.high_ratio = float(high_ratio)
        self.smooth = smooth
        h = max(in_channels // 4, 16) if hidden == 0 else hidden
        self.mlp = nn.Sequential(nn.Conv2d(1, h, 1), nn.GELU(), nn.Conv2d(h, 2 * in_channels, 1))

    @torch.no_grad()
    def _gate_from_energy(self, x):
        gate = _fft2_energy(x, self.high_ratio)  # [B,1,H,W]
        if self.smooth:
            gate = F.avg_pool2d(gate, 3, 1, 1)
        return gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.enable) or x.dim() != 4:
            return x
        gate = self._gate_from_energy(x)
        gamma_beta = self.mlp(gate)
        g, b = torch.chunk(gamma_beta, 2, dim=1)
        return x * (1 + g * gate) + b * gate


# --------- B) Direction-consistent gate ----------
class DirectionConsistentGate(nn.Module):
    """
    Use Sobel-based orientation to gate fusion of low/high level features.
    """

    def __init__(self, in_channels: int, mode: str = "mul", min_gate: float = 0.1, enable: bool = True):
        super().__init__()
        self.enable = enable
        self.mode = mode
        self.min_gate = float(min_gate)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))
        self.head = nn.Sequential(nn.Conv2d(1, 8, 1), nn.GELU(), nn.Conv2d(8, 1, 1), nn.Sigmoid())

    def _orientation(self, x: torch.Tensor) -> torch.Tensor:
        xg = x.mean(1, keepdim=True)
        gx = F.conv2d(xg, self.kx, padding=1)
        gy = F.conv2d(xg, self.ky, padding=1)
        ori = torch.atan2(gy, gx)  # [-pi, pi]
        ori = (ori + 3.14159265) / (2 * 3.14159265)
        return ori

    def forward(self, f_low: torch.Tensor, f_high: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return f_low + f_high
        assert f_low.shape[-2:] == f_high.shape[-2:], "resolution must match"
        ori_low = self._orientation(f_low)
        ori_high = self._orientation(f_high)
        diff = torch.cos((ori_low - ori_high) * 2 * 3.14159265)  # [-1,1]
        diff = (diff + 1) * 0.5  # [0,1]
        g = self.head(diff)  # [B,1,H,W]
        g = torch.clamp(g, min=self.min_gate)
        if self.mode == "mul":
            out = f_low * g + f_high * (1 - g)
        else:
            w_low = g
            w_high = 1 - g
            out = f_low * w_low + f_high * w_high
        return out


# --------- C) Spectrum-aware residual head with tiny spectral attention ----------
class FreqResidualHead(nn.Module):
    """
    Learnable band filtering + lightweight spectral attention, then IFFT residual.
    """

    def __init__(
        self,
        in_channels: int,
        reduce: Optional[int] = None,
        high_ratio: float = 0.5,
        with_edge_head: bool = True,
        enable: bool = True,
        attn_scale: float = 0.3,
        mode: str = "full",
    ):
        super().__init__()
        self.enable = enable
        self.high_ratio = float(high_ratio)
        self.attn_scale = float(attn_scale)
        self.mode = "full"
        self.set_mode(mode)
        c_red = in_channels // 2 if reduce is None else reduce
        self.reduce = nn.Conv2d(in_channels, c_red, 1)
        self.expand = nn.Conv2d(c_red, in_channels, 1)
        self.band_gen = nn.Sequential(
            nn.Conv2d(1, c_red, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_red, c_red, 3, padding=1),
            nn.Sigmoid(),
        )
        self.spec_attn = nn.Sequential(
            nn.Conv2d(c_red, c_red, 3, padding=1, groups=c_red),
            nn.GELU(),
            nn.Conv2d(c_red, c_red, 1),
            nn.Tanh(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.with_edge_head = with_edge_head
        if with_edge_head:
            self.edge_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(in_channels // 2, 1, 1),
                nn.Sigmoid(),
            )

    def set_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in VALID_FR_MODES:
            raise ValueError(f"Unknown FR mode '{mode}'. Expected one of {VALID_FR_MODES}.")
        self.mode = mode

    def _spectral_refinement(self, f: torch.Tensor) -> Dict[str, torch.Tensor]:
        amp, phase = _fft2_amp_phase(f)
        band = torch.log1p(amp.mean(1, keepdim=True))

        if self.mode == "fft_identity":
            mask = torch.ones_like(amp)
            amp_band = amp
            gain = torch.zeros_like(amp)
            amp_ref = amp
        else:
            mask = self.band_gen(band)
            amp_band = amp * mask
            if self.mode == "amplitude":
                gain = torch.zeros_like(amp_band)
                amp_ref = amp_band
            else:
                gain = self.spec_attn(torch.log1p(amp_band + 1e-6))
                amp_ref = amp_band * (1 + self.attn_scale * gain)

        spectrum_refined = amp_ref * torch.exp(1j * phase)
        return {
            "amp": amp,
            "phase": phase,
            "band_summary": band,
            "band_mask": mask,
            "masked_amp": amp_band,
            "gain": gain,
            "refined_amp": amp_ref,
            "refined_spectrum": spectrum_refined,
        }

    def forward(self, x: torch.Tensor):
        if (not self.enable) or self.mode == "off":
            return x, None

        orig_dtype = x.dtype
        needs_cast = orig_dtype in (torch.float16, torch.bfloat16)

        f = self.reduce(x)
        if needs_cast:
            f = f.to(torch.float32)

        trace = self._spectral_refinement(f)
        # Keep the original shift convention for backward numerical
        # compatibility with existing FSR-SAM checkpoints.
        f_rec = torch.fft.ifft2(
            torch.fft.ifftshift(trace["refined_spectrum"]),
            norm="ortho",
        ).real

        if needs_cast:
            f_rec = f_rec.to(orig_dtype)

        f_rec = self.expand(f_rec)
        out = x + torch.tanh(self.alpha) * f_rec

        edge = None
        if self.with_edge_head:
            edge = self.edge_head(out)
        return out, edge


# --------- D) DCT-based utilities and residual head (optional) ----------
def _dct2(x: torch.Tensor) -> torch.Tensor:
    """Compute 2D DCT-II per channel. Falls back to FFT magnitude if DCT not available."""
    if hasattr(torch.fft, "dct"):
        x32 = x.float()
        y = torch.fft.dct(
            torch.fft.dct(x32, type=2, n=None, dim=-1, norm="ortho"),
            type=2,
            n=None,
            dim=-2,
            norm="ortho",
        )
        return y.to(x.dtype)
    X = torch.fft.fft2(x, norm="ortho")
    return X.real


def _idct2(X: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.fft, "idct"):
        X32 = X.float()
        y = torch.fft.idct(
            torch.fft.idct(X32, type=2, n=None, dim=-1, norm="ortho"),
            type=2,
            n=None,
            dim=-2,
            norm="ortho",
        )
        return y.to(X.dtype)
    x = torch.fft.ifft2(X.to(torch.complex64), norm="ortho").real
    return x.to(X.dtype)


def _dct2_highpass(x: torch.Tensor, high_ratio: float = 0.5) -> torch.Tensor:
    B, C, H, W = x.shape
    X = _dct2(x)
    mask = _radial_mask(H, W, high_ratio, device=x.device).unsqueeze(0).unsqueeze(0)
    Y = X * mask
    y = _idct2(Y)
    return y


class DCTResidualHead(nn.Module):
    def __init__(
        self, in_channels: int, reduce: Optional[int] = None, with_edge_head: bool = False, enable: bool = False, high_ratio: float = 0.5
    ):
        super().__init__()
        self.enable = enable
        self.high_ratio = float(high_ratio)
        c_red = in_channels // 2 if reduce is None else reduce
        self.reduce = nn.Conv2d(in_channels, c_red, 1)
        self.expand = nn.Conv2d(c_red, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.with_edge_head = bool(with_edge_head)
        if with_edge_head:
            self.edge_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(in_channels // 2, 1, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor):
        if not self.enable:
            return x, None
        f = self.reduce(x)
        f_hp = _dct2_highpass(f, self.high_ratio)
        f_rec = self.expand(f_hp)
        out = x + torch.tanh(self.alpha) * f_rec
        edge = None
        if self.with_edge_head:
            edge = self.edge_head(out)
        return out, edge
