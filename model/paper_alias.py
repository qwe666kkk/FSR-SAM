from typing import Optional

import torch
import torch.nn as nn

from .mb_adapt import MBA as MBAdapt
from .block import DetailEnhancement, MLFusion
from .freq_addons import DirectionConsistentGate, FreqResidualHead
from .semantic_fusion import SFEM as _SFEM


class MBA(nn.Module):
    """MBA: wrapper for MBA adapter."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.adapter = MBAdapt(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class MCFM(nn.Module):
    """MCFM: wrapper for MLFusion."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.fusion = MLFusion(*args, **kwargs)

    def forward(self, features):
        return self.fusion(features)


class OCM(nn.Module):
    """OCM: orientation correction through direction-consistent gating."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.gate = DirectionConsistentGate(*args, **kwargs)

    def forward(self, f_low: torch.Tensor, f_high: torch.Tensor) -> torch.Tensor:
        return self.gate(f_low, f_high)


class FR(nn.Module):
    """FR: wrapper for frequency-domain residual head."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.head = FreqResidualHead(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.head(x)


class DR_BE(nn.Module):
    """DR-BE: wrapper for detail refinement and background exclusion."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.block = DetailEnhancement(*args, **kwargs)

    def forward(
        self,
        img: torch.Tensor,
        feature: torch.Tensor,
        b_feature: torch.Tensor,
        loc_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.block(img, feature, b_feature, loc_prior=loc_prior)


SFEM = _SFEM
SFEM.__doc__ = (
    "SFEM: semantic fusion and enhancement. Internally uses PPM-style extraction, "
    "cross-scale weighting, and fusion to produce refined semantics and guidance."
)
