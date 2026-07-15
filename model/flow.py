"""Structured outputs that mirror the modules and data flow in the paper."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class EncoderFlow:
    """Encoder-side flow: SAM+MBA -> BandFiLM/OCM."""

    mba_features: Tuple[Tensor, ...]
    deepest_feature_after_bandfilm: Tensor
    ocm_features: Tuple[Tensor, ...]


@dataclass(frozen=True)
class PredictionFlow:
    """Prediction-side flow: SFEM -> SAM Decoder -> FGPM -> final logits."""

    base_fusion: Tensor
    sfem_feature: Tensor
    sfem_guidance: Tensor
    semantic_guidance: Tensor
    decoder_feature: Tensor
    frequency_refined_feature: Tensor
    foreground_confidence: Tensor
    coarse_logits: Tensor
    final_logits: Tensor


@dataclass(frozen=True)
class FSRSAMFlow:
    """Paper-facing structured result without changing the legacy tuple API."""

    encoder: EncoderFlow
    prediction: PredictionFlow
    auxiliary: Dict[str, Tensor]


def build_flow(final_logits: Tensor, coarse_logits: Tensor, auxiliary: Dict) -> FSRSAMFlow:
    """Build a validated paper-facing flow from ``FSRSAM.forward`` diagnostics."""

    if "analysis" not in auxiliary:
        raise ValueError("Structured flow requires forward(..., return_intermediates=True).")

    analysis = auxiliary["analysis"]
    required = {
        "encoder_features_before_post_refinement",
        "encoder_f4_after_bandfilm",
        "encoder_features_after_ocm",
        "mcfm_output",
        "sfem_output",
        "sfem_guidance",
        "semantic_guidance",
        "decoder_before_fr",
        "decoder_after_fr",
        "foreground_localization",
    }
    missing = sorted(required.difference(analysis))
    if missing:
        raise RuntimeError(f"Missing paper-flow intermediates: {missing}")

    foreground = analysis["foreground_localization"]
    if foreground is None:
        raise RuntimeError("FGPM foreground localization is disabled; enable_loc must be True.")

    encoder = EncoderFlow(
        mba_features=tuple(analysis["encoder_features_before_post_refinement"]),
        deepest_feature_after_bandfilm=analysis["encoder_f4_after_bandfilm"],
        ocm_features=tuple(analysis["encoder_features_after_ocm"]),
    )
    prediction = PredictionFlow(
        base_fusion=analysis["mcfm_output"],
        sfem_feature=analysis["sfem_output"],
        sfem_guidance=analysis["sfem_guidance"],
        semantic_guidance=analysis["semantic_guidance"],
        decoder_feature=analysis["decoder_before_fr"],
        frequency_refined_feature=analysis["decoder_after_fr"],
        foreground_confidence=foreground,
        coarse_logits=coarse_logits,
        final_logits=final_logits,
    )
    clean_aux = {key: value for key, value in auxiliary.items() if key != "analysis"}
    return FSRSAMFlow(encoder=encoder, prediction=prediction, auxiliary=clean_aux)
