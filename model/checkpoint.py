"""Minimal SAM-B checkpoint adapter for the paper model."""

from collections import OrderedDict
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F


def _resize_absolute_position(value: torch.Tensor, token_size: int) -> torch.Tensor:
    if value.ndim == 4 and value.shape[1:3] != (token_size, token_size):
        value = F.interpolate(
            value.permute(0, 3, 1, 2),
            size=(token_size, token_size),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).contiguous()
    return value


def _resize_global_relative_position(
    key: str, value: torch.Tensor, token_size: int
) -> torch.Tensor:
    if not any(f"blocks.{index}." in key for index in (2, 5, 8, 11)):
        return value
    target_length = token_size * 2 - 1
    if value.ndim == 2 and value.shape[0] != target_length:
        value = F.interpolate(
            value.t().unsqueeze(0),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0).t().contiguous()
    return value


def load_sam_checkpoint(
    model: torch.nn.Module,
    checkpoint: Union[str, Path],
    img_size: int = 512,
):
    """Load compatible encoder and mask-decoder weights from SAM ViT-B."""
    checkpoint_object = torch.load(str(checkpoint), map_location="cpu")
    source = checkpoint_object.get("model", checkpoint_object) if isinstance(checkpoint_object, dict) else checkpoint_object
    token_size = int(img_size // 16)
    target = OrderedDict()

    for key, value in source.items():
        key = key[7:] if key.startswith("module.") else key
        if key == "image_encoder.pos_embed":
            target[key] = _resize_absolute_position(value, token_size)
        elif "image_encoder" in key and "rel_pos" in key:
            target[key] = _resize_global_relative_position(key, value, token_size)
        elif key.startswith("image_encoder.neck."):
            suffix = key[len("image_encoder.neck."):]
            for index in range(4):
                target[f"image_encoder.neck.{index}.{suffix}"] = value
        elif key.startswith("image_encoder."):
            target[key] = value
        elif key.startswith("mask_decoder.transformer.") or key == "mask_decoder.iou_token.weight":
            target[key] = value
        elif key == "mask_decoder.mask_tokens.weight":
            target[key] = value[:1]
        elif key.startswith("mask_decoder.output_upscaling."):
            mapping = {
                "mask_decoder.output_upscaling.0.": "mask_decoder.deconv1.",
                "mask_decoder.output_upscaling.1.": "mask_decoder.deconv1_norm.",
                "mask_decoder.output_upscaling.3.": "mask_decoder.deconv2.",
            }
            for old_prefix, new_prefix in mapping.items():
                if key.startswith(old_prefix):
                    target[new_prefix + key[len(old_prefix):]] = value
                    break
        elif key.startswith("mask_decoder.output_hypernetworks_mlps.0."):
            target[key.replace("output_hypernetworks_mlps.0.", "output_hypernetworks_mlps.")] = value

    return model.load_state_dict(target, strict=False)
