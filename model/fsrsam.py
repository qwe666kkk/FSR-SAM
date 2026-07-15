from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .fftconv_min import FFTConv2d
from .freq_addons import BandFiLM2d
from .flow import FSRSAMFlow, build_flow
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .paper_alias import DR_BE, FR, MCFM, OCM, SFEM
from .transformer import TwoWayTransformer

class FSRSAM(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        norm: Type[nn.Module] = nn.BatchNorm2d,
        act: Type[nn.Module] = nn.ReLU,
        enable_loc: bool = True,
        fr_mode: str = "full",
        fr_decoder_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.pe_layer = PositionEmbeddingRandom(256 // 2)

        self.image_embedding_size = [img_size // 16, img_size // 16]
        self.img_size = img_size

        self.image_encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=img_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
        )

        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            norm=norm,
            act=act,
        )

        self.deep_feautre_conv = nn.Sequential(
            nn.Conv2d(256, 32, 3, padding=1, bias=False),
            norm(32),
            act(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        )
        self.mcfm = MCFM(norm=norm, act=act)

        self.dr_be = DR_BE(
            img_dim=32, feature_dim=32, norm=norm, act=act
        )
        self.fallback_reduce = nn.Conv2d(768, 256, kernel_size=1, bias=False)

        # Frequency-aware add-ons
        self.band_film = BandFiLM2d(
            in_channels=256, high_ratio=0.55, smooth=True, enable=True
        )
        self.ocm = OCM(in_channels=256, enable=True)
        self.fr_decoder = FR(
            in_channels=32,
            with_edge_head=False,
            enable=True,
            mode=fr_decoder_mode or fr_mode,
        )

        # Semantic fusion & enhancement
        self.sfem = SFEM(in_ch=256)

        # ---------------- FFTConv-based multi-level foreground locator ----------------
        # Legacy 1_16/1_8/1_4 names are retained in parameter keys for checkpoint
        # compatibility. The four SAM ViT outputs share one token resolution; these
        # branches consume different encoder depths, not true spatial pyramid scales.
        self.enable_loc = bool(enable_loc)
        self.loc_head_1_16 = LocHeadFFTConv(
            in_ch=256, mid_ch=64, kernel_size=51, name="loc_1_16"
        )
        self.loc_head_1_8 = LocHeadFFTConv(
            in_ch=256, mid_ch=64, kernel_size=31, name="loc_1_8"
        )
        # Eq. (FGPM P_loc): two normalized level weights for F3' and F4'.
        # Legacy head names are retained for checkpoint readability: 1_16 is
        # F4' and 1_8 is F3'. The third legacy head is not part of the paper path.
        self.loc_level_logits = nn.Parameter(torch.zeros(2))

    def set_frequency_ablation(
        self,
        mode: str,
        stages: Iterable[str] = ("decoder",),
    ) -> None:
        """Switch FR computation without changing checkpoint parameter keys."""
        stages = set(stages)
        unknown = stages.difference({"decoder"})
        if unknown:
            raise ValueError(f"Unknown frequency stages: {sorted(unknown)}")
        if "decoder" in stages:
            self.fr_decoder.head.set_mode(mode)

    def set_mba_enabled(self, enabled: bool) -> None:
        """Enable/disable all MBA adapters for encoder-only spectral analysis."""
        for block in self.image_encoder.blocks:
            adapter = getattr(block, "adapter", None)
            if adapter is not None and hasattr(adapter, "enable"):
                adapter.enable = bool(enabled)

    def frequency_config(self) -> Dict[str, str]:
        return {
            "decoder": str(self.fr_decoder.head.mode),
        }

    def paper_modules(self) -> Dict[str, Tuple[nn.Module, ...]]:
        """Return non-owning module groups using the names in the paper.

        The returned tuples reference modules already registered on this model;
        they do not alter the module tree or checkpoint parameter names.
        """

        mba_modules = tuple(
            block.adapter
            for block in self.image_encoder.blocks
            if getattr(block, "adapter", None) is not None
        )
        return {
            "MBA": mba_modules,
            "OCM": (self.band_film, self.ocm),
            "SFEM": (self.mcfm, self.sfem),
            "SAM Decoder": (self.mask_decoder,),
            "FGPM": (
                self.loc_head_1_16,
                self.loc_head_1_8,
                self.fr_decoder,
                self.deep_feautre_conv,
                self.dr_be,
            ),
        }

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward_flow(self, img: torch.Tensor) -> FSRSAMFlow:
        """Run FSR-SAM and return stages named after the paper modules.

        The regular :meth:`forward` method intentionally keeps returning
        ``(final_logits, coarse_logits, auxiliary)`` so existing training,
        evaluation, and checkpoint code remains compatible.
        """

        final_logits, coarse_logits, auxiliary = self.forward(
            img, return_intermediates=True
        )
        return build_flow(final_logits, coarse_logits, auxiliary)

    def _normalize_last_feat(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Normalize the last feature from the encoder to shape (B,256,t,t).
        """
        if feat.dim() == 4:
            # (B,C,H,W) or (B,H,W,C)
            if feat.shape[1] == 256:
                return feat.contiguous()
            if feat.shape[1] == 768:
                return self.fallback_reduce(feat)
            if feat.shape[-1] == 256:
                return feat.permute(0, 3, 1, 2).contiguous()
            if feat.shape[-1] == 768:
                feat = feat.permute(0, 3, 1, 2).contiguous()
                return self.fallback_reduce(feat)
            raise RuntimeError(
                f"[fsrsam] Got last feature with shape {tuple(feat.shape)}, "
                f"expect channels in {{256,768}}."
            )
        if feat.dim() == 3:
            # (B, N, C) or (B, C, N)
            if feat.shape[-1] in (256, 768):
                C = feat.shape[-1]
                t = (
                    self.img_size // 16
                    if hasattr(self, "img_size")
                    else int(feat.shape[1] ** 0.5)
                )
                feat = feat.transpose(1, 2).reshape(feat.shape[0], C, t, t).contiguous()
                if C == 768:
                    feat = self.fallback_reduce(feat)
                return feat
            if feat.shape[1] in (256, 768):
                C = feat.shape[1]
                t = (
                    self.img_size // 16
                    if hasattr(self, "img_size")
                    else int(feat.shape[-1] ** 0.5)
                )
                feat = feat.reshape(feat.shape[0], C, t, t).contiguous()
                if C == 768:
                    feat = self.fallback_reduce(feat)
                return feat
            raise RuntimeError(
                f"[fsrsam] Got last feature with shape {tuple(feat.shape)}, "
                f"expect last/second dim in {{256,768}}."
            )
        raise RuntimeError(
            f"[fsrsam] Got last feature with shape {tuple(feat.shape)}, "
            f"expect (B,256,t,t). Please check image_encoder outputs."
        )

    def forward(self, img: torch.Tensor, return_intermediates: bool = False):
        # Encoder + MBA (inside ImageEncoderViT)
        features_list = self.image_encoder(img)

        analysis = {} if return_intermediates else None
        if analysis is not None:
            analysis["encoder_features_before_post_refinement"] = tuple(features_list)

        feat = self._normalize_last_feat(features_list[-1])

        if analysis is not None:
            analysis["encoder_f4_before_bandfilm"] = feat

        # BandFiLM modulation on 1/16 features
        feat = self.band_film(feat)

        if analysis is not None:
            analysis["encoder_f4_after_bandfilm"] = feat

        if not isinstance(features_list, list):
            features_list = list(features_list)
        features_list[-1] = feat

        while len(features_list) < 4:
            features_list.append(features_list[-1])
        if len(features_list) > 4:
            features_list = features_list[-4:]

        if len(features_list) >= 2:
            low_res = features_list[-2]
            high_res = feat
            if high_res.shape[-2:] != low_res.shape[-2:]:
                high_res = F.interpolate(
                    high_res,
                    size=low_res.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            features_list[-2] = self.ocm(low_res, high_res)

        if analysis is not None:
            analysis["encoder_features_after_ocm"] = tuple(features_list)

        # ---------------- Multi-level FFTConv foreground locator (optional) ----------------
        loc_multi = None
        loc_1_16 = None
        loc_1_8 = None
        if self.enable_loc:
            # Paper-strict locator: L4=Hloc(F4') and L3=Hloc(F3').
            f_1_16 = feat
            if len(features_list) >= 2:
                f_1_8 = features_list[-2]
            else:
                f_1_8 = F.interpolate(
                    f_1_16, scale_factor=2.0, mode="bilinear", align_corners=False
                )
            loc_1_16 = self.loc_head_1_16(f_1_16)
            loc_1_8 = self.loc_head_1_8(f_1_8)

            H16, W16 = loc_1_16.shape[-2], loc_1_16.shape[-1]
            loc_1_8_to_16 = F.interpolate(
                loc_1_8, size=(H16, W16), mode="bilinear", align_corners=False
            )
            omega = torch.softmax(self.loc_level_logits, dim=0)
            loc_fused = omega[0] * loc_1_8_to_16 + omega[1] * loc_1_16
            loc_multi = torch.sigmoid(loc_fused)

        # Deep feature branch (for DR-BE)
        deep_feature = self.deep_feautre_conv(feat)  # 256 -> 32

        # MCFM: multi-level fusion
        img_feature = self.mcfm(features_list)
        if analysis is not None:
            analysis["mcfm_output"] = img_feature
        # SFEM: semantic fusion + guidance
        img_feature, sem_g2 = self.sfem(img_feature, features_list)
        if analysis is not None:
            analysis["sfem_output"] = img_feature
            analysis["sfem_guidance"] = sem_g2
        img_pe = self.get_dense_pe()
        # Eq. (SFEM): S_m alone drives the SAM Decoder. P_loc is deliberately
        # excluded here and is used only by the final FGPM prediction stage.
        sem_g = sem_g2
        if analysis is not None:
            analysis["semantic_guidance"] = sem_g

        coarse_mask, feature, _ = self.mask_decoder(
            img_feature,
            img_pe,
            sem_g=sem_g,
        )
        if analysis is not None:
            analysis["decoder_before_fr"] = feature
        # FR: frequency refinement at decoder stage
        feature, _ = self.fr_decoder(feature)
        if analysis is not None:
            analysis["decoder_after_fr"] = feature
        coarse_mask = F.interpolate(
            coarse_mask,
            [self.img_size, self.img_size],
            mode="bilinear",
            align_corners=False,
        )

        # DR-BE: detail refinement
        mask = self.dr_be(img, feature, deep_feature, loc_prior=loc_multi)

        aux = {}
        # Expose multi-level foreground priors for training-time loss / weighting
        if loc_multi is not None:
            aux["loc_f4"] = loc_1_16
            aux["loc_f3"] = loc_1_8
            aux["loc_multi"] = loc_multi
        aux["sem_g_all"] = sem_g
        if analysis is not None:
            analysis["foreground_localization"] = loc_multi
            analysis["coarse_logits"] = coarse_mask
            analysis["final_logits"] = mask
            aux["analysis"] = analysis

        return mask, coarse_mask, aux


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class LocHeadFFTConv(nn.Module):
    """
    Single-level FFTConv-based foreground locator.
    Operates on a feature map (B, C, H, W) and outputs a probability map (B,1,H,W).
    """

    def __init__(
        self,
        in_ch: int = 256,
        mid_ch: int = 64,
        kernel_size: int = 31,
        name: str = "loc",
    ) -> None:
        super().__init__()
        self.name = name
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
        )
        self.fft_block = nn.Sequential(
            FFTConv2d(
                mid_ch, mid_ch, kernel_size=kernel_size, padding="same", groups=mid_ch
            ),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
        )
        self.out_conv = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.fft_block(x)
        x = self.out_conv(x)
        # Return a response logit. The sigmoid is applied once, after normalized
        # level fusion, exactly as in the paper's P_loc equation.
        return x

