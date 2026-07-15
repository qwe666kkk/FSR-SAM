# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type, Optional, Dict

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        norm : Type[nn.Module] = nn.BatchNorm2d,
        act : Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(1, transformer_dim)

        # Two-stage upscaling with an injection point between stages
        self.deconv1 = nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2)
        self.deconv1_norm = LayerNorm2d(transformer_dim // 4)
        self.deconv1_act = activation()

        self.deconv2 = nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2)
        self.deconv2_act = activation()

        # FiLM residual gating scalars (applied with semantic guidance map)
        self.film_gamma = nn.Parameter(torch.zeros(1))
        self.film_beta = nn.Parameter(torch.zeros(1))

        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sem_g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, upscaled_embedding, mids = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sem_g=sem_g,
        )

        # Select the correct mask or masks for output

        # Prepare output
        return masks, upscaled_embedding, mids

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sem_g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(image_embeddings.shape[0],-1,-1)

        src = image_embeddings# + prompt_token# + prompt_token
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, output_tokens)

        # Use the transformer-updated mask token, as in the SAM mask decoder.
        # Using the input embedding here would bypass token/image interaction.
        hs = hs[:, 1:, :]

        hs = self.output_hypernetworks_mlps(hs)

        # Upscale mask embeddings in two stages and predict masks
        src = src.transpose(1, 2).view(b, c, h, w).contiguous()

        # Stage 1
        x = self.deconv1(src)
        x = self.deconv1_norm(x)
        x = self.deconv1_act(x)

        # Inject semantic guidance via FiLM residual gating
        if sem_g is not None:
            sg = F.interpolate(sem_g, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = x * (1 + self.film_gamma * sg) + self.film_beta * sg

        # Stage 2
        x = self.deconv2(x)
        x = self.deconv2_act(x)
        upscaled_embedding = x

        bb, cc, hh, ww = upscaled_embedding.shape
        masks_hyper = (hs @ upscaled_embedding.view(bb, cc, hh * ww)).view(bb, -1, hh, ww)

        if masks_hyper.shape[1] != 1:
            base_logit = masks_hyper.mean(dim=1, keepdim=True)
        else:
            base_logit = masks_hyper
        return base_logit, upscaled_embedding, {}


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
