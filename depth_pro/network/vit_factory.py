# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Factory functions to build and load ViT models.


from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import timm
import torch
import torch.nn as nn

from .vit import (
    forward_features_eva_fixed,
    make_vit_b16_backbone,
    resize_patch_embed,
    resize_vit,
)

LOGGER = logging.getLogger(__name__)

def create_vit(
    name: str,
    params: dict,
    use_pretrained: bool = False,
    checkpoint_uri: str | None = None,
    use_grad_checkpointing: bool = False,
) -> nn.Module:
    """Create and load a VIT backbone module.

    Args:
    ----
        preset: The VIT preset to load the pre-defined config.
        use_pretrained: Load pretrained weights if True, default is False.
        checkpoint_uri: Checkpoint to load the wights from.
        use_grad_checkpointing: Use grandient checkpointing.

    Returns:
    -------
        A Torch ViT backbone module.

    """

    img_size = (params.img_size, params.img_size)
    patch_size = (params.patch_size, params.patch_size)

    if "eva02" in name:
        model = timm.create_model(params.timm_preset, pretrained=use_pretrained)
        model.forward_features = types.MethodType(forward_features_eva_fixed, model)
    else:
        model = timm.create_model(
            params.timm_preset, pretrained=use_pretrained, dynamic_img_size=True
        )
    model = make_vit_b16_backbone(
        model,
        encoder_feature_dims=params.encoder_feature_dims,
        encoder_feature_layer_ids=params.encoder_feature_layer_ids,
        vit_features=params.embed_dim,
        use_grad_checkpointing=use_grad_checkpointing,
    )
    if params.patch_size != params.timm_patch_size:
        model.model = resize_patch_embed(model.model, new_patch_size=patch_size)
    if params.img_size != params.timm_img_size:
        model.model = resize_vit(model.model, img_size=img_size)

    if checkpoint_uri is not None:
        state_dict = torch.load(checkpoint_uri, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=state_dict, strict=False
        )

        if len(unexpected_keys) != 0:
            raise KeyError(f"Found unexpected keys when loading vit: {unexpected_keys}")
        if len(missing_keys) != 0:
            raise KeyError(f"Keys are missing when loading vit: {missing_keys}")

    LOGGER.info(model)
    return model.model
