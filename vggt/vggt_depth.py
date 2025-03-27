# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from .models.aggregator import Aggregator
from .heads.dpt_head import DPTHead

from lib.depthlib.modelbase import ModelBase


class VGGT(ModelBase, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.amp.autocast('cuda',enabled=False):

            depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        predictions["images"] = images

        return predictions

    @torch.no_grad()
    def infer_image(self, raw_image, transform=None):
        h, w = raw_image.shape[:2]
        if transform is not None:
            image = transform(raw_image).to(self.device)
        else:
            image = raw_image

        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        # _, _, H, W = image.shape

        predictions = self.forward(image)

        depth = predictions["depth"].squeeze(-1)
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return {
            'images': raw_image,
            "depth": depth,
            "depth_conf": predictions["depth_conf"]
        }