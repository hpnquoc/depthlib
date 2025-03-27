# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Depth Pro: Sharp Monocular Metric Depth in Less Than a Second


from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import torch
from torch import nn

from lib.depthlib.modelbase import ModelBase

from .network.decoder import MultiresConvDecoder
from .network.encoder import DepthProEncoder
from .network.fov import FOVNetwork
from .network.vit_factory import create_vit

class DepthPro(ModelBase):
    """DepthPro network."""

    def __init__(
        self,
        enc_conf: dict,
        dec_conf: dict,
        last_dims: tuple[int, int],
        use_fov_head: bool = True,
        fov_conf: Optional[dict] = None,
    ):
        """Initialize DepthPro.

        Args:
        ----
            encoder: The DepthProEncoder backbone.
            decoder: The MultiresConvDecoder decoder.
            last_dims: The dimension for the last convolution layers.
            use_fov_head: Whether to use the field-of-view head.
            fov_encoder: A separate encoder for the field of view.

        """
        super().__init__()

        self.encoder = self._init_enc(enc_conf)
        self.decoder = self._init_dec(dec_conf)
    
        dim_decoder = self.decoder.dim_decoder
        self.head = nn.Sequential(
            nn.Conv2d(
                dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(
                dim_decoder // 2,
                last_dims[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        # Set the final convolution layer's bias to be 0.
        self.head[4].bias.data.fill_(0)

        # Set the FOV estimation head.
        if use_fov_head:
            self.fov = self._init_fov(fov_conf)

    def _init_enc(self, enc_conf: dict) -> DepthProEncoder:
        return DepthProEncoder(**enc_conf)
    
    def _init_dec(self, dec_conf: dict) -> MultiresConvDecoder:
        return MultiresConvDecoder(**dec_conf)
    
    def _init_fov(self, fov_conf: dict) -> FOVNetwork:
        return FOVNetwork(**fov_conf)

    @property
    def img_size(self) -> int:
        """Return the internal image size of the network."""
        return self.encoder.img_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode by projection and fusion of multi-resolution encodings.

        Args:
        ----
            x (torch.Tensor): Input image.

        Returns:
        -------
            The canonical inverse depth map [m] and the optional estimated field of view [deg].

        """
        _, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        encodings = self.encoder(x)
        features, features_0 = self.decoder(encodings)
        canonical_inverse_depth = self.head(features)

        fov_deg = None
        if hasattr(self, "fov"):
            fov_deg = self.fov.forward(x, features_0.detach())

        return canonical_inverse_depth, fov_deg

    @torch.no_grad()
    def infer_image(
        self,
        x,
        transform = None,
        f_px: Optional[Union[float, torch.Tensor]] = None,
        interpolation_mode="bilinear",
    ) -> Mapping[str, torch.Tensor]:
        """Infer depth and fov for a given image.

        If the image is not at network resolution, it is resized to 1536x1536 and
        the estimated depth is resized to the original image resolution.
        Note: if the focal length is given, the estimated value is ignored and the provided
        focal length is use to generate the metric depth values.

        Args:
        ----
            x (torch.Tensor): Input image
            f_px (torch.Tensor): Optional focal length in pixels corresponding to `x`.
            interpolation_mode (str): Interpolation function for downsampling/upsampling. 

        Returns:
        -------
            Tensor dictionary (torch.Tensor): depth [m], focallength [pixels].

        """
        H, W = x.shape[:2]
        if transform is not None:
            x = transform(x).to(self.device)
        else:
            x = x

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # if len(x.shape) == 3:
        #     x = x.unsqueeze(0)
        # _, _, H, W = x.shape
        # resize = H != self.img_size or W != self.img_size

        # if resize:
        #     x = nn.functional.interpolate(
        #         x,
        #         size=(self.img_size, self.img_size),
        #         mode=interpolation_mode,
        #         align_corners=False,
        #     )

        canonical_inverse_depth, fov_deg = self.forward(x)
        if f_px is None:
            f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
        
        inverse_depth = canonical_inverse_depth * (W / f_px)
        f_px = f_px.squeeze()

        # if resize:
        inverse_depth = nn.functional.interpolate(
            inverse_depth, size=(H, W), mode=interpolation_mode, align_corners=False
        )

        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)

        return {
            "depth": depth.squeeze(),
            "focallength_px": f_px,
        }
