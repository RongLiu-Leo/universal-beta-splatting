#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        timestamp=0.0,
        data_device="cuda",
        resolution=None,
        image_path="",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.timestamp = timestamp
        self.resolution = resolution

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = image
        self.image_width = resolution[0]
        self.image_height = resolution[1]

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale))
            .transpose(0, 1)
            .to(self.data_device)
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(self.data_device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def to(self, device=None):
        """Move all tensor attributes of this Camera to the specified device (default: cuda)."""
        if device is None:
            device = torch.device("cuda")

        for name, value in self.__dict__.items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
            elif isinstance(value, list):
                setattr(
                    self,
                    name,
                    [v.to(device) if torch.is_tensor(v) else v for v in value],
                )
            elif isinstance(value, dict):
                setattr(
                    self,
                    name,
                    {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in value.items()
                    },
                )

        self.data_device = device
        return self

    def cuda(self):
        """Shortcut to move Camera tensors to CUDA."""
        return self.to(torch.device("cuda"))
