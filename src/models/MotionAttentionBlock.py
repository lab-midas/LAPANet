#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Aya

script for Local All Pass Attention Network (LAPANet)

Developed at the University Hospital of Tübingen.
Copyright © 2026 University Hospital of Tübingen.

If you'd like to use or share this code, please get in touch with
Aya Ghoul <aya.ghoul@med.uni-tuebingen.de>.
"""

import torch
from torch import nn
from .GlobalResidualBlock import Identity
import torch.nn.functional as F


class MotionAttention(nn.Module):
    def __init__(self,  filters, out_channels=2, in_channels=2, scale_factor=2):
        super(MotionAttention, self).__init__()
        self.upsampling = Upsampling(filters, out_channels)
        self.refinement = EstimationRefinement(in_channels, scale_factor)

    def forward(self, x, x_up):
        x = self.upsampling(x)
        x_up = self.refinement(x + x_up)
        return x, x_up

class EstimationRefinement(nn.Module):
    def __init__(self, in_channels=2, scale_factor=2):
        super(EstimationRefinement, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, 3, padding='same')
        self.conv1_1 = nn.Conv2d(1, 1, 1, padding='same')
        self.conv2 = nn.Conv2d(in_channels, 1, 3, padding='same')
        self.conv2 = nn.Conv2d(in_channels, 1, 1, padding='same')
        self.conv2_1 = nn.Conv2d(1, 1, 1, padding='same')
        self.sigmoid = nn.Sigmoid()
        if scale_factor!=1:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.up = Identity()

    def forward(self, x):
        # Perform independent convolution on two channels
        weight_map1 = self.sigmoid(self.conv1_1(self.conv1(x)))
        weight_map2 = self.sigmoid(self.conv2_1(self.conv2(x)))

        # Multiply input channels by corresponding weight maps
        channel1_deformed = x[:, 0, None] * weight_map1
        channel2_deformed = x[:, 1, None] * weight_map2

        # Combine the deformed channels to form a new deformation field
        deformation_field = torch.cat([channel1_deformed, channel2_deformed], dim=1)
        deformation_field  = self.up(deformation_field)

        return deformation_field


class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        out = self.conv3(x1)

        return out