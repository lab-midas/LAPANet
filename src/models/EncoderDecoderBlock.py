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

from torch import nn
import torch
import torch.nn.functional as F

class ChannelIntegration(nn.Module):
    def __init__(self, out_channels, num_heads):
        super().__init__()
        self.attention_output = Attention(channels=out_channels, num_heads=num_heads)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.dilated_module = DilatedFusionModule(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x1 = self.conv1(x1)
        x2 = torch.add(x1, x)
        x3 = x2.permute(0, 2, 3, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 3, 1, 2)
        x3 = self.dilated_module(x3)
        x3 = torch.add(x2, x3)
        return x3

class DilatedFusionModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x1 = F.dropout(x1, 0.1)
        x2 = self.layer2(x)
        x2 = F.dropout(x2, 0.1)
        x3 = self.layer3(x)
        x3 = F.dropout(x3, 0.1)
        added = torch.add(x1, x2)
        added = torch.add(added, x3)
        x_out = self.layer4(added)
        x_out = F.dropout(x_out, 0.1)
        return x_out


class Encoder(nn.Module):
    def __init__(self, type, in_channels, out_channels, att_heads):
        super().__init__()
        self.type = type

        if ((self.type == "first") or (self.type == "bottleneck")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.integrate = ChannelIntegration(out_channels, att_heads)
        elif self.type == "intermediate":
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.integrate = ChannelIntegration(out_channels, att_heads)



    def forward(self, x, scale_img="none"):
        if ((self.type == "first") or (self.type == "bottleneck")):
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = self.conv1(x1)
            x1 = F.relu(self.conv2(x1))
            x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool2d(x1, (2, 2))
            out = self.integrate(x1)
        elif (self.type == "intermediate")  :
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = torch.cat((scale_img, x1), axis=1)
            x1 = F.relu(self.conv2(x1))
            x1 = F.relu(self.conv3(x1))
            x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool2d(x1, (2, 2))
            out = self.integrate(x1)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, scale_factor=2, filter_num=0):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels * 2 + filter_num, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.channel_integration = ChannelIntegration(out_channels, att_heads)

    def forward(self, x, skip):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.upsample(x1)
        x1 = F.relu(self.conv1(x1))
        x1 = torch.cat((skip, x1), axis=1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = F.dropout(x1, 0.3)
        out = self.channel_integration(x1)
        return out

class Attention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop
        self.conv_q = nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias, groups=channels)
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,groups=channels)
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,groups=channels)
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)
        self.attention = nn.MultiheadAttention(embed_dim=channels,bias=attention_bias, batch_first=True, num_heads=self.num_heads)

    def channelwise_projection(self, x, qkv):
        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)
        return proj

    def forward_conv(self, x):
        q = self.channelwise_projection(x, "q")
        k = self.channelwise_projection(x, "k")
        v = self.channelwise_projection(x, "v")
        return q, k, v

    def forward(self, x):
        q, k, v = self.forward_conv(x)
        q = q.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        k = k.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        v = v.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)
        MM,NN = x.shape[-2:]
        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], MM, NN)
        x1 = F.dropout(x1, self.proj_drop)
        return x1