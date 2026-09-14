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
import torch.nn as nn
from .GlobalResidualBlock import GlobalResidual
from .EncoderDecoderBlock import Encoder, Decoder
from .MotionAttentionBlock import  MotionAttention


class LAPANet2D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # attention heads and filters per block
        att_heads = self.args.att_heads
        global_filters = self.args.global_filters
        filters = self.args.encoder_decoder_filters
        print('LAPANet has the following filter number in the encoder and decoder levels:', filters)
        print('LAPANet has the following attention heads in the encoder and decoder levels:', att_heads)

        # training mode
        self.deep_supervision = self.args.deep_supervision
        print('training using deep supervision:', self.deep_supervision)
        self.return_translation = self.args.return_translation
        print('returning shift:', self.return_translation)
        self.shift_input =  self.args.shift_input
        print('shifting input:', self.shift_input)



        # Global Residual blocks
        self.global1 = GlobalResidual(dim=self.args.num_coils*4, dim_out=global_filters[0], groups=4, use_gca=True)
        self.global2 = GlobalResidual(dim=self.args.num_coils*4, dim_out=global_filters[1], groups=4, use_gca=True, pool=2)
        self.global3 = GlobalResidual(dim=self.args.num_coils*4, dim_out=global_filters[2], groups=4, use_gca=True, pool=4)
        self.global4 = GlobalResidual(dim=self.args.num_coils*4, dim_out=global_filters[3], groups=4, use_gca=True, pool=8)


        # Encoder/Decoder
        self.encoder1 = Encoder("first", 4, filters[0], att_heads[0])
        self.encoder2 = Encoder("intermediate", filters[0], filters[1], att_heads[1])
        self.encoder3 = Encoder("intermediate", filters[1], filters[2], att_heads[2])
        self.encoder4 = Encoder("intermediate", filters[2], filters[3], att_heads[3])
        self.bottleneck = Encoder("bottleneck", filters[3], filters[4], att_heads[4])
        self.decoder1 = Decoder(filters[4], filters[5], att_heads[5])
        self.decoder2 = Decoder(filters[5], filters[6], att_heads[6])
        self.decoder3 = Decoder(filters[6], filters[7], att_heads[7])
        self.decoder4 = Decoder(filters[7], filters[8], att_heads[8])

        # Estimation/Refinement
        self.motion1 = MotionAttention(filters[5], 2)
        self.motion2 = MotionAttention(filters[6], 2)
        self.motion3 = MotionAttention(filters[7], 2)
        self.motion4 = MotionAttention(filters[8], 2, scale_factor=1)

        # shift
        if self.return_translation:
            self.max = nn.MaxPool2d(kernel_size=5)
            self.conv_shift = nn.Conv2d(filters[4], 2, 1)


    def forward(self, k_ref, k_mov):
        # stack real and imaginary
        cat_ksp = torch.cat((k_ref.real, k_ref.imag, k_mov.real, k_mov.imag), dim=1).float()
        shifted_ksp = torch.fft.ifftshift(cat_ksp, dim=[-2,-1])  if self.shift_input else cat_ksp

        # tapering like operation
        scale_ksp_1 = self.global1(shifted_ksp)
        scale_ksp_2 = self.global2(shifted_ksp)
        scale_ksp_3 = self.global3(shifted_ksp)
        scale_ksp_4 = self.global4(shifted_ksp)

        # encoding
        x = self.encoder1(scale_ksp_1)
        skip1 = x
        x = self.encoder2(x, scale_ksp_2)
        skip2 = x
        x = self.encoder3(x, scale_ksp_3)
        skip3 = x
        x = self.encoder4(x, scale_ksp_4)
        skip4 = x
        x = self.bottleneck(x)
        skip5 = x

        # translation
        if self.return_translation:
            translation = self.max(skip5)
            translation = self.conv_shift(translation)
            translation = torch.squeeze(translation)

        # decoding
        x = self.decoder1(x, skip4)
        skip6=x
        x = self.decoder2(x, skip3)
        skip7 = x
        x = self.decoder3(x, skip2)
        skip8 = x
        x = self.decoder4(x, skip1)
        skip9 = x

        # estimation and refinement
        df1, df1_up = self.motion1(skip6, 0)
        df2, df2_up = self.motion2(skip7, df1_up)
        df3, df3_up = self.motion3(skip8, df2_up)
        df4, df4_up = self.motion4(skip9, df3_up)

        out_flo = [df1, df2, df3, df4_up] if self.deep_supervision else df4_up

        return (out_flo, translation) if self.return_translation else out_flo
