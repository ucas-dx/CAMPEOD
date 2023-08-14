#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# @Project : CAMPEOD
# @Time    : 2023/8/10 11:50
# @Author  : Deng xun
# @Email   : 38694034@qq.com
# @File    : net.py
# @Software: PyCharm 
# -------------------------------------------------------------------------------
"""
Constructs a swin_v2_tiny architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/pdf/2111.09883>`_.

        From PyTorch:

        Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
        Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
        Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
        Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
        Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
        Copyright (c) 2011-2013 NYU                      (Clement Farabet)
        Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
        Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
        Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

        From Caffe2:

        Copyright (c) 2016-present, Facebook Inc. All rights reserved.

        All contributions by Facebook:
        Copyright (c) 2016 Facebook Inc.

        All contributions by Google:
        Copyright (c) 2015 Google Inc.
        All rights reserved.

        All contributions by Yangqing Jia:
        Copyright (c) 2015 Yangqing Jia
        All rights reserved.

        All contributions by Kakao Brain:
        Copyright 2019-2020 Kakao Brain

        All contributions by Cruise LLC:
        Copyright (c) 2022 Cruise LLC.
        All rights reserved.

        All contributions from Caffe:
        Copyright(c) 2013, 2014, 2015, the respective contributors
        All rights reserved.

        All other contributions:
        Copyright(c) 2015, 2016 the respective contributors
        All rights reserved.

        Caffe2 uses a copyright model similar to Caffe: each contributor holds
        copyright over their contributions to Caffe2. The project versioning records
        all such contribution and copyright details. If a contributor wants to further
        mark their specific copyright on a particular contribution, they should
        indicate their copyright solely in the commit message of the change when it is
        committed.

        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright
           notice, this list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright
           notice, this list of conditions and the following disclaimer in the
           documentation and/or other materials provided with the distribution.

        3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
           and IDIAP Research Institute nor the names of its contributors may be
           used to endorse or promote products derived from this software without
           specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
        ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
        LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
        CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
        INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
        CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
        ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
        POSSIBILITY OF SUCH DAMAGE.
"""

import math
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
import torchsummary
from torch import nn, Tensor

from torchvision.ops.misc import MLP, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

import model
from model import MultiheadAttention

__all__ = [
    "SwinTransformer",
    "Swin_T_Weights",
    "Swin_S_Weights",
    "Swin_B_Weights",
    "Swin_V2_T_Weights",
    "Swin_V2_S_Weights",
    "Swin_V2_B_Weights",
    "swin_t",
    "swin_s",
    "swin_b",
    "swin_v2_t",
    "swin_v2_s",
    "swin_v2_b",
]




def _patch_merging_pad(x: torch.Tensor) -> torch.Tensor:
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
    x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
    x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
    x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
    return x


torch.fx.wrap("_patch_merging_pad")


def _get_relative_position_bias(
        relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


torch.fx.wrap("_get_relative_position_bias")


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        _log_api_usage_once(self)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        return x


class PatchMergingV2(nn.Module):
    """Patch Merging Layer for Swin Transformer V2.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        _log_api_usage_once(self)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)  # difference

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        x = self.norm(x)
        return x


def shifted_window_attention(
        input: Tensor,
        qkv_weight: Tensor,
        proj_weight: Tensor,
        relative_position_bias: Tensor,
        window_size: List[int],
        num_heads: int,
        shift_size: List[int],
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        qkv_bias: Optional[Tensor] = None,
        proj_bias: Optional[Tensor] = None,
        logit_scale: Optional[torch.Tensor] = None,
        training: bool = True,
) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length: 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0]: h[1], w[0]: w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention")


class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
            self,
            dim: int,
            window_size: List[int],
            shift_size: List[int],
            num_heads: int,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            attention_dropout: float = 0.0,
            dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )


class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    """
    See :func:`shifted_window_attention_v2`.
    """

    def __init__(
            self,
            dim: int,
            window_size: List[int],
            shift_size: List[int],
            num_heads: int,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            attention_dropout: float = 0.0,
            dropout: float = 0.0,
    ):
        super().__init__(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )
        if qkv_bias:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length: 2 * length].data.zero_()

    def define_relative_position_bias_table(self):
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
                torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        )
        self.register_buffer("relative_coords_table", relative_coords_table)

    def get_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = _get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
            training=self.training,
        )


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: List[int],
            shift_size: List[int],
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            stochastic_depth_prob: float = 0.0,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()
        _log_api_usage_once(self)

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class SwinTransformerBlockV2(SwinTransformerBlock):
    """
    Swin Transformer V2 Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttentionV2.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: List[int],
            shift_size: List[int],
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            stochastic_depth_prob: float = 0.0,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_layer: Callable[..., nn.Module] = ShiftedWindowAttentionV2,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
        )

    def forward(self, x: Tensor):
        # Here is the difference, we apply norm after the attention in V2.
        # In V1 we applied norm before the attention.
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x


class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
            self,
            patch_size: List[int],
            embed_dim: int,
            depths: List[int],
            num_heads: List[int],
            window_size: List[int],
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            stochastic_depth_prob: float = 0.1,
            num_classes: int = 1000,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            block: Optional[Callable[..., nn.Module]] = None,
            downsample_layer: Callable[..., nn.Module] = PatchMerging,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2 ** i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


def _swin_transformer(
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        stochastic_depth_prob: float,
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> SwinTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "categories": _IMAGENET_CATEGORIES,
}


class Swin_T_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_t-704ceda3.pth",
        transforms=partial(
            ImageClassification, crop_size=224, resize_size=232, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 28288354,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.474,
                    "acc@5": 95.776,
                }
            },
            "_ops": 4.491,
            "_file_size": 108.19,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class Swin_S_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_s-5e29d889.pth",
        transforms=partial(
            ImageClassification, crop_size=224, resize_size=246, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 49606258,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.196,
                    "acc@5": 96.360,
                }
            },
            "_ops": 8.741,
            "_file_size": 189.786,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class Swin_B_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_b-68c6b09e.pth",
        transforms=partial(
            ImageClassification, crop_size=224, resize_size=238, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 87768224,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.582,
                    "acc@5": 96.640,
                }
            },
            "_ops": 15.431,
            "_file_size": 335.364,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class Swin_V2_T_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
        transforms=partial(
            ImageClassification, crop_size=256, resize_size=260, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 28351570,
            "min_size": (256, 256),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.072,
                    "acc@5": 96.132,
                }
            },
            "_ops": 5.94,
            "_file_size": 108.626,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class Swin_V2_S_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth",
        transforms=partial(
            ImageClassification, crop_size=256, resize_size=260, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 49737442,
            "min_size": (256, 256),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.712,
                    "acc@5": 96.816,
                }
            },
            "_ops": 11.546,
            "_file_size": 190.675,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class Swin_V2_B_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_v2_b-781e5279.pth",
        transforms=partial(
            ImageClassification, crop_size=256, resize_size=272, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 87930848,
            "min_size": (256, 256),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 84.112,
                    "acc@5": 96.864,
                }
            },
            "_ops": 20.325,
            "_file_size": 336.372,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


#@register_model()
@handle_legacy_interface(weights=("pretrained", Swin_T_Weights.IMAGENET1K_V1))
def swin_t(*, weights: Optional[Swin_T_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_tiny architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_T_Weights
        :members:
    """
    weights = Swin_T_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        weights=weights,
        progress=progress,
        **kwargs,
    )


#@register_model()
@handle_legacy_interface(weights=("pretrained", Swin_S_Weights.IMAGENET1K_V1))
def swin_s(*, weights: Optional[Swin_S_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_small architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_S_Weights
        :members:
    """
    weights = Swin_S_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.3,
        weights=weights,
        progress=progress,
        **kwargs,
    )


#@register_model()
@handle_legacy_interface(weights=("pretrained", Swin_B_Weights.IMAGENET1K_V1))
def swin_b(*, weights: Optional[Swin_B_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_base architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_B_Weights
        :members:
    """
    weights = Swin_B_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        weights=weights,
        progress=progress,
        **kwargs,
    )


#@register_model()
@handle_legacy_interface(weights=("pretrained", Swin_V2_T_Weights.IMAGENET1K_V1))
def swin_v2_t(*, weights: Optional[Swin_V2_T_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_tiny architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/pdf/2111.09883>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_V2_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_V2_T_Weights
        :members:
    """
    weights = Swin_V2_T_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )


#@register_model()
@handle_legacy_interface(weights=("pretrained", Swin_V2_S_Weights.IMAGENET1K_V1))
def swin_v2_s(*, weights: Optional[Swin_V2_S_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_small architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/pdf/2111.09883>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_V2_S_Weights
        :members:
    """
    weights = Swin_V2_S_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )


#@register_model()
@handle_legacy_interface(weights=("pretrained", Swin_V2_B_Weights.IMAGENET1K_V1))
def swin_v2_b(*, weights: Optional[Swin_V2_B_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_base architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/pdf/2111.09883>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_V2_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_V2_B_Weights
        :members:
    """
    weights = Swin_V2_B_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )


class decodeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decodeConv, self).__init__()
        self.quadruple_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2 * out_channels, kernel_size=3, padding="same", bias=True),
            nn.BatchNorm2d(2 * out_channels),
            nn.GELU(),
            nn.Conv2d(2 * out_channels, 1 * out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(1 * out_channels),
            nn.GELU(),
            nn.Conv2d(1 * out_channels, 1 * out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(1 * out_channels),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x1 = self.quadruple_conv1(x)
        return x1


class swinv2encode1(nn.Module):
    """
    Input :3，512，512->128,128,outchangel
    changel=128
    """

    def __init__(self):
        super(swinv2encode1, self).__init__()
        model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children()))[0]
        model[0][0] = nn.Conv2d(3, 128, kernel_size=(2, 2), stride=(2, 2))
        self.modeldown = model[0]
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x_128 = self.modeldown(x)
        x_128 = self.model[1][:2](x_128)
        x_64 = self.model[2](x_128)
        x_64 = self.model[3][:2](x_64)
        x_32 = self.model[4](x_64)
        x_32 = self.model[5][:6](x_32)
        return x_128, x_64, x_32


class swinv2encode2(nn.Module):
    """
    Input :3，512，512->128,128,outchangel
    """

    def __init__(self):
        super(swinv2encode2, self).__init__()
        model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children()))[0]
        self.model = model
        # modeldown = self.model[0]
        model[0][0] = nn.Conv2d(3, 128, kernel_size=(8, 8), stride=(8, 8))
        self.modeldown = model[0]
        self.modelswin = model[1][:2]
        # for param in self.model.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        x_128 = self.modeldown(x)
        x_128 = self.modelswin(x_128)
        # x_64 = self.model_train[2](x_128)
        # x_64 = self.model_train[3][:2](x_64)
        # x_32 = self.model_train[4](x_64)
        # x_32 = self.model_train[5][:6](x_32)
        return x_128  # , x_64, x_32


class up_feature(nn.Module):
    def __init__(self, inc, outc, scale, channel_last=True, out_channel_last=True):
        super(up_feature, self).__init__()
        self.channel_last = channel_last
        self.outchannel_last = out_channel_last
        self.quadruple_conv1 = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=1))
        self.layer_norm = nn.LayerNorm(outc)
        self.act = nn.GELU()
        # self.upsam = nn.ConvTranspose2d(inc,outc,kernel_size=scale,stride=scale)
        self.upsam = nn.Sequential(nn.Upsample(scale_factor=scale, mode="bilinear"),
                                   nn.Conv2d(inc, outc, kernel_size=1, stride=1), nn.GELU())

    def forward(self, x):
        if self.channel_last == True:
            x = x.permute(0, 3, 1, 2)
        else:
            pass
        x = self.upsam(x)
        # x=self.quadruple_conv1(x).permute(0,2,3,1)
        # x=self.layer_norm(x).permute(0,3,1,2)
        # x=self.act(x)
        if self.outchannel_last == True:
            x = x.permute(0, 2, 3, 1)
        else:
            pass
        return x


class up_sample(nn.Module):
    def __init__(self, inc, outc, scale):
        super(up_sample, self).__init__()
        self.up_conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=1)
                                     , nn.BatchNorm2d(outc), nn.GELU())
        # self.upsam=nn.Upsample(scale_factor=scale,mode='bilinear',align_corners=True)
        self.upsam = nn.Sequential(nn.Upsample(scale_factor=scale, mode="bilinear"),
                                   nn.Conv2d(inc, outc, kernel_size=1, stride=1), nn.GELU())

    def forward(self, x):
        x = self.upsam(x)
        x = self.up_conv(x)
        return x


class mering_a(nn.Module):
    def __init__(self, inc, outc):
        super(mering_a, self).__init__()
        self.conv3 = nn.Sequential(nn.Conv2d(inc, inc, kernel_size=3, padding="same", stride=1)
                                   , nn.BatchNorm2d(inc), nn.GELU(),
                                   nn.Conv2d(inc, outc, kernel_size=3, padding="same", stride=1)
                                   , nn.BatchNorm2d(outc)
                                   )
        self.conv1 = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding="same"), nn.BatchNorm2d(outc))
        # self.out=nn.Sequential(nn.Conv2d(2*inc,outc,kernel_size=1,stride=1,padding="same"),nn.BatchNorm2d(outc), nn.GELU())

    def forward(self, x, y):
        x, y = x.transpose(-1, -3), y.transpose(-1, -3)
        x_cat = torch.cat([x, y], dim=1)
        x1 = self.conv3(x_cat)
        x2 = self.conv1(x_cat)
        # out=torch.cat([x1,x2],dim=1)
        # out=self.out(out)
        return x2.transpose(-1, -3)


class mering_b(nn.Module):
    def __init__(self, inc, outc):
        super(mering_b, self).__init__()
        self.Linear = nn.Sequential(nn.Linear(inc, inc), nn.LayerNorm(inc), nn.GELU(), nn.Linear(inc, outc))

    def forward(self, x, y):
        x = torch.cat([x, y], dim=3)
        x = self.Linear(x)
        return x


class mering_c(nn.Module):
    def __init__(self, inc, outc):
        super(mering_c, self).__init__()
        self.up_conv = nn.Sequential(nn.Conv2d(inc, 2 * inc, kernel_size=1, padding="same", stride=1)
                                     , nn.BatchNorm2d(2 * inc), nn.GELU(),
                                     nn.Conv2d(2 * inc, outc, kernel_size=3, padding="same", stride=1)
                                     , nn.BatchNorm2d(outc)
                                     )
        self.Linear = nn.Sequential(nn.Linear(inc, inc), nn.LayerNorm(inc), nn.GELU(), nn.Linear(inc, outc))

    def forward(self, x, y):
        # x,y=x.transpose(-1,-3),y.transpose(-1,-3)
        x = torch.cat([x, y], dim=3)
        # x = self.up_conv(x)
        x = self.Linear(x)
        return x


class swin_att(nn.Module):
    """
    input 3，512，512->256,256,128
                    ->128,128,256
                    ->64, 64, 512
    """

    def __init__(self):
        super(swin_att, self).__init__()
        model = swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children()))[0][5][0:1]
        # print(model)
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.model(x)
        return out


class CrossAttention(nn.Module):
    def __init__(self, channel1, channel2, dim_feedforward=512, dropout=0.1,add=False):
        super().__init__()
        self.add=add
        # 注意力得分计算的线性变换
        self.q_lin = nn.Linear(channel1, channel2)
        self.k_lin = nn.Linear(channel2, channel2)
        self.v_lin = nn.Linear(channel2, channel2)

        # 多头注意力机制
        self.multihead_attn = model.MultiheadAttention(channel2, num_heads=8, dropout=dropout, batch_first=True)

        # 前馈网络
        self.linear1 = nn.Linear(channel2, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, channel2)

    def forward(self, x1, x2):
        # 输入形状变换

        x1, x2 = x1.permute(0, 3, 1, 2), x2.permute(0, 3, 1, 2)
        batch_size, _, height1, width1 = x1.shape
        _, _, height2, width2 = x2.shape
        x1 = x1.view(batch_size, -1, height1 * width1).permute(0, 2, 1)
        x2 = x2.view(batch_size, -1, height2 * width2).permute(0, 2, 1)

        q = self.q_lin(x1)
        k = self.k_lin(x2)
        v = self.v_lin(x2)

        # 多头注意力层
        attn_output, _ = self.multihead_attn(q, k, v)

        # 前馈网络
        output = F.relu(self.linear1(attn_output))
        output = self.dropout(output)
        output = self.linear2(output)

        # 输出形状变换
        output = output.permute(0, 2, 1).view(batch_size, -1, height2, width2)
        return output.permute(0, 2, 3, 1)+int(self.add)*k.permute(0, 2, 1).view(batch_size, -1, height2, width2).permute(0, 2, 3,1)


class CampeodNet(nn.Module):
    def __init__(self):
        super(CampeodNet, self).__init__()
        self.rgb = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=3, stride=1, kernel_size=3, padding="same"),
                                 nn.BatchNorm2d(3))
        self.swin64 = swinv2encode2()
        self.swindencode1 = swinv2encode1()
        # self.swin_mering=swin_att()
        self.up_feature_64_128 = up_feature(512, 256, 2, channel_last=True, out_channel_last=True)
        # self.decode32=decodeConv(512,512)#512,32,32->512,32,32
        # self.up_feature_32_512 = up_feature(512, 64, 16,channel_last=False,out_channel_last=False)
        self.up_feature_128_256 = up_feature(256, 128, 2, channel_last=True, out_channel_last=True)
        # self.decode64 = decodeConv(256, 256)
        # self.up_feature_64_512 = up_feature(256, 64,8,channel_last=False,out_channel_last=False)
        # self.cat = mering_a(256 + 128, 256)
        self.decode256 = decodeConv(128, 128)
        self.crossatt = CrossAttention(128, 512)

        # self.decode256 = decodeConv(128, 128)
        self.up_feature_256_512 = up_feature(128, 64, 2, channel_last=False,
                                             out_channel_last=False)

        self.out = nn.Sequential(nn.Conv2d(64, 64, 1, 1), nn.BatchNorm2d(64),
                                 nn.GELU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        x = self.rgb(x)
        x_swin64 = self.swin64(x)
        x_swin = self.swindencode1(x)
        x_256, x_128, x_64 = x_swin
        x_64_1 = self.crossatt(x_swin64, x_64) + x_64
        x_64up128 = self.up_feature_64_128(x_64_1)
        x128 = x_128 + x_64up128
        # x64 = torch.cat([x64, x_swin128], dim=3)
        # x64=self.swin_mering(x64)
        # x64=self.cat(x64,x_swin128)
        # x64=self.crossatt(x_swin128,x64)
        x_128up256 = self.up_feature_128_256(x128)
        x256 = self.decode256(x_256 + x_128up256)
        x_256_512 = self.up_feature_256_512(x256)
        # x_256=self.decode256(x_256.transpose(-1,-3))
        # x256=torch.cat([x_128_256,x_256],dim=1)
        # x512=self.up_feature_256_512(x256)

        # x64=self.decode64(x64)
        # x_64_512=self.up_feature_64_512(x64)
        # x32=self.decode32(x_32)
        # x_32_512=self.up_feature_32_512(x32)
        # x_fuse512=torch.cat([x_32_512,x_64_512,x_128_512],dim=1)
        #
        # #**********************
        # x_128_1, x_64_1, x_32_1 =self.swindencode2(x)
        # x_32up64_1 = self.up_feature_32_64_1(x_32_1+x_32)
        # x64_1 = x_64_1 + x_32up64_1
        # x_64up128_1 = self.up_feature_64_128_1(x64_1+x64)
        # x128_1 = self.decode128_1(x_128_1 + x_64up128_1)
        # x_128_512_1 = self.up_feature_128_512_1(x128_1+x_128)
        # x_up32_1 = self.up_feature_32_1(x_32_1)+x_up32  # 512,32,32
        # x32_up64_1 = self.up_sample_64_1(x_up32_1)  # 512,64,64
        # x_up64_1 = self.up_feature_64_1(x_64_1) + x32_up64_1+x32_up64
        # x64_up128_1 = self.up_sample_128_1(x_up64_1)  # 256,128,128
        # x_up128_1 = self.up_feature_128_1(x_128_1) + x64_up128_1+x64_up128
        # x256_up512_1 = self.up_sample_256_1(x_up128_1) # 3,512,512
        #
        # out=self.out(x256_up512_1)
        # weight1,weight2=torch.sigmoid(self.weight)
        out = self.out(x_256_512)
        return out,x_64_1

# if __name__=="__main__":
#     data=torch.ones(1,1,512,512)
#     model=CampeodNet()
#     print(model(data))
#     torchsummary.summary(model,(1,512,512),1,device='cpu')
#     print(model)