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
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        dilation=1,
        groups = 8,
        norm = True
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        if dilation == 1:
            self.project = nn.Conv2d(dim, dim_out, 3, padding = 1, dilation=dilation)
        else:
            self.project = nn.Conv2d(dim, dim_out, 3,dilation=dilation, padding='same')

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)

class GlobalContext(nn.Module):
    def __init__(
        self,
        *,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        x, context = map(lambda t: rearrange(t, 'b n ... -> b n (...)'), (x, context))
        out = einsum('b i n, b c n -> b c i', context.softmax(dim = -1), x)
        out = rearrange(out, '... -> ... 1')
        return self.net(out)


class GlobalResidual(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        groups = 8,
        pool=0,
        linear_attn = False,
        use_gca = False,
        squeeze_excite = False,
        **attn_kwargs
    ):
        super().__init__()

        self.time_mlp = None

        self.pool = pool

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = attn_klass(
                dim = dim_out,
                context_dim = cond_dim,
                **attn_kwargs
            )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)

        self.gca = GlobalContext(dim_in = dim_out, dim_out = dim_out) if use_gca else Always(1)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()

        if self.pool:
            self.maxpool = nn.MaxPool2d(kernel_size=pool, stride=pool)


    def forward(self, x, time_emb = None, cond = None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x)

        if exists(self.cross_attn):
            assert exists(cond)
            h = rearrange(h, 'b c h w -> b h w c')
            h, ps = pack([h], 'b * c')
            h = self.cross_attn(h, context = cond) + h
            h, = unpack(h, ps, 'b * c')
            h = rearrange(h, 'b h w c -> b c h w')

        h = self.block2(h, scale_shift = scale_shift)

        h = h * self.gca(h)

        h = h + self.res_conv(x)

        if self.pool:
            h = self.maxpool(h)

        return h


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        norm_context = False,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> b h 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # cosine sim attention

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearCrossAttention(CrossAttention):
    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (q, k, v))

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> (b h) 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # masking

        max_neg_value = -torch.finfo(x.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b n -> b n 1')
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.)

        # linear attention

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = self.heads)
        return self.to_out(out)

def analyze_pattern(pattern: str, opname: str):
    # Maybe some validation of identifiers?
    axes = pattern.split()
    n_axes_before = axes.index('*')
    n_axes_after = len(axes) - n_axes_before - 1
    min_axes = n_axes_before + n_axes_after
    return n_axes_before, n_axes_after, min_axes

def pack(tensors, pattern):

    n_axes_before, n_axes_after, min_axes = analyze_pattern(pattern, 'pack')

    reshaped_tensors = []
    packed_shapes = []
    for i, tensor in enumerate(tensors):
        shape = tensor.shape

        axis_after_packed_axes = len(shape) - n_axes_after
        packed_shapes.append(shape[n_axes_before:axis_after_packed_axes])
        reshaped_tensors.append(
            torch.reshape(tensor, (*shape[:n_axes_before], -1, *shape[axis_after_packed_axes:]))
        )

    return torch.cat(reshaped_tensors, dim=n_axes_before), packed_shapes

def unpack(tensor, packed_shapes, pattern):
    def prod(x):
        result = 1
        for i in x:
            result *= i
        return result

    n_axes_before, n_axes_after, min_axes = analyze_pattern(pattern, opname='unpack')
    input_shape = tensor.shape


    unpacked_axis: int = n_axes_before

    lengths_of_composed_axes = [
        -1 if -1 in p_shape else prod(p_shape)
        for p_shape in packed_shapes
    ]

    n_unknown_composed_axes = sum(x == -1 for x in lengths_of_composed_axes)

    split_positions = [0] * len(packed_shapes) + [input_shape[unpacked_axis]]
    if n_unknown_composed_axes == 0:
        for i, x in enumerate(lengths_of_composed_axes[:-1]):
            split_positions[i + 1] = split_positions[i] + x
    else:
        unknown_composed_axis: int = lengths_of_composed_axes.index(-1)
        for i in range(unknown_composed_axis):
            split_positions[i + 1] = split_positions[i] + lengths_of_composed_axes[i]
        for j in range(unknown_composed_axis + 1, len(lengths_of_composed_axes))[::-1]:
            split_positions[j] = split_positions[j + 1] - lengths_of_composed_axes[j]

    shape_start = input_shape[:unpacked_axis]
    shape_end = input_shape[unpacked_axis + 1:]
    slice_filler = (slice(None, None),) * unpacked_axis
    try:
        return [
            torch.reshape(
                # shortest way slice arbitrary axis
                tensor[(*slice_filler, slice(split_positions[i], split_positions[i + 1]))],
                (*shape_start, *element_shape, *shape_end)
            )
            for i, element_shape in enumerate(packed_shapes)
        ]
    except BaseException:
        # this hits if there is an error during reshapes, which means passed shapes were incorrect
        raise RuntimeError(f'Error during unpack(..., "{pattern}"): could not split axis of size {split_positions[-1]}'
                           f' into requested {packed_shapes}')

class LayerNorm(nn.Module):
    def __init__(self, feats, stable = False, dim = -1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)


def exists(val):
    return val is not None

def l2norm(t):
    return F.normalize(t, dim = -1)

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val