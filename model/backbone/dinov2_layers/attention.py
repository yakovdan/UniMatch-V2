# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os

from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


# XFORMERS_DISABLED forces the plain-attention path in MemEffAttention.forward below.
#
# Why it is needed: xformers 0.0.35 has no fp32 attention kernel for Blackwell. Every fp32
# backend (cutlassF-pt, fa3F) requires compute capability <= 9.0, and fa2F is bf16/fp16
# only, so on an RTX 5090 (capability 12.0) an fp32 forward raises
#   NotImplementedError: No operator found for `memory_efficient_attention_forward`
# Training under --bf16 is unaffected (fa2F handles it), which is why this surfaces only at
# the first epoch boundary: supervised.evaluate runs in fp32. On an RTX 4090 (8.9) cutlassF
# covers fp32, so it never reproduces locally.
#
# xformers' own XFORMERS_DISABLED does not help -- as of 0.0.35 the import still succeeds --
# and the vendored dinov2 code keys the fallback purely off ImportError, so the switch has
# to live here. The fallback is mathematically the same attention, just without the fused
# kernel: slower, and O(N^2) memory for the attention matrix.
#
# block.py and swiglu_ffn.py import xformers too and are deliberately left alone: block.py
# only needs it for nested-tensor (list) inputs, which DPT never passes, and SwiGLU is used
# by the giant backbone only.
if os.environ.get("XFORMERS_DISABLED"):
    logger.warning("xFormers disabled by XFORMERS_DISABLED; using plain attention")
    XFORMERS_AVAILABLE = False
else:
    try:
        from xformers.ops import memory_efficient_attention, unbind, fmha

        XFORMERS_AVAILABLE = True
    except ImportError:
        logger.warning("xFormers not available")
        XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        