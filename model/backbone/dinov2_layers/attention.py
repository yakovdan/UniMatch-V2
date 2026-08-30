# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from torch import Tensor
from torch import nn
from torch.nn import functional as F


# MemEffAttention uses torch F.scaled_dot_product_attention (SDPA) instead of
# xformers' memory_efficient_attention. The two are numerically interchangeable
# (verified 2026-08-30 on a 4090, torch 2.10 / xformers 0.0.35: bf16 fwd rel err
# vs fp64 is 6.29e-3 for both, at the 6.05e-3 floor set by rounding the inputs
# to bf16; fp32 ~1.2e-6 for both), save the same O(N) state for backward, and
# run within ~5% of each other (per layer at micro-batch 8: xformers 1.29 ms
# fwd+bwd, SDPA-flash 1.35 ms).
#
# SDPA's dispatcher picks the best available backend per dtype/arch -- flash
# (FlashAttention-2) for bf16/fp16, mem-efficient for fp32 on Ada -- and falls
# back to the math backend instead of raising when no fused kernel exists. That
# retires the XFORMERS_DISABLED workaround that used to live here: xformers
# 0.0.35 has no fp32 kernel above capability 9.0, so fp32 eval on Blackwell
# raised NotImplementedError, and the escape hatch forced the plain-attention
# path below -- which under bf16 autocast also rounds the pre-softmax scores to
# bf16, doubling the numerical error (1.25e-2 vs 6.29e-3) and materialising the
# O(N^2) attention matrix. With SDPA neither failure mode exists; the
# XFORMERS_DISABLED export in docker/onstart.sh is now a no-op for this file.
#
# block.py and swiglu_ffn.py still import xformers: block.py only needs it for
# nested-tensor (list) inputs, which DPT never passes, and SwiGLU is used by
# the giant backbone only.


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
        if attn_bias is not None:
            raise NotImplementedError(
                "attn_bias is not supported by the SDPA path; it was only ever an "
                "xformers BlockDiagonalMask for nested-tensor list inputs, which "
                "DPT never passes"
            )
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, num_heads, N, head_dim)

        # dropout_p: the old xformers call silently ignored attn_drop; DINOv2
        # configures attn_drop=0.0, so passing it here changes nothing today
        # but keeps the semantics of Attention if it is ever set.
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
