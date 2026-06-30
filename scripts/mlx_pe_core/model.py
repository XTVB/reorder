"""MLX implementation of Meta PE-Core-bigG-14-448 vision encoder.

Architecture (from timm/Meta cross-check):
- 50 transformer blocks, width 1536, 16 heads (head_dim 96), MLP hidden 8960 GELU
- 32x32 patch grid (no CLS, no register tokens)
- Both learned absolute pos_embed AND 2D-axial RoPE active
- Pre-norm, fused QKV, no LayerScale, no QK-norm
- Attention pool: 1 latent query, 8 heads (head_dim 192), residual MLP only
- Final projection: Linear(1536 -> 1280)

Loaded from a converted state_dict via mlx_pe_core.convert.
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from rope import build_rope_buf


PE_CORE_BIGG_CONFIG = dict(
    image_size=448,
    patch_size=14,
    embed_dim=1536,
    depth=50,
    num_heads=16,
    mlp_hidden=8960,
    output_dim=1280,
    attn_pool_heads=8,
    attn_pool_mlp_hidden=6144,
    rope_theta=10000.0,
    ln_eps=1e-5,
)


def apply_rope_2d_cat(x: mx.array, rope_emb: mx.array) -> mx.array:
    """Apply 2D-axial RoPE (interleaved pair rotation, cat layout).

    x: (B, H, N, D) — query or key tensor.
    rope_emb: (N, 2*D) — first D dims sin, last D dims cos (D = head_dim).
    Returns rotated x.

    Pair-rotation (interleaved): (x_2k, x_2k+1) -> (x_2k*cos - x_2k+1*sin,
                                                     x_2k+1*cos + x_2k*sin).
    Equivalent to x*cos + rot(x)*sin where rot interleaves: rot(x)[..., 2k] = -x[..., 2k+1],
    rot(x)[..., 2k+1] = x[..., 2k].
    """
    sin_emb, cos_emb = mx.split(rope_emb, 2, axis=-1)  # (N, D), (N, D)
    # Broadcast: (1, 1, N, D)
    sin_emb = sin_emb[None, None, :, :]
    cos_emb = cos_emb[None, None, :, :]

    # rot(x): pair-wise (a, b) -> (-b, a). Reshape to pairs, swap+negate, restore.
    B, H, N, D = x.shape
    x_pairs = x.reshape(B, H, N, D // 2, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    rotated = mx.stack([-x1, x0], axis=-1).reshape(B, H, N, D)

    return x * cos_emb + rotated * sin_emb


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=3, out_channels=embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )
        self.grid = image_size // patch_size  # 32

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, 3) NHWC. Conv2d expects NHWC in MLX.
        x = self.proj(x)  # (B, gH, gW, embed_dim)
        # Flatten spatial: (B, gH*gW, embed_dim)
        B, gH, gW, D = x.shape
        return x.reshape(B, gH * gW, D)


class Attention(nn.Module):
    """EvaAttention-with-RoPE: fused QKV, RoPE on Q/K, no QK-norm, identity inner-norm."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, rope_emb: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*C)
        # PyTorch did: reshape (B, N, 3, H, D) -> permute (3, B, H, N, D) -> unbind 0
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, N, D)

        q = apply_rope_2d_cat(q, rope_emb)
        k = apply_rope_2d_cat(k, rope_emb)

        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # x: (B, H, N, D) -> (B, N, C)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    """Plain GELU MLP (timm.layers.Mlp), not SwiGLU."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # MLX nn.gelu is exact (matches PyTorch nn.GELU(approximate='none')).
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden: int, ln_eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=ln_eps)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=ln_eps)
        self.mlp = Mlp(dim, mlp_hidden)

    def __call__(self, x: mx.array, rope_emb: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), rope_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionPoolLatent(nn.Module):
    """timm.layers.AttentionPoolLatent — 1 latent query, 8 heads (head_dim 192).

    Forward: q from learned latent, k/v from x (fused KV), single SDPA, proj,
    then x = x + mlp(norm(x)). Pool='token' returns x[:, 0].
    """

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int, ln_eps: float):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 192 for PE-G pooler
        self.scale = self.head_dim ** -0.5

        self.latent = mx.zeros((1, 1, dim))  # learnable

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.norm = nn.LayerNorm(dim, eps=ln_eps)
        self.mlp = Mlp(dim, mlp_hidden)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        latent = mx.broadcast_to(self.latent, (B, 1, C))

        q = self.q(latent).reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim)
        kv = kv.transpose(2, 0, 3, 1, 4)  # (2, B, H, N, D)
        k, v = kv[0], kv[1]

        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(0, 2, 1, 3).reshape(B, 1, C)
        x = self.proj(x)
        x = x + self.mlp(self.norm(x))
        return x[:, 0]  # pool='token'


class PECoreBigG(nn.Module):
    def __init__(self, cfg: dict = PE_CORE_BIGG_CONFIG):
        super().__init__()
        self.cfg = cfg

        D = cfg["embed_dim"]
        head_dim = D // cfg["num_heads"]
        grid = cfg["image_size"] // cfg["patch_size"]
        num_patches = grid * grid

        self.patch_embed = PatchEmbed(cfg["image_size"], cfg["patch_size"], D)
        self.pos_embed = mx.zeros((1, num_patches, D))  # learned

        self.norm_pre = nn.LayerNorm(D, eps=cfg["ln_eps"])
        self.blocks = [
            Block(D, cfg["num_heads"], cfg["mlp_hidden"], cfg["ln_eps"])
            for _ in range(cfg["depth"])
        ]
        self.norm = nn.LayerNorm(D, eps=cfg["ln_eps"])
        self.attn_pool = AttentionPoolLatent(
            D, cfg["attn_pool_heads"], cfg["attn_pool_mlp_hidden"], cfg["ln_eps"]
        )
        self.head = nn.Linear(D, cfg["output_dim"], bias=True)

        # Static RoPE buffer — not a parameter, built from constants.
        rope_np = build_rope_buf(
            grid_h=grid, grid_w=grid, head_dim=head_dim,
            theta=cfg["rope_theta"], grid_offset=0.0,
        )
        self._rope_emb = mx.array(rope_np)  # (num_patches, 2*head_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, 3) float in [0, 1] already preprocessed to mean/std.
        x = self.patch_embed(x)              # (B, N, D)
        x = x + self.pos_embed
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x, self._rope_emb)
        x = self.norm(x)
        x = self.attn_pool(x)                # (B, D)
        x = self.head(x)                     # (B, output_dim)
        return x

    def _pool_tokens(self, x: mx.array, pooling: str) -> mx.array:
        """Pool a block's token sequence (B, N, D) -> (B, D)."""
        if pooling == "mean":
            return x.mean(axis=1)
        if pooling == "gem3":
            xf = x.astype(mx.float32)
            return (mx.maximum(xf, 1e-6) ** 3).mean(axis=1) ** (1.0 / 3.0)
        if pooling == "attnpool":
            # the model's own pooling head (final LayerNorm + attn-pool) applied to
            # the intermediate tokens — matches the deployed final-layer pooling.
            return self.attn_pool(self.norm(x))
        raise ValueError(f"unknown pooling {pooling!r}")

    def forward_and_layer(self, x: mx.array, layer_idx: int, pooling: str = "attnpool"):
        """Single forward returning BOTH the final image embedding (B, output_dim)
        and one intermediate block's pooled features (B, D) — the production
        in-pass capture for the learned-head layer input. Identical math to
        __call__ for the final output; the capture rides along free. Pooling uses
        the SAME _pool_tokens path as the offline forward_capture the head trained
        on, so inference features match training exactly."""
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.norm_pre(x)
        cap = None
        for idx, blk in enumerate(self.blocks):
            x = blk(x, self._rope_emb)
            if idx == layer_idx:
                cap = self._pool_tokens(x, pooling)
        x = self.norm(x)
        x = self.attn_pool(x)
        x = self.head(x)
        return x, cap

    def forward_capture(self, x: mx.array, layers, poolings=("mean", "gem3", "attnpool")) -> dict:
        """Run the vision forward and return {(block_idx, pooling): (B, D)} for the
        requested 0-based transformer blocks, captured in the single pass (no
        recompute). Used by the intermediate-layer extraction experiment."""
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.norm_pre(x)
        want = set(layers)
        out = {}
        for idx, blk in enumerate(self.blocks):
            x = blk(x, self._rope_emb)
            if idx in want:
                for p in poolings:
                    out[(idx, p)] = self._pool_tokens(x, p)
        return out
