"""Validate MLX PE-Core-bigG against PyTorch open_clip on real images.

Runs both backends on a set of test images, asserts cosine similarity >= 0.999.
On mismatch, drills down to identify which stage diverges.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

# Local imports (make script-relative imports work)
sys.path.insert(0, os.path.dirname(__file__))
from model import PECoreBigG, PE_CORE_BIGG_CONFIG
import mlx.core as mx
import mlx.nn as nn


def load_mlx_model(weights_path: str, dtype=mx.float32) -> PECoreBigG:
    model = PECoreBigG()
    print(f"Loading MLX weights from {weights_path}", flush=True)
    # strict=False: our _rope_emb constant isn't in the file (built from constants).
    model.load_weights(weights_path, strict=False)
    if dtype != mx.float32:
        model.set_dtype(dtype)
    model.eval()
    mx.eval(model.parameters())
    return model


def preprocess_for_torch_and_mlx(image_paths, preprocess):
    """Run open_clip's preprocess on images and produce both NCHW and NHWC tensors."""
    pt_imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        pt_imgs.append(preprocess(img))
    pt_batch = torch.stack(pt_imgs)  # (B, 3, 448, 448)
    # MLX: NHWC
    np_batch = pt_batch.numpy().transpose(0, 2, 3, 1)  # (B, 448, 448, 3)
    return pt_batch, mx.array(np_batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.expanduser("~/.cache/mlx-pe-core-bigg.safetensors"))
    ap.add_argument("--images", default="benchmark", help="Image directory")
    ap.add_argument("--n", type=int, default=8, help="Number of images to test")
    ap.add_argument("--threshold", type=float, default=0.999)
    args = ap.parse_args()

    # ── Gather image paths ───────────────────────────────────────────────
    img_dir = os.path.abspath(args.images)
    files = sorted(f for f in os.listdir(img_dir)
                   if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".webp"})[: args.n]
    paths = [os.path.join(img_dir, f) for f in files]
    print(f"Testing on {len(paths)} images from {img_dir}", flush=True)

    # ── Load PyTorch reference ───────────────────────────────────────────
    print("Loading PyTorch open_clip PE-Core-bigG (fp32)...", flush=True)
    import open_clip
    pt_model, _, preprocess = open_clip.create_model_and_transforms(
        "PE-Core-bigG-14-448", pretrained="meta"
    )
    pt_model.eval()

    pt_batch, mlx_batch = preprocess_for_torch_and_mlx(paths, preprocess)

    print("Running PyTorch forward...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        pt_emb = pt_model.encode_image(pt_batch).numpy()  # (B, 1280)
    print(f"  PyTorch done in {time.time()-t0:.1f}s, embedding shape {pt_emb.shape}")
    del pt_model

    # ── Load MLX model ───────────────────────────────────────────────────
    mlx_model = load_mlx_model(args.weights, dtype=mx.float32)

    print("Running MLX forward...", flush=True)
    t0 = time.time()
    mlx_emb = mlx_model(mlx_batch)
    mx.eval(mlx_emb)
    mlx_emb_np = np.array(mlx_emb)
    print(f"  MLX done in {time.time()-t0:.1f}s, embedding shape {mlx_emb_np.shape}")

    # ── Compare ──────────────────────────────────────────────────────────
    print("\n=== Per-image embedding comparison ===")
    pt_norm = pt_emb / np.linalg.norm(pt_emb, axis=-1, keepdims=True)
    mlx_norm = mlx_emb_np / np.linalg.norm(mlx_emb_np, axis=-1, keepdims=True)
    cos_per_img = (pt_norm * mlx_norm).sum(axis=-1)
    abs_per_img = np.abs(pt_emb - mlx_emb_np).max(axis=-1)
    rel_per_img = np.linalg.norm(pt_emb - mlx_emb_np, axis=-1) / np.linalg.norm(pt_emb, axis=-1)

    for i, f in enumerate(files):
        flag = "OK" if cos_per_img[i] >= args.threshold else "FAIL"
        print(f"  [{flag}] {f}: cos={cos_per_img[i]:.6f}  max_abs={abs_per_img[i]:.4f}  rel_l2={rel_per_img[i]:.4f}")

    min_cos = cos_per_img.min()
    print(f"\nmin cosine: {min_cos:.6f}, threshold: {args.threshold}")
    if min_cos >= args.threshold:
        print("VALIDATION PASSED")
        return 0
    else:
        print("VALIDATION FAILED — running stage-by-stage diagnostic...")
        return diagnose(paths, preprocess, args.weights)


def diagnose(paths, preprocess, weights_path):
    """Stage-by-stage comparison when validation fails."""
    print("\n=== STAGE-BY-STAGE DIAGNOSTIC ===")

    # Reload PyTorch with hooks
    import open_clip
    pt_model, _, _ = open_clip.create_model_and_transforms(
        "PE-Core-bigG-14-448", pretrained="meta"
    )
    pt_model.eval()
    trunk = pt_model.visual.trunk

    captured = {}
    def hook(name):
        def _h(_m, _i, out):
            captured[name] = out.detach() if isinstance(out, torch.Tensor) else out
        return _h

    trunk.patch_embed.register_forward_hook(hook("patch_embed"))
    trunk.norm_pre.register_forward_hook(hook("norm_pre"))
    trunk.blocks[0].register_forward_hook(hook("block0"))
    trunk.blocks[-1].register_forward_hook(hook("block_last"))
    trunk.norm.register_forward_hook(hook("norm_final"))
    trunk.attn_pool.register_forward_hook(hook("attn_pool"))

    pt_batch, mlx_batch = preprocess_for_torch_and_mlx(paths[:2], preprocess)
    with torch.no_grad():
        _ = pt_model.encode_image(pt_batch)

    # Now run MLX with manual intermediate captures
    sys.path.insert(0, os.path.dirname(__file__))
    from model import PECoreBigG
    mlx_model = PECoreBigG()
    mlx_model.load_weights(weights_path, strict=False)
    mlx_model.eval()
    mx.eval(mlx_model.parameters())

    x = mlx_batch
    x = mlx_model.patch_embed(x)
    mx.eval(x)
    pt_pe = captured["patch_embed"].numpy()
    mlx_pe = np.array(x)
    print(f"  patch_embed: pt {pt_pe.shape} mlx {mlx_pe.shape}  max_abs={np.abs(pt_pe - mlx_pe).max():.4e}")

    x = x + mlx_model.pos_embed
    x = mlx_model.norm_pre(x)
    mx.eval(x)
    pt_np = captured["norm_pre"].numpy()
    mlx_np = np.array(x)
    print(f"  norm_pre:    pt {pt_np.shape} mlx {mlx_np.shape}  max_abs={np.abs(pt_np - mlx_np).max():.4e}")

    x = mlx_model.blocks[0](x, mlx_model._rope_emb)
    mx.eval(x)
    pt_b0 = captured["block0"].numpy()
    mlx_b0 = np.array(x)
    print(f"  block[0]:    pt {pt_b0.shape} mlx {mlx_b0.shape}  max_abs={np.abs(pt_b0 - mlx_b0).max():.4e}")

    for i in range(1, mlx_model.cfg["depth"]):
        x = mlx_model.blocks[i](x, mlx_model._rope_emb)
    mx.eval(x)
    pt_bl = captured["block_last"].numpy()
    mlx_bl = np.array(x)
    print(f"  block[-1]:   pt {pt_bl.shape} mlx {mlx_bl.shape}  max_abs={np.abs(pt_bl - mlx_bl).max():.4e}")

    x = mlx_model.norm(x)
    mx.eval(x)
    pt_n = captured["norm_final"].numpy()
    mlx_n = np.array(x)
    print(f"  norm:        pt {pt_n.shape} mlx {mlx_n.shape}  max_abs={np.abs(pt_n - mlx_n).max():.4e}")

    x = mlx_model.attn_pool(x)
    mx.eval(x)
    pt_ap = captured["attn_pool"].numpy()
    mlx_ap = np.array(x)
    print(f"  attn_pool:   pt {pt_ap.shape} mlx {mlx_ap.shape}  max_abs={np.abs(pt_ap - mlx_ap).max():.4e}")

    x = mlx_model.head(x)
    mx.eval(x)
    mlx_final = np.array(x)
    with torch.no_grad():
        pt_final = pt_model.encode_image(pt_batch).numpy()
    print(f"  head/final:  pt {pt_final.shape} mlx {mlx_final.shape}  max_abs={np.abs(pt_final - mlx_final).max():.4e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
