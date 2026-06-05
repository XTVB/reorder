#!/usr/bin/env python3
"""
Pre-extract K augmented views per image as PE-G + color feature vectors.
Pairs with the projection-head training loop, which can sample views as
same-shoot positives — addresses the holdout-generalization failure mode
where the head over-fits to specific crops/lighting of each training shoot.

Augmentations per view (applied to the raw PIL image, before PE-G's
ToTensor+Normalize):
  - RandomResizedCrop(448, scale=(0.65, 1.0), ratio=(0.75, 1.333))
  - RandomHorizontalFlip(p=0.5)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

Each augmented view goes through:
  - PE-Core-bigG-14-448 → 1280-d L2-normalized feature
  - Color histogram (3×3 HSV + RGB moments, same as extract_features.py) → 693d

Only images in a group of 2..MAX_GROUP_SIZE are extracted — that's all the
trainer samples views for. Other rows stay zero-filled; `view_indices` in
views_meta.json lists the real ones so training can fall back to the original.

Outputs in {cache_dir}:
  - pecore_g_views.npy   shape (N, K, 1280) float32  (non-target rows are zero)
  - color_views.npy      shape (N, K, 693)  float32  (non-target rows are zero)
  - views_meta.json      {n_views, view_indices, completed_through, version}

Resume: re-running picks up from `completed_through` in views_meta.json
(images < that index are skipped). Checkpoint is written every CHECKPOINT_EVERY
images so an interrupted run loses at most that many.

Usage:
  python scripts/extract_augmented_views.py <target_dir> \\
      [--n-views 3] [--cache-dir DIR] [--batch-size 8] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Reuse color histogram logic from extract_features.py (still lives in ../scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from extract_features import extract_color_features  # noqa: E402

PEG_HW = 448
PEG_DIM = 1280
COLOR_DIM = 693
CHECKPOINT_EVERY = 200
# Larger groups already have enough real variety; augmenting them adds little.
MAX_GROUP_SIZE = 100
VIEWS_VERSION = "aug-v1-crop0.65-jitter0.2-hflip0.5"

# PE-Core preprocessing constants (verified in extract_features.py — bit-exact
# with open_clip.create_model_and_transforms('PE-Core-bigG-14-448')).
PEG_MEAN = (0.48145466, 0.4578275, 0.40821073)
PEG_STD = (0.26862954, 0.26130258, 0.27577711)


def build_aug_pipeline() -> transforms.Compose:
    """Pixel-level augmentations (PIL → PIL). PE-G's normalize happens after."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            PEG_HW,
            scale=(0.65, 1.0),
            ratio=(0.75, 1.333),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05,
        ),
    ])


PEG_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=PEG_MEAN, std=PEG_STD),
])


def load_pecore_g_mlx():
    """MLX backend (matches extract_features.py default). Returns a callable
    that takes a torch tensor (B, 3, H, W) and returns (B, 1280) numpy."""
    import mlx.core as mx
    mlx_pe_dir = os.path.join(SCRIPTS_DIR, "mlx_pe_core")
    if mlx_pe_dir not in sys.path:
        sys.path.insert(0, mlx_pe_dir)
    from model import PECoreBigG

    weights_path = os.path.expanduser("~/.cache/mlx-pe-core-bigg.safetensors")
    if not os.path.exists(weights_path):
        sys.exit(f"MLX PE-Core weights not found at {weights_path}. Run extract_features.py once to convert them.")

    model = PECoreBigG()
    model.load_weights(weights_path, strict=False)
    model.set_dtype(mx.float32)
    model.eval()
    mx.eval(model.parameters())

    def encode(batch_torch: torch.Tensor) -> np.ndarray:
        # MLX expects NHWC; torch gives NCHW. Convert.
        arr = batch_torch.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        out = model(mx.array(arr))
        mx.eval(out)
        return np.array(out, dtype=np.float32)

    return encode


def load_pecore_g_torch():
    """PyTorch open_clip fallback. fp16 on MPS for speed."""
    import open_clip
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(
        "PE-Core-bigG-14-448", pretrained="meta", device=device,
    )
    model.eval()
    model = model.half()

    @torch.no_grad()
    def encode(batch_torch: torch.Tensor) -> np.ndarray:
        out = model.encode_image(batch_torch.to(device).half())
        out = out.float().cpu().numpy()
        return out
    return encode


def load_meta(cache_dir: str, n: int, n_views: int, view_indices: list[int]) -> dict:
    path = os.path.join(cache_dir, "views_meta.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                meta = json.load(f)
            # view_indices is part of the cache identity — if the group set changed,
            # the partial run is stale and must restart.
            if (meta.get("version") == VIEWS_VERSION and meta.get("n_images") == n
                    and meta.get("n_views") == n_views and meta.get("view_indices") == view_indices):
                return meta
            print(f"  views_meta.json mismatch (version/n_images/n_views/view_indices), starting fresh", file=sys.stderr)
        except Exception as e:
            print(f"  could not read views_meta.json: {e}, starting fresh", file=sys.stderr)
    return {"version": VIEWS_VERSION, "n_images": n, "n_views": n_views,
            "view_indices": view_indices, "completed_through": 0}


def save_meta(cache_dir: str, meta: dict):
    path = os.path.join(cache_dir, "views_meta.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target_dir")
    ap.add_argument("--n-views", type=int, default=3)
    ap.add_argument("--cache-dir", default=None,
                    help="defaults to {target_dir}/.reorder-cache")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pecore-g-backend", default="mlx", choices=["mlx", "pytorch"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=None,
                    help="cap at this many images (for smoke tests); leaves rest as zeros")
    args = ap.parse_args()

    cache_dir = args.cache_dir or os.path.join(args.target_dir, ".reorder-cache")
    if not os.path.isdir(cache_dir):
        sys.exit(f"cache dir not found: {cache_dir}")
    ch_path = os.path.join(cache_dir, "content_hashes.json")
    if not os.path.exists(ch_path):
        sys.exit(f"content_hashes.json not found at {ch_path}; run extract_features.py first")

    with open(ch_path) as f:
        content_hashes: dict[str, str] = json.load(f)
    filenames = sorted(content_hashes.keys())
    fn_to_idx = {fn: i for i, fn in enumerate(filenames)}
    n = len(filenames)
    k = args.n_views

    # Only grouped images get views (see docstring). No groups file → extract all.
    groups_path = os.path.join(args.target_dir, ".reorder-groups.json")
    if os.path.exists(groups_path):
        with open(groups_path) as f:
            groups_raw = json.load(f)
        groups = groups_raw if isinstance(groups_raw, list) else groups_raw.get("groups", [])
        target_set: set[int] = set()
        for g in groups:
            imgs = g.get("images", [])
            if len(imgs) < 2 or len(imgs) > MAX_GROUP_SIZE:
                continue
            for fn in imgs:
                j = fn_to_idx.get(fn)
                if j is not None:
                    target_set.add(j)
        print(f"  {len(target_set)}/{n} images in 2–{MAX_GROUP_SIZE}-image groups (pixel-aug targets); "
              f"skipping {n - len(target_set)} singleton/ungrouped/large-group", file=sys.stderr)
    else:
        print(f"  WARN: {groups_path} not found — extracting views for ALL images", file=sys.stderr)
        target_set = set(range(n))
    target_indices = sorted(target_set)

    print(f"  target: {len(target_indices)} images × {k} views = {len(target_indices)*k} augmented embeddings", file=sys.stderr)

    # Seeded RNG so augmentations are reproducible per run.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resume
    meta = load_meta(cache_dir, n, k, target_indices)
    start_idx = meta["completed_through"]

    # Output arrays
    peg_path = os.path.join(cache_dir, "pecore_g_views.npy")
    col_path = os.path.join(cache_dir, "color_views.npy")
    if start_idx > 0 and os.path.exists(peg_path) and os.path.exists(col_path):
        peg_views = np.load(peg_path)
        col_views = np.load(col_path)
        if peg_views.shape != (n, k, PEG_DIM) or col_views.shape != (n, k, COLOR_DIM):
            print(f"  shape mismatch on resume — restarting from 0", file=sys.stderr)
            peg_views = np.zeros((n, k, PEG_DIM), dtype=np.float32)
            col_views = np.zeros((n, k, COLOR_DIM), dtype=np.float32)
            start_idx = 0
            meta["completed_through"] = 0
        else:
            print(f"  resuming from image {start_idx}/{n}", file=sys.stderr)
    else:
        peg_views = np.zeros((n, k, PEG_DIM), dtype=np.float32)
        col_views = np.zeros((n, k, COLOR_DIM), dtype=np.float32)
        start_idx = 0

    if start_idx >= n:
        print("  already complete; nothing to do", file=sys.stderr)
        return

    # Load PE-G model
    print(f"  loading PE-G ({args.pecore_g_backend})...", file=sys.stderr)
    if args.pecore_g_backend == "mlx":
        encode = load_pecore_g_mlx()
    else:
        encode = load_pecore_g_torch()

    aug = build_aug_pipeline()
    bs = args.batch_size
    t_start = time.time()
    images_done = 0

    # Outer loop: image indices. Inner loop: views. We process all K views of
    # the same batch together so PIL loading cost is amortized.
    cap = min(n, args.max_images) if args.max_images is not None else n
    i = start_idx
    while i < cap:
        batch_filenames = filenames[i:i + bs]
        bs_actual = len(batch_filenames)

        # Load originals once per image (PIL is cheap relative to PE-G). Non-target
        # images get None, which skips all per-view compute and leaves their rows zero.
        originals: list[Image.Image] = []
        for bi, fn in enumerate(batch_filenames):
            if (i + bi) not in target_set:
                originals.append(None)
                continue
            try:
                originals.append(Image.open(os.path.join(args.target_dir, fn)).convert("RGB"))
            except Exception as e:
                print(f"  WARN: failed to open {fn}: {e}", file=sys.stderr)
                originals.append(None)

        for v in range(k):
            tensors = []
            color_arrs = []
            kept_idxs = []  # which positions in the batch had a valid image
            for bi, img in enumerate(originals):
                if img is None:
                    tensors.append(None)
                    color_arrs.append(None)
                    continue
                img_aug = aug(img)
                tensors.append(PEG_TO_TENSOR(img_aug))
                color_arrs.append(extract_color_features(img_aug))
                kept_idxs.append(bi)

            if not kept_idxs:
                continue
            stacked = torch.stack([tensors[bi] for bi in kept_idxs])
            peg_feats = encode(stacked)
            # L2-normalize PE-G output (matches the production cache convention).
            peg_feats = peg_feats / np.linalg.norm(peg_feats, axis=1, keepdims=True).clip(min=1e-8)

            for out_pos, bi in enumerate(kept_idxs):
                peg_views[i + bi, v] = peg_feats[out_pos]
                col_views[i + bi, v] = color_arrs[bi]

        i += bs_actual
        images_done += bs_actual

        if images_done % CHECKPOINT_EVERY == 0 or i >= n:
            np.save(peg_path, peg_views)
            np.save(col_path, col_views)
            meta["completed_through"] = i
            save_meta(cache_dir, meta)
            elapsed = time.time() - t_start
            ips = images_done / max(elapsed, 1e-6)
            remaining_sec = (n - i) / max(ips, 1e-6)
            print(f"  {i}/{n} images done ({ips:.2f} img/s, ~{remaining_sec/60:.1f} min remaining)", file=sys.stderr)

    # Final save (in case the last batch wasn't a checkpoint boundary)
    np.save(peg_path, peg_views)
    np.save(col_path, col_views)
    meta["completed_through"] = i  # last image index actually processed (may overshoot `cap` by < batch_size)
    save_meta(cache_dir, meta)
    print(f"  done: {i} images × {k} views saved to {cache_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
