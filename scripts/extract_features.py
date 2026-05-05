#!/usr/bin/env python3
"""Extract CLIP + PE-Core + color features from images, cached by content hash.

Only does feature extraction — no clustering. Outputs a manifest JSON to stdout.
Progress is reported on stderr.

Models:
  - CLIP ViT-B/32 (512-dim) — kept for TF-IDF auto-naming
  - DINOv2 ViT-L/14 (1024-dim)
  - DINOv3 ViT-B/16 (768-dim CLS + 49 × 768 pooled patches + 196 × 768 full-res patches)
  - PE-Core-L-14-336 (1024-dim) — Meta Perception Encoder, large variant
  - PE-Core-bigG-14-448 (1280-dim) — Meta Perception Encoder, giant variant
  - Color histograms (693-dim) — 3x3 spatial grid of HSV + RGB moments (77 per cell)

Usage:
    python3 extract_features.py <image_dir> [--cache-dir <dir>] [--batch-size N]

Cache is stored in <image_dir>/.reorder-cache/ by default.
Features are keyed by content hash (blake2b of first 16KB + file size),
so renaming files does not invalidate the cache.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque

import numpy as np

# Per-model version keys. Only models whose version changed get re-extracted.
# The cache stores "_v_<key>" for each model. Missing or mismatched → re-extract that model only.
MODEL_VERSIONS = {
    "clip": "ViT-B-32-laion2b-v1",
    "dino": "dinov2-vitl14-v1",
    "pecore_l": "PE-Core-L-14-336-meta-v1",
    "pecore_g": "PE-Core-bigG-14-448-meta-v1",
    "color": "hsv-rgb-3x3-693d-v1",
    "dinov3": "dinov3-vitb16-7x7pool-v2",
}

# All embedding arrays stored in the npz
EMB_KEYS = list(MODEL_VERSIONS.keys())

# Color feature constants
COLOR_GRID = 3                    # 3x3 spatial grid
COLOR_CELL_DIM = 36 + 16 + 16 + 9 # HSV histogram (H/S/V) + RGB moments (mean/std/skew per channel)
COLOR_DIM = COLOR_GRID * COLOR_GRID * COLOR_CELL_DIM  # 693
COLOR_THUMB_SIZE = 144            # divisible by COLOR_GRID; 48x48 per cell

# DINOv3 constants
DINOV3_CLS_DIM = 768
DINOV3_PATCH_DIM = 768
DINOV3_N_PATCHES = 49  # 14x14 avg-pooled to 7x7
DINOV3_N_PATCHES_FULL = 196  # 14x14 full resolution

# DINOv3 local weights path (downloaded from Kaggle)
DINOV3_WEIGHTS = os.environ.get(
    "DINOV3_WEIGHTS",
    os.path.expanduser("~/.cache/dinov3-weights/facebook/dinov3-vitb16-pretrain-lvd1689m"),
)

CHECKPOINT_SEC = 300  # periodic cache save interval during extraction


def content_hash(filepath: str) -> str:
    """Fast content-based hash: blake2b(first 16KB + file size)."""
    size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        head = f.read(16384)
    return hashlib.blake2b(head, digest_size=16, key=size.to_bytes(8, "big")).hexdigest()


def _cell_color_features(arr_cell, hsv_cell):
    """Per-cell 77-dim HSV histogram + RGB color moments."""
    feats = []
    for ch, bins in [(0, 36), (1, 16), (2, 16)]:
        h, _ = np.histogram(hsv_cell[:, :, ch], bins=bins, range=(0, 256))
        h = h.astype(np.float32) / (h.sum() + 1e-10)
        feats.extend(h)
    for ch in range(3):
        d = arr_cell[:, :, ch]
        mu, sigma = d.mean(), d.std()
        feats.extend([
            mu / 256.0,
            sigma / 128.0,
            float(np.mean(((d - mu) / max(sigma, 1.0)) ** 3)) / 5.0,
        ])
    return feats


def extract_color_features(img_rgb, thumb_size=COLOR_THUMB_SIZE):
    """Spatial 3x3 grid of HSV + RGB color features (693 dimensions, row-major top-left → bottom-right)."""
    thumb = img_rgb.resize((thumb_size, thumb_size))
    arr = np.array(thumb, dtype=np.float32)
    hsv = np.array(thumb.convert("HSV"), dtype=np.float32)

    cell = thumb_size // COLOR_GRID
    feats = []
    for gy in range(COLOR_GRID):
        for gx in range(COLOR_GRID):
            y0, x0 = gy * cell, gx * cell
            y1, x1 = y0 + cell, x0 + cell
            feats.extend(_cell_color_features(arr[y0:y1, x0:x1], hsv[y0:y1, x0:x1]))
    return np.array(feats, dtype=np.float32)


def _color_extract_worker(args):
    """Module-level worker for ProcessPoolExecutor — must be picklable.
    Returns (feature_array, error_message_or_None)."""
    image_dir, fname = args
    try:
        from PIL import Image
        return (extract_color_features(Image.open(os.path.join(image_dir, fname)).convert("RGB")), None)
    except Exception as e:
        return (None, repr(e))


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def merge_cached_array(existing_arr, existing_hashes, new_arr, new_hashes,
                       hash_universe=None):
    """Merge (existing, new) arrays keyed by hash, preserving sorted hash order.

    Args:
        existing_arr: (N_old, ...) array or None
        existing_hashes: list/dict of N_old hashes (dict: hash→old_idx; list: positional)
        new_arr: (N_new, ...) array
        new_hashes: list of N_new hashes
        hash_universe: optional set forcing the output hash list to exactly this
            set (sorted). Hashes in the universe with no data are zero-filled;
            existing/new hashes outside the universe are dropped (for pruning).
            If None, the union of existing+new hashes is used.

    Returns:
        (merged_arr, merged_hash_list) where merged_hash_list is sorted.
        Rows from `new_arr` take precedence over `existing_arr` on hash collisions.
        Missing hashes are zero-filled.
    """
    # Normalize existing_hashes to a hash→index map
    if isinstance(existing_hashes, dict):
        existing_h2i = existing_hashes
    else:
        existing_h2i = {h: i for i, h in enumerate(existing_hashes or [])}

    # Determine output hash set
    if hash_universe is not None:
        out_set = set(hash_universe)
    else:
        out_set = set(existing_h2i.keys())
        out_set.update(new_hashes)
    out_hashes = sorted(out_set)
    h2i = {h: i for i, h in enumerate(out_hashes)}
    n = len(out_hashes)

    # Determine output dim from whichever array has it
    if existing_arr is not None and len(existing_arr) > 0:
        shape_tail = existing_arr.shape[1:]
        dtype = existing_arr.dtype
    elif new_arr is not None and len(new_arr) > 0:
        shape_tail = new_arr.shape[1:]
        dtype = new_arr.dtype
    else:
        # Nothing to merge
        return np.zeros((n, 0), dtype=np.float32), out_hashes

    out = np.zeros((n,) + shape_tail, dtype=dtype)

    # Fill from existing
    if existing_arr is not None and len(existing_arr) > 0:
        for h, old_i in existing_h2i.items():
            tgt = h2i.get(h)
            if tgt is not None:
                out[tgt] = existing_arr[old_i]

    # Overwrite from new (new takes precedence)
    if new_arr is not None and len(new_arr) > 0:
        for i, h in enumerate(new_hashes):
            tgt = h2i.get(h)
            if tgt is not None:
                out[tgt] = new_arr[i]

    return out, out_hashes


def _report_progress(label, done, total, t0):
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(f"  {label}: {done}/{total} ({done/total*100:.0f}%) "
          f"- {rate:.1f} img/s - ETA {eta:.0f}s", file=sys.stderr)


# ────────────────────────────────────────────────────────────────────────────
# Cache loading
# ────────────────────────────────────────────────────────────────────────────


def _load_existing_cache(hash_cache_path):
    """Load the multi-model NPZ cache, returning (cached_hashes, cached_arrays,
    models_to_extract, mem_versions)."""
    cached_hashes = {}
    cached = {}
    models_to_extract = set(EMB_KEYS)
    mem_versions = {}
    if not os.path.exists(hash_cache_path):
        return cached_hashes, cached, models_to_extract, mem_versions

    try:
        data = np.load(hash_cache_path, allow_pickle=True)
        if "hashes" in data:
            cached_hash_list = list(data["hashes"])
            cached_hashes = {h: i for i, h in enumerate(cached_hash_list)}

            # Migrate old monolithic _model_version to per-model versions.
            old_version = str(data["_model_version"]) if "_model_version" in data else None
            old_compat = {
                "clip": old_version and "clip-ViT-B-32" in old_version,
                "dino": old_version and "dinov2-vitl14" in old_version,
                "color": old_version and "color77" in old_version,
            }

            for k in EMB_KEYS:
                version_key = f"_v_{k}"
                stored = str(data[version_key]) if version_key in data else None
                if stored == MODEL_VERSIONS[k] and k in data:
                    cached[k] = data[k]
                    models_to_extract.discard(k)
                    print(f"  {k}: cache valid ({len(cached_hashes)} entries)", file=sys.stderr)
                elif old_compat.get(k) and k in data:
                    cached[k] = data[k]
                    models_to_extract.discard(k)
                    print(f"  {k}: migrated from old cache ({len(cached_hashes)} entries)", file=sys.stderr)
                else:
                    print(f"  {k}: needs extraction (stored={stored!r}, current={MODEL_VERSIONS[k]!r})", file=sys.stderr)

            for k2 in EMB_KEYS:
                vk = f"_v_{k2}"
                if vk in data:
                    mem_versions[vk] = data[vk]
            if "_model_version" in data:
                mem_versions["_model_version"] = data["_model_version"]
    except Exception as e:
        print(f"  Hash cache corrupt, rebuilding: {e}", file=sys.stderr)
        cached_hashes = {}
        cached = {}
        models_to_extract = set(EMB_KEYS)
        mem_versions = {}

    return cached_hashes, cached, models_to_extract, mem_versions


def _detect_zero_fill(cached, cached_hashes, current_hashes, models_to_extract,
                     patches_cache_hashes_path, patches_cache_data_path):
    """Detect images with zero embeddings (from prior partial extraction).
    Returns dict of model_key → [(fname, hash), ...]."""
    zero_fill_needed = {}
    current_hash_set = set(current_hashes.values())
    hash_to_fname = {h: f for f, h in current_hashes.items()}
    current_cached = {h: i for h, i in cached_hashes.items() if h in current_hash_set}
    for k in list(cached.keys()):
        if k in models_to_extract:
            continue
        arr = cached[k]
        missing = [(hash_to_fname[h], h) for h, idx in current_cached.items()
                   if not np.any(arr[idx])]
        if missing:
            zero_fill_needed[k] = missing
            print(f"  {k}: {len(missing)} zero embeddings (will extract incrementally)", file=sys.stderr)

    # DINOv3 patches cache may be missing entries the CLS cache has.
    if "dinov3" not in models_to_extract:
        _patches_cache_h = {}
        _patches_cache_arr = None
        if os.path.exists(patches_cache_hashes_path) and os.path.exists(patches_cache_data_path):
            try:
                with open(patches_cache_hashes_path) as f:
                    _pl = json.load(f)
                _patches_cache_arr = np.load(patches_cache_data_path)
                _patches_cache_h = {h: i for i, h in enumerate(_pl)}
            except Exception:
                pass
        patches_missing = [
            (hash_to_fname[h], h) for h in current_hash_set
            if h not in _patches_cache_h
            or (_patches_cache_arr is not None and not np.any(_patches_cache_arr[_patches_cache_h[h]]))
        ]
        if patches_missing:
            existing = set(h for _, h in zero_fill_needed.get("dinov3", []))
            extra = [(f, h) for f, h in patches_missing if h not in existing]
            if extra:
                zero_fill_needed.setdefault("dinov3", []).extend(extra)
                print(f"  dinov3: {len(patches_missing)} missing patches "
                      f"({len(extra)} beyond CLS zero-fill)", file=sys.stderr)
    return zero_fill_needed


# ────────────────────────────────────────────────────────────────────────────
# Generic batched-inference pass (works for both torch MPS and MLX backends)
# ────────────────────────────────────────────────────────────────────────────


def _run_pass(items, transform, fallback_hw, batch_size, label, inference_fn,
              ctx, *, mlx_dtype=None, on_checkpoint=None):
    """Run batched inference with multi-batch-ahead prefetch.

    `inference_fn(batch)` receives a torch.Tensor (already on device for the
    pytorch path) or a numpy NHWC batch (for the MLX path; signaled by
    `mlx_dtype` being non-None). Must return a numpy float32 array of shape
    (B, dim), already L2-normalized.

    `ctx` carries shared state: image_dir, interrupted flag, checkpoint timer.
    Respects ctx.interrupted — breaks early.
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    n = len(items)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    def _prepare_batch(indices):
        tensors = []
        for i in indices:
            fname, _ = items[i]
            path = os.path.join(ctx.image_dir, fname)
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(transform(img))
            except Exception as e:
                print(f"  WARNING: skipping {fname}: {e}", file=sys.stderr)
                tensors.append(torch.zeros(3, fallback_hw, fallback_hw))
        return torch.stack(tensors)

    results = []
    t0 = time.time()
    prefetch_depth = 4

    with ThreadPoolExecutor(max_workers=prefetch_depth) as pool:
        futures = deque()
        batch_starts = list(range(0, n, batch_size))
        for bs in batch_starts[:prefetch_depth]:
            be = min(bs + batch_size, n)
            futures.append(pool.submit(_prepare_batch, range(bs, be)))

        submitted = min(prefetch_depth, len(batch_starts))

        for batch_start in batch_starts:
            batch_end = min(batch_start + batch_size, n)
            batch_torch = futures.popleft().result()

            if submitted < len(batch_starts) and not ctx.interrupted:
                bs = batch_starts[submitted]
                be = min(bs + batch_size, n)
                futures.append(pool.submit(_prepare_batch, range(bs, be)))
                submitted += 1

            if mlx_dtype is not None:
                # MLX path: torch tensor → NHWC numpy → mx.array → model → numpy
                import mlx.core as mx
                np_batch = batch_torch.numpy().transpose(0, 2, 3, 1)
                mlx_batch = mx.array(np_batch).astype(mlx_dtype)
                embs = inference_fn(mlx_batch)
                mx.eval(embs)
                embs_np = np.array(embs).astype(np.float32)
                norms = np.linalg.norm(embs_np, axis=-1, keepdims=True)
                embs_np = embs_np / np.maximum(norms, 1e-12)
                results.append(embs_np)
            else:
                # PyTorch path
                batch_tensor = batch_torch.to(ctx.device)
                with torch.no_grad():
                    embs = inference_fn(batch_tensor)
                    embs = embs / embs.norm(dim=-1, keepdim=True)
                results.append(embs.cpu().numpy().astype(np.float32))

            _report_progress(label, batch_end, n, t0)

            if on_checkpoint and ctx.should_checkpoint():
                partial = np.vstack(results)
                on_checkpoint(partial, items[:batch_end])
                print(f"  Checkpoint saved: {batch_end}/{n}", file=sys.stderr)

            if ctx.interrupted:
                break

    return np.vstack(results) if results else np.zeros((0, 0), dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Per-model extractors
# ────────────────────────────────────────────────────────────────────────────


def extract_color(items, ctx):
    """Color histogram pass (CPU, multiprocess)."""
    from concurrent.futures import ProcessPoolExecutor
    n = len(items)
    n_workers = os.cpu_count() or 4
    print(f"  [Pass {ctx.next_pass()}/{ctx.total_passes}] Color histograms "
          f"({n} images, {n_workers} workers)", file=sys.stderr)

    color_results = []
    t0 = time.time()
    ex = ProcessPoolExecutor(max_workers=n_workers)
    try:
        futures = [ex.submit(_color_extract_worker, (ctx.image_dir, f)) for f, _ in items]
        for i, fut in enumerate(futures):
            if ctx.interrupted:
                for pending in futures[i:]:
                    pending.cancel()
                break
            feat, err = fut.result()
            if err is not None:
                print(f"  WARNING: skipping {items[i][0]}: {err}", file=sys.stderr)
                color_results.append(np.zeros(COLOR_DIM, dtype=np.float32))
            else:
                color_results.append(feat)
            if (i + 1) % 200 == 0 or i == n - 1:
                _report_progress("Color", i + 1, n, t0)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    n_done = len(color_results)
    new_arr = np.array(color_results, dtype=np.float32) if color_results else np.zeros((0, COLOR_DIM), dtype=np.float32)
    return new_arr, items[:n_done]


def extract_open_clip(key, model_name, pretrained, hw, batch_size_eff, label,
                     items, ctx, args, save_to_cache):
    """CLIP / PE-Core-L / PE-Core-G extraction. Switches MLX vs torch for PE-Core-G."""
    print(f"  [Pass {ctx.next_pass()}/{ctx.total_passes}] {label} ({len(items)} images)",
          file=sys.stderr)

    use_mlx_pecore_g = key == "pecore_g" and args.pecore_g_backend == "mlx"

    if use_mlx_pecore_g:
        import mlx.core as mx
        from torchvision import transforms
        mlx_pe_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx_pe_core")
        if mlx_pe_dir not in sys.path:
            sys.path.insert(0, mlx_pe_dir)
        from model import PECoreBigG

        weights_path = os.path.expanduser("~/.cache/mlx-pe-core-bigg.safetensors")
        if not os.path.exists(weights_path):
            print(f"  MLX weights not found at {weights_path}", file=sys.stderr)
            print(f"  Running one-time conversion (~5 min, writes ~7.5GB)...", file=sys.stderr)
            from convert import convert as _mlx_convert
            _mlx_convert(weights_path, dtype="float32")

        # Verified bit-exact vs open_clip.create_model_and_transforms('PE-Core-bigG-14-448').
        preprocess = transforms.Compose([
            transforms.Resize(hw, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(hw),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        mlx_dtype = getattr(mx, args.pecore_g_mlx_dtype)
        mlx_model = PECoreBigG()
        mlx_model.load_weights(weights_path, strict=False)
        mlx_model.set_dtype(mlx_dtype)
        mlx_model.eval()
        mx.eval(mlx_model.parameters())

        embs = _run_pass(
            items, preprocess, hw, batch_size_eff, label,
            inference_fn=mlx_model, ctx=ctx, mlx_dtype=mlx_dtype,
            on_checkpoint=lambda e, i: save_to_cache(key, e, i),
        )
        ctx.free_model(mlx_model)
        return embs

    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=ctx.device,
    )
    model.eval()

    # PE-Core-G runs in fp16 on MPS: ~2x faster, embeddings re-cast to fp32.
    is_fp16 = key == "pecore_g"
    if is_fp16:
        model = model.half()
    encode_fn = (lambda x, _m=model: _m.encode_image(x.half())) if is_fp16 else model.encode_image

    embs = _run_pass(
        items, preprocess, hw, batch_size_eff, label,
        inference_fn=encode_fn, ctx=ctx,
        on_checkpoint=lambda e, i: save_to_cache(key, e, i),
    )
    ctx.free_model(model, preprocess)
    return embs


def extract_dinov2(items, ctx, args, save_to_cache):
    """DINOv2 ViT-L/14 extraction (torch.hub)."""
    import contextlib
    import torch
    from torchvision import transforms
    print(f"  [Pass {ctx.next_pass()}/{ctx.total_passes}] DINOv2 ViT-L/14 ({len(items)} images)",
          file=sys.stderr)
    with contextlib.redirect_stdout(sys.stderr):
        dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    dino_model = dino_model.to(ctx.device).eval()
    dino_preprocess = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embs = _run_pass(
        items, dino_preprocess, 518, max(1, args.batch_size // 2), "DINOv2",
        inference_fn=dino_model, ctx=ctx,
        on_checkpoint=lambda e, i: save_to_cache("dino", e, i),
    )
    ctx.free_model(dino_model, dino_preprocess)
    return embs


def extract_dinov3(items, ctx, args, save_to_cache, save_patches_cache):
    """DINOv3 ViT-B/16 extraction (transformers, local weights).

    Special-cased — produces three outputs per batch (CLS, pooled patches,
    full-res patches), so it doesn't go through `_run_pass`.
    Returns (cls_embs, patch_embs_pooled, patch_embs_full, items_done).
    """
    import torch
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image
    print(f"  [Pass {ctx.next_pass()}/{ctx.total_passes}] DINOv3 ViT-B/16 ({len(items)} images)",
          file=sys.stderr)

    dinov3_model = AutoModel.from_pretrained(DINOV3_WEIGHTS)
    dinov3_processor = AutoImageProcessor.from_pretrained(DINOV3_WEIGHTS)
    dinov3_model = dinov3_model.to(ctx.device).eval()

    n = len(items)
    cls_results = []
    patch_results = []
    patch_full_results = []
    t0 = time.time()
    bs = max(1, args.batch_size // 4)  # smaller batches for patch memory

    for batch_start in range(0, n, bs):
        batch_end = min(batch_start + bs, n)
        images = []
        for i in range(batch_start, batch_end):
            fname, _h = items[i]
            path = os.path.join(ctx.image_dir, fname)
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                print(f"  WARNING: skipping {fname}: {e}", file=sys.stderr)
                images.append(Image.new("RGB", (224, 224)))

        inputs = dinov3_processor(images=images, return_tensors="pt").to(ctx.device)
        with torch.no_grad():
            outputs = dinov3_model(**inputs)
            hidden = outputs.last_hidden_state  # [B, 1+4+196, 768]
            cls_tokens = hidden[:, 0, :]  # [B, 768]
            patch_tokens = hidden[:, 5:, :]  # [B, 196, 768] — skip CLS + 4 registers

            # L2-normalize CLS
            cls_tokens = cls_tokens / cls_tokens.norm(dim=-1, keepdim=True)

            # Full-res patches: L2-normalize the raw 14x14 grid
            patch_tokens_full = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)

            # Average-pool 14x14 patch grid to 7x7 for efficiency
            B = patch_tokens.shape[0]
            grid = patch_tokens.view(B, 14, 14, DINOV3_PATCH_DIM).permute(0, 3, 1, 2)
            pooled = torch.nn.functional.avg_pool2d(grid, kernel_size=2, stride=2)
            patch_tokens = pooled.permute(0, 2, 3, 1).reshape(B, DINOV3_N_PATCHES, DINOV3_PATCH_DIM)
            patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)

        cls_results.append(cls_tokens.cpu().numpy().astype(np.float32))
        patch_results.append(patch_tokens.cpu().numpy().astype(np.float32))
        patch_full_results.append(patch_tokens_full.cpu().numpy().astype(np.float32))
        _report_progress("DINOv3", batch_end, n, t0)

        if ctx.should_checkpoint():
            partial_cls = np.vstack(cls_results)
            partial_patches = np.vstack(patch_results)
            partial_patches_full = np.vstack(patch_full_results)
            items_done = items[:batch_end]
            save_to_cache("dinov3", partial_cls, items_done)
            save_patches_cache(partial_patches, partial_patches_full, items_done)
            print(f"  Checkpoint saved: {batch_end}/{n}", file=sys.stderr)

        if ctx.interrupted:
            break

    n_done = sum(r.shape[0] for r in cls_results)
    cls_arr = np.vstack(cls_results) if cls_results else np.zeros((0, DINOV3_CLS_DIM), dtype=np.float32)
    patch_arr = np.vstack(patch_results) if patch_results else np.zeros((0, DINOV3_N_PATCHES, DINOV3_PATCH_DIM), dtype=np.float32)
    patch_full_arr = np.vstack(patch_full_results) if patch_full_results else np.zeros((0, DINOV3_N_PATCHES_FULL, DINOV3_PATCH_DIM), dtype=np.float32)

    ctx.free_model(dinov3_model, dinov3_processor)
    return cls_arr, patch_arr, patch_full_arr, items[:n_done]


# ────────────────────────────────────────────────────────────────────────────
# Extraction context — bundles shared state for extractors
# ────────────────────────────────────────────────────────────────────────────


class ExtractCtx:
    """Shared state passed to per-model extractors."""

    def __init__(self, image_dir, total_passes):
        self.image_dir = image_dir
        self.total_passes = total_passes
        self._pass_num = 0
        self.interrupted = False
        self._last_checkpoint = time.time()
        self.device = None  # set after torch import

    def next_pass(self):
        self._pass_num += 1
        return self._pass_num

    def should_checkpoint(self):
        now = time.time()
        if now - self._last_checkpoint >= CHECKPOINT_SEC:
            self._last_checkpoint = now
            return True
        return False

    def free_model(self, *objs):
        import gc
        for o in objs:
            del o
        try:
            import torch
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except ImportError:
            pass
        gc.collect()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP + PE-Core + color features")
    parser.add_argument("image_dir", help="Directory containing images")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache directory (default: <image_dir>/.reorder-cache)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--models", default=None,
                        help="Comma-separated list of models to force re-extract (e.g. 'pecore_l,pecore_g'). "
                             "Default: extract all models with missing/outdated cache.")
    parser.add_argument("--required", default=None,
                        help="Comma-separated list of required models. Only these (if missing/outdated) "
                             "will be extracted; others are skipped even if missing.")
    parser.add_argument("--pecore-g-backend", default="mlx", choices=["pytorch", "mlx"],
                        help="Backend for PE-Core-G inference. mlx (default) uses our native "
                             "MLX port (requires running scripts/mlx_pe_core/convert.py once); "
                             "pytorch falls back to open_clip's MPS path in fp16.")
    parser.add_argument("--pecore-g-mlx-dtype", default="float32", choices=["float16", "float32"],
                        help="MLX dtype for PE-Core-G. fp32 default, fp16 saves a small amount of time")
    args = parser.parse_args()

    forced_models = set(args.models.split(",")) if args.models else None
    required_set = set(args.required.split(",")) if args.required else None

    image_dir = os.path.abspath(args.image_dir)
    cache_dir = args.cache_dir or os.path.join(image_dir, ".reorder-cache")
    os.makedirs(cache_dir, exist_ok=True)

    hash_cache_path = os.path.join(cache_dir, "clip_hash_cache.npz")
    hash_cache_order_path = os.path.join(cache_dir, "hash_cache_order.json")
    # Lock-gap fix: write to .tmp; the TS side (src/cluster/pipeline.ts) renames
    # to the final path under the FS lock so concurrent /api/save can't observe
    # the cache mid-write.
    content_hashes_tmp_path = os.path.join(cache_dir, "content_hashes.json.tmp")
    patches_cache_hashes_path = os.path.join(cache_dir, "dinov3_patches_hashes.json")
    patches_cache_data_path = os.path.join(cache_dir, "dinov3_patches_hash_cache.npy")
    patches_full_cache_data_path = os.path.join(cache_dir, "dinov3_patches_full_hash_cache.npy")

    # Find image files
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts
    )
    print(f"Found {len(image_files)} images in {image_dir}", file=sys.stderr)

    # Compute content hashes
    print("Hashing files...", file=sys.stderr)
    t0 = time.time()
    current_hashes = {}
    for f in image_files:
        current_hashes[f] = content_hash(os.path.join(image_dir, f))
    print(f"  Hashed {len(current_hashes)} files in {time.time()-t0:.1f}s", file=sys.stderr)

    # Write content hashes bridge file (filename → content hash) to tmp.
    with open(content_hashes_tmp_path, "w") as fp:
        json.dump(current_hashes, fp)

    # Load existing cache with per-model version check.
    cached_hashes, cached, models_to_extract, _initial_versions = _load_existing_cache(hash_cache_path)

    # Compute zero-fill / forced/required filtering
    zero_fill_needed = {}
    if forced_models:
        invalid = forced_models - set(EMB_KEYS)
        if invalid:
            print(f"  WARNING: unknown models: {invalid}", file=sys.stderr)
        models_to_extract = forced_models & set(EMB_KEYS)
        print(f"  Requested models: {models_to_extract}", file=sys.stderr)
    else:
        zero_fill_needed = _detect_zero_fill(
            cached, cached_hashes, current_hashes, models_to_extract,
            patches_cache_hashes_path, patches_cache_data_path,
        )

    if required_set and not forced_models:
        skipped = models_to_extract - required_set
        if skipped:
            print(f"  Skipping unrequired models: {skipped}", file=sys.stderr)
            models_to_extract &= required_set

    # Find which content hashes need extraction (new images not in cache)
    needed = []
    for f in image_files:
        h = current_hashes[f]
        if h not in cached_hashes:
            needed.append((f, h))

    needs_new_images = len(needed) > 0
    needs_model_reextract = len(models_to_extract) > 0
    needs_zero_fill = bool(zero_fill_needed) and not forced_models
    if needs_zero_fill and required_set:
        needs_zero_fill = bool(set(zero_fill_needed.keys()) & required_set)

    if not needs_new_images and not needs_model_reextract and not needs_zero_fill:
        print("All features cached, nothing to extract.", file=sys.stderr)
        json.dump({
            "total": len(image_files), "cached": len(image_files),
            "extracted": 0, "cachePath": hash_cache_path,
        }, sys.stdout)
        return

    if needs_model_reextract:
        print(f"Models to re-extract: {models_to_extract}", file=sys.stderr)
    if needs_zero_fill:
        zf_summary = {k: len(v) for k, v in zero_fill_needed.items()}
        print(f"Models to incrementally fill: {zf_summary}", file=sys.stderr)
    if needs_new_images:
        print(f"New images to extract: {len(needed)}", file=sys.stderr)

    # ── Interrupt handling ───────────────────────────────────────────────────
    # Ctrl+C saves partial results so the next run resumes from where it stopped.
    # Second Ctrl+C forces immediate exit.
    import signal

    all_items = [(f, current_hashes[f]) for f in image_files]

    # Determine items per model
    def _items_for(model_key):
        if model_key in models_to_extract:
            return all_items  # full re-extraction (version mismatch)
        if forced_models:
            return []
        if required_set and model_key not in required_set:
            return []
        zero_items = zero_fill_needed.get(model_key, [])
        combined = list(needed) + zero_items
        return combined if combined else []

    items_map = {k: _items_for(k) for k in EMB_KEYS}
    total_passes = sum(1 for k in EMB_KEYS if items_map[k])

    ctx = ExtractCtx(image_dir, total_passes)

    def _handle_sigint(signum, frame):
        if ctx.interrupted:
            sys.exit(1)
        ctx.interrupted = True
        print("\n  Interrupted — saving after current batch... (Ctrl+C again to force quit)", file=sys.stderr)

    prev_sigint = signal.signal(signal.SIGINT, _handle_sigint)

    # Only import torch if a neural model pass is needed.
    _neural_keys = {"clip", "pecore_l", "pecore_g", "dino", "dinov3"}
    if any(items_map[k] for k in _neural_keys):
        import torch
        ctx.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"  Using device: {ctx.device}", file=sys.stderr)

    # In-memory cache state — avoids reloading the full NPZ on every save
    _mem_hashes = dict(cached_hashes)
    _mem_arrays = dict(cached)
    _mem_versions = dict(_initial_versions)

    def _save_model_to_cache(key, new_arr, items_used):
        """Incrementally save one model's results to the hash cache.

        Uses merge_cached_array to compute the active model's merged array;
        other models are reindexed to the same hash list.
        """
        nonlocal _mem_hashes, _mem_arrays, _mem_versions
        new_hashes = [h for _, h in items_used]

        # Merge active model — establishes the new hash list
        merged_arr, all_h_list = merge_cached_array(
            _mem_arrays.get(key), _mem_hashes,
            new_arr, new_hashes,
        )
        h2i = {h: i for i, h in enumerate(all_h_list)}
        n = len(all_h_list)

        save_arrays = {key: merged_arr}

        # Reindex other already-cached models to the new hash list (no new data)
        for k2 in EMB_KEYS:
            if k2 == key or k2 not in _mem_arrays:
                continue
            old_arr = _mem_arrays[k2]
            out = np.zeros((n,) + old_arr.shape[1:], dtype=np.float32)
            for h, old_i in _mem_hashes.items():
                tgt = h2i.get(h)
                if tgt is not None:
                    out[tgt] = old_arr[old_i]
            save_arrays[k2] = out

        _mem_versions[f"_v_{key}"] = np.array(MODEL_VERSIONS[key])

        np.savez_compressed(
            hash_cache_path,
            hashes=np.array(all_h_list),
            **_mem_versions,
            **save_arrays,
        )
        with open(hash_cache_order_path, "w") as f:
            json.dump(all_h_list, f)

        _mem_hashes = h2i
        _mem_arrays = save_arrays
        print(f"  Saved {key} to cache ({n} entries)", file=sys.stderr)

    # ── Patches cache (DINOv3) ───────────────────────────────────────────────
    # Load existing
    pc_hashes = {}
    pc_arr = None
    if os.path.exists(patches_cache_hashes_path) and os.path.exists(patches_cache_data_path):
        try:
            with open(patches_cache_hashes_path) as f:
                _pl = json.load(f)
            pc_arr = np.load(patches_cache_data_path)
            pc_hashes = {h: i for i, h in enumerate(_pl)}
        except Exception:
            pass

    pcf_arr = None
    if os.path.exists(patches_cache_hashes_path) and os.path.exists(patches_full_cache_data_path):
        try:
            pcf_arr = np.load(patches_full_cache_data_path)
        except Exception:
            pass

    def _save_patches_cache(new_patches, new_patches_full, items_used):
        """Save DINOv3 patches (pooled + full-res) to hash-keyed cache."""
        nonlocal pc_hashes, pc_arr, pcf_arr
        new_h = [h for _, h in items_used]
        new_h_set = set(new_h)

        # Fast path: in-place update if hash set unchanged
        if pc_arr is not None and pcf_arr is not None and new_h_set.issubset(pc_hashes.keys()):
            new_h2r = {hh: i for i, hh in enumerate(new_h)}
            dst = np.array([pc_hashes[hh] for hh in new_h2r], dtype=np.intp)
            src = np.array(list(new_h2r.values()), dtype=np.intp)
            pc_arr[dst] = new_patches[src]
            np.save(patches_cache_data_path, pc_arr)
            pcf_arr[dst] = new_patches_full[src]
            np.save(patches_full_cache_data_path, pcf_arr)
            return

        # Full reindex via merge_cached_array
        merged_pooled, out_hashes = merge_cached_array(
            pc_arr, pc_hashes, new_patches, new_h,
        )
        merged_full, _ = merge_cached_array(
            pcf_arr, pc_hashes, new_patches_full, new_h,
        )
        np.save(patches_cache_data_path, merged_pooled)
        np.save(patches_full_cache_data_path, merged_full)
        with open(patches_cache_hashes_path, "w") as f:
            json.dump(out_hashes, f)
        pc_hashes = {h: i for i, h in enumerate(out_hashes)}
        pc_arr = merged_pooled
        pcf_arr = merged_full

    # ── Run extractors ───────────────────────────────────────────────────────

    new_arrays = {}
    new_color = np.zeros((0, COLOR_DIM), dtype=np.float32)

    if items_map["color"] and not ctx.interrupted:
        new_color, items_done = extract_color(items_map["color"], ctx)
        _save_model_to_cache("color", new_color, items_done)

    OPEN_CLIP_MODELS = [
        ("clip",     "ViT-B-32",           "laion2b_s34b_b79k", 224, 4, 1,  512, "CLIP"),
        ("pecore_l", "PE-Core-L-14-336",   "meta",              336, 1, 2, 1024, "PE-Core-L"),
        ("pecore_g", "PE-Core-bigG-14-448","meta",              448, 1, 8, 1280, "PE-Core-G"),
    ]

    for key, model_name, pretrained, hw, batch_mult, batch_div, dim, label in OPEN_CLIP_MODELS:
        model_items = items_map[key]
        if model_items and not ctx.interrupted:
            batch_size_eff = max(1, args.batch_size * batch_mult // batch_div)
            embs = extract_open_clip(
                key, model_name, pretrained, hw, batch_size_eff, label,
                model_items, ctx, args, _save_model_to_cache,
            )
            n_done = embs.shape[0]
            new_arrays[key] = embs
            _save_model_to_cache(key, embs, model_items[:n_done])
        else:
            new_arrays[key] = np.zeros((0, dim), dtype=np.float32)

    new_clip = new_arrays["clip"]
    new_pecore_l = new_arrays["pecore_l"]
    new_pecore_g = new_arrays["pecore_g"]

    # DINOv2
    dino_items = items_map["dino"]
    if dino_items and not ctx.interrupted:
        new_dino = extract_dinov2(dino_items, ctx, args, _save_model_to_cache)
        n_done = new_dino.shape[0]
        _save_model_to_cache("dino", new_dino, dino_items[:n_done])
    else:
        new_dino = np.zeros((0, 1024), dtype=np.float32)

    # DINOv3
    dinov3_items = items_map["dinov3"]
    if dinov3_items and not ctx.interrupted:
        new_dinov3, dinov3_patches, dinov3_patches_full, items_done = extract_dinov3(
            dinov3_items, ctx, args, _save_model_to_cache, _save_patches_cache,
        )
        _save_model_to_cache("dinov3", new_dinov3, items_done)
        if dinov3_patches.shape[0] > 0:
            _save_patches_cache(dinov3_patches, dinov3_patches_full, items_done)
    else:
        new_dinov3 = np.zeros((0, DINOV3_CLS_DIM), dtype=np.float32)

    # ── Interrupted? Skip final merge — intermediate saves are the checkpoint ──
    if ctx.interrupted:
        signal.signal(signal.SIGINT, prev_sigint)
        print(f"\n  Extraction interrupted. Partial results saved to cache.", file=sys.stderr)
        print(f"  Re-run to continue from where it left off.", file=sys.stderr)
        json.dump({"interrupted": True, "total": len(image_files), "extracted": 0}, sys.stdout)
        return

    signal.signal(signal.SIGINT, prev_sigint)

    # ── Final merge with existing cache ──────────────────────────────────────
    # For each model: if it was fully re-extracted, the new array IS the complete data.
    # If only new images were extracted, merge with existing cache.
    # Constrain to current images only — prune stale entries from
    # deleted/moved files to prevent unbounded cache growth.
    new_hashes_map = {k: [h for _, h in items_map[k]] for k in EMB_KEYS}
    new_data_map = {
        "clip": new_clip, "dino": new_dino,
        "pecore_l": new_pecore_l, "pecore_g": new_pecore_g,
        "color": new_color, "dinov3": new_dinov3,
    }

    current_hash_set = set(current_hashes.values())
    n_pruned = len(set(cached_hashes.keys()) - current_hash_set)
    if n_pruned:
        print(f"  Pruning {n_pruned} stale entries from hash cache", file=sys.stderr)

    all_arrays = {}
    final_hash_list = None
    for k in EMB_KEYS:
        new_arr = new_data_map[k]
        new_hashes = new_hashes_map[k]

        if k in models_to_extract:
            # Full re-extraction: only the new array is authoritative
            existing_arr_for_merge = None
            existing_hashes_for_merge = []
        elif len(new_arr) > 0:
            # Incremental: cached + new
            existing_arr_for_merge = cached.get(k)
            existing_hashes_for_merge = cached_hashes
        else:
            # Fully cached, just reindex (skip if not in cache)
            if k not in cached:
                continue
            existing_arr_for_merge = cached[k]
            existing_hashes_for_merge = cached_hashes

        out, out_hashes = merge_cached_array(
            existing_arr_for_merge, existing_hashes_for_merge,
            new_arr, new_hashes,
            hash_universe=current_hash_set,
        )
        all_arrays[k] = out
        if final_hash_list is None:
            final_hash_list = out_hashes

    if final_hash_list is None:
        # No models had any data; build an empty hash list constrained to current.
        final_hash_list = sorted(current_hash_set)

    # Save hash-keyed cache with per-model version keys
    version_keys = {f"_v_{k}": np.array(v) for k, v in MODEL_VERSIONS.items() if k in all_arrays}
    np.savez_compressed(
        hash_cache_path,
        hashes=np.array(final_hash_list),
        **version_keys,
        **all_arrays,
    )
    print(f"  Saved hash cache: {hash_cache_path}", file=sys.stderr)

    with open(hash_cache_order_path, "w") as f:
        json.dump(final_hash_list, f)

    # Prune patches cache to current images only
    if pc_arr is not None and pc_hashes:
        pruned_hashes = sorted(h for h in pc_hashes if h in current_hash_set)
        if len(pruned_hashes) < len(pc_hashes):
            pruned_h2i = {h: i for i, h in enumerate(pruned_hashes)}
            pruned_arr = np.zeros(
                (len(pruned_hashes), DINOV3_N_PATCHES, DINOV3_PATCH_DIM), dtype=np.float32,
            )
            for h, new_i in pruned_h2i.items():
                pruned_arr[new_i] = pc_arr[pc_hashes[h]]
            np.save(patches_cache_data_path, pruned_arr)
            with open(patches_cache_hashes_path, "w") as f:
                json.dump(pruned_hashes, f)
            print(f"  Pruned patches cache: {len(pc_hashes)} → {len(pruned_hashes)}", file=sys.stderr)

    json.dump({
        "total": len(image_files),
        "cached": len(image_files) - len(needed),
        "extracted": len(needed),
        "cachePath": hash_cache_path,
    }, sys.stdout)


if __name__ == "__main__":
    main()
