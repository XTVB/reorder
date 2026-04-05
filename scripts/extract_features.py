#!/usr/bin/env python3
"""Extract CLIP + PE-Core + color features from images, cached by content hash.

Only does feature extraction — no clustering. Outputs a manifest JSON to stdout.
Progress is reported on stderr.

Models:
  - CLIP ViT-B/32 (512-dim) — kept for TF-IDF auto-naming
  - PE-Core-L-14-336 (1024-dim) — Meta Perception Encoder, large variant
  - PE-Core-bigG-14-448 (1280-dim) — Meta Perception Encoder, giant variant
  - Color histograms (77-dim) — HSV + RGB moments

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

import numpy as np

# Per-model version keys. Only models whose version changed get re-extracted.
# The cache stores "_v_<key>" for each model. Missing or mismatched → re-extract that model only.
MODEL_VERSIONS = {
    "clip": "ViT-B-32-laion2b-v1",
    "dino": "dinov2-vitl14-v1",
    "pecore_l": "PE-Core-L-14-336-meta-v1",
    "pecore_g": "PE-Core-bigG-14-448-meta-v1",
    "color": "hsv-rgb-77d-v1",
    "dinov3": "dinov3-vitb16-7x7pool-v2",
}

# All embedding arrays stored in the npz
EMB_KEYS = list(MODEL_VERSIONS.keys())

# DINOv3 local weights path (downloaded from Kaggle)
DINOV3_WEIGHTS = os.environ.get(
    "DINOV3_WEIGHTS",
    "/tmp/dinov3-weights/facebook/dinov3-vitb16-pretrain-lvd1689m",
)


def content_hash(filepath: str) -> str:
    """Fast content-based hash: blake2b(first 16KB + file size)."""
    size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        head = f.read(16384)
    return hashlib.blake2b(head, digest_size=16, key=size.to_bytes(8, "big")).hexdigest()


def extract_color_features(img_rgb, thumb_size=128):
    """Extract HSV histogram + RGB color moments (77 dimensions)."""
    thumb = img_rgb.resize((thumb_size, thumb_size))
    arr = np.array(thumb, dtype=np.float32)
    hsv = np.array(thumb.convert("HSV"), dtype=np.float32)

    feats = []
    for ch, bins in [(0, 36), (1, 16), (2, 16)]:
        h, _ = np.histogram(hsv[:, :, ch], bins=bins, range=(0, 256))
        h = h.astype(np.float32) / (h.sum() + 1e-10)
        feats.extend(h)
    for ch in range(3):
        d = arr[:, :, ch]
        mu, sigma = d.mean(), d.std()
        feats.extend([
            mu / 256.0,
            sigma / 128.0,
            float(np.mean(((d - mu) / max(sigma, 1.0)) ** 3)) / 5.0,
        ])
    return np.array(feats, dtype=np.float32)


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
    args = parser.parse_args()

    # Parse set arguments once up front
    forced_models = set(args.models.split(",")) if args.models else None
    required_set = set(args.required.split(",")) if args.required else None

    image_dir = os.path.abspath(args.image_dir)
    cache_dir = args.cache_dir or os.path.join(image_dir, ".reorder-cache")
    os.makedirs(cache_dir, exist_ok=True)

    hash_cache_path = os.path.join(cache_dir, "clip_hash_cache.npz")
    npz_path = os.path.join(cache_dir, "clip_embeddings.npz")
    filenames_path = os.path.join(cache_dir, "clip_embeddings.filenames.json")

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

    # Load existing cache with per-model version check.
    # Models whose version matches are kept; mismatched or missing models are re-extracted.
    cached_hashes = {}
    cached = {}  # key -> np array
    models_to_extract = set(EMB_KEYS)  # start assuming all need extraction
    if os.path.exists(hash_cache_path):
        try:
            data = np.load(hash_cache_path, allow_pickle=True)
            if "hashes" in data:
                cached_hash_list = list(data["hashes"])
                cached_hashes = {h: i for i, h in enumerate(cached_hash_list)}

                # Migrate old monolithic _model_version to per-model versions.
                # Old caches have clip, dino, color arrays + a single _model_version string.
                old_version = str(data["_model_version"]) if "_model_version" in data else None
                old_compat = {
                    "clip": old_version and "clip-ViT-B-32" in old_version,
                    "dino": old_version and "dinov2-vitl14" in old_version,
                    "color": old_version and "color77" in old_version,
                }

                # Check each model's version individually
                for k in EMB_KEYS:
                    version_key = f"_v_{k}"
                    stored = str(data[version_key]) if version_key in data else None
                    # Accept either new per-model key or old monolithic compatibility
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
        except Exception as e:
            print(f"  Hash cache corrupt, rebuilding: {e}", file=sys.stderr)
            cached_hashes = {}
            cached = {}
            models_to_extract = set(EMB_KEYS)

    # If --models is specified, force re-extract those specific models.
    # If --required is specified, only extract models in that set (if missing/outdated).
    if forced_models:
        invalid = forced_models - set(EMB_KEYS)
        if invalid:
            print(f"  WARNING: unknown models: {invalid}", file=sys.stderr)
        # Force re-extraction of requested models, keep others as-is
        models_to_extract = forced_models & set(EMB_KEYS)
        print(f"  Requested models: {models_to_extract}", file=sys.stderr)
    else:
        # Auto/required mode: detect zero-padded entries from previous partial --models runs.
        # L2-normalized embeddings can never be all-zero; zeros mean the entry was
        # never actually extracted (hash-list expansion zero-padded it).
        for k in list(cached.keys()):
            if k in models_to_extract or k not in {"clip", "dino", "pecore_l", "pecore_g"}:
                continue
            arr = cached[k]
            if np.any(~np.any(arr, axis=1)):
                models_to_extract.add(k)
                del cached[k]
                print(f"  {k}: zero embeddings detected (incomplete extraction), will re-extract", file=sys.stderr)

    # --required: limit extraction to only the required models (don't extract others
    # even if missing). Unlike --models, this doesn't force re-extraction.
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

    # If any model needs extraction, ALL images need to go through that model
    # (even if their content hash is cached for other models)
    needs_new_images = len(needed) > 0
    needs_model_reextract = len(models_to_extract) > 0
    if needs_model_reextract:
        # For models that need re-extraction, we need ALL images
        all_needed = [(f, current_hashes[f]) for f in image_files]
    else:
        all_needed = needed

    if not needs_new_images and not needs_model_reextract:
        print("All features cached, nothing to extract.", file=sys.stderr)
        ordered = {}
        for k in EMB_KEYS:
            if k not in cached:
                continue
            arr = cached[k]
            out = np.zeros((len(image_files), arr.shape[1]), dtype=np.float32)
            for i, f in enumerate(image_files):
                out[i] = arr[cached_hashes[current_hashes[f]]]
            ordered[k] = out
        np.savez_compressed(npz_path, **ordered)
        with open(filenames_path, "w") as fp:
            json.dump(image_files, fp)
        json.dump({
            "total": len(image_files), "cached": len(image_files),
            "extracted": 0, "cache_path": npz_path,
        }, sys.stdout)
        return

    if needs_model_reextract:
        print(f"Models to re-extract: {models_to_extract}", file=sys.stderr)
    if needs_new_images:
        print(f"New images to extract: {len(needed)}", file=sys.stderr)

    from PIL import Image
    import gc

    # For models needing full re-extraction, process ALL images.
    # For models only needing new images, process just `needed`.
    all_items = [(f, current_hashes[f]) for f in image_files]

    # ── Prefetch helper ──────────────────────────────────────────────────────

    def _run_pass(items, transform, fallback_hw, batch_size, label, encode_fn, extract_color=False):
        """Run batched inference with 1-batch-ahead prefetch. `items` is [(fname, hash)]."""
        n = len(items)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32), []

        def _prepare_batch(indices):
            tensors = []
            colors = []
            for i in indices:
                fname, h = items[i]
                path = os.path.join(image_dir, fname)
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(transform(img))
                    if extract_color:
                        colors.append(extract_color_features(img))
                except Exception as e:
                    print(f"  WARNING: skipping {fname}: {e}", file=sys.stderr)
                    tensors.append(torch.zeros(3, fallback_hw, fallback_hw))
                    if extract_color:
                        colors.append(np.zeros(77, dtype=np.float32))
            return torch.stack(tensors), colors

        results = []
        all_colors = []
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=1) as pool:
            first_end = min(batch_size, n)
            future = pool.submit(_prepare_batch, range(0, first_end))

            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                batch_tensor, colors = future.result()
                batch_tensor = batch_tensor.to(device)

                next_start = batch_start + batch_size
                if next_start < n:
                    next_end = min(next_start + batch_size, n)
                    future = pool.submit(_prepare_batch, range(next_start, next_end))

                with torch.no_grad():
                    embs = encode_fn(batch_tensor)
                    embs = embs / embs.norm(dim=-1, keepdim=True)
                results.append(embs.cpu().numpy().astype(np.float32))
                all_colors.extend(colors)
                _report_progress(label, batch_end, n, t0)

        return np.vstack(results), all_colors

    # For each model, decide what to extract:
    # - Model in models_to_extract → run on ALL images (all_items)
    # - Model NOT in models_to_extract but new images exist → run on `needed` only
    #   (only in pure auto mode — not when --models or --required limits the scope)
    # - Otherwise → skip (already cached or not requested)
    def _items_for(model_key):
        if model_key in models_to_extract:
            return all_items  # full re-extraction
        elif needed and not forced_models and not required_set:
            return needed     # just new images (pure auto mode only)
        elif needed and required_set and model_key in required_set:
            return needed     # new images for a required model that's already cached
        else:
            return []         # fully cached or not requested

    items_map = {k: _items_for(k) for k in EMB_KEYS}

    # DINOv3 patches are stored as a single 3D array in filename order, not a
    # hash-keyed cache, so we cannot incrementally merge in new rows. If we
    # would run partial extraction, promote to a full re-extraction instead.
    if items_map["dinov3"] and items_map["dinov3"] is not all_items:
        print("  DINOv3 is incremental → promoting to full re-extraction (patches cache is not hash-keyed)", file=sys.stderr)
        items_map["dinov3"] = all_items
        models_to_extract.add("dinov3")

    # Only import heavy ML libraries if a neural model pass is actually needed.
    _neural_keys = {"clip", "pecore_l", "pecore_g", "dino", "dinov3"}
    if any(items_map[k] for k in _neural_keys):
        import torch
        import open_clip
        from concurrent.futures import ThreadPoolExecutor
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"  Using device: {device}", file=sys.stderr)

    def _report_progress(label, done, total, t0):
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  {label}: {done}/{total} ({done/total*100:.0f}%) "
              f"- {rate:.1f} img/s - ETA {eta:.0f}s", file=sys.stderr)

    def _free_model(*objs):
        for o in objs:
            del o
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        gc.collect()

    def _save_model_to_cache(key, new_arr, items_used):
        """Incrementally save one model's results to the hash cache immediately."""
        new_hashes = [h for _, h in items_used]
        # Reload current cache state (may have been updated by previous pass)
        cur_cached = {}
        cur_hashes = {}
        if os.path.exists(hash_cache_path):
            try:
                data = np.load(hash_cache_path, allow_pickle=True)
                if "hashes" in data:
                    cur_hashes = {h: i for i, h in enumerate(list(data["hashes"]))}
                    for k2 in EMB_KEYS:
                        if k2 in data:
                            cur_cached[k2] = data[k2]
            except Exception:
                pass

        # Build unified hash list
        all_h = set(cur_hashes.keys())
        all_h.update(new_hashes)
        all_h_list = sorted(all_h)
        h2i = {h: i for i, h in enumerate(all_h_list)}
        n = len(all_h_list)

        # Build arrays for all models
        save_arrays = {}
        for k2 in EMB_KEYS:
            if k2 == key:
                # This is the model we just extracted
                dim = new_arr.shape[1]
                out = np.zeros((n, dim), dtype=np.float32)
                # Fill from old cache first (for incremental)
                if k2 in cur_cached:
                    for h, old_i in cur_hashes.items():
                        if h in h2i:
                            out[h2i[h]] = cur_cached[k2][old_i]
                # Overwrite with new
                new_h2r = {h: i for i, h in enumerate(new_hashes)}
                for h, row in new_h2r.items():
                    if h in h2i:
                        out[h2i[h]] = new_arr[row]
                save_arrays[k2] = out
            elif k2 in cur_cached:
                # Reindex existing cached array
                dim = cur_cached[k2].shape[1]
                out = np.zeros((n, dim), dtype=np.float32)
                for h, old_i in cur_hashes.items():
                    if h in h2i:
                        out[h2i[h]] = cur_cached[k2][old_i]
                save_arrays[k2] = out
            # else: model not yet extracted, skip (don't create empty placeholder)

        # Build version keys — keep existing, update current model
        version_keys = {}
        if os.path.exists(hash_cache_path):
            try:
                data = np.load(hash_cache_path, allow_pickle=True)
                for k2 in EMB_KEYS:
                    vk = f"_v_{k2}"
                    if vk in data:
                        version_keys[vk] = data[vk]
                # Also keep old _model_version for compat
                if "_model_version" in data:
                    version_keys["_model_version"] = data["_model_version"]
            except Exception:
                pass
        version_keys[f"_v_{key}"] = np.array(MODEL_VERSIONS[key])

        np.savez_compressed(
            hash_cache_path,
            hashes=np.array(all_h_list),
            **version_keys,
            **save_arrays,
        )
        print(f"  Saved {key} to cache ({n} entries)", file=sys.stderr)

    # ── Standalone color extraction (when requested without CLIP) ──────────
    # Color features are just HSV histograms + RGB moments — no model needed.
    # Normally color piggybacks on the CLIP pass, but when requested independently
    # via --models, extract standalone with just PIL.
    standalone_color = (
        "color" in models_to_extract
        and "clip" not in models_to_extract
        and not items_map["clip"]
    )

    pass_num = 0
    # Color doesn't have its own pass unless standalone
    total_passes = sum(1 for k in EMB_KEYS if k != "color" and items_map[k])
    if standalone_color:
        total_passes += 1

    new_arrays = {}
    new_color = np.zeros((0, 77), dtype=np.float32)

    if standalone_color:
        color_items = items_map["color"]
        pass_num += 1
        print(f"  [Pass {pass_num}/{total_passes}] Color histograms ({len(color_items)} images)", file=sys.stderr)
        color_results = []
        t0 = time.time()
        n = len(color_items)
        for i, (fname, h) in enumerate(color_items):
            path = os.path.join(image_dir, fname)
            try:
                img = Image.open(path).convert("RGB")
                color_results.append(extract_color_features(img))
            except Exception as e:
                print(f"  WARNING: skipping {fname}: {e}", file=sys.stderr)
                color_results.append(np.zeros(77, dtype=np.float32))
            if (i + 1) % 200 == 0 or i == n - 1:
                _report_progress("Color", i + 1, n, t0)
        new_color = np.array(color_results, dtype=np.float32) if color_results else np.zeros((0, 77), dtype=np.float32)
        _save_model_to_cache("color", new_color, color_items)

    # ── open_clip models (data-driven) ───────────────────────────────────────
    # (key, model_name, pretrained, input_hw, batch_mult, batch_div, dim, label)
    OPEN_CLIP_MODELS = [
        ("clip",     "ViT-B-32",           "laion2b_s34b_b79k", 224, 4, 1,  512, "CLIP"),
        ("pecore_l", "PE-Core-L-14-336",   "meta",              336, 1, 2, 1024, "PE-Core-L"),
        ("pecore_g", "PE-Core-bigG-14-448","meta",              448, 1, 8, 1280, "PE-Core-G"),
    ]

    for key, model_name, pretrained, hw, batch_mult, batch_div, dim, label in OPEN_CLIP_MODELS:
        model_items = items_map[key]
        if model_items:
            pass_num += 1
            # Extract color during CLIP pass (unless already done standalone)
            extract_color = (key == "clip") and not standalone_color
            print(f"  [Pass {pass_num}/{total_passes}] {label} ({len(model_items)} images)", file=sys.stderr)
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            model.eval()
            embs, colors = _run_pass(
                model_items, preprocess, hw, max(1, args.batch_size * batch_mult // batch_div),
                label, model.encode_image, extract_color=extract_color,
            )
            new_arrays[key] = embs
            if extract_color:
                new_color = np.array(colors, dtype=np.float32)
                _save_model_to_cache("color", new_color, model_items)
            _save_model_to_cache(key, embs, model_items)
            _free_model(model, preprocess)
        else:
            new_arrays[key] = np.zeros((0, dim), dtype=np.float32)

    new_clip = new_arrays["clip"]
    new_pecore_l = new_arrays["pecore_l"]
    new_pecore_g = new_arrays["pecore_g"]

    # ── DINOv2 ViT-L/14 (uses torch.hub, not open_clip) ─────────────────────
    dino_items = items_map["dino"]
    if dino_items:
        pass_num += 1
        import contextlib
        from torchvision import transforms
        print(f"  [Pass {pass_num}/{total_passes}] DINOv2 ViT-L/14 ({len(dino_items)} images)", file=sys.stderr)
        with contextlib.redirect_stdout(sys.stderr):
            dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        dino_model = dino_model.to(device).eval()
        dino_preprocess = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        new_dino, _ = _run_pass(
            dino_items, dino_preprocess, 518, max(1, args.batch_size // 2), "DINOv2", dino_model,
        )
        _save_model_to_cache("dino", new_dino, dino_items)
        _free_model(dino_model, dino_preprocess)
    else:
        new_dino = np.zeros((0, 1024), dtype=np.float32)

    # ── DINOv3 ViT-B/16 (uses transformers, local weights) ─────────────────
    # Extracts CLS token (768d) as the global embedding stored in the NPZ,
    # and full patch tokens (196 × 768) stored in a separate .npy file for
    # patch-level group similarity matching.
    DINOV3_CLS_DIM = 768
    DINOV3_PATCH_DIM = 768
    DINOV3_N_PATCHES = 49  # 14x14 avg-pooled to 7x7
    dinov3_items = items_map["dinov3"]
    dinov3_patches_path = os.path.join(cache_dir, "dinov3_patches.npy")
    if dinov3_items:
        pass_num += 1
        from torchvision import transforms as tv_transforms
        from transformers import AutoModel, AutoImageProcessor
        print(f"  [Pass {pass_num}/{total_passes}] DINOv3 ViT-B/16 ({len(dinov3_items)} images)", file=sys.stderr)

        dinov3_model = AutoModel.from_pretrained(DINOV3_WEIGHTS)
        dinov3_processor = AutoImageProcessor.from_pretrained(DINOV3_WEIGHTS)
        dinov3_model = dinov3_model.to(device).eval()

        # DINOv3 needs its own run loop because we extract both CLS and patches
        n_dinov3 = len(dinov3_items)
        dinov3_cls_results = []
        dinov3_patch_results = []
        t0 = time.time()
        bs = max(1, args.batch_size // 4)  # smaller batches for patch memory

        for batch_start in range(0, n_dinov3, bs):
            batch_end = min(batch_start + bs, n_dinov3)
            images = []
            for i in range(batch_start, batch_end):
                fname, h = dinov3_items[i]
                path = os.path.join(image_dir, fname)
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception as e:
                    print(f"  WARNING: skipping {fname}: {e}", file=sys.stderr)
                    images.append(Image.new("RGB", (224, 224)))

            inputs = dinov3_processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = dinov3_model(**inputs)
                hidden = outputs.last_hidden_state  # [B, 1+4+196, 768]
                cls_tokens = hidden[:, 0, :]  # [B, 768]
                patch_tokens = hidden[:, 5:, :]  # [B, 196, 768] — skip CLS + 4 registers

                # L2-normalize CLS
                cls_tokens = cls_tokens / cls_tokens.norm(dim=-1, keepdim=True)

                # Average-pool 14x14 patch grid to 7x7 for efficiency
                B = patch_tokens.shape[0]
                grid = patch_tokens.view(B, 14, 14, DINOV3_PATCH_DIM).permute(0, 3, 1, 2)
                pooled = torch.nn.functional.avg_pool2d(grid, kernel_size=2, stride=2)
                patch_tokens = pooled.permute(0, 2, 3, 1).reshape(B, DINOV3_N_PATCHES, DINOV3_PATCH_DIM)
                # Re-normalize after averaging
                patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)

            dinov3_cls_results.append(cls_tokens.cpu().numpy().astype(np.float32))
            dinov3_patch_results.append(patch_tokens.cpu().numpy().astype(np.float32))
            _report_progress("DINOv3", batch_end, n_dinov3, t0)

        new_dinov3 = np.vstack(dinov3_cls_results)
        new_dinov3_patches = np.vstack(dinov3_patch_results)  # [N, 196, 768]
        _save_model_to_cache("dinov3", new_dinov3, dinov3_items)
        _free_model(dinov3_model, dinov3_processor)
    else:
        new_dinov3 = np.zeros((0, DINOV3_CLS_DIM), dtype=np.float32)
        new_dinov3_patches = None

    # ── Merge with existing cache ────────────────────────────────────────────
    # For each model: if it was fully re-extracted, the new array IS the complete data.
    # If only new images were extracted, merge with existing cache.
    # Build a unified hash list from all images we have data for.
    new_hashes_map = {k: [h for _, h in items_map[k]] for k in EMB_KEYS}
    new_data_map = {
        "clip": new_clip, "dino": new_dino,
        "pecore_l": new_pecore_l, "pecore_g": new_pecore_g,
        "color": new_color, "dinov3": new_dinov3,
    }

    # Build unified hash list (union of cached + any new hashes)
    all_hash_set = set(cached_hashes.keys())
    for hashes in new_hashes_map.values():
        all_hash_set.update(hashes)
    all_hash_list = sorted(all_hash_set)
    hash_to_idx = {h: i for i, h in enumerate(all_hash_list)}
    n_total = len(all_hash_list)

    # For each model, build a complete array indexed by all_hash_list
    all_arrays = {}
    for k in EMB_KEYS:
        new_arr = new_data_map[k]
        new_hashes = new_hashes_map[k]

        if k in models_to_extract:
            # Full re-extraction: new_arr has all images, indexed by all_items order
            dim = new_arr.shape[1]
            out = np.zeros((n_total, dim), dtype=np.float32)
            new_hash_to_row = {h: i for i, h in enumerate(new_hashes)}
            for h, idx in hash_to_idx.items():
                if h in new_hash_to_row:
                    out[idx] = new_arr[new_hash_to_row[h]]
            all_arrays[k] = out
        elif len(new_arr) > 0:
            # Incremental: merge cached + new
            dim = cached[k].shape[1] if k in cached else new_arr.shape[1]
            out = np.zeros((n_total, dim), dtype=np.float32)
            # Fill from cache
            if k in cached:
                for h, old_idx in cached_hashes.items():
                    if h in hash_to_idx:
                        out[hash_to_idx[h]] = cached[k][old_idx]
            # Overwrite/fill from new
            new_hash_to_row = {h: i for i, h in enumerate(new_hashes)}
            for h, row in new_hash_to_row.items():
                if h in hash_to_idx:
                    out[hash_to_idx[h]] = new_arr[row]
            all_arrays[k] = out
        else:
            # Fully cached, just reindex (skip if model not in cache at all)
            if k not in cached:
                continue
            dim = cached[k].shape[1]
            out = np.zeros((n_total, dim), dtype=np.float32)
            for h, old_idx in cached_hashes.items():
                if h in hash_to_idx:
                    out[hash_to_idx[h]] = cached[k][old_idx]
            all_arrays[k] = out

    # Save hash-keyed cache with per-model version keys (only for models with data)
    version_keys = {f"_v_{k}": np.array(v) for k, v in MODEL_VERSIONS.items() if k in all_arrays}
    np.savez_compressed(
        hash_cache_path,
        hashes=np.array(all_hash_list),
        **version_keys,
        **all_arrays,
    )
    print(f"  Saved hash cache: {hash_cache_path}", file=sys.stderr)

    # Write filename-ordered arrays for Rust
    ordered = {}
    for k in EMB_KEYS:
        if k not in all_arrays:
            continue
        arr = all_arrays[k]
        out = np.zeros((len(image_files), arr.shape[1]), dtype=np.float32)
        for i, f in enumerate(image_files):
            out[i] = arr[hash_to_idx[current_hashes[f]]]
        ordered[k] = out

    np.savez_compressed(npz_path, **ordered)
    with open(filenames_path, "w") as fp:
        json.dump(image_files, fp)

    print(f"  Saved Rust embeddings: {npz_path}", file=sys.stderr)
    print(f"  Saved filenames: {filenames_path}", file=sys.stderr)

    # DINOv3 patches: saved separately (3D, ~1GB for 7k images) in filename order
    # for the group-similarity Rust tool.
    if new_dinov3_patches is not None:
        dinov3_hash_to_row = {h: i for i, (_, h) in enumerate(dinov3_items)}
        patches_ordered = np.zeros(
            (len(image_files), DINOV3_N_PATCHES, DINOV3_PATCH_DIM), dtype=np.float32
        )
        for i, f in enumerate(image_files):
            h = current_hashes[f]
            if h in dinov3_hash_to_row:
                patches_ordered[i] = new_dinov3_patches[dinov3_hash_to_row[h]]
        del new_dinov3_patches  # free ~1GB before save+caller reads result
        np.save(dinov3_patches_path, patches_ordered)
        print(f"  Saved DINOv3 patches: {dinov3_patches_path} "
              f"({patches_ordered.shape})", file=sys.stderr)

    json.dump({
        "total": len(image_files),
        "cached": len(image_files) - len(needed),
        "extracted": len(needed),
        "cache_path": npz_path,
    }, sys.stdout)


if __name__ == "__main__":
    main()
