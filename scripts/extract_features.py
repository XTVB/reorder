#!/usr/bin/env python3
"""Extract PE-Core-G + DINOv3 + color features from images, cached by content hash.

Only does feature extraction — no clustering. Outputs a manifest JSON to stdout.
Progress is reported on stderr.

Models:
  - DINOv3 ViT-B/16 (768-dim CLS + 49 × 768 pooled patches + 196 × 768 full-res patches)
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


def _learned_head_groups(cfg):
    """Normalize learned_head.json to a list of head groups:
    [{npz_key, input_mods, input_dim, head_files}]. Legacy configs (no "heads"
    key) describe a single joint-input group under the original field names."""
    if "heads" in cfg:
        return cfg["heads"]
    return [{
        "npz_key": "learned_proj",
        "input_mods": "peg,color",
        "input_dim": cfg["input_dim_total"],
        "head_files": cfg.get("head_files", ["learned_head.pt"]),
    }]


def _pe_layer_enabled(cfg):
    """Toggle: config default `use_pe_layer` (true) AND/OVERRIDDEN by the
    REORDER_USE_PE_LAYER env var ('0'/'false' → off, '1'/'true' → on)."""
    env = os.environ.get("REORDER_USE_PE_LAYER")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(cfg.get("use_pe_layer", True))


def _deployed_pe_layer():
    """The intermediate PE layer the installed head wants captured in-pass:
    (block_idx, pooling, spec) from learned_head.json's group `pe_layer`, or None
    when no head / no layer group / toggle off. Drives the production capture."""
    head_dir = os.environ.get("REORDER_HEAD_DIR", os.path.expanduser("~/.cache/reorder"))
    head_cfg_path = os.path.join(head_dir, "learned_head.json")
    if not os.path.exists(head_cfg_path):
        return None
    with open(head_cfg_path) as f:
        cfg = json.load(f)
    if not _pe_layer_enabled(cfg):
        return None
    for g in _learned_head_groups(cfg):
        spec = g.get("pe_layer")
        if spec and g.get("pe_layer_head_files"):
            L, pool = spec.split(":")
            return int(L), pool, spec
    return None


def _full_coverage(arr, n_images):
    """True iff `arr` covers every image (right row count, no zero rows)."""
    return (arr is not None and arr.shape[0] == n_images
            and not bool((np.abs(arr).sum(axis=1) == 0).any()))


def _load_full_pe_layer(cache_dir, spec, n_images):
    """Load the OFFLINE pe_layers_L<NN>_<pool>.npy (npz-hash row order) ONLY when
    it covers EVERY image. Returns the (n, d) array or None. A grouped-only
    extraction (the offline default) leaves ungrouped rows zero → rejected."""
    if not cache_dir or not spec:
        return None
    L, pool = spec.split(":")
    path = os.path.join(cache_dir, f"pe_layers_L{int(L):02d}_{pool}.npy")
    if not os.path.exists(path):
        return None
    arr = np.load(path).astype(np.float32, copy=False)
    return arr if _full_coverage(arr, n_images) else None


def _resolve_pe_layer(spec, cache_dir, pe_layer_npz, n_images):
    """Find a fully-covering layer array for `spec`: prefer the in-pass capture
    stored in the npz (production path), else the offline .npy (benchmarks).
    Returns the array or None (→ plain-head fallback). pe_layer_npz is
    (stored_spec, array) read from the hash-cache NPZ, or None."""
    if pe_layer_npz is not None:
        stored_spec, arr = pe_layer_npz
        if stored_spec == spec and _full_coverage(arr, n_images):
            return arr.astype(np.float32, copy=False)
    return _load_full_pe_layer(cache_dir, spec, n_images)


def _expected_learned_version(cfg, cache_dir, n_images, pe_layer_npz=None):
    """The version string _compute_learned_proj WOULD return for this dataset —
    cfg["version"] plus a "+peL<spec>" marker iff a group's layer is enabled and
    fully extracted here. Used to detect a stale cache without re-projecting."""
    head_dir = os.environ.get("REORDER_HEAD_DIR", os.path.expanduser("~/.cache/reorder"))
    marker = ""
    if _pe_layer_enabled(cfg):
        for g in _learned_head_groups(cfg):
            if not (g.get("pe_layer") and g.get("pe_layer_head_files")):
                continue
            if not all(os.path.exists(os.path.join(head_dir, hf)) for hf in g["pe_layer_head_files"]):
                continue
            if _resolve_pe_layer(g["pe_layer"], cache_dir, pe_layer_npz, n_images) is not None:
                marker = f"+peL{g['pe_layer']}"
    return cfg["version"] + marker


def _compute_learned_proj(peg_arr, color_arr, cache_dir=None, pe_layer_npz=None):
    """Project features through the trained head group(s) at ~/.cache/reorder/.
    Returns ({npz_key: proj_array}, version_string) or None if no head installed.

    Groups (learned_head.json "heads"): the joint head reads PE-G ⊕ color, the
    split single-modality heads read one block each (see LEARNED_HEAD.md "Split
    single-modality heads"). Within a group, each seed's L2-normed projection is
    one block; blocks are concatenated and scaled by 1/√n_heads, so rows stay
    unit-norm and their dot product equals the ensemble-MEAN cosine similarity —
    the downstream blend needs no changes.

    Intermediate PE-layer (LEARNED_HEAD.md "PE-layer head input"): a group may
    declare `pe_layer` + `pe_layer_head_files` (a parallel head trained on PE-G ⊕
    color ⊕ the L2-normed layer). It's used ONLY when the toggle is on and the
    layer is extracted FULLY for this dataset (every image covered); otherwise we
    fall back to the plain head transparently. The chosen path is recorded in the
    returned version so a toggle flip / coverage change re-projects the cache.

    PE-G is L2-renormalized before use (defensive — should already be unit
    norm but extract paths vary across backends)."""
    head_dir = os.environ.get("REORDER_HEAD_DIR", os.path.expanduser("~/.cache/reorder"))
    head_cfg_path = os.path.join(head_dir, "learned_head.json")
    if not os.path.exists(head_cfg_path):
        return None
    with open(head_cfg_path) as f:
        cfg = json.load(f)
    groups = _learned_head_groups(cfg)
    for g in groups:
        paths = [os.path.join(head_dir, hf) for hf in g["head_files"]]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            if g["head_files"] == ["learned_head.pt"]:
                return None  # no head installed at all
            raise FileNotFoundError(
                f"learned_head.json lists heads for {g['npz_key']} but {missing} missing "
                f"— rerun train_final_head.py"
            )

    # Lazy imports — torch is heavy; only pay the cost if we have a head.
    import torch  # noqa: PLC0415
    refinement_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "clusteringRefinement",
    )
    if refinement_dir not in sys.path:
        sys.path.insert(0, refinement_dir)
    from train_projection_head import ProjectionHead  # noqa: PLC0415

    def _l2(a):
        return a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)

    peg = _l2(peg_arr.astype(np.float32, copy=False))
    color = color_arr.astype(np.float32, copy=False)
    n = peg.shape[0]
    inputs_by_mods = {
        "peg,color": lambda: np.concatenate([peg, color], axis=1),
        "peg": lambda: peg,
        "color": lambda: color,
    }
    pe_layer_on = _pe_layer_enabled(cfg)

    def _project(head, x, in_dim):
        out = np.empty((x.shape[0], cfg["out_dim"]), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, x.shape[0], 512):
                out[i:i + 512] = head(torch.from_numpy(x[i:i + 512])).numpy()
        return out  # already L2-normed by the head's forward

    result = {}
    layer_used = None  # spec of the layer actually used, for the version marker
    for g in groups:
        # Decide layer vs plain for this group + dataset.
        layer_arr = None
        if pe_layer_on and g.get("pe_layer") and g.get("pe_layer_head_files"):
            layer_heads = [os.path.join(head_dir, hf) for hf in g["pe_layer_head_files"]]
            if all(os.path.exists(p) for p in layer_heads):
                layer_arr = _resolve_pe_layer(g["pe_layer"], cache_dir, pe_layer_npz, n)
        if layer_arr is not None:
            head_files = g["pe_layer_head_files"]
            in_dim = g["pe_layer_input_dim"]
            feats = np.concatenate([peg, color, _l2(layer_arr)], axis=1)
            layer_used = g["pe_layer"]
        else:
            head_files = g["head_files"]
            in_dim = g["input_dim"]
            feats = inputs_by_mods[g["input_mods"]]()
        assert feats.shape[1] == in_dim, (
            f"{g['npz_key']} input dim mismatch: got {feats.shape[1]}, head expects {in_dim}"
        )
        blocks = []
        for hf in head_files:
            head = ProjectionHead(
                in_dim=in_dim,
                hidden=cfg["hidden"],
                out_dim=cfg["out_dim"],
                dropout=cfg.get("dropout", 0.1),
            )
            head.load_state_dict(torch.load(os.path.join(head_dir, hf),
                                            map_location="cpu", weights_only=True))
            head.eval()
            blocks.append(_project(head, feats, in_dim))
        # NumPy 2 (NEP 50): dividing the f32 blocks by the np.float64 sqrt scalar
        # silently promotes the whole array to float64. The NPZ must stay <f4 —
        # the Rust readers are dtype-strict and skip an <f8 array entirely.
        result[g["npz_key"]] = (np.concatenate(blocks, axis=1)
                                / np.sqrt(len(blocks))).astype(np.float32, copy=False)
    # Encode the layer decision in the version so a toggle flip or a change in
    # layer coverage invalidates the per-dataset cached learned_proj.
    version = cfg["version"] + (f"+peL{layer_used}" if layer_used else "")
    return result, version


def _maybe_update_learned_proj(npz_path):
    """Idempotently ensure the NPZ contains every learned_proj* array matching
    the current head version. No-op when the head isn't installed or all arrays
    are already current. Called by main() in both the early-exit (cache fully
    valid) and full-save paths so the projections stay in sync with the live
    head. One `_v_learned_proj` version key covers all groups — retraining any
    head refreshes every array."""
    if not os.path.exists(npz_path):
        return
    head_dir = os.environ.get("REORDER_HEAD_DIR", os.path.expanduser("~/.cache/reorder"))
    head_cfg_path = os.path.join(head_dir, "learned_head.json")
    if not os.path.exists(head_cfg_path):
        return
    with open(head_cfg_path) as f:
        head_cfg = json.load(f)
    expected_keys = [g["npz_key"] for g in _learned_head_groups(head_cfg)]

    data = np.load(npz_path, allow_pickle=True)
    if "pecore_g" not in data.files or "color" not in data.files:
        return
    cache_dir = os.path.dirname(npz_path)
    # In-pass layer captured into the same NPZ (production path).
    pe_layer_npz = None
    if "pe_layer" in data.files and "_v_pe_layer" in data.files:
        pe_layer_npz = (str(data["_v_pe_layer"]), data["pe_layer"])
    # Expected version must mirror the layer decision _compute_learned_proj will
    # make for THIS dataset (toggle + full-coverage), so a toggle flip or a newly
    # complete layer extraction is detected as stale and re-projected.
    current_version = _expected_learned_version(
        head_cfg, cache_dir, data["pecore_g"].shape[0], pe_layer_npz=pe_layer_npz)
    stored_version = str(data["_v_learned_proj"]) if "_v_learned_proj" in data.files else None
    if stored_version == current_version and all(
        # A non-f4 array (float64 from a pre-fix NumPy 2 promotion bug) is
        # unusable by the dtype-strict Rust readers — treat it as stale so the
        # next run rewrites it.
        k in data.files and data[k].dtype == np.float32
        for k in expected_keys
    ):
        # Already current
        return

    result = _compute_learned_proj(data["pecore_g"], data["color"],
                                   cache_dir=cache_dir, pe_layer_npz=pe_layer_npz)
    if result is None:
        return
    lp_arrays, lp_version = result

    # Reconstruct NPZ with the learned_proj* arrays added/updated. Preserve all
    # other keys (including version keys for the other models). Stale split
    # arrays from a previous schema are dropped, not orphaned.
    stale = {"_v_learned_proj", "learned_proj", "learned_proj_peg", "learned_proj_color"}
    arrays = {k: data[k] for k in data.files if k not in stale}
    arrays.update(lp_arrays)
    arrays["_v_learned_proj"] = np.array(lp_version)
    np.savez(npz_path, **arrays)
    desc = ", ".join(f"{k} {v.shape[0]}×{v.shape[1]}d" for k, v in lp_arrays.items())
    print(f"  learned_proj: updated NPZ ({desc}; head {lp_version})", file=sys.stderr)


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
    pytorch path) or an mx.array NHWC batch (for the MLX path; signaled by
    `mlx_dtype` being non-None). Must return a numpy float32 array of shape
    (B, dim), already L2-normalized.

    For the MLX path, `transform` must map a PIL image to a numpy HWC float32
    array — torch is never imported. For the pytorch path it's the usual
    torchvision Compose producing a CHW tensor.

    `ctx` carries shared state: image_dir, interrupted flag, checkpoint timer.
    Respects ctx.interrupted — breaks early.
    """
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    is_mlx = mlx_dtype is not None
    if not is_mlx:
        import torch

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
                if is_mlx:
                    tensors.append(np.zeros((fallback_hw, fallback_hw, 3), dtype=np.float32))
                else:
                    tensors.append(torch.zeros(3, fallback_hw, fallback_hw))
        return np.stack(tensors) if is_mlx else torch.stack(tensors)

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
            batch_data = futures.popleft().result()

            if submitted < len(batch_starts) and not ctx.interrupted:
                bs = batch_starts[submitted]
                be = min(bs + batch_size, n)
                futures.append(pool.submit(_prepare_batch, range(bs, be)))
                submitted += 1

            if is_mlx:
                # MLX path: NHWC numpy → mx.array → model → numpy
                import mlx.core as mx
                mlx_batch = mx.array(batch_data).astype(mlx_dtype)
                embs = inference_fn(mlx_batch)
                mx.eval(embs)
                embs_np = np.array(embs).astype(np.float32)
                norms = np.linalg.norm(embs_np, axis=-1, keepdims=True)
                embs_np = embs_np / np.maximum(norms, 1e-12)
                results.append(embs_np)
            else:
                # PyTorch path
                batch_tensor = batch_data.to(ctx.device)
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
                     items, ctx, args, save_to_cache, capture_layer=None):
    """CLIP / PE-Core-L / PE-Core-G extraction. Switches MLX vs torch for PE-Core-G.

    Returns (embs, layer_arr). When `capture_layer=(block_idx, pooling)` and the
    MLX backend is used, the intermediate layer is captured in the SAME forward
    pass (free) and returned as `layer_arr` (raw pooled features, items order);
    otherwise `layer_arr` is None and the caller falls back to the plain head."""
    print(f"  [Pass {ctx.next_pass()}/{ctx.total_passes}] {label} ({len(items)} images)",
          file=sys.stderr)

    use_mlx_pecore_g = key == "pecore_g" and args.pecore_g_backend == "mlx"

    if use_mlx_pecore_g:
        import mlx.core as mx
        from PIL import Image
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

        # Pure PIL+numpy clone of open_clip's preprocess (Resize shorter side →
        # CenterCrop → scale → normalize). Verified bit-exact vs torchvision's
        # Compose — torchvision dispatches PIL inputs to these same PIL calls.
        # Keeps torch/torchvision out of the MLX path entirely.
        _mean = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
        _std = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)

        def preprocess(img, _hw=hw):
            w, h = img.size
            # Long-side size truncates (int(), not round()) — torchvision's
            # _compute_resized_output_size semantics, required for bit-exactness.
            if w <= h:
                nw, nh = _hw, int(h * _hw / w)
            else:
                nh, nw = _hw, int(w * _hw / h)
            img = img.resize((nw, nh), Image.BICUBIC)
            left = int(round((nw - _hw) / 2.0))
            top = int(round((nh - _hw) / 2.0))
            img = img.crop((left, top, left + _hw, top + _hw))
            arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
            return (arr - _mean) / _std
        mlx_dtype = getattr(mx, args.pecore_g_mlx_dtype)
        mlx_model = PECoreBigG()
        mlx_model.load_weights(weights_path, strict=False)
        mlx_model.set_dtype(mlx_dtype)
        mlx_model.eval()
        mx.eval(mlx_model.parameters())

        # In-pass layer capture: wrap the model so each batch's intermediate
        # block features are stashed (raw, items order) alongside the final
        # embedding — one forward, no recompute.
        captured = []
        inference_fn = mlx_model
        if capture_layer is not None:
            lidx, lpool = capture_layer
            print(f"    capturing layer L{lidx}:{lpool} in-pass", file=sys.stderr)

            def inference_fn(mlx_batch, _m=mlx_model, _i=lidx, _p=lpool, _acc=captured):
                emb, cap = _m.forward_and_layer(mlx_batch, _i, _p)
                mx.eval(emb, cap)
                _acc.append(np.array(cap).astype(np.float32))
                return emb

        embs = _run_pass(
            items, preprocess, hw, batch_size_eff, label,
            inference_fn=inference_fn, ctx=ctx, mlx_dtype=mlx_dtype,
            on_checkpoint=lambda e, i: save_to_cache(key, e, i),
        )
        ctx.free_model(mlx_model)
        layer_arr = np.vstack(captured) if captured else None
        if layer_arr is not None:
            layer_arr = layer_arr[:embs.shape[0]]  # align if interrupted mid-pass
        return embs, layer_arr

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
    # Torch path has no in-pass layer capture (only the MLX port exposes the
    # intermediate blocks); the head falls back to the plain joint variant.
    return embs, None


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
        if "torch" in sys.modules:  # don't pay the import just to clear a cache
            import torch
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        gc.collect()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extract PE-G + DINOv3 + color features")
    parser.add_argument("image_dir", help="Directory containing images")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache directory (default: <image_dir>/.reorder-cache)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--models", default=None,
                        help="Comma-separated list of models to force re-extract (e.g. 'pecore_g,color'). "
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

    hash_cache_path = os.path.join(cache_dir, "embeddings_hash_cache.npz")
    # Migrate the old CLIP-era filename in place so existing caches survive the rename.
    legacy_hash_cache = os.path.join(cache_dir, "clip_hash_cache.npz")
    if os.path.exists(legacy_hash_cache) and not os.path.exists(hash_cache_path):
        os.rename(legacy_hash_cache, hash_cache_path)
        print(f"Migrated cache: {legacy_hash_cache} → {hash_cache_path}", file=sys.stderr)
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

    # Snapshot any previously-captured in-pass PE layer BEFORE extraction starts —
    # the intermediate _save_model_to_cache checkpoints rewrite the NPZ without it,
    # so the final merge must reuse this snapshot, not re-read the (clobbered) file.
    cached_pe_layer, cached_pe_layer_spec = None, None
    if os.path.exists(hash_cache_path):
        try:
            _snap = np.load(hash_cache_path, allow_pickle=True)
            if "pe_layer" in _snap and "_v_pe_layer" in _snap:
                cached_pe_layer = _snap["pe_layer"]
                cached_pe_layer_spec = str(_snap["_v_pe_layer"])
        except Exception:
            pass

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
        _maybe_update_learned_proj(hash_cache_path)
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

    # Only import torch if a pass actually runs on the torch/MPS stack.
    # PE-Core-G on the (default) MLX backend doesn't need torch at all —
    # preprocessing is pure PIL+numpy and inference is MLX.
    _needs_torch = bool(items_map["dinov3"]) or (
        bool(items_map["pecore_g"]) and args.pecore_g_backend == "pytorch"
    )
    if _needs_torch:
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

        # Uncompressed savez: float embeddings deflate by only ~20% but cost
        # ~1s/2.8k images per write, and this runs once per model per run.
        # np.load reads both formats, so old compressed caches stay valid.
        np.savez(
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
        ("pecore_g", "PE-Core-bigG-14-448","meta",              448, 1, 8, 1280, "PE-Core-G"),
    ]

    # The installed head may want an intermediate PE layer captured during the
    # pecore_g pass (free); spec read from learned_head.json.
    pe_layer_want = _deployed_pe_layer()  # (idx, pool, spec) or None
    new_pe_layer = None          # captured rows (items order), or None
    pe_layer_items = []          # the pecore_g items they correspond to

    for key, model_name, pretrained, hw, batch_mult, batch_div, dim, label in OPEN_CLIP_MODELS:
        model_items = items_map[key]
        if model_items and not ctx.interrupted:
            batch_size_eff = max(1, args.batch_size * batch_mult // batch_div)
            cap = (pe_layer_want[0], pe_layer_want[1]) if (key == "pecore_g" and pe_layer_want) else None
            embs, layer_arr = extract_open_clip(
                key, model_name, pretrained, hw, batch_size_eff, label,
                model_items, ctx, args, _save_model_to_cache, capture_layer=cap,
            )
            n_done = embs.shape[0]
            new_arrays[key] = embs
            _save_model_to_cache(key, embs, model_items[:n_done])
            if key == "pecore_g" and layer_arr is not None:
                new_pe_layer = layer_arr
                pe_layer_items = model_items[:layer_arr.shape[0]]
        else:
            new_arrays[key] = np.zeros((0, dim), dtype=np.float32)

    new_pecore_g = new_arrays["pecore_g"]

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
        "pecore_g": new_pecore_g,
        "color": new_color,
        "dinov3": new_dinov3,
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

    # Intermediate PE layer captured in-pass during pecore_g — cached like the
    # models so incremental re-extracts keep full coverage. Cached rows are
    # reused only when the spec is unchanged; rows for images with neither cached
    # nor freshly-captured layer stay zero (→ the head falls back to plain until
    # a full pecore_g re-extract or an offline `extract_pe_layers --all-images`).
    pe_layer_spec = pe_layer_want[2] if pe_layer_want else None
    if pe_layer_spec:
        # Reuse the pre-extraction snapshot (the checkpoints clobbered the file),
        # only when its spec matches the currently-wanted layer.
        cached_layer = cached_pe_layer if cached_pe_layer_spec == pe_layer_spec else None
        new_layer = new_pe_layer if new_pe_layer is not None else np.zeros((0, 0), np.float32)
        new_layer_h = [h for _, h in pe_layer_items]
        if cached_layer is not None or len(new_layer):
            merged_layer, _ = merge_cached_array(
                cached_layer, cached_hashes if cached_layer is not None else [],
                new_layer, new_layer_h, hash_universe=current_hash_set,
            )
            all_arrays["pe_layer"] = merged_layer.astype(np.float32, copy=False)
            covered = int((np.abs(all_arrays["pe_layer"]).sum(axis=1) > 0).sum())
            print(f"  pe_layer (L{pe_layer_spec}): {covered}/{len(final_hash_list)} images "
                  f"covered in-pass", file=sys.stderr)

    # Save hash-keyed cache with per-model version keys (uncompressed — see
    # _save_model_to_cache for rationale)
    version_keys = {f"_v_{k}": np.array(v) for k, v in MODEL_VERSIONS.items() if k in all_arrays}
    if "pe_layer" in all_arrays:
        version_keys["_v_pe_layer"] = np.array(pe_layer_spec)
    np.savez(
        hash_cache_path,
        hashes=np.array(final_hash_list),
        **version_keys,
        **all_arrays,
    )
    print(f"  Saved hash cache: {hash_cache_path}", file=sys.stderr)
    _maybe_update_learned_proj(hash_cache_path)

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
