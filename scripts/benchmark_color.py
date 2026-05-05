#!/usr/bin/env python3
"""Benchmark + correctness check for the parallel/draft/BILINEAR color extractor.

Compares the new fast path against a reference that uses the same spatial 3x3 layout
but with BICUBIC resize, no JPEG draft, and a serial loop. Both paths produce 693-dim
spatial features, so any numerical drift comes from (a) BILINEAR vs BICUBIC resize and
(b) JPEG draft-mode decode vs full decode.

Usage: python3 scripts/benchmark_color.py <image_dir> [--limit N]
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import (
    COLOR_DIM,
    COLOR_GRID,
    COLOR_THUMB_SIZE,
    _cell_color_features,
    _color_extract_worker,
    extract_color_features,
)


def reference_extract(img_rgb, thumb_size=COLOR_THUMB_SIZE):
    """Old path: BICUBIC resize, no JPEG draft, spatial 3x3 layout."""
    thumb = img_rgb.resize((thumb_size, thumb_size), Image.Resampling.BICUBIC)
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


def _reference_worker(args):
    """Module-level worker for the BICUBIC + no-draft reference path."""
    image_dir, fname = args
    try:
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        return (reference_extract(img), None)
    except Exception as e:
        return (None, repr(e))


def _reference_batch_worker(args):
    """Process a chunk of images in one task. Reduces pickling round-trips
    (one return per chunk instead of one per image)."""
    image_dir, fnames = args
    out = []
    for f in fnames:
        try:
            img = Image.open(os.path.join(image_dir, f)).convert("RGB")
            out.append((reference_extract(img), None))
        except Exception as e:
            out.append((None, repr(e)))
    return out


def run_serial_reference(image_dir, fnames):
    feats = np.zeros((len(fnames), COLOR_DIM), dtype=np.float32)
    t0 = time.time()
    for i, f in enumerate(fnames):
        img = Image.open(os.path.join(image_dir, f)).convert("RGB")
        feats[i] = reference_extract(img)
        if (i + 1) % 250 == 0:
            print(f"  reference: {i+1}/{len(fnames)} ({(i+1)/(time.time()-t0):.1f} img/s)",
                  file=sys.stderr)
    return feats, time.time() - t0


def run_parallel(image_dir, fnames, n_workers, worker_fn, label):
    from concurrent.futures import ProcessPoolExecutor
    feats = np.zeros((len(fnames), COLOR_DIM), dtype=np.float32)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(worker_fn, (image_dir, f)) for f in fnames]
        for i, fut in enumerate(futures):
            feat, err = fut.result()
            if err is not None:
                print(f"  WARN {fnames[i]}: {err}", file=sys.stderr)
                continue
            feats[i] = feat
            if (i + 1) % 250 == 0:
                print(f"  {label}: {i+1}/{len(fnames)} ({(i+1)/(time.time()-t0):.1f} img/s)",
                      file=sys.stderr)
    return feats, time.time() - t0


def run_serial_fast(image_dir, fnames):
    """Serial timing of the new code path — isolates the parallelism speedup from
    the BILINEAR + JPEG-draft micro-optimization speedup."""
    feats = np.zeros((len(fnames), COLOR_DIM), dtype=np.float32)
    t0 = time.time()
    for i, f in enumerate(fnames):
        feat, err = _color_extract_worker((image_dir, f))
        if err is None:
            feats[i] = feat
        if (i + 1) % 250 == 0:
            print(f"  serial-fast: {i+1}/{len(fnames)} ({(i+1)/(time.time()-t0):.1f} img/s)",
                  file=sys.stderr)
    return feats, time.time() - t0


def cosine_sim(a, b):
    na = np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    nb = np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return ((a / na) * (b / nb)).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = ap.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    fnames = sorted(f for f in os.listdir(args.image_dir)
                    if os.path.splitext(f)[1].lower() in exts)
    if args.limit:
        fnames = fnames[: args.limit]
    n = len(fnames)
    print(f"Benchmarking on {n} images from {args.image_dir}", file=sys.stderr)

    from concurrent.futures import ProcessPoolExecutor

    def run_batched(image_dir, fnames, n_workers, batch_size, label):
        feats = np.zeros((len(fnames), COLOR_DIM), dtype=np.float32)
        chunks = [fnames[i:i + batch_size] for i in range(0, len(fnames), batch_size)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_reference_batch_worker, (image_dir, c)) for c in chunks]
            done = 0
            for ci, fut in enumerate(futures):
                results = fut.result()
                base = ci * batch_size
                for k, (feat, err) in enumerate(results):
                    if err is None:
                        feats[base + k] = feat
                done += len(results)
                if done % 500 < batch_size or done == len(fnames):
                    print(f"  {label}: {done}/{len(fnames)} "
                          f"({done/(time.time()-t0):.1f} img/s)", file=sys.stderr)
        return feats, time.time() - t0

    configs = [
        ("serial",                          lambda: run_serial_reference(args.image_dir, fnames)),
        ("parallel  8w  per-image",         lambda: run_parallel(args.image_dir, fnames,  8, _reference_worker, "par 8")),
        ("parallel 10w  per-image",         lambda: run_parallel(args.image_dir, fnames, 10, _reference_worker, "par 10")),
        ("parallel 14w  per-image",         lambda: run_parallel(args.image_dir, fnames, 14, _reference_worker, "par 14")),
        ("parallel 10w  batch=16",          lambda: run_batched(args.image_dir, fnames, 10, 16, "p10b16")),
        ("parallel 14w  batch=16",          lambda: run_batched(args.image_dir, fnames, 14, 16, "p14b16")),
        ("parallel 14w  batch=32",          lambda: run_batched(args.image_dir, fnames, 14, 32, "p14b32")),
        ("parallel 14w  batch=64",          lambda: run_batched(args.image_dir, fnames, 14, 64, "p14b64")),
    ]

    results = []
    for i, (label, fn) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {label} ...", file=sys.stderr)
        feats, elapsed = fn()
        results.append((label, feats, elapsed))

    ref_feats = results[0][1]
    print()
    print("=" * 70)
    print(f"{'config':<35} {'time':>8} {'img/s':>10} {'speedup':>9}  {'cos vs ref':>11}")
    print("-" * 70)
    ref_t = results[0][2]
    for label, feats, elapsed in results:
        cs = cosine_sim(ref_feats, feats)
        print(f"{label:<35} {elapsed:>7.2f}s {n/elapsed:>9.1f}  {ref_t/elapsed:>7.2f}x   "
              f"mean={cs.mean():.6f} min={cs.min():.6f}")


if __name__ == "__main__":
    main()
