#!/usr/bin/env python3
"""
Blend a learned condensed-distance matrix with the baseline PE-G + color
cosine-distance matrix the rust cluster-tool would compute, and write the
result in cluster-tool's binary format. We do the blend in Python because the
rust binary's --dist-matrix codepath hard-skips embedding loading (main.rs).

Replicates rust's distance exactly: for each active model, L2-normalize if
needed, scale by weight, concat → unit-norm not enforced → cosine distance.

Usage:
  python scripts/blend_dist_matrix.py <target_dir> \\
      --learned <learned_dist_matrix.bin> \\
      --learned-weight 0.5 \\
      --output <blended.bin> \\
      [--peg-weight 1.0] [--color-weight 0.8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def load_baseline_features(target_dir: str, peg_weight: float, color_weight: float) -> tuple[np.ndarray, list[str]]:
    """Build the concat'd feature vector the rust cluster-tool would build."""
    cache = os.path.join(target_dir, ".reorder-cache")
    npz = np.load(os.path.join(cache, "embeddings_hash_cache.npz"), allow_pickle=False)
    with open(os.path.join(cache, "content_hashes.json")) as f:
        content_hashes = json.load(f)
    filenames = sorted(content_hashes.keys())
    fn_to_idx = {f: i for i, f in enumerate(filenames)}
    hashes = list(npz["hashes"])
    hash_to_row = {h: i for i, h in enumerate(hashes)}
    n = len(filenames)

    # Reindex hash-ordered arrays into sorted-filename order. Apply weights and
    # per-row L2-normalize color (rust's `needs_l2_norm` is True for color only).
    parts: list[np.ndarray] = []
    if peg_weight > 0:
        peg_hash = npz["pecore_g"]
        peg = np.empty((n, peg_hash.shape[1]), dtype=np.float32)
        for i, fn in enumerate(filenames):
            peg[i] = peg_hash[hash_to_row[content_hashes[fn]]]
        # PE-G already L2-normalized in extract_features.py
        parts.append(peg * peg_weight)
    if color_weight > 0:
        col_hash = npz["color"]
        col = np.empty((n, col_hash.shape[1]), dtype=np.float32)
        for i, fn in enumerate(filenames):
            col[i] = col_hash[hash_to_row[content_hashes[fn]]]
        norms = np.linalg.norm(col, axis=1, keepdims=True).clip(min=1e-10)
        col = col / norms
        parts.append(col * color_weight)
    feat = np.concatenate(parts, axis=1)
    return feat, filenames


def compute_condensed_cosine_dist(feat: np.ndarray) -> np.ndarray:
    """Match rust's cosine distance: 1 - dot(a,b)/(|a||b|), clamp ≥0, condensed."""
    n = feat.shape[0]
    norms = np.linalg.norm(feat, axis=1).clip(min=1e-20)
    # Normalize once, then dot products → cos sims
    feat_n = feat / norms[:, None]
    # Compute full sims matrix once. For ≤10k images this is fine in memory.
    sims = (feat_n.astype(np.float64)) @ (feat_n.astype(np.float64).T)
    out = np.empty(n * (n - 1) // 2, dtype=np.float64)
    off = 0
    for i in range(n - 1):
        row = sims[i, i + 1:]
        d = np.clip(1.0 - row, 0.0, None)
        out[off:off + d.shape[0]] = d
        off += d.shape[0]
    return out


def read_dist_matrix(path: str) -> tuple[int, np.ndarray]:
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
        data = np.frombuffer(f.read(), dtype=np.float64)
    expected = n * (n - 1) // 2
    if data.shape[0] != expected:
        raise ValueError(f"size mismatch: n={n} expects {expected} pairs, got {data.shape[0]}")
    return n, data


def write_dist_matrix(path: str, n: int, dists: np.ndarray):
    with open(path, "wb") as f:
        f.write(np.uint64(n).tobytes())
        f.write(dists.astype(np.float64, copy=False).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target_dir")
    ap.add_argument("--learned", required=True, help="learned dist matrix .bin")
    ap.add_argument("--weights", required=True,
                    help="comma-separated learned-weights, e.g. 0.3,0.5,0.7. 1.0 = learned only, 0.0 = baseline only")
    ap.add_argument("--output-pattern", required=True,
                    help="output path with {w} placeholder, e.g. /tmp/blended_{w}.bin")
    ap.add_argument("--peg-weight", type=float, default=1.0)
    ap.add_argument("--color-weight", type=float, default=0.8)
    args = ap.parse_args()

    n_learned, learned = read_dist_matrix(args.learned)
    feat, filenames = load_baseline_features(args.target_dir, args.peg_weight, args.color_weight)
    if feat.shape[0] != n_learned:
        sys.exit(f"size mismatch: features have {feat.shape[0]} images, learned matrix has {n_learned}")
    print(f"  computing baseline cosine on {feat.shape[0]} × {feat.shape[1]}-d features...", file=sys.stderr)
    baseline = compute_condensed_cosine_dist(feat)
    for w_str in args.weights.split(","):
        w = float(w_str)
        blended = (w * learned + (1.0 - w) * baseline).astype(np.float64)
        out = args.output_pattern.replace("{w}", w_str)
        write_dist_matrix(out, n_learned, blended)
        print(f"  wrote {out}  (learned_weight={w})", file=sys.stderr)


if __name__ == "__main__":
    main()
