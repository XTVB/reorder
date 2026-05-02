#!/usr/bin/env python3
"""Precompute the k-reciprocal re-ranking distance matrix.

Reads weighted features from clip_hash_cache.npz, builds the kNN graph using
cosine similarity, computes R*-expanded reciprocal neighbor sets, applies local
query expansion, and writes the resulting Jaccard-style distance matrix in the
binary format expected by cluster-tool's --dist-matrix flag.

Output format (matches patch_dist_matrix.bin):
    [u64 LE: n][f64 LE × n*(n-1)/2 condensed upper triangle]

Algorithm reference:
    Zhong et al., "Re-ranking Person Re-identification with k-reciprocal
    Encoding" (CVPR 2017). Implemented with binary R* indicator + LQE +
    cosine-style kernel of weighted indicators (a tractable proxy for
    weighted Jaccard that stays in [0, 1]).
"""

import argparse
import json
import os
import struct
import sys
import time
import warnings

import numpy as np
from scipy import sparse

# numpy/BLAS may emit harmless overflow warnings during the cosine matmul on
# very high-dim feature vectors; the results are clamped/normalized later.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


def load_weighted_features(cache_dir: str, weights: dict[str, float]):
    """Concatenate weighted features from the cache, L2-normalize the result.

    `weights` maps model key (e.g. "pecore_g", "color", "dinov3") → weight.
    Skips models with weight 0 or missing from cache.
    """
    npz = np.load(os.path.join(cache_dir, "clip_hash_cache.npz"), allow_pickle=True)
    with open(os.path.join(cache_dir, "content_hashes.json")) as f:
        ch = json.load(f)
    with open(os.path.join(cache_dir, "hash_cache_order.json")) as f:
        hash_order = json.load(f)
    hash_to_row = {h: i for i, h in enumerate(hash_order)}
    fnames = sorted(ch.keys())
    fname_to_row = np.array([hash_to_row[ch[f]] for f in fnames])

    chunks = []
    used = []
    for key, w in weights.items():
        if w is None or w <= 0:
            continue
        if key not in npz.files:
            print(f"  WARNING: '{key}' not in cache, skipping (weight {w})", file=sys.stderr)
            continue
        arr = npz[key][fname_to_row]
        # color is not L2-normalized at extraction; everything else is
        if key == "color":
            arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-10)
        chunks.append(w * arr)
        used.append((key, w, arr.shape[1]))

    if not chunks:
        raise SystemExit("No features with non-zero weight; nothing to compute.")

    print(f"  Active features: {used}", file=sys.stderr)
    feat = np.concatenate(chunks, axis=1).astype(np.float32)
    feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-10)
    return fnames, feat


def k_reciprocal_neighbors(sim: np.ndarray, k: int):
    """Return list of k-reciprocal NN sets (np.array of indices per image)."""
    n = sim.shape[0]
    nn_idx = np.argpartition(-sim, k, axis=1)[:, :k]
    nn_sets = [set(nn_idx[i].tolist()) for i in range(n)]
    return [
        np.array([j for j in nn_idx[i] if i in nn_sets[j]], dtype=np.int32)
        for i in range(n)
    ]


def expand_R_star(R_k, R_half, threshold: float = 2.0 / 3.0):
    """Standard R* expansion: add R(j, k/2) members if overlap with R(i, k) is high."""
    n = len(R_k)
    R_k_sets = [set(r.tolist()) for r in R_k]
    R_half_sets = [set(r.tolist()) for r in R_half]
    out = []
    for i in range(n):
        s = set(R_k[i].tolist())
        for j in R_k[i]:
            half_j = R_half_sets[j]
            if not half_j:
                continue
            inter = len(R_k_sets[i] & half_j)
            if inter >= threshold * len(half_j):
                s |= half_j
        out.append(np.array(sorted(s), dtype=np.int32))
    return out


def compute_rerank_distance(feat: np.ndarray, k1: int, k2: int) -> np.ndarray:
    n = feat.shape[0]
    print(f"  cosine sim ({n}x{n})...", file=sys.stderr)
    t0 = time.time()
    sim = (feat @ feat.T).astype(np.float32)
    np.fill_diagonal(sim, -2)
    print(f"    {time.time()-t0:.1f}s", file=sys.stderr)

    print(f"  k-reciprocal sets (k1={k1}, k1/2={max(2, k1 // 2)})...", file=sys.stderr)
    t0 = time.time()
    R_k = k_reciprocal_neighbors(sim, k1)
    R_half = k_reciprocal_neighbors(sim, max(2, k1 // 2))
    print(f"    {time.time()-t0:.1f}s", file=sys.stderr)

    print("  R* expansion...", file=sys.stderr)
    t0 = time.time()
    R_star = expand_R_star(R_k, R_half)
    print(f"    {time.time()-t0:.1f}s", file=sys.stderr)

    rows, cols = [], []
    for i in range(n):
        for c in R_star[i]:
            rows.append(i)
            cols.append(int(c))
    inc = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    )

    if k2 > 1:
        print(f"  local query expansion (k2={k2})...", file=sys.stderr)
        nn2 = np.argpartition(-sim, k2, axis=1)[:, :k2]
        avg_data = np.full(n * k2, 1.0 / k2, dtype=np.float32)
        avg_rows = np.repeat(np.arange(n), k2)
        avg_cols = nn2.flatten()
        avg = sparse.csr_matrix((avg_data, (avg_rows, avg_cols)), shape=(n, n))
        V = avg @ inc
    else:
        V = inc

    print("  computing kernel V @ V.T...", file=sys.stderr)
    t0 = time.time()
    K = (V @ V.T).toarray().astype(np.float64)
    diag = np.diag(K).copy()
    norm_outer = np.sqrt(np.outer(diag, diag)) + 1e-10
    sim_kernel = np.clip(K / norm_outer, 0.0, 1.0)
    np.fill_diagonal(sim_kernel, 1.0)
    d = 1.0 - sim_kernel
    np.fill_diagonal(d, 0.0)
    print(f"    {time.time()-t0:.1f}s", file=sys.stderr)
    return d


def write_dist_matrix(d: np.ndarray, path: str):
    """Write condensed upper-triangle in the same .bin format as patch_dist_matrix.bin."""
    n = d.shape[0]
    n_pairs = n * (n - 1) // 2
    out = np.empty(n_pairs, dtype=np.float64)
    idx = 0
    for i in range(n):
        m = n - i - 1
        out[idx : idx + m] = d[i, i + 1 :]
        idx += m
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(out.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True,
                    help="Path to .reorder-cache directory")
    ap.add_argument("--output", required=True,
                    help="Output .bin path")
    ap.add_argument("--weights", required=True,
                    help='JSON object: {"pecore_g": 1.0, "color": 0.7, ...}')
    ap.add_argument("--k1", type=int, default=65,
                    help="Primary k for k-reciprocal NN sets (default 65)")
    ap.add_argument("--k2", type=int, default=4,
                    help="Local query expansion neighborhood size (default 4)")
    args = ap.parse_args()

    weights = json.loads(args.weights)
    print(f"Loading features from {args.cache_dir}", file=sys.stderr)
    fnames, feat = load_weighted_features(args.cache_dir, weights)
    n = len(fnames)
    print(f"  {n} images, dim={feat.shape[1]}", file=sys.stderr)

    print(f"Computing re-rank distance (k1={args.k1}, k2={args.k2})", file=sys.stderr)
    d = compute_rerank_distance(feat, args.k1, args.k2)

    print(f"Writing {args.output}", file=sys.stderr)
    write_dist_matrix(d, args.output)

    print(json.dumps({"n": n, "output": args.output}))


if __name__ == "__main__":
    main()
