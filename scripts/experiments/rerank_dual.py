#!/usr/bin/env python3
"""Dual-feature re-ranking: combine re-rank distances from PE-G+color AND DINOv3.

Hypothesis: even though DINOv3 alone is roughly neutral on this benchmark,
its re-rank distance captures DIFFERENT group structure (different feature
space → different kNN graph → different mistakes). Blending the two re-rank
distances should outperform either alone.

Tests:
  1. re-rank(PE-G + color) — current best
  2. re-rank(DINOv3) alone
  3. linear blend: d = α·rerank_pec + (1-α)·rerank_dino  (sweep α)
  4. consensus blend: only NN edges in BOTH features' kNN graphs count
  5. concatenated features re-rank (both feature spaces in one kNN)
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_merge import load_ground_truth, metrics
from rerank_distance import (
    expand_R_star,
    k_reciprocal_neighbors,
    rerank_jaccard_distance,
)


def load_pec_and_dino(folder: str, w_pe: float = 1.0, w_col: float = 0.7):
    """Returns: fnames, pec_feat (PE-G+color, normalized), dino_feat (DINOv3 CLS, normalized)."""
    import json
    cache = os.path.join(folder, ".reorder-cache")
    d = np.load(os.path.join(cache, "clip_hash_cache.npz"), allow_pickle=True)
    with open(os.path.join(cache, "content_hashes.json")) as f:
        ch = json.load(f)
    with open(os.path.join(cache, "hash_cache_order.json")) as f:
        hash_order = json.load(f)
    hash_to_row = {h: i for i, h in enumerate(hash_order)}
    fnames = sorted(ch.keys())
    fname_to_row = np.array([hash_to_row[ch[f]] for f in fnames])

    pe = d["pecore_g"][fname_to_row]
    co = d["color"][fname_to_row]
    co = co / np.maximum(np.linalg.norm(co, axis=1, keepdims=True), 1e-10)
    pec = np.concatenate([w_pe * pe, w_col * co], axis=1)
    pec = pec / np.maximum(np.linalg.norm(pec, axis=1, keepdims=True), 1e-10)

    dino = d["dinov3"][fname_to_row]  # already L2-normed in extraction
    dino = dino / np.maximum(np.linalg.norm(dino, axis=1, keepdims=True), 1e-10)

    return fnames, pec.astype(np.float32), dino.astype(np.float32)


def consensus_rerank(sim_a, sim_b, k1, k2):
    """Re-rank where R(i, k1) only includes neighbors that are k-reciprocal in BOTH feature spaces.

    Reduces false positives — only neighbors confirmed by both views count.
    """
    n = sim_a.shape[0]
    R_k_a = k_reciprocal_neighbors(sim_a, k1)
    R_k_b = k_reciprocal_neighbors(sim_b, k1)

    # Intersect: R_k(i) = R_k_a(i) ∩ R_k_b(i)
    R_k = []
    for i in range(n):
        s_a = set(R_k_a[i].tolist())
        s_b = set(R_k_b[i].tolist())
        R_k.append(np.array(sorted(s_a & s_b), dtype=np.int32))

    # Half-k for expansion (also intersection)
    R_half_a = k_reciprocal_neighbors(sim_a, max(2, k1 // 2))
    R_half_b = k_reciprocal_neighbors(sim_b, max(2, k1 // 2))
    R_half = []
    for i in range(n):
        s_a = set(R_half_a[i].tolist())
        s_b = set(R_half_b[i].tolist())
        R_half.append(np.array(sorted(s_a & s_b), dtype=np.int32))

    R_star = expand_R_star(R_k, R_half)

    # Build indicator matrix
    rows, cols = [], []
    for i in range(n):
        for c in R_star[i]:
            rows.append(i)
            cols.append(c)
    inc = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    )

    # LQE: average indicator over top-k2 cosine NNs in PE-G+color (the "primary" view)
    if k2 > 1:
        nn2 = np.argpartition(-sim_a, k2, axis=1)[:, :k2]
        avg_data = np.full(n * k2, 1.0 / k2, dtype=np.float32)
        avg_rows = np.repeat(np.arange(n), k2)
        avg_cols = nn2.flatten()
        avg_kernel = sparse.csr_matrix((avg_data, (avg_rows, avg_cols)), shape=(n, n))
        V = avg_kernel @ inc
    else:
        V = inc

    K = (V @ V.T).toarray().astype(np.float64)
    diag = np.diag(K).copy()
    norm = np.sqrt(np.outer(diag, diag)) + 1e-10
    sim_kernel = np.clip(K / norm, 0.0, 1.0)
    np.fill_diagonal(sim_kernel, 1.0)
    d = 1.0 - sim_kernel
    np.fill_diagonal(d, 0.0)
    return d


def evaluate(d, gt, methods=("ward", "average"), ns=(150, 200, 250, 300, 400)):
    out = []
    d = np.clip(d, 0.0, None); d = (d + d.T) / 2.0; np.fill_diagonal(d, 0)
    cd = squareform(d, checks=False)
    for method in methods:
        Z = linkage(cd, method=method)
        for nc in ns:
            labels = fcluster(Z, t=nc, criterion="maxclust") - 1
            m = metrics(labels, gt)
            f1 = 2 * m["weighted_group_recall"] * m["weighted_cluster_purity"] / max(
                m["weighted_group_recall"] + m["weighted_cluster_purity"], 1e-10
            )
            out.append((method, nc, m["n_clusters"], m["ari"],
                        m["weighted_group_recall"], m["weighted_cluster_purity"], f1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    args = ap.parse_args()

    print(f"Loading {args.folder}", file=sys.stderr)
    fnames, pec, dino = load_pec_and_dino(args.folder)
    n = len(fnames)
    gt, gnames = load_ground_truth(args.folder, fnames)
    print(f"  {n} images, {len(gnames)} groups; pec_dim={pec.shape[1]} dino_dim={dino.shape[1]}",
          file=sys.stderr)

    # cosine sim matrices
    print("Computing cosine sim matrices...", file=sys.stderr)
    sim_pec = (pec @ pec.T).astype(np.float32); np.fill_diagonal(sim_pec, -2)
    sim_dino = (dino @ dino.T).astype(np.float32); np.fill_diagonal(sim_dino, -2)
    cos_pec = (1.0 - sim_pec).astype(np.float64); np.fill_diagonal(cos_pec, 0.0)
    cos_dino = (1.0 - sim_dino).astype(np.float64); np.fill_diagonal(cos_dino, 0.0)

    K1, K2 = 65, 4
    print(f"\nComputing re-rank(PE-G+color) k1={K1} k2={K2}...", file=sys.stderr)
    t0 = time.time()
    rr_pec = rerank_jaccard_distance(sim_pec, k1=K1, k2=K2)
    print(f"  {time.time()-t0:.1f}s", file=sys.stderr)

    print(f"Computing re-rank(DINOv3) k1={K1} k2={K2}...", file=sys.stderr)
    t0 = time.time()
    rr_dino = rerank_jaccard_distance(sim_dino, k1=K1, k2=K2)
    print(f"  {time.time()-t0:.1f}s", file=sys.stderr)

    rows = []

    # Baselines
    print("\nBaselines:", file=sys.stderr)
    for tag, d in [("cos_pec", cos_pec), ("cos_dino", cos_dino)]:
        for r in evaluate(d, gt, methods=("ward",), ns=(200,)):
            rows.append((f"{tag}", r))

    # Re-rank each alone, blended with own cosine at λ=0.7
    print("Re-rank alone (blended with own cosine, λ=0.7):", file=sys.stderr)
    for tag, cd, rd in [("rerank_pec", cos_pec, rr_pec), ("rerank_dino", cos_dino, rr_dino)]:
        d = 0.3 * cd + 0.7 * rd
        for r in evaluate(d, gt):
            rows.append((f"{tag} λ=0.7", r))

    # Linear blend of two re-rank distances (simple)
    print("Linear blend rr_pec + rr_dino:", file=sys.stderr)
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        d = alpha * rr_pec + (1 - alpha) * rr_dino
        for r in evaluate(d, gt):
            rows.append((f"blend(rr_pec α={alpha:.1f} + rr_dino)", r))

    # Three-way blend with cosine
    print("Three-way blend cos_pec + rr_pec + rr_dino:", file=sys.stderr)
    for w_cos in [0.2, 0.3]:
        for w_pec in [0.3, 0.4, 0.5]:
            w_dino = 1 - w_cos - w_pec
            if w_dino <= 0:
                continue
            d = w_cos * cos_pec + w_pec * rr_pec + w_dino * rr_dino
            for r in evaluate(d, gt):
                rows.append((f"3way cos={w_cos} pec={w_pec} dino={w_dino:.1f}", r))

    # Consensus re-rank: NN edges must be reciprocal in BOTH feature spaces
    print("Consensus re-rank...", file=sys.stderr)
    for k1 in [50, 65, 100]:
        rr_consensus = consensus_rerank(sim_pec, sim_dino, k1=k1, k2=K2)
        for lam in [0.5, 0.7, 1.0]:
            d = (1 - lam) * cos_pec + lam * rr_consensus
            for r in evaluate(d, gt):
                rows.append((f"consensus k1={k1} λ={lam}", r))

    # Print sorted
    print()
    print(f"{'method':50s} {'link':>5s} {'N':>5s} {'n_cl':>5s} {'ari':>6s} "
          f"{'rec':>6s} {'pur':>6s} {'F1':>6s}")
    print("-" * 100)
    flat = [(name, *r) for name, r in rows]  # (name, method, nc_target, n_cl, ari, rec, pur, f1)
    flat.sort(key=lambda x: -x[4])
    for name, method, nc_t, n_cl, ari, rec, pur, f1 in flat[:35]:
        print(f"{name:50s} {method:>5s} {nc_t:5d} {n_cl:5d} {ari:6.3f} {rec:6.3f} {pur:6.3f} {f1:6.3f}")


if __name__ == "__main__":
    main()
