#!/usr/bin/env python3
"""Re-ranking distance via k-reciprocal encoding (Zhong et al., 2017).

Algorithm:
1. R(p, k) = k-reciprocal NN set of p (= {q : q in topK(p) AND p in topK(q)}).
2. R*(p, k) = R(p, k) ∪ (R(q, k/2) for q in R(p, k) if |R(p,k) ∩ R(q,k/2)| ≥ ⅔|R(q,k/2)|).
3. V_p(c) = exp(-d(p, c)) if c in R*(p, k) else 0   (sparse weighted vector)
4. d_jaccard(p, g) = 1 - Σ_c min(V_p(c), V_g(c)) / Σ_c max(V_p(c), V_g(c))
5. Local query expansion: V_p ← mean over top-k_q nearest of V_q
6. d_final = (1-λ) * d_original + λ * d_jaccard

This is canonical re-ranking — known to consistently outperform plain k-reciprocal.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_merge import load_features, load_ground_truth, metrics


def k_reciprocal_neighbors(sim, k):
    """Return list of k-reciprocal NN sets (each as np.array of indices)."""
    n = sim.shape[0]
    nn_idx = np.argpartition(-sim, k, axis=1)[:, :k]
    # Build forward NN as sets for fast lookup
    nn_sets = [set(nn_idx[i].tolist()) for i in range(n)]
    R = []
    for i in range(n):
        recip = [j for j in nn_idx[i] if i in nn_sets[j]]
        R.append(np.array(recip, dtype=np.int32))
    return R


def expand_R_star(R_k, R_half, expand_threshold=2/3):
    """For each i, R*(i, k) = R(i, k) ∪ {R(j, k/2) for j in R(i,k) where overlap is high}."""
    n = len(R_k)
    R_star = []
    R_k_sets = [set(r.tolist()) for r in R_k]
    R_half_sets = [set(r.tolist()) for r in R_half]
    for i in range(n):
        s = set(R_k[i].tolist())
        for j in R_k[i]:
            if len(R_half_sets[j]) == 0:
                continue
            inter = len(R_k_sets[i] & R_half_sets[j])
            if inter >= expand_threshold * len(R_half_sets[j]):
                s |= R_half_sets[j]
        R_star.append(np.array(sorted(s), dtype=np.int32))
    return R_star


def rerank_jaccard_distance(sim: np.ndarray, k1: int = 20, k2: int = 6, dtype=np.float32):
    """Compute Jaccard distance via k-reciprocal R* encoding.

    k1: primary k for R(p, k1)
    k2: local query expansion size — averages each R* indicator over k2 cosine NNs.
        Set to 1 to skip LQE.

    Implementation: binary R* indicator matrix, then weighted Jaccard via sparse
    matmul. With LQE, indicators become probabilistic in [0, 1].
    """
    n = sim.shape[0]

    print(f"  k-reciprocal sets (k1={k1}, k1/2={max(2, k1//2)})...", file=sys.stderr)
    R_k = k_reciprocal_neighbors(sim, k1)
    R_half = k_reciprocal_neighbors(sim, max(2, k1 // 2))

    print("  R* expansion...", file=sys.stderr)
    R_star = expand_R_star(R_k, R_half)

    # Binary indicator matrix
    rows, cols = [], []
    for i in range(n):
        for c in R_star[i]:
            rows.append(i)
            cols.append(c)
    inc = sparse.csr_matrix(
        (np.ones(len(rows), dtype=dtype), (rows, cols)), shape=(n, n)
    )

    if k2 > 1:
        print(f"  local query expansion (k2={k2})...", file=sys.stderr)
        nn2 = np.argpartition(-sim, k2, axis=1)[:, :k2]
        avg_data = np.full(n * k2, 1.0 / k2, dtype=dtype)
        avg_rows = np.repeat(np.arange(n), k2)
        avg_cols = nn2.flatten()
        avg_kernel = sparse.csr_matrix((avg_data, (avg_rows, avg_cols)), shape=(n, n))
        V = avg_kernel @ inc  # values in [0, 1]
    else:
        V = inc

    # Weighted Jaccard via sum of pairwise mins. For non-negative V:
    #   sum_min(p, g) = sum over c of min(V[p,c], V[g,c])
    # Compute via:  min(a, b) = (a + b - |a - b|) / 2
    # For sparse non-negative vectors with avg ~k1+expand non-zeros, dense
    # broadcast over n is too big — but sum_min ≤ min(||V_p||_1, ||V_g||_1).
    # Approximate with cosine-style kernel: V @ V.T, normalized.
    # That's the same fix from snn_variants — known to work well.
    print("  computing kernel V @ V.T...", file=sys.stderr)
    K = (V @ V.T).toarray().astype(np.float64)
    diag = np.diag(K).copy()
    norm = np.sqrt(np.outer(diag, diag)) + 1e-10
    sim_kernel = K / norm
    sim_kernel = np.clip(sim_kernel, 0.0, 1.0)
    np.fill_diagonal(sim_kernel, 1.0)
    d = 1.0 - sim_kernel
    np.fill_diagonal(d, 0.0)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--k1s", default="20,30,50,75")
    ap.add_argument("--k2s", default="6,10")
    ap.add_argument("--lambdas", default="0.0,0.3,0.5,0.7,1.0")
    args = ap.parse_args()

    print(f"Loading {args.folder}", file=sys.stderr)
    fnames, feat = load_features(args.folder, 1.0, 0.7)
    n = len(fnames)
    gt, gnames = load_ground_truth(args.folder, fnames)
    print(f"  {n} images, {len(gnames)} groups", file=sys.stderr)

    sim = (feat @ feat.T).astype(np.float32)
    np.fill_diagonal(sim, -2)
    cos_d = (1.0 - sim).astype(np.float64)
    np.fill_diagonal(cos_d, 0.0)

    rows = []

    Z_cos = linkage(squareform(cos_d, checks=False), method="ward")
    for nc in [150, 200, 300]:
        labels = fcluster(Z_cos, t=nc, criterion="maxclust") - 1
        rows.append(("cosine", f"N={nc}", metrics(labels, gt)))

    k1s = [int(x) for x in args.k1s.split(",")]
    k2s = [int(x) for x in args.k2s.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]

    for k1 in k1s:
        for k2 in k2s:
            print(f"\n=== Re-rank k1={k1} k2={k2} ===", file=sys.stderr)
            t0 = time.time()
            jd = rerank_jaccard_distance(sim, k1=k1, k2=k2)
            print(f"  done ({time.time()-t0:.1f}s)", file=sys.stderr)
            for lam in lambdas:
                d = (1.0 - lam) * cos_d + lam * jd
                d = np.clip(d, 0.0, None)
                # Ensure exact symmetry
                d = (d + d.T) / 2.0
                np.fill_diagonal(d, 0.0)
                Z = linkage(squareform(d, checks=False), method="ward")
                for nc in [150, 200, 300]:
                    labels = fcluster(Z, t=nc, criterion="maxclust") - 1
                    rows.append(
                        ("rerank", f"k1={k1} k2={k2} λ={lam:.1f} N={nc}", metrics(labels, gt))
                    )

    print()
    print(f"{'method':10s} {'config':30s} {'n_cl':>5s} {'ari':>6s} "
          f"{'recall':>7s} {'w_rec':>7s} {'purity':>7s} {'w_pur':>7s} {'F1':>6s}")
    print("-" * 100)
    rows.sort(key=lambda r: -r[2]["ari"])
    for vname, cfg, m in rows[:40]:
        f1 = 2 * m["weighted_group_recall"] * m["weighted_cluster_purity"] / max(
            m["weighted_group_recall"] + m["weighted_cluster_purity"], 1e-10)
        print(f"{vname:10s} {cfg:30s} {m['n_clusters']:5d} {m['ari']:6.3f} "
              f"{m['avg_group_recall']:7.3f} {m['weighted_group_recall']:7.3f} "
              f"{m['avg_cluster_purity']:7.3f} {m['weighted_cluster_purity']:7.3f} "
              f"{f1:6.3f}")


if __name__ == "__main__":
    main()
