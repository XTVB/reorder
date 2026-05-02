#!/usr/bin/env python3
"""Benchmark cosine + shared-nearest-neighbors (SNN) distance combinations.

Shared-NN distance: d_snn(a, b) = 1 - |kNN(a) ∩ kNN(b)| / |kNN(a) ∪ kNN(b)|
Combined distance:  d = (1-α) × cosine + α × d_snn

Evaluates clustering quality (ARI, weighted recall, weighted purity) using each
distance, with Ward's linkage and fixed-N cuts.
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


def compute_snn_distance(feat: np.ndarray, k: int) -> np.ndarray:
    """Compute pairwise SNN distance via sparse incidence matrix product.

    Returns dense (n, n) matrix where d[a, b] = 1 - jaccard(NN_k(a), NN_k(b)).
    """
    n = feat.shape[0]
    # cosine similarity matrix (feat is L2-normalized)
    print(f"  cosine sim matrix ({n}x{n})...", file=sys.stderr)
    sim = feat @ feat.T
    np.fill_diagonal(sim, -2)

    # top-K indices per row
    print(f"  top-{k} per image...", file=sys.stderr)
    nn_idx = np.argpartition(-sim, k, axis=1)[:, :k]

    # Build sparse incidence matrix: incidence[i, j] = 1 if j in NN_k(i)
    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.flatten()
    data = np.ones(n * k, dtype=np.float32)
    incidence = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Shared-NN count: A @ A.T where A is incidence
    print("  sparse matmul for shared-NN counts...", file=sys.stderr)
    shared = (incidence @ incidence.T).toarray()  # (n, n) int counts

    # Jaccard: |A ∩ B| / |A ∪ B| = inter / (k + k - inter) = inter / (2k - inter)
    union = 2.0 * k - shared
    jaccard = np.where(union > 0, shared / np.maximum(union, 1), 0.0)
    np.fill_diagonal(jaccard, 1.0)
    snn_dist = 1.0 - jaccard
    return snn_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--w-pe", type=float, default=1.0)
    ap.add_argument("--w-col", type=float, default=0.7)
    ap.add_argument("--ks", type=str, default="15,30,50",
                    help="kNN sizes for SNN (comma-separated)")
    ap.add_argument("--alphas", type=str, default="0.0,0.3,0.5,0.7,1.0",
                    help="α blending weights (0=cosine only, 1=SNN only)")
    args = ap.parse_args()

    print(f"Loading {args.folder}", file=sys.stderr)
    fnames, feat = load_features(args.folder, args.w_pe, args.w_col)
    n = len(fnames)
    gt, gnames = load_ground_truth(args.folder, fnames)
    print(f"  {n} images, {len(gnames)} groups", file=sys.stderr)

    print("Cosine pdist...", file=sys.stderr)
    t0 = time.time()
    cos_d = squareform(pdist(feat, metric="cosine")).astype(np.float64)
    print(f"  {time.time()-t0:.1f}s", file=sys.stderr)

    ks = [int(x) for x in args.ks.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    rows = []

    # Cosine baseline
    print("Cosine baseline (Ward's)...", file=sys.stderr)
    Z = linkage(squareform(cos_d, checks=False), method="ward")
    for nc in [100, 150, 200, 300, 500]:
        labels = fcluster(Z, t=nc, criterion="maxclust") - 1
        m = metrics(labels, gt)
        rows.append(("cosine", f"N={nc}", m))

    # SNN sweep
    for k in ks:
        print(f"\n--- Building SNN distance for k={k} ---", file=sys.stderr)
        t0 = time.time()
        snn_d = compute_snn_distance(feat, k)
        print(f"  done ({time.time()-t0:.1f}s)", file=sys.stderr)

        for alpha in alphas:
            d = (1.0 - alpha) * cos_d + alpha * snn_d
            np.fill_diagonal(d, 0.0)
            cd = squareform(d, checks=False)
            Z = linkage(cd, method="ward")
            for nc in [100, 150, 200, 300, 500]:
                labels = fcluster(Z, t=nc, criterion="maxclust") - 1
                m = metrics(labels, gt)
                rows.append(("blend", f"k={k} α={alpha:.1f} N={nc}", m))

    # Print
    print()
    print(f"{'method':10s} {'config':28s} {'n_cl':>6s} {'ari':>6s} "
          f"{'recall':>7s} {'w_rec':>7s} {'purity':>7s} {'w_pur':>7s} {'F1':>6s}")
    print("-" * 100)
    for method, cfg, m in rows:
        f1 = 2 * m["weighted_group_recall"] * m["weighted_cluster_purity"] / max(
            m["weighted_group_recall"] + m["weighted_cluster_purity"], 1e-10)
        print(f"{method:10s} {cfg:28s} {m['n_clusters']:6d} {m['ari']:6.3f} "
              f"{m['avg_group_recall']:7.3f} {m['weighted_group_recall']:7.3f} "
              f"{m['avg_cluster_purity']:7.3f} {m['weighted_cluster_purity']:7.3f} "
              f"{f1:6.3f}")


if __name__ == "__main__":
    main()
