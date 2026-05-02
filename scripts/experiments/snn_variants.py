#!/usr/bin/env python3
"""Benchmark variants of shared-nearest-neighbors distance:
  - vanilla SNN (binary kNN, Jaccard)
  - k-reciprocal SNN (only mutual edges)
  - rank-weighted SNN (weight neighbors by 1/(1+rank))
  - multi-scale SNN (blend Jaccard at multiple k values)
  - mutual-reachability distance (HDBSCAN-style core distance)
  - two-pass refined SNN (drop hub neighbors before re-computing)

Each variant is blended with cosine: d = (1-α) * cosine + α * variant.
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


def topk_per_row(sim: np.ndarray, k: int):
    """Return (top_k_indices [n,k], top_k_sims [n,k]) sorted by sim descending."""
    n = sim.shape[0]
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    # Sort within selected k
    rows = np.arange(n)[:, None]
    sims_topk = sim[rows, idx]
    order = np.argsort(-sims_topk, axis=1)
    idx_sorted = idx[rows, order]
    sims_sorted = sims_topk[rows, order]
    return idx_sorted, sims_sorted


def vanilla_snn(sim: np.ndarray, k: int):
    n = sim.shape[0]
    nn_idx = np.argpartition(-sim, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.flatten()
    inc = sparse.csr_matrix(
        (np.ones(n * k, dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    shared = (inc @ inc.T).toarray()
    union = 2.0 * k - shared
    jaccard = np.where(union > 0, shared / np.maximum(union, 1), 0.0)
    np.fill_diagonal(jaccard, 1.0)
    return 1.0 - jaccard


def kreciprocal_snn(sim: np.ndarray, k: int):
    """Only count edges that are mutual: i in NN_k(j) AND j in NN_k(i)."""
    n = sim.shape[0]
    nn_idx = np.argpartition(-sim, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.flatten()
    inc = sparse.csr_matrix(
        (np.ones(n * k, dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    # Mutual edges only: element-wise min of inc and inc.T
    mutual = inc.minimum(inc.T)
    shared = (mutual @ mutual.T).toarray()
    deg = np.asarray(mutual.sum(axis=1)).flatten()
    union = deg[:, None] + deg[None, :] - shared
    jaccard = np.where(union > 0, shared / np.maximum(union, 1), 0.0)
    np.fill_diagonal(jaccard, 1.0)
    return 1.0 - jaccard


def rank_weighted_snn(sim: np.ndarray, k: int, decay: str = "inv_rank"):
    """Rank-weighted SNN via cosine-style kernel of weighted incidence vectors.

    For each pair (a,b), the kernel is Σ_c w_a(c) * w_b(c) over shared NNs,
    normalized by sqrt(||w_a||² * ||w_b||²). This is a tractable proxy for
    weighted Jaccard and stays in [0,1].
    """
    n = sim.shape[0]
    nn_idx, _ = topk_per_row(sim, k)
    if decay == "inv_rank":
        weights = 1.0 / (1.0 + np.arange(k, dtype=np.float32))
    elif decay == "exp":
        weights = np.exp(-np.arange(k, dtype=np.float32) / (k / 4))
    else:
        weights = np.ones(k, dtype=np.float32)

    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.flatten()
    data = np.tile(weights, n)
    inc = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    kernel = (inc @ inc.T).toarray()
    diag = np.diag(kernel).copy()
    norm = np.sqrt(np.outer(diag, diag) + 1e-10)
    sim_kernel = kernel / norm
    np.fill_diagonal(sim_kernel, 1.0)
    return 1.0 - sim_kernel


def multiscale_snn(sim: np.ndarray, ks: list):
    """Average SNN distance over multiple k values."""
    out = np.zeros(sim.shape, dtype=np.float64)
    for k in ks:
        out += vanilla_snn(sim, k)
    out /= len(ks)
    return out


def mutual_reach_distance(cos_d: np.ndarray, k: int):
    """Mutual reachability: max(d(a,b), core_k(a), core_k(b))."""
    sorted_d = np.sort(cos_d, axis=1)
    core = sorted_d[:, k]  # k-th smallest, excluding self at index 0
    mr = np.maximum(cos_d, np.maximum(core[:, None], core[None, :]))
    return mr


def two_pass_snn(sim: np.ndarray, k: int, hub_drop_pct: float = 0.05):
    """Round 1: identify hub images (appear in many top-K NN lists). Round 2: drop hubs from NN lists, recompute SNN."""
    n = sim.shape[0]
    nn_idx = np.argpartition(-sim, k, axis=1)[:, :k]
    # Count how often each image appears as someone else's NN
    counts = np.bincount(nn_idx.flatten(), minlength=n)
    # Hubs: top hub_drop_pct most-frequent NNs
    n_hubs = max(1, int(n * hub_drop_pct))
    hub_threshold = np.sort(counts)[-n_hubs]
    is_hub = counts >= hub_threshold

    # Round 2: build kNN excluding hubs as TARGETS
    sim_no_hub = sim.copy()
    sim_no_hub[:, is_hub] = -2  # never select hubs as NN
    return vanilla_snn(sim_no_hub, k), int(is_hub.sum())


def benchmark(folder, alphas=(0.0, 0.3, 0.5, 0.7), ns=(150, 200, 300)):
    print(f"Loading {folder}", file=sys.stderr)
    fnames, feat = load_features(folder, 1.0, 0.7)
    n = len(fnames)
    gt, gnames = load_ground_truth(folder, fnames)
    print(f"  {n} images, {len(gnames)} groups", file=sys.stderr)

    print("Cosine sim matrix...", file=sys.stderr)
    t0 = time.time()
    sim = (feat @ feat.T).astype(np.float32)
    np.fill_diagonal(sim, -2)
    cos_d = (1.0 - sim).astype(np.float64)
    np.fill_diagonal(cos_d, 0.0)
    print(f"  {time.time()-t0:.1f}s", file=sys.stderr)

    rows = []

    # Cosine baseline
    print("Cosine baseline...", file=sys.stderr)
    Z = linkage(squareform(cos_d, checks=False), method="ward")
    for nc in ns:
        labels = fcluster(Z, t=nc, criterion="maxclust") - 1
        rows.append(("cosine", f"N={nc}", metrics(labels, gt)))

    variants = [
        ("vanilla_k50", lambda: vanilla_snn(sim, 50)),
        ("vanilla_k75", lambda: vanilla_snn(sim, 75)),
        ("kreciprocal_k50", lambda: kreciprocal_snn(sim, 50)),
        ("kreciprocal_k75", lambda: kreciprocal_snn(sim, 75)),
        ("kreciprocal_k100", lambda: kreciprocal_snn(sim, 100)),
        ("rankweight_k50", lambda: rank_weighted_snn(sim, 50, "inv_rank")),
        ("rankweight_k75", lambda: rank_weighted_snn(sim, 75, "inv_rank")),
        ("rankweight_exp_k75", lambda: rank_weighted_snn(sim, 75, "exp")),
        ("multiscale_30_60_120", lambda: multiscale_snn(sim, [30, 60, 120])),
        ("multiscale_50_100", lambda: multiscale_snn(sim, [50, 100])),
        ("twopass_k50", lambda: two_pass_snn(sim, 50)[0]),
        ("twopass_k75", lambda: two_pass_snn(sim, 75)[0]),
        ("mutual_reach_k15", lambda: mutual_reach_distance(cos_d, 15)),
        ("mutual_reach_k30", lambda: mutual_reach_distance(cos_d, 30)),
    ]

    for vname, vfn in variants:
        print(f"Computing {vname}...", file=sys.stderr)
        t0 = time.time()
        vd = vfn()
        print(f"  {time.time()-t0:.1f}s", file=sys.stderr)
        for alpha in alphas:
            d = (1.0 - alpha) * cos_d + alpha * vd
            np.fill_diagonal(d, 0.0)
            Z = linkage(squareform(d, checks=False), method="ward")
            for nc in ns:
                labels = fcluster(Z, t=nc, criterion="maxclust") - 1
                rows.append((vname, f"α={alpha:.1f} N={nc}", metrics(labels, gt)))

    # Print sorted by ARI desc
    print()
    print(f"{'method':22s} {'config':18s} {'n_cl':>6s} {'ari':>6s} "
          f"{'recall':>7s} {'w_rec':>7s} {'purity':>7s} {'w_pur':>7s} {'F1':>6s}")
    print("-" * 110)
    rows.sort(key=lambda r: -r[2]["ari"])
    for vname, cfg, m in rows[:60]:  # top 60
        f1 = 2 * m["weighted_group_recall"] * m["weighted_cluster_purity"] / max(
            m["weighted_group_recall"] + m["weighted_cluster_purity"], 1e-10)
        print(f"{vname:22s} {cfg:18s} {m['n_clusters']:6d} {m['ari']:6.3f} "
              f"{m['avg_group_recall']:7.3f} {m['weighted_group_recall']:7.3f} "
              f"{m['avg_cluster_purity']:7.3f} {m['weighted_cluster_purity']:7.3f} "
              f"{f1:6.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    args = ap.parse_args()
    benchmark(args.folder)
