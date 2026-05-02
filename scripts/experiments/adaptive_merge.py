#!/usr/bin/env python3
"""Prototype for approach #5: mixed-granularity adaptive merge.

Walks Ward's linkage tree in distance order, accepting each merge only when the
merge distance is <= β × max(density_a, density_b). Density of a cluster is
tracked as the maximum merge distance in its history (a proxy for internal
scale). Singletons bootstrap with a configurable density floor.

Compared against fixed-N and HDBSCAN-style adaptive cuts on a labeled dataset
(every image belongs to exactly one ground-truth group).

Metrics: per-group recall (largest output cluster ∩ group / |group|),
per-cluster purity (largest GT-group ∩ cluster / |cluster|), Adjusted Rand
Index, and number of output clusters.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score


def load_features(folder: str, w_pe: float = 1.0, w_col: float = 0.7):
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

    feat = np.concatenate([w_pe * pe, w_col * co], axis=1)
    feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-10)
    return fnames, feat


def load_ground_truth(folder: str, fnames):
    """Returns labels[i] = group index for fnames[i], or -1 if ungrouped."""
    with open(os.path.join(folder, ".reorder-groups.json")) as f:
        raw = json.load(f)
    groups = raw if isinstance(raw, list) else raw.get("groups", [])
    fname_to_idx = {f: i for i, f in enumerate(fnames)}
    labels = np.full(len(fnames), -1, dtype=np.int64)
    for gi, g in enumerate(groups):
        for fn in g["images"]:
            if fn in fname_to_idx:
                labels[fname_to_idx[fn]] = gi
    return labels, [g.get("name", g.get("id", str(i))) for i, g in enumerate(groups)]


def adaptive_merge_walk(Z: np.ndarray, beta: float, density_floor: float,
                         min_cluster_size: int = 1):
    """Walk linkage steps in execution order, accepting each merge only when
    the merge distance is <= β × max(density_a, density_b, density_floor).

    Designed for COMPLETE or AVERAGE linkage trees, where the merge distance
    is a meaningful inter-cluster cosine quantity and the new cluster's
    diameter equals the merge distance (complete) or grows monotonically
    (average). Ward's trees have a size-scaling artifact that breaks the β
    interpretation; pass an average/complete tree for best results.

    `min_cluster_size` floor: any merge where either side is smaller than this
    is forced through, so stragglers don't strand as singletons.
    """
    n = Z.shape[0] + 1
    n_total = n + Z.shape[0]
    parent = np.arange(n_total)
    size = np.ones(n_total, dtype=np.int64)
    intra_density = np.zeros(n_total)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    next_id = n
    n_accepted = 0
    n_rejected = 0
    for a, b, d, _ in Z:
        a, b = int(a), int(b)
        ra, rb = find(a), find(b)
        if ra == rb:
            next_id += 1
            continue
        sa, sb = size[ra], size[rb]
        da, db = intra_density[ra], intra_density[rb]

        force = sa < min_cluster_size or sb < min_cluster_size
        scale = max(da, db, density_floor)
        accept = force or d <= beta * scale

        if accept:
            new_id = next_id
            parent[ra] = new_id
            parent[rb] = new_id
            parent[new_id] = new_id
            size[new_id] = sa + sb
            intra_density[new_id] = max(da, db, d)
            n_accepted += 1
        else:
            n_rejected += 1
        next_id += 1

    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels, n_accepted, n_rejected


def metrics(pred_labels, gt_labels, fnames=None):
    """Compute per-group recall, per-cluster purity, ARI, and cluster count."""
    valid = gt_labels >= 0
    gt = gt_labels[valid]
    pred = pred_labels[valid]
    ari = adjusted_rand_score(gt, pred)
    n_clusters = len(np.unique(pred_labels))

    # Per-group recall: for each GT group, find the most-populated cluster it
    # falls into; fraction of group in that cluster.
    group_recalls = []
    group_dominant_cluster = {}
    for g in np.unique(gt):
        mask = gt == g
        members_pred = pred[mask]
        most_common = Counter(members_pred).most_common(1)[0]
        recall = most_common[1] / mask.sum()
        group_recalls.append((int(g), int(mask.sum()), float(recall),
                              int(most_common[0]), int(most_common[1])))
        group_dominant_cluster[int(g)] = int(most_common[0])

    # Per-cluster purity: for each cluster, what fraction came from one GT group?
    cluster_purities = []
    for c in np.unique(pred):
        mask = pred == c
        members_gt = gt[mask]
        most_common = Counter(members_gt).most_common(1)[0]
        purity = most_common[1] / mask.sum()
        cluster_purities.append((int(c), int(mask.sum()), float(purity),
                                  int(most_common[0]), int(most_common[1])))

    avg_recall = float(np.mean([r[2] for r in group_recalls]))
    avg_purity = float(np.mean([p[2] for p in cluster_purities]))
    weighted_recall = float(np.average([r[2] for r in group_recalls],
                                        weights=[r[1] for r in group_recalls]))
    weighted_purity = float(np.average([p[2] for p in cluster_purities],
                                        weights=[p[1] for p in cluster_purities]))

    return {
        "ari": ari,
        "n_clusters": int(n_clusters),
        "avg_group_recall": avg_recall,
        "avg_cluster_purity": avg_purity,
        "weighted_group_recall": weighted_recall,
        "weighted_cluster_purity": weighted_purity,
        "group_recalls": group_recalls,
        "cluster_purities": cluster_purities,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Image folder with .reorder-cache and .reorder-groups.json")
    ap.add_argument("--w-pe", type=float, default=1.0)
    ap.add_argument("--w-col", type=float, default=0.7)
    ap.add_argument("--betas", type=str, default="1.1,1.2,1.3,1.4,1.5,1.7,2.0",
                    help="Comma-separated β values to sweep")
    ap.add_argument("--floors", type=str, default="0.05,0.10,0.15,0.20",
                    help="Density floor values to sweep")
    ap.add_argument("--mcs-floor", type=int, default=3,
                    help="Force-merge clusters smaller than this (bootstrap)")
    ap.add_argument("--detail", action="store_true", help="Print worst groups")
    args = ap.parse_args()

    print(f"Loading features from {args.folder}", file=sys.stderr)
    t0 = time.time()
    fnames, feat = load_features(args.folder, args.w_pe, args.w_col)
    n = len(fnames)
    print(f"  Loaded {n} images, dim={feat.shape[1]} ({time.time()-t0:.1f}s)", file=sys.stderr)

    print("Computing pairwise cosine distances...", file=sys.stderr)
    t0 = time.time()
    D = pdist(feat, metric="cosine")
    print(f"  pdist done ({time.time()-t0:.1f}s)", file=sys.stderr)

    print("Running linkage trees...", file=sys.stderr)
    t0 = time.time()
    Z_ward = linkage(D, method="ward")
    Z_complete = linkage(D, method="complete")
    Z_average = linkage(D, method="average")
    print(f"  linkage done ({time.time()-t0:.1f}s)", file=sys.stderr)

    gt, group_names = load_ground_truth(args.folder, fnames)
    n_groups = (gt >= 0).sum()
    print(f"Ground truth: {n_groups} labeled images, {len(group_names)} groups", file=sys.stderr)

    rows = []

    # ── Baselines ────────────────────────────────────────────────────────────
    for nc in [50, 100, 200, 300, 500, 800]:
        labels = fcluster(Z_ward, t=nc, criterion="maxclust") - 1
        m = metrics(labels, gt)
        rows.append(("fixed_N", f"ward N={nc}", m))

    # HDBSCAN-style stability extraction
    try:
        import hdbscan
        from scipy.spatial.distance import squareform
        Dsq = squareform(D).astype(np.float64)
        np.fill_diagonal(Dsq, 0)
        for mcs in [3, 5, 10, 15, 20]:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs, metric="precomputed", allow_single_cluster=False
            )
            labels = clusterer.fit_predict(Dsq)
            mx = labels.max() + 1
            for i in range(len(labels)):
                if labels[i] == -1:
                    labels[i] = mx
                    mx += 1
            m = metrics(labels, gt)
            rows.append(("hdbscan", f"mcs={mcs}", m))
    except ImportError:
        print("(hdbscan not installed, skipping)", file=sys.stderr)

    # ── Adaptive merge sweep ─────────────────────────────────────────────────
    betas = [float(x) for x in args.betas.split(",")]
    floors = [float(x) for x in args.floors.split(",")]
    trees = [("complete", Z_complete), ("average", Z_average), ("ward", Z_ward)]
    for tree_name, Z in trees:
        for floor in floors:
            for beta in betas:
                labels, n_acc, n_rej = adaptive_merge_walk(
                    Z, beta=beta, density_floor=floor,
                    min_cluster_size=args.mcs_floor,
                )
                m = metrics(labels, gt)
                rows.append(("adaptive", f"{tree_name} β={beta:.2f} floor={floor:.2f}", m))

    # ── Print summary ────────────────────────────────────────────────────────
    print()
    print(f"{'method':10s} {'config':28s} {'n_clust':>7s} {'ari':>6s} "
          f"{'recall':>7s} {'w_rec':>7s} {'purity':>7s} {'w_pur':>7s}")
    print("-" * 96)
    for method, cfg, m in rows:
        print(f"{method:10s} {cfg:28s} "
              f"{m['n_clusters']:7d} {m['ari']:6.3f} "
              f"{m['avg_group_recall']:7.3f} {m['weighted_group_recall']:7.3f} "
              f"{m['avg_cluster_purity']:7.3f} {m['weighted_cluster_purity']:7.3f}")

    if args.detail:
        # Show the best adaptive run's worst groups
        best = max((r for r in rows if r[0] == "adaptive"),
                   key=lambda r: r[2]["weighted_group_recall"] * r[2]["weighted_cluster_purity"])
        print(f"\nBest adaptive run: {best[1]}")
        print("Worst-recall ground-truth groups:")
        worst = sorted(best[2]["group_recalls"], key=lambda r: r[2])[:10]
        for gi, sz, rec, dom_c, dom_n in worst:
            print(f"  GT group {gi:3d} ({group_names[gi]:50s}): size={sz:3d} "
                  f"recall={rec:.2f}  largest cluster has {dom_n}/{sz}")
        print("\nLeast-pure output clusters:")
        worst_p = sorted(best[2]["cluster_purities"], key=lambda r: r[2])[:10]
        for ci, sz, pur, dom_g, dom_n in worst_p:
            print(f"  Cluster {ci:4d}: size={sz:3d} purity={pur:.2f}  "
                  f"dominant GT={group_names[dom_g] if dom_g < len(group_names) else dom_g}")


if __name__ == "__main__":
    main()
