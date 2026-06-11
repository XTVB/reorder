#!/usr/bin/env python3
"""Ward-init boundary-polish A/B (LOMO folds).

Keep Ward@N structure; reassign individual images to the cluster they fit best,
for a few damped iterations. Rationale: oracle_ceiling.tsv shows the residual
error is in local kNN edges (perfect k=40 edge classification → ~0.95), i.e.
misassigned boundary images, not merge ordering (overcluster_merge_eval falsified
that). A kNN-vote reassignment is the practical analog of that oracle.

Caution encoded in the design: standalone spherical k-means scores far below
Ward (algo_comparison.tsv), so full convergence to a reassignment fixed point is
bad — we polish with small steps and track ARI per iteration to find where the
trajectory peaks.

Scoring rule per (image i, cluster c): mean of the top-K similarities between i
and members of c (K=cluster-local neighbors; K=inf → plain mean = spherical
k-means direction). Move i only if best − current > MARGIN.

Output columns: <agg>_it<t> for t = 1..ITERS.
"""
from __future__ import annotations
import argparse
import sys
import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import COLOR_W, EVAL_SET, PEG_W, ALL, load_fold  # noqa: E402

B = 0.60
ITERS = 5
AGGS = ["top3", "top5", "top10", "mean"]
MARGIN = 0.0


def deployed_sim(peg, col, proj):
    Gzs = (PEG_W**2 * (peg @ peg.T) + COLOR_W**2 * (col @ col.T)) / (PEG_W**2 + COLOR_W**2)
    return (1 - B) * Gzs + B * (proj @ proj.T)


def ari_of(labels, true):
    g = true >= 0
    return adjusted_rand_score(true[g], labels[g])


def fit_matrix(S, labels, agg):
    """(n, C) score of every image against every cluster. Self-sim excluded."""
    n = len(labels)
    uniq = np.unique(labels)
    F = np.full((n, len(uniq)), -np.inf, dtype=np.float32)
    Sx = S.copy()
    np.fill_diagonal(Sx, -np.inf)  # never count self
    for ci, c in enumerate(uniq):
        idx = np.where(labels == c)[0]
        X = Sx[:, idx]  # (n, |c|)
        if agg == "mean":
            # mean over valid entries (members of c excluding self)
            cnt = np.full(n, len(idx), dtype=np.float32)
            cnt[idx] -= 1
            cnt = np.maximum(cnt, 1)
            F[:, ci] = np.where(X == -np.inf, 0, X).sum(axis=1) / cnt
        else:
            k = min(int(agg[3:]), X.shape[1])
            top = -np.partition(-X, k - 1, axis=1)[:, :k]
            valid = top != -np.inf
            F[:, ci] = np.where(valid, top, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    return F, uniq


def polish(S, labels0, true, agg, iters, margin):
    """Run damped reassignment; return ARI after each iteration."""
    labels = labels0.copy()
    out = []
    for _ in range(iters):
        F, uniq = fit_matrix(S, labels, agg)
        pos = {c: i for i, c in enumerate(uniq)}
        cur = F[np.arange(len(labels)), [pos[l] for l in labels]]
        best_ci = F.argmax(axis=1)
        best = F[np.arange(len(labels)), best_ci]
        move = best - cur > margin
        labels = labels.copy()
        labels[move] = uniq[best_ci[move]]
        out.append((ari_of(labels, true), int(move.sum())))
        if move.sum() == 0:
            while len(out) < iters:
                out.append(out[-1])
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--aggs", nargs="*", default=AGGS)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for tgt in args.targets or ALL:
        try:
            peg, col, proj, true, N = load_fold(tgt)
        except FileNotFoundError as e:
            print(f"{tgt}: skipping ({e})", file=sys.stderr)
            continue
        S = deployed_sim(peg, col, proj)
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(np.clip(0.5 * (D + D.T), 0, None), checks=False), method="ward")
        labels0 = fcluster(Z, t=N, criterion="maxclust")
        res = {"dataset": tgt, "base": ari_of(labels0, true)}
        moved = {}
        for agg in args.aggs:
            traj = polish(S, labels0, true, agg, args.iters, args.margin)
            for t, (a, mv) in enumerate(traj, 1):
                res[f"{agg}_it{t}"] = a
            moved[agg] = [mv for _, mv in traj]
        rows.append(res)
        summ = "  ".join(f"{agg}:{res[f'{agg}_it1'] - res['base']:+.3f}@1"
                         f"/{res[f'{agg}_it{args.iters}'] - res['base']:+.3f}@{args.iters}"
                         f"(mv{moved[agg][0]})" for agg in args.aggs)
        print(f"{tgt}: base={res['base']:.4f}  {summ}", flush=True)

    if not rows:
        sys.exit("no folds scored")
    keys = [k for k in rows[0] if k not in ("dataset",)]
    ev = [r for r in rows if r["dataset"] in EVAL_SET]
    print(f"\n=== means ({len(rows)} scored, {len(ev)} in EVAL_SET) ===")
    be = np.mean([r["base"] for r in ev])
    print(f"{'config':<14}{'eval':>9}{'Δeval':>9}{'wins':>7}")
    for k in keys:
        e = np.mean([r[k] for r in ev])
        if k == "base":
            print(f"{k:<14}{e:>9.4f}{'—':>9}")
            continue
        wins = sum(1 for r in ev if r[k] > r["base"])
        print(f"{k:<14}{e:>9.4f}{e - be:>+9.4f}{wins:>5}/{len(ev)}")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\t".join(["dataset"] + keys) + "\n")
            for r in rows:
                fh.write("\t".join([r["dataset"]] + [f"{r[k]:.6f}" for k in keys]) + "\n")
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
