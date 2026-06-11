#!/usr/bin/env python3
"""Over-cluster → merge-back A/B (LOMO folds).

Hypothesis: Ward's late greedy merges are where oracle-N ARI is lost. Instead of
trusting Ward to the end, cut the tree at m·N clusters (m > 1) and agglomerate
back to N using a *robust pairwise statistic* between groups (the same shape of
score the production merge-suggestions mode uses: median / p75 of all cross-pair
similarities), rather than Ward's centroid-variance criterion.

Both arms share the identical deployed distance matrix (blend 0.60, color 0.7),
so any delta is purely the clustering algorithm. Scored as in
verifier_accuracy_sweep.py: Ward on the full image set, ARI over labeled images.

Criteria:
  q25/median/q75/q90 : percentile of cross-pair cosine sims (subsampled at CAP)
  mean               : avg cross-pair sim (≈ centroid cosine, average-linkage-like)
  max                : best single pair (single-linkage-like; chaining control)
  relmedian          : median(A,B) / sqrt(within-median(A) · within-median(B)) —
                       density-normalized, adapts to per-group tightness

Usage:
  python overcluster_merge_eval.py                  # all folds present under $LOMO_ROOT
  python overcluster_merge_eval.py --targets M5 M17 # subset
  python overcluster_merge_eval.py --out overcluster_merge.tsv
"""
from __future__ import annotations
import argparse
import os
import sys
import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import COLOR_W, EVAL_SET, PEG_W, ALL, load_fold  # noqa: E402

B = 0.60                 # deployed inference blend
CAP = 4096               # max sampled cross-pair values per group pair (quantile est.)
M_FACTORS = [1.5, 2.0, 3.0, 4.0]
CRITS = ["q25", "median", "q75", "q90", "mean", "max", "relmedian"]
SEED = 0


def deployed_sim(peg, col, proj):
    Gzs = (PEG_W**2 * (peg @ peg.T) + COLOR_W**2 * (col @ col.T)) / (PEG_W**2 + COLOR_W**2)
    return (1 - B) * Gzs + B * (proj @ proj.T)


def ward_tree(S):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = np.clip(0.5 * (D + D.T), 0.0, None)
    return linkage(squareform(D, checks=False), method="ward")


def ari_of(labels, true):
    grouped = true >= 0
    return adjusted_rand_score(true[grouped], labels[grouped])


def _sample(vals, rng):
    v = vals.ravel()
    if v.size > CAP:
        v = v[rng.choice(v.size, CAP, replace=False)]
    return v


def pair_stats(S, A, Bi, rng):
    """All base statistics for one group pair, from (subsampled) cross-pair sims."""
    v = _sample(S[np.ix_(A, Bi)], rng)
    q25, med, q75, q90 = np.percentile(v, [25, 50, 75, 90])
    return {"q25": q25, "median": med, "q75": q75, "q90": q90,
            "mean": float(v.mean()), "max": float(v.max())}


def within_median(S, A, rng):
    if len(A) < 2:
        return None
    sub = S[np.ix_(A, A)]
    iu = np.triu_indices(len(A), k=1)
    return float(np.median(_sample(sub[iu], rng)))


def merge_to_n(S, labels0, N, crit, rng, global_within):
    """Greedy agglomeration from the labels0 partition down to N clusters."""
    uniq = np.unique(labels0)
    clusters = [np.where(labels0 == c)[0] for c in uniq]
    K = len(clusters)
    if K <= N:
        return labels0

    within = [within_median(S, A, rng) for A in clusters]

    def score(i, j):
        st = pair_stats(S, clusters[i], clusters[j], rng)
        if crit == "relmedian":
            wi = within[i] if within[i] is not None else global_within
            wj = within[j] if within[j] is not None else global_within
            denom = np.sqrt(max(wi, 1e-6) * max(wj, 1e-6))
            return st["median"] / denom
        return st[crit]

    sc = np.full((K, K), -np.inf, dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            sc[i, j] = score(i, j)

    alive = np.ones(K, dtype=bool)
    n_alive = K
    while n_alive > N:
        flat = np.argmax(sc)
        i, j = divmod(flat, K)
        # merge j into i
        clusters[i] = np.concatenate([clusters[i], clusters[j]])
        within[i] = within_median(S, clusters[i], rng)
        alive[j] = False
        n_alive -= 1
        sc[j, :] = -np.inf
        sc[:, j] = -np.inf
        for k in np.where(alive)[0]:
            if k == i:
                continue
            a, b = (i, k) if i < k else (k, i)
            sc[a, b] = score(a, b)

    out = np.empty(len(labels0), dtype=np.int64)
    for ci, idx in enumerate(c for c, a in zip(clusters, alive) if a):
        out[idx] = ci
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--m", nargs="*", type=float, default=M_FACTORS)
    ap.add_argument("--crits", nargs="*", default=CRITS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    targets = args.targets or ALL
    cols = [f"m{m:g}_{c}" for m in args.m for c in args.crits]
    rows = []
    for tgt in targets:
        try:
            peg, col, proj, true, N = load_fold(tgt)
        except FileNotFoundError as e:
            print(f"{tgt}: fold incomplete, skipping ({e})", file=sys.stderr)
            continue
        rng = np.random.default_rng(SEED)
        S = deployed_sim(peg, col, proj)
        Z = ward_tree(S)
        base = ari_of(fcluster(Z, t=N, criterion="maxclust"), true)

        # Dataset-level within-group density fallback for relmedian singletons:
        # median over the baseline cut's per-cluster within medians.
        base_labels = fcluster(Z, t=N, criterion="maxclust")
        wms = [within_median(S, np.where(base_labels == c)[0], rng)
               for c in np.unique(base_labels)]
        wms = [w for w in wms if w is not None]
        global_within = float(np.median(wms)) if wms else 1.0

        res = {"dataset": tgt, "base": base}
        for m in args.m:
            K0 = min(int(np.ceil(m * N)), len(true) - 1)
            labels0 = fcluster(Z, t=K0, criterion="maxclust")
            for c in args.crits:
                res[f"m{m:g}_{c}"] = ari_of(
                    merge_to_n(S, labels0, N, c, rng, global_within), true)
        rows.append(res)
        deltas = " ".join(f"{k.split('_', 1)[0]}/{k.split('_', 1)[1]}:{res[k] - base:+.3f}"
                          for k in cols if k in res)
        print(f"{tgt}: base={base:.4f}  {deltas}", flush=True)

    if not rows:
        print("No complete folds found.", file=sys.stderr)
        sys.exit(1)

    # Summary means: full set of scored targets, and the EVAL_SET subset (no outliers).
    def mean_over(subset, key):
        vals = [r[key] for r in rows if r["dataset"] in subset and key in r]
        return sum(vals) / len(vals) if vals else float("nan")

    scored = [r["dataset"] for r in rows]
    eval_scored = [t for t in scored if t in EVAL_SET]
    print(f"\n=== means over {len(scored)} scored ({len(eval_scored)} in EVAL_SET) ===")
    print(f"{'config':<16} {'full':>8} {'eval':>8} {'Δfull':>8} {'Δeval':>8}")
    bf, be = mean_over(scored, "base"), mean_over(eval_scored, "base")
    print(f"{'base (ward@N)':<16} {bf:>8.4f} {be:>8.4f} {'—':>8} {'—':>8}")
    for k in cols:
        f, e = mean_over(scored, k), mean_over(eval_scored, k)
        print(f"{k:<16} {f:>8.4f} {e:>8.4f} {f - bf:>+8.4f} {e - be:>+8.4f}")

    if args.out:
        with open(args.out, "w") as fh:
            hdr = ["dataset", "base"] + cols
            fh.write("\t".join(hdr) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(h, "NA")) for h in hdr) + "\n")
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
