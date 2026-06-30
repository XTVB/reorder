#!/usr/bin/env python3
"""Bootstrap-stability color-w selector (Phase 4b of the auto-color-w study).

Classic model selection (Ben-Hur/Lange): the right metric makes clustering
reproducible under resampling. Per (dataset, w): draw R pairs of 80% subsamples
(capped at SUBCAP images), Ward each at the dataset's N, ARI between the two
labelings on the intersection; stability(w) = mean over pairs. Selector:
argmax_w stability, with the same shrink-to-0.7 margin as colorw_internal.

Uses oracle N for the cut (same caveat as Phase 4a; N-sensitivity checked
separately). Scored by lookup into the oracle ARI matrix.

Usage: python colorw_stability.py [--reps 3] [--jobs 6]
"""
from __future__ import annotations
import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import EVAL_SET, ALL, load_fold  # noqa: E402
from colorw_oracle import B, load_ens_sim  # noqa: E402
from colorw_select import GRID, DEPLOYED, read_tsv  # noqa: E402
from colorw_internal import pick  # noqa: E402

SUBCAP = 3000
FRAC = 0.8


def ward_labels(S, N):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = np.clip(0.5 * (D + D.T), 0.0, None)
    Z = linkage(squareform(D, checks=False), method="ward")
    return fcluster(Z, t=N, criterion="maxclust")


def stability_for(tgt, reps):
    peg, col, _, _, N = load_fold(tgt)
    Sens = load_ens_sim(tgt)
    Speg = peg @ peg.T
    Scol = col @ col.T
    n = Speg.shape[0]
    rng = np.random.default_rng(1)
    m = min(int(FRAC * n), SUBCAP)
    out = {w: [] for w in GRID}
    for _ in range(reps):
        a = rng.choice(n, m, replace=False)
        bsub = rng.choice(n, m, replace=False)
        inter = np.intersect1d(a, bsub)
        if len(inter) < 50:
            continue
        pa = {v: i for i, v in enumerate(a)}
        pb = {v: i for i, v in enumerate(bsub)}
        ia = np.array([pa[v] for v in inter])
        ib = np.array([pb[v] for v in inter])
        for w in GRID:
            S = (1 - B) * (Speg + w * w * Scol) / (1 + w * w) + B * Sens
            la = ward_labels(S[np.ix_(a, a)], N)
            lb = ward_labels(S[np.ix_(bsub, bsub)], N)
            out[w].append(adjusted_rand_score(la[ia], lb[ib]))
    return tgt, {w: float(np.mean(v)) if v else 0.0 for w, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="colorw_oracle_ens3.tsv")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--margins", nargs="*", type=float, default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--out", default="colorw_stability.tsv")
    args = ap.parse_args()

    o = read_tsv(args.oracle)
    targets = [t for t in (args.targets or ALL) if t in o]
    curves_ari = {d: np.array([float(o[d][f"w{w}"]) for w in GRID]) for d in targets}

    stab = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(stability_for, t, args.reps): t for t in targets}
        for fut in as_completed(futs):
            tgt, s = fut.result()
            stab[tgt] = s
            print(f"{tgt}: pick={pick(s, 0.0)}  "
                  + " ".join(f"{w}:{s[w]:.3f}" for w in GRID), flush=True)

    i07 = GRID.index(DEPLOYED)
    ev = [d for d in targets if d in EVAL_SET and d in stab]
    base = np.mean([curves_ari[d][i07] for d in ev])
    print(f"\neval n={len(ev)}  fixed@0.7={base:.4f}")
    print(f"{'margin':>7}{'evalARI':>9}{'delta':>9}{'win':>5}{'loss':>5}  worst")
    for m in args.margins:
        deltas = {d: curves_ari[d][GRID.index(pick(stab[d], m))] - curves_ari[d][i07]
                  for d in ev}
        md = float(np.mean(list(deltas.values())))
        wins = sum(1 for v in deltas.values() if v > 0.002)
        losses = sum(1 for v in deltas.values() if v < -0.002)
        worst = min(deltas, key=deltas.get)
        print(f"{m:>7.2f}{base + md:>9.4f}{md:>+9.4f}{wins:>5}{losses:>5}"
              f"  {worst} {deltas[worst]:+.4f}")

    with open(args.out, "w") as fh:
        fh.write("dataset\t" + "\t".join(f"w{w}" for w in GRID) + "\n")
        for d in targets:
            if d in stab:
                fh.write(d + "\t" + "\t".join(f"{stab[d][w]:.6f}" for w in GRID) + "\n")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
