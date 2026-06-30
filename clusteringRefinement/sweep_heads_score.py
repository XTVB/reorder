#!/usr/bin/env python3
"""Score split-head hyperparam variants inside the winning 3-head blend.

For each candidate fold root (trained by the sweep driver with one hyperparam
changed), swap that head into joint .55 + peg .30 + color .15 — the other two
components stay at the seed-42 controls — Ward @ oracle N, ARI. Reports the
candidate's blend delta vs the all-control blend, plus head-solo ARI.

Usage: python sweep_heads_score.py [--targets M2 M5 M9 M11] [--root ~/.cache/reorder/sweep_heads]
"""
from __future__ import annotations
import argparse
import os
import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import LOMO, l2, load_fold  # noqa: E402

DEV = ["M2", "M5", "M9", "M11"]
CTRL_PEG = os.path.expanduser("~/.cache/reorder/lomo_pegonly")
CTRL_COL = os.path.expanduser("~/.cache/reorder/lomo_coloronly")
W = {"hb": 0.55, "hp": 0.30, "hc": 0.15}


def proj_sim(root, tgt):
    p = f"{root}/{tgt}/{tgt}_proj.npy"
    if not os.path.exists(p):
        return None
    pr = l2(np.load(p).astype(np.float32))
    return pr @ pr.T


def ward_ari(S, true, N):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = np.clip(0.5 * (D + D.T), 0.0, None)
    Z = linkage(squareform(D, checks=False), method="ward")
    g = true >= 0
    return adjusted_rand_score(true[g], fcluster(Z, t=N, criterion="maxclust")[g])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=DEV)
    ap.add_argument("--root", default=os.path.expanduser("~/.cache/reorder/sweep_heads"))
    ap.add_argument("--out", default="sweep_heads_dev.tsv")
    args = ap.parse_args()

    cands = sorted(d for d in os.listdir(args.root)
                   if os.path.isdir(f"{args.root}/{d}"))
    rows = {}  # cand -> {tgt: (blend_ari, solo_ari)}
    ctrl = {}  # tgt -> blend ari with all controls
    for tgt in args.targets:
        _, _, _, true, N = load_fold(tgt)
        sims = {"hb": proj_sim(LOMO, tgt),
                "hp": proj_sim(CTRL_PEG, tgt),
                "hc": proj_sim(CTRL_COL, tgt)}
        blend = lambda s: sum(W[k] * s[k] for k in W) / sum(W.values())  # noqa: E731
        ctrl[tgt] = ward_ari(blend(sims), true, N)
        print(f"{tgt}: ctrl-blend={ctrl[tgt]:.4f}", flush=True)
        for c in cands:
            Sc = proj_sim(f"{args.root}/{c}", tgt)
            if Sc is None:
                continue
            slot = "hc" if c.startswith("c_") else "hp"
            swapped = dict(sims)
            swapped[slot] = Sc
            rows.setdefault(c, {})[tgt] = (ward_ari(blend(swapped), true, N),
                                           ward_ari(Sc, true, N))
            b, s = rows[c][tgt]
            print(f"  {c}: blend={b:.4f} ({b - ctrl[tgt]:+.4f})  solo={s:.4f}", flush=True)

    print(f"\n=== dev-set means (n={len(args.targets)}; ctrl blend "
          f"{np.mean([ctrl[t] for t in args.targets]):.4f}) ===")
    print(f"{'candidate':<16}{'blend':>8}{'delta':>9}{'solo':>8}{'n':>3}")
    lines = []
    for c in sorted(rows, key=lambda c: -np.mean([v[0] for v in rows[c].values()])):
        ts = [t for t in args.targets if t in rows[c]]
        b = np.mean([rows[c][t][0] for t in ts])
        s = np.mean([rows[c][t][1] for t in ts])
        d = b - np.mean([ctrl[t] for t in ts])
        print(f"{c:<16}{b:>8.4f}{d:>+9.4f}{s:>8.4f}{len(ts):>3}")
        lines.append((c, b, d, s, ts))
    with open(args.out, "w") as fh:
        fh.write("candidate\tblend\tdelta_vs_ctrl\tsolo\ttargets\n")
        for c, b, d, s, ts in lines:
            fh.write(f"{c}\t{b:.6f}\t{d:+.6f}\t{s:.6f}\t{','.join(ts)}\n")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
