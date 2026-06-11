#!/usr/bin/env python3
"""Seed-ensemble A/B: does averaging the learned-proj similarity over heads
trained at different seeds beat a single head?

Per fold: S_proj^(k) = proj_k @ proj_k.T per replica root, ensemble = mean over
roots (equivalent to concatenating the L2-normed projections), blended at the
deployed 0.60 and Ward-cut at oracle N. Reports each replica solo (= the seed
noise floor) and cumulative ensembles.

Usage: python seed_ensemble_eval.py [--roots R1 R2 ...] [--out f.tsv]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import COLOR_W, EVAL_SET, PEG_W, ALL, LOMO, l2, load_fold  # noqa: E402

B = 0.60
DEFAULT_ROOTS = [LOMO] + [os.path.expanduser(f"~/.cache/reorder/lomo_v26_r{r}") for r in (2, 3, 4)]


def ward_ari(S, true, N):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = np.clip(0.5 * (D + D.T), 0.0, None)
    Z = linkage(squareform(D, checks=False), method="ward")
    g = true >= 0
    return adjusted_rand_score(true[g], fcluster(Z, t=N, criterion="maxclust")[g])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    R = len(args.roots)

    rows = []
    for tgt in args.targets or ALL:
        try:
            peg, col, _, true, N = load_fold(tgt)
            fns0 = json.load(open(f"{args.roots[0]}/{tgt}/{tgt}_filenames.json"))
            projs = []
            for root in args.roots:
                fns = json.load(open(f"{root}/{tgt}/{tgt}_filenames.json"))
                assert fns == fns0, f"{tgt}: filename order differs in {root}"
                projs.append(l2(np.load(f"{root}/{tgt}/{tgt}_proj.npy").astype(np.float32)))
        except (FileNotFoundError, AssertionError) as e:
            print(f"{tgt}: skipping ({e})", file=sys.stderr)
            continue

        Gzs = (PEG_W**2 * (peg @ peg.T) + COLOR_W**2 * (col @ col.T)) / (PEG_W**2 + COLOR_W**2)
        sims = [p @ p.T for p in projs]
        res = {"dataset": tgt}
        for i, s in enumerate(sims):
            res[f"solo{i + 1}"] = ward_ari((1 - B) * Gzs + B * s, true, N)
        for k in range(2, R + 1):
            ens = np.mean(sims[:k], axis=0)
            res[f"ens{k}"] = ward_ari((1 - B) * Gzs + B * ens, true, N)
        rows.append(res)
        print(f"{tgt}: " + "  ".join(f"{k}={v:.4f}" for k, v in res.items() if k != "dataset"),
              flush=True)

    if not rows:
        sys.exit("no folds scored")
    keys = [k for k in rows[0] if k != "dataset"]
    ev = [r for r in rows if r["dataset"] in EVAL_SET]
    print(f"\n=== means ({len(rows)} scored, {len(ev)} in EVAL_SET) ===")
    solo_keys = [k for k in keys if k.startswith("solo")]
    for sub, name in ((rows, "full"), (ev, "eval")):
        solo_avg = np.mean([[r[k] for k in solo_keys] for r in sub])
        line = "  ".join(f"{k}={np.mean([r[k] for r in sub]):.4f}" for k in keys)
        print(f"{name:<5} {line}   mean-solo={solo_avg:.4f}")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\t".join(["dataset"] + keys) + "\n")
            for r in rows:
                fh.write("\t".join([r["dataset"]] + [f"{r[k]:.6f}" for k in keys]) + "\n")
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
