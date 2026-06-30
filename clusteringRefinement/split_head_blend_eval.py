#!/usr/bin/env python3
"""Split-head blend sweep: zs-PE-G / zs-color / learned-both / learned-peg /
learned-color as five similarity components, swept over weight configs.

Heads: learned-both = the v26 deployed-config fold (seed 42); learned-peg /
learned-color = single-modality heads trained with --input-mods (same recipe,
drop-color disabled for the color head), fold roots ~/.cache/reorder/
lomo_{pegonly,coloronly}. All single-seed, so deltas are head-vs-head fair
(the deployed ens3 adds ~+0.01 on top of any winner via seed-ensembling).

S(cfg) = sum(w_i * S_i) / sum(w_i), Ward @ oracle N, ARI on labeled images.
The deployed reference in this notation: 0.4*(zs_peg + 0.49*zs_col)/1.49 +
0.6*S_both, i.e. weights (0.2685, 0.1315, 0.6, 0, 0).

Usage: python split_head_blend_eval.py [--jobs 6] [--out split_head_blend.tsv]
"""
from __future__ import annotations
import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import EVAL_SET, ALL, LOMO, l2, load_fold  # noqa: E402

def _roots(env, default):
    return [os.path.expanduser(p) for p in os.environ.get(env, default).split(",")]

# Comma-list env overrides turn any head into a seed ensemble (mean of sims).
PEG_ROOTS = _roots("SPLIT_PEG_ROOTS", "~/.cache/reorder/lomo_pegonly")
COL_ROOTS = _roots("SPLIT_COL_ROOTS", "~/.cache/reorder/lomo_coloronly")
BOTH_ROOTS = _roots("SPLIT_BOTH_ROOTS", LOMO)

# zs split inside a 0.4 zero-shot share, deployed composition (peg 1.0, color 0.7):
ZP, ZC = 0.4 / 1.49, 0.4 * 0.49 / 1.49   # 0.2685, 0.1315

# (name, (zs_peg, zs_col, h_both, h_peg, h_col)) — normalized in compose().
CONFIGS = [
    ("deployed(b0.6)",      (ZP, ZC, 0.60, 0.00, 0.00)),
    ("zs-only",             (1.0, 0.49, 0, 0, 0)),
    ("both-only",           (0, 0, 1, 0, 0)),
    ("hpeg-only",           (0, 0, 0, 1, 0)),
    ("hcol-only",           (0, 0, 0, 0, 1)),
    ("split r=1.0",         (ZP, ZC, 0, 0.60, 0.00)),
    ("split r=0.83",        (ZP, ZC, 0, 0.50, 0.10)),
    ("split r=0.67",        (ZP, ZC, 0, 0.40, 0.20)),
    ("split r=0.50",        (ZP, ZC, 0, 0.30, 0.30)),
    ("split r=0.33",        (ZP, ZC, 0, 0.20, 0.40)),
    ("split r=0.17",        (ZP, ZC, 0, 0.10, 0.50)),
    ("split r=0.0",         (ZP, ZC, 0, 0.00, 0.60)),
    ("tri b.3 p.15 c.15",   (ZP, ZC, 0.30, 0.15, 0.15)),
    ("tri b.2 p.2 c.2",     (ZP, ZC, 0.20, 0.20, 0.20)),
    ("tri b.4 p.1 c.1",     (ZP, ZC, 0.40, 0.10, 0.10)),
    ("tri b.4 p.2",         (ZP, ZC, 0.40, 0.20, 0.00)),
    ("tri b.4 c.2",         (ZP, ZC, 0.40, 0.00, 0.20)),
    ("tri b.48 p.12",       (ZP, ZC, 0.48, 0.12, 0.00)),
    ("tri b.48 c.12",       (ZP, ZC, 0.48, 0.00, 0.12)),
    ("tri b.52 c.08",       (ZP, ZC, 0.52, 0.00, 0.08)),
    ("noZS b.6 p.25 c.15",  (0, 0, 0.60, 0.25, 0.15)),
    ("noZSc b.6 c.12 zp.28", (0.28, 0, 0.60, 0.00, 0.12)),
    ("hvy b.7 c.1",         (0.75 * ZP, 0.75 * ZC, 0.70, 0.00, 0.10)),
    ("lrn.7 b.5 p.1 c.1",   (0.75 * ZP, 0.75 * ZC, 0.50, 0.10, 0.10)),
    ("lrn.5 b.3 p.1 c.1",   (1.25 * ZP, 1.25 * ZC, 0.30, 0.10, 0.10)),
]

# Override the grid without editing the file: SPLIT_CONFIGS_JSON='[["name",[w1..w5]],...]'
# (env propagates to the ProcessPool workers, unlike a patched module global).
if os.environ.get("SPLIT_CONFIGS_JSON"):
    import json as _json
    CONFIGS = [(n, tuple(w)) for n, w in _json.loads(os.environ["SPLIT_CONFIGS_JSON"])]


def load_proj_sim(roots, tgt):
    S = None
    for root in roots:
        p = f"{root}/{tgt}/{tgt}_proj.npy"
        if not os.path.exists(p):
            return None
        pr = l2(np.load(p).astype(np.float32))
        S = pr @ pr.T if S is None else S + pr @ pr.T
    return S / len(roots)


def ward_ari(S, true, N):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = np.clip(0.5 * (D + D.T), 0.0, None)
    Z = linkage(squareform(D, checks=False), method="ward")
    g = true >= 0
    return adjusted_rand_score(true[g], fcluster(Z, t=N, criterion="maxclust")[g])


def score_fold(tgt):
    peg, col, _, true, N = load_fold(tgt)
    sims = {
        "zp": peg @ peg.T,
        "zc": col @ col.T,
        "hb": load_proj_sim(BOTH_ROOTS, tgt),
        "hp": load_proj_sim(PEG_ROOTS, tgt),
        "hc": load_proj_sim(COL_ROOTS, tgt),
    }
    if sims["hb"] is None or sims["hp"] is None or sims["hc"] is None:
        return tgt, None
    out = {}
    for name, w in CONFIGS:
        ws = np.array(w, dtype=np.float64)
        S = sum(wi * sims[k] for wi, k in zip(ws, ("zp", "zc", "hb", "hp", "hc")) if wi > 0)
        out[name] = ward_ari(S / ws.sum(), true, N)
    return tgt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--out", default="split_head_blend.tsv")
    args = ap.parse_args()
    targets = args.targets or ALL

    res = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(score_fold, t): t for t in targets}
        for fut in as_completed(futs):
            tgt = futs[fut]
            try:
                _, o = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"{tgt}: FAILED ({e})", file=sys.stderr)
                continue
            if o is None:
                print(f"{tgt}: missing single-mod head fold — skipped", file=sys.stderr)
                continue
            res[tgt] = o
            ref = "deployed(b0.6)" if "deployed(b0.6)" in o else CONFIGS[0][0]
            print(f"{tgt}: {ref}={o[ref]:.4f}  "
                  f"best={max(o, key=o.get)} ({max(o.values()):.4f})", flush=True)

    if not res:
        sys.exit("no folds scored")
    names = [n for n, _ in CONFIGS]
    ref = "deployed(b0.6)" if "deployed(b0.6)" in names else names[0]
    ev = [t for t in res if t in EVAL_SET]
    dep = np.mean([res[t][ref] for t in ev])
    print(f"\n=== eval-set means (n={len(ev)}; ref [{ref}] {dep:.4f}) ===")
    print(f"{'config':<22}{'evalARI':>9}{'delta':>9}{'win':>5}{'loss':>5}")
    for n in names:
        m = np.mean([res[t][n] for t in ev])
        wins = sum(1 for t in ev if res[t][n] > res[t][ref] + 0.002)
        losses = sum(1 for t in ev if res[t][n] < res[t][ref] - 0.002)
        print(f"{n:<22}{m:>9.4f}{m - dep:>+9.4f}{wins:>5}{losses:>5}")

    with open(args.out, "w") as fh:
        fh.write("dataset\t" + "\t".join(names) + "\n")
        for t in res:
            fh.write(t + "\t" + "\t".join(f"{res[t][n]:.6f}" for n in names) + "\n")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
