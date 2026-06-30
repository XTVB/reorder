#!/usr/bin/env python3
"""Per-dataset color-weight oracle curves (Phase 1 of the auto-color-w study).

For every LOMO fold and every color_w on the sweep grid, blends
  S(w) = (1-B) * (peg + w^2 * color) / (1 + w^2)  +  B * S_proj(ens3)
(Ward at oracle N, ARI on labeled images) — the deployed 3-seed-ensemble config,
matching run_pegcolor_sweep.sh's grid but scored in-python like the v26 evals.

Outputs:
  colorw_oracle_ens3.tsv          dataset x color_w ARI matrix + best/deployed cols
  ~/.cache/reorder/colorw_auto/<tgt>_labels.npz   Ward labels per w (for internal-
                                                  criterion selectors; oracle-N cut)

The headline number is the ORACLE CEILING: mean(best-w ARI) - mean(ARI @ 0.7).
If that is within seed noise (~±0.006), per-dataset color-w selection cannot pay
no matter how good the selector — cheap falsification before building one.

Usage: python colorw_oracle.py [--targets M1 ...] [--jobs 6]
"""
from __future__ import annotations
import argparse
import json
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

B = 0.60
CW_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 2.0]
# Override roots (comma-separated) to score against a different head seed, e.g.
# COLORW_ROOTS=~/.cache/reorder/lomo_v26_r4 for the seed-stability diagnostic.
ENS_ROOTS = ([os.path.expanduser(p) for p in os.environ["COLORW_ROOTS"].split(",")]
             if os.environ.get("COLORW_ROOTS")
             else [LOMO] + [os.path.expanduser(f"~/.cache/reorder/lomo_v26_r{r}") for r in (2, 3)])
OUTDIR = os.environ.get("COLORW_OUTDIR", os.path.expanduser("~/.cache/reorder/colorw_auto"))


def load_ens_sim(tgt):
    """Mean proj similarity over the 3 seed roots (deployed ensemble shape)."""
    fns0 = json.load(open(f"{ENS_ROOTS[0]}/{tgt}/{tgt}_filenames.json"))
    S = None
    for root in ENS_ROOTS:
        fns = json.load(open(f"{root}/{tgt}/{tgt}_filenames.json"))
        assert fns == fns0, f"{tgt}: filename order differs in {root}"
        p = l2(np.load(f"{root}/{tgt}/{tgt}_proj.npy").astype(np.float32))
        S = p @ p.T if S is None else S + p @ p.T
    return S / len(ENS_ROOTS)


def score_dataset(tgt):
    peg, col, _, true, N = load_fold(tgt)
    Sens = load_ens_sim(tgt)
    Speg = peg @ peg.T
    Scol = col @ col.T
    del peg, col
    g = true >= 0
    aris, labels_by_w, trees_by_w = {}, {}, {}
    for w in CW_GRID:
        S = (1 - B) * (Speg + w * w * Scol) / (1 + w * w) + B * Sens
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
        D = np.clip(0.5 * (D + D.T), 0.0, None)
        Z = linkage(squareform(D, checks=False), method="ward")
        lab = fcluster(Z, t=N, criterion="maxclust")
        aris[w] = adjusted_rand_score(true[g], lab[g])
        labels_by_w[f"w{w}"] = lab.astype(np.int32)
        trees_by_w[f"Z_w{w}"] = Z  # tiny; lets later evals re-cut at any N
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez_compressed(f"{OUTDIR}/{tgt}_labels.npz", true=true, N=N,
                        **labels_by_w, **trees_by_w)
    return tgt, aris


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="colorw_oracle_ens3.tsv")
    args = ap.parse_args()
    targets = args.targets or ALL

    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(score_dataset, t): t for t in targets}
        for fut in as_completed(futs):
            tgt = futs[fut]
            try:
                _, aris = fut.result()
            except Exception as e:  # noqa: BLE001 — keep scoring the other folds
                print(f"{tgt}: FAILED ({e})", file=sys.stderr)
                continue
            results[tgt] = aris
            best_w = max(aris, key=aris.get)
            print(f"{tgt}: best w={best_w} ({aris[best_w]:.4f})  "
                  f"@0.7={aris[0.7]:.4f}  delta={aris[best_w] - aris[0.7]:+.4f}", flush=True)

    if not results:
        sys.exit("no folds scored")
    rows = [t for t in targets if t in results]
    with open(args.out, "w") as fh:
        fh.write("dataset\t" + "\t".join(f"w{w}" for w in CW_GRID) + "\tbest_w\tbest\tdeployed\n")
        for t in rows:
            a = results[t]
            bw = max(a, key=a.get)
            fh.write(t + "\t" + "\t".join(f"{a[w]:.6f}" for w in CW_GRID)
                     + f"\t{bw}\t{a[bw]:.6f}\t{a[0.7]:.6f}\n")
    print(f"\nSaved: {args.out}")

    for sub, name in ((rows, "full"), ([t for t in rows if t in EVAL_SET], "eval")):
        if not sub:
            continue
        dep = np.mean([results[t][0.7] for t in sub])
        orc = np.mean([max(results[t].values()) for t in sub])
        gbest_w = max(CW_GRID, key=lambda w: np.mean([results[t][w] for t in sub]))
        gbest = np.mean([results[t][gbest_w] for t in sub])
        print(f"{name:<5} n={len(sub)}  @0.7={dep:.4f}  global-best w={gbest_w} ({gbest:.4f})  "
              f"oracle={orc:.4f}  CEILING vs 0.7: {orc - dep:+.4f}")


if __name__ == "__main__":
    main()
