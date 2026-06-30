#!/usr/bin/env python3
"""Grouped-only paired scoring of the PE-layer head-input LOMO
(run_pelayer_headinput_lomo.sh). Self-contained: everything is read in each
arm's own filename order (both arms share it), so peg/color/proj/labels align
without depending on any external fold order.

Per target: ctrl vs layer head proj, blended with the SAME zero-shot peg+color
at b=0.60, Ward @ #groups on the labeled images. Paired Δ (layer − ctrl)."""
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
from lomo_common import BASE, COLOR_W, NAMES, PEG_W, l2  # noqa: E402

B = 0.60


def ward_ari(S, true, N):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(np.clip(0.5 * (D + D.T), 0, None), checks=False), "ward")
    return adjusted_rand_score(true, fcluster(Z, t=N, criterion="maxclust"))


def load_zs_and_labels(tgt, fns):
    """peg, col (L2-normed, in fns order) + int group label per fns row (-1 ungrouped)."""
    d = f"{BASE}/ClusteringBenchmark{NAMES[tgt]}"
    cache = f"{d}/.reorder-cache"
    z = np.load(f"{cache}/embeddings_hash_cache.npz")
    ch = json.load(open(f"{cache}/content_hashes.json"))
    hrow = {h: i for i, h in enumerate(z["hashes"])}
    idx = np.array([hrow[ch[f]] for f in fns])
    peg = l2(z["pecore_g"][idx].astype(np.float32))
    col = l2(z["color"][idx].astype(np.float32))
    graw = json.load(open(f"{d}/.reorder-groups.json"))
    glist = graw if isinstance(graw, list) else graw.get("groups", [])
    fn2lab = {fn: gi for gi, g in enumerate(glist) for fn in g["images"]}
    true = np.array([fn2lab.get(f, -1) for f in fns])
    return peg, col, true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/.cache/reorder/pelayer_hi_lomo"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for tgt in sorted(os.listdir(f"{args.root}/ctrl"), key=lambda s: int(s[1:])):
        cdir, ldir = f"{args.root}/ctrl/{tgt}", f"{args.root}/layer/{tgt}"
        if not (os.path.exists(f"{cdir}/{tgt}_proj.npy") and os.path.exists(f"{ldir}/{tgt}_proj.npy")):
            print(f"{tgt}: missing arm, skip", file=sys.stderr)
            continue
        fns_c = json.load(open(f"{cdir}/{tgt}_filenames.json"))
        fns_l = json.load(open(f"{ldir}/{tgt}_filenames.json"))
        if fns_c != fns_l:
            print(f"{tgt}: arm filename order differs, skip", file=sys.stderr)
            continue
        peg, col, true = load_zs_and_labels(tgt, fns_c)
        ctrl = l2(np.load(f"{cdir}/{tgt}_proj.npy").astype(np.float32))
        layer = l2(np.load(f"{ldir}/{tgt}_proj.npy").astype(np.float32))

        g = true >= 0
        peg_g, col_g, true_g = peg[g], col[g], true[g]
        N = len(np.unique(true_g))
        Gzs = (PEG_W**2 * (peg_g @ peg_g.T) + COLOR_W**2 * (col_g @ col_g.T)) / (PEG_W**2 + COLOR_W**2)
        cg, lg = ctrl[g], layer[g]
        res = {"dataset": tgt,
               "ctrl": ward_ari((1 - B) * Gzs + B * (cg @ cg.T), true_g, N),
               "layer": ward_ari((1 - B) * Gzs + B * (lg @ lg.T), true_g, N)}
        res["delta"] = res["layer"] - res["ctrl"]
        rows.append(res)
        print(f"{tgt}: ctrl={res['ctrl']:.4f} layer={res['layer']:.4f} Δ={res['delta']:+.4f}", flush=True)

    if not rows:
        sys.exit("no folds scored")
    d = np.array([r["delta"] for r in rows])
    print(f"\n=== {len(rows)} datasets, head-input layer vs ctrl (grouped-only, paired) ===")
    print(f"ctrl mean={np.mean([r['ctrl'] for r in rows]):.4f}  "
          f"layer mean={np.mean([r['layer'] for r in rows]):.4f}")
    print(f"mean Δ={d.mean():+.4f}  median Δ={np.median(d):+.4f}  "
          f"wins={int((d > 1e-9).sum())}/{len(rows)}  (±0.0062 = 95% LOMO noise band)")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write("dataset\tctrl\tlayer\tdelta\n")
            for r in rows:
                fh.write(f"{r['dataset']}\t{r['ctrl']:.6f}\t{r['layer']:.6f}\t{r['delta']:.6f}\n")
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
