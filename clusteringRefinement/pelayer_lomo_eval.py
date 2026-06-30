#!/usr/bin/env python3
"""Full-LOMO test of the intermediate PE-G layer signal, reusing the v26 trained
folds (NO retraining). Two integration modes, both scored grouped-only so the
ungrouped images (which the grouped-only extractor leaves zero-filled, an
artifact absent in production where every image would get layer features) never
enter the clustering:

  zsL_w<w> : add the L2-normed layer cosine to the ZERO-SHOT side at weight w,
             keeping the deployed head blend (b=0.60) on top. This is the
             cheap, no-retrain integration — matches last session's
             pe_layers_zs_eval (+0.0095 on M1/M3/M5/M6).

Per fold: restrict to grouped images, Ward @ (#groups), ARI. Paired Δ vs the
deployed base on the IDENTICAL grouped set. Datasets whose fold lacks clean
layer coverage on its grouped images are skipped loudly.
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
from lomo_common import BASE, COLOR_W, EVAL_SET, NAMES, PEG_W, LOMO, l2, load_fold  # noqa: E402

B = 0.60
WS = [0.25, 0.5, 1.0]


def ward_ari(S, true, N):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(np.clip(0.5 * (D + D.T), 0, None), checks=False), "ward")
    return adjusted_rand_score(true, fcluster(Z, t=N, criterion="maxclust"))


def load_layer(tgt, fold_fns, pe_layer):
    lno, pool = pe_layer.split(":")
    cache = f"{BASE}/ClusteringBenchmark{NAMES[tgt]}/.reorder-cache"
    arr = f"{cache}/pe_layers_L{int(lno):02d}_{pool}.npy"
    if not os.path.exists(arr):
        return None
    z = np.load(f"{cache}/embeddings_hash_cache.npz")
    if np.load(arr, mmap_mode="r").shape[0] != len(z["hashes"]):
        return None
    ch = json.load(open(f"{cache}/content_hashes.json"))
    hrow = {h: i for i, h in enumerate(z["hashes"])}
    idx = np.array([hrow[ch[f]] for f in fold_fns])
    return np.load(arr).astype(np.float32)[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--pe-layer", default="47:attnpool")
    ap.add_argument("--ws", nargs="*", type=float, default=WS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for tgt in args.targets or EVAL_SET:
        try:
            peg, col, proj, true, _ = load_fold(tgt)
        except FileNotFoundError as e:
            print(f"{tgt}: no fold, skip ({e})", file=sys.stderr)
            continue
        fns = json.load(open(f"{LOMO}/{tgt}/{tgt}_filenames.json"))
        layer = load_layer(tgt, fns, args.pe_layer)
        if layer is None:
            print(f"{tgt}: no/stale layer, skip", file=sys.stderr)
            continue

        g = true >= 0
        # any grouped image with a zero layer row means incomplete extraction on
        # a labeled image — that IS corruption (not the ungrouped artifact); bail.
        zero_grouped = int((np.abs(layer[g]).sum(axis=1) == 0).sum())
        if zero_grouped:
            print(f"{tgt}: {zero_grouped} grouped imgs missing layer — skip", file=sys.stderr)
            continue

        peg_g, col_g, proj_g, layer_g = peg[g], col[g], proj[g], l2(layer[g])
        true_g = true[g]
        N = len(np.unique(true_g))
        Speg, Scol, Sproj, SL = (peg_g @ peg_g.T, col_g @ col_g.T,
                                 proj_g @ proj_g.T, layer_g @ layer_g.T)
        Gzs = (PEG_W**2 * Speg + COLOR_W**2 * Scol) / (PEG_W**2 + COLOR_W**2)
        res = {"dataset": tgt, "n_grouped": int(g.sum()), "N": N,
               "base": ward_ari((1 - B) * Gzs + B * Sproj, true_g, N)}
        for w in args.ws:
            GzsL = ((PEG_W**2 * Speg + w**2 * SL + COLOR_W**2 * Scol)
                    / (PEG_W**2 + w**2 + COLOR_W**2))
            res[f"zsL_w{w:g}"] = ward_ari((1 - B) * GzsL + B * Sproj, true_g, N)
        rows.append(res)
        deltas = "  ".join(f"w{w:g}:{res[f'zsL_w{w:g}'] - res['base']:+.4f}" for w in args.ws)
        print(f"{tgt}: base={res['base']:.4f}  {deltas}", flush=True)

    if not rows:
        sys.exit("no folds scored")
    cols = [k for k in rows[0] if k not in ("dataset", "n_grouped", "N")]
    print(f"\n=== means over {len(rows)} datasets (grouped-only) ===")
    bm = np.mean([r["base"] for r in rows])
    for k in cols:
        m = np.mean([r[k] for r in rows])
        if k == "base":
            print(f"{k:<12} {m:.4f}")
        else:
            wins = sum(1 for r in rows if r[k] > r["base"])
            print(f"{k:<12} {m:.4f}  Δ={m - bm:+.4f}  wins {wins}/{len(rows)}")
    if args.out:
        hdr = ["dataset", "n_grouped", "N"] + cols
        with open(args.out, "w") as fh:
            fh.write("\t".join(hdr) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[h]) for h in hdr) + "\n")
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
