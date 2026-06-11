#!/usr/bin/env python3
"""Score the paired pe-layer mini-LOMO (run_pelayer_minilomo.sh).

Per target (M1/M3/M5/M6) and seed: ward@N ARI at blend 0.60 for
  ctrl       : control head (peg+color input)
  ctrl+zsL   : control head, with L47-attnpool added to the ZERO-SHOT side (w=0.5)
  layer      : layer-augmented head (peg+color+L47-attnpool input)
  layer+zsL  : both
Reports seed-means and the paired Δ vs ctrl.
"""
from __future__ import annotations
import argparse
import json
import os
import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import BASE, NAMES, PEG_W, COLOR_W, l2, load_fold  # noqa: E402

B = 0.60
ZSL_W = 0.5
TARGETS = ["M1", "M3", "M5", "M6"]


def ward_ari(S, true, N):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(np.clip(0.5 * (D + D.T), 0, None), checks=False), "ward")
    g = true >= 0
    return adjusted_rand_score(true[g], fcluster(Z, t=N, criterion="maxclust")[g])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/.cache/reorder/pelayer_minilomo"))
    ap.add_argument("--seeds", nargs="*", default=["42", "43"])
    ap.add_argument("--pe-layer", default="47:attnpool")
    args = ap.parse_args()
    lno, pool = args.pe_layer.split(":")

    table = {}
    for tgt in TARGETS:
        peg, col, _, true, N = load_fold(tgt)
        cache = f"{BASE}/ClusteringBenchmark{NAMES[tgt]}/.reorder-cache"
        z = np.load(f"{cache}/embeddings_hash_cache.npz")
        ch = json.load(open(f"{cache}/content_hashes.json"))
        fns = json.load(open(f"{args.root}/ctrl/{tgt}/s{args.seeds[0]}/{tgt}_filenames.json"))
        hrow = {h: i for i, h in enumerate(z["hashes"])}
        idx = np.array([hrow[ch[f]] for f in fns])
        X = l2(np.load(f"{cache}/pe_layers_L{int(lno):02d}_{pool}.npy").astype(np.float32)[idx])
        Speg, Scol, SL = peg @ peg.T, col @ col.T, X @ X.T
        Gzs = (PEG_W**2 * Speg + COLOR_W**2 * Scol) / (PEG_W**2 + COLOR_W**2)
        GzsL = ((PEG_W**2 * Speg + ZSL_W**2 * SL + COLOR_W**2 * Scol)
                / (PEG_W**2 + ZSL_W**2 + COLOR_W**2))

        res = {}
        for arm in ("ctrl", "layer"):
            for seed in args.seeds:
                p = f"{args.root}/{arm}/{tgt}/s{seed}/{tgt}_proj.npy"
                if not os.path.exists(p):
                    continue
                pr = l2(np.load(p).astype(np.float32))
                Sp = pr @ pr.T
                res.setdefault(arm, []).append(ward_ari((1 - B) * Gzs + B * Sp, true, N))
                res.setdefault(arm + "+zsL", []).append(
                    ward_ari((1 - B) * GzsL + B * Sp, true, N))
        table[tgt] = {k: float(np.mean(v)) for k, v in res.items()}
        print(f"{tgt}: " + "  ".join(f"{k}={v:.4f}" for k, v in table[tgt].items()), flush=True)

    arms = ["ctrl", "ctrl+zsL", "layer", "layer+zsL"]
    print(f"\n{'arm':<11}{'mean':>9}{'Δ vs ctrl':>11}  wins")
    base = np.mean([table[t]["ctrl"] for t in TARGETS])
    for a in arms:
        if not all(a in table[t] for t in TARGETS):
            continue
        m = np.mean([table[t][a] for t in TARGETS])
        wins = sum(1 for t in TARGETS if table[t][a] > table[t]["ctrl"])
        print(f"{a:<11}{m:>9.4f}{m - base:>+11.4f}  {wins}/{len(TARGETS)}")


if __name__ == "__main__":
    main()
