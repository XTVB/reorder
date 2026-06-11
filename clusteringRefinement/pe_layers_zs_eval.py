#!/usr/bin/env python3
"""Score cached intermediate PE-G layer features (pe_layers_L*_<pool>.npy) as a
zero-shot blend component: peg ⊕ w·layer ⊕ 0.7·color vs the peg ⊕ 0.7·color
baseline, Ward @ oracle N.

Only datasets whose pe_layers row count exactly matches the current
embeddings_hash_cache.npz hash list are scored (stale extractions — dataset
changed since — are skipped loudly, not NaN-patched). Zero rows are reported.
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
from lomo_common import BASE, NAMES, PEG_W, COLOR_W, LOMO, l2, load_fold  # noqa: E402

LAYERS = [42, 44, 46, 47]
POOLS = ["mean", "gem3", "attnpool"]
WS = [0.25, 0.5, 1.0]


def ward_ari(S, true, N):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(np.clip(0.5 * (D + D.T), 0, None), checks=False), "ward")
    g = true >= 0
    return adjusted_rand_score(true[g], fcluster(Z, t=N, criterion="maxclust")[g])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["M1", "M3", "M5", "M6"])
    args = ap.parse_args()

    results = {}
    for tgt in args.targets:
        cache = f"{BASE}/ClusteringBenchmark{NAMES[tgt]}/.reorder-cache"
        peg, col, _, true, N = load_fold(tgt)
        z = np.load(f"{cache}/embeddings_hash_cache.npz")
        n_hashes = len(z["hashes"])
        ch = json.load(open(f"{cache}/content_hashes.json"))
        fns = json.load(open(f"{LOMO}/{tgt}/{tgt}_filenames.json"))
        hrow = {h: i for i, h in enumerate(z["hashes"])}
        idx = np.array([hrow[ch[f]] for f in fns])

        Scol = col @ col.T
        Speg = peg @ peg.T
        Gzs = (PEG_W**2 * Speg + COLOR_W**2 * Scol) / (PEG_W**2 + COLOR_W**2)
        res = {"base": ward_ari(Gzs, true, N)}
        for L in LAYERS:
            for P in POOLS:
                path = f"{cache}/pe_layers_L{L:02d}_{P}.npy"
                if not os.path.exists(path):
                    continue
                X = np.load(path)
                if X.shape[0] != n_hashes:
                    print(f"{tgt}: SKIP L{L}_{P} — stale ({X.shape[0]} rows vs "
                          f"{n_hashes} hashes)", file=sys.stderr)
                    continue
                X = l2(X.astype(np.float32)[idx])
                nz = int((np.abs(X).sum(axis=1) == 0).sum())
                if nz > 0.005 * len(X):
                    print(f"{tgt}: SKIP L{L}_{P} — {nz} zero rows", file=sys.stderr)
                    continue
                SL = X @ X.T
                res[f"L{L}_{P}_solo"] = ward_ari(
                    (PEG_W**2 * SL + COLOR_W**2 * Scol) / (PEG_W**2 + COLOR_W**2), true, N)
                for w in WS:
                    S = ((PEG_W**2 * Speg + w**2 * SL + COLOR_W**2 * Scol)
                         / (PEG_W**2 + w**2 + COLOR_W**2))
                    res[f"L{L}_{P}_w{w:g}"] = ward_ari(S, true, N)
        results[tgt] = res
        print(f"{tgt} done ({len(res) - 1} configs)", flush=True)

    targets = list(results)
    bases = [results[t]["base"] for t in targets]
    print(f"\nzero-shot base (peg+0.7col): mean={np.mean(bases):.4f}  "
          + "  ".join(f"{t}={b:.3f}" for t, b in zip(targets, bases)))
    keys = sorted({k for r in results.values() for k in r if k != "base"})
    print(f"{'config':<22}{'mean':>8}{'Δ':>9}  wins")
    for k in keys:
        if not all(k in results[t] for t in targets):
            continue
        v = [results[t][k] for t in targets]
        wins = sum(1 for t in targets if results[t][k] > results[t]["base"])
        print(f"{k:<22}{np.mean(v):>8.4f}{np.mean(v) - np.mean(bases):>+9.4f}  {wins}/{len(targets)}")


if __name__ == "__main__":
    main()
