#!/usr/bin/env python3
"""
Global vs 3x3-grid color histograms — zero-shot sweep over EVAL_SET (excl. M7/M14/M15).

The 3x3 spatial color grid replaced global histograms on the hypothesis that
position information helps; that switch was never benchmarked. This scores both
variants as the color modality in the deployed zero-shot blend
(peg ⊕ w·color → unit-norm → cosine → Ward @ oracle N, ARI on grouped subset),
sweeping the color weight so neither variant is handicapped by the 0.7 chosen
for the other. Also scores a handful of grid+global combos in case they're
complementary. Deterministic, no training.

Requires the global_color_cache.npz sidecars (extract_global_color.py).

  python clusteringRefinement/global_color_eval.py
"""
from __future__ import annotations
import json
import os
import time
import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import BASE, NAMES, LOMO, EVAL_SET, PEG_W, COLOR_W, l2, load_fold  # noqa: E402

WS = [0.35, 0.5, 0.7, 1.0]

CONFIGS = [("peg_only", {})]
CONFIGS += [(f"grid3x3_w{w:g}", {"grid": w}) for w in WS]
CONFIGS += [(f"global_w{w:g}", {"glob": w}) for w in WS]
CONFIGS += [(f"both_g{wg:g}_x{w3:g}", {"glob": wg, "grid": w3})
            for wg in [0.35, 0.5] for w3 in [0.35, 0.5, 0.7]]
BASELINE = f"grid3x3_w{COLOR_W:g}"  # the deployed blend


def load_global(tgt):
    """color_global in the same fold order as load_fold (L2-normed)."""
    fns = json.load(open(f"{LOMO}/{tgt}/{tgt}_filenames.json"))
    cache = f"{BASE}/ClusteringBenchmark{NAMES[tgt]}/.reorder-cache"
    ch = json.load(open(f"{cache}/content_hashes.json"))
    sc = np.load(f"{cache}/global_color_cache.npz", allow_pickle=False)
    h2r = {h: i for i, h in enumerate(sc["hashes"].tolist())}
    idx = np.array([h2r[ch[f]] for f in fns])
    return l2(sc["color_global"][idx].astype(np.float32))


def ward_ari(S, true, N):
    d = 1.0 - S
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(np.clip(0.5 * (d + d.T), 0.0, None), checks=False), "ward")
    g = true >= 0
    return adjusted_rand_score(true[g], fcluster(Z, t=N, criterion="maxclust")[g])


def main():
    rows = {}  # rows[tgt][cfgname]
    print(f"Global vs 3x3 color — zero-shot, Ward@oracle-N, ARI on grouped subset")
    print(f"{'tgt':<5}{'n':>6}{'N':>5}   ({len(CONFIGS)} configs)")
    for tgt in EVAL_SET:
        t0 = time.time()
        peg, col, _, true, N = load_fold(tgt)
        glo = load_global(tgt)
        grams = {"grid": col @ col.T, "glob": glo @ glo.T}
        Speg = peg @ peg.T
        rows[tgt] = {}
        for name, w in CONFIGS:
            num = PEG_W**2 * Speg + sum(v * v * grams[m] for m, v in w.items())
            den = PEG_W**2 + sum(v * v for v in w.values())
            rows[tgt][name] = ward_ari(num / den, true, N)
        print(f"{tgt:<5}{len(true):>6}{N:>5}   ({time.time()-t0:.0f}s)", flush=True)

    def mean(name):
        return float(np.mean([rows[t][name] for t in EVAL_SET]))

    base = mean(BASELINE)
    print(f"\n[self-check] {BASELINE} (deployed zero-shot blend) {len(EVAL_SET)}-set mean = {base:.4f}")

    print(f"\n================ AVERAGE OVERVIEW — {len(EVAL_SET)}-set mean ARI (zero-shot) ================")
    print(f"{'config':<18}{'mean':>9}{'Δ vs deployed':>15}  wins")
    for name, _ in CONFIGS:
        m = mean(name)
        wins = sum(1 for t in EVAL_SET if rows[t][name] > rows[t][BASELINE])
        print(f"{name:<18}{m:>9.4f}{m - base:>+15.4f}  {wins}/{len(EVAL_SET)}")

    best_grid = max((n for n, _ in CONFIGS if n.startswith("grid")), key=mean)
    best_glob = max((n for n, _ in CONFIGS if n.startswith("global")), key=mean)
    print(f"\n================ PER-DATASET  (best grid={best_grid}, best global={best_glob}) ================")
    print(f"{'tgt':<6}{'peg_only':>10}{best_grid:>16}{best_glob:>16}{'Δ(glob-grid)':>14}")
    for t in EVAL_SET:
        g3, gl = rows[t][best_grid], rows[t][best_glob]
        print(f"{t:<6}{rows[t]['peg_only']:>10.3f}{g3:>16.3f}{gl:>16.3f}{gl - g3:>+14.3f}")
    print("-" * 62)
    print(f"{'MEAN':<6}{mean('peg_only'):>10.4f}{mean(best_grid):>16.4f}{mean(best_glob):>16.4f}"
          f"{mean(best_glob) - mean(best_grid):>+14.4f}")

    out = os.path.join(os.path.dirname(__file__), "global_color_eval.tsv")
    with open(out, "w") as f:
        f.write("dataset\t" + "\t".join(n for n, _ in CONFIGS) + "\n")
        for t in EVAL_SET:
            f.write(f"{t}\t" + "\t".join(f"{rows[t][n]:.4f}" for n, _ in CONFIGS) + "\n")
    print(f"\nTSV: {out}")


if __name__ == "__main__":
    main()
