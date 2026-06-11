#!/usr/bin/env python3
"""
(grid color, global color, learned blend) combo sweep — finds whether any blend of
the 3x3-grid color, the global 77-d color (extract_global_color.py sidecars), and
the seed-ensembled learned head beats the deployed combo (grid 0.7, no global,
b=0.60).

Similarity scored:  S = (1−b)·Gzs + b·Gproj_ens
  Gzs       = (peg² + w3²·Sgrid + wg²·Sglob) / (1 + w3² + wg²)
  Gproj_ens = mean over seed-replica roots of proj @ proj.T  (the ens4 estimator
              from seed_ensemble_eval.py — same protocol: Ward @ oracle N, ARI on
              the grouped subset)

Two head arms: ctrl (deployed grid-color-input head, lomo_v26 + _r2.._r4) and
concat (--global-color head, lomo_globcolor + _r2.._r4). Datasets are processed
in parallel; each worker computes its Grams once and re-runs only Ward per config.

  python clusteringRefinement/blend_combo_sweep.py [--workers 4]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import BASE, NAMES, LOMO, ALL, EVAL_SET, l2, load_fold  # noqa: E402

CACHE_ROOT = os.path.expanduser("~/.cache/reorder")
ARMS = {
    "ctrl": [LOMO] + [f"{CACHE_ROOT}/lomo_v26_r{r}" for r in (2, 3, 4)],
    "concat": [f"{CACHE_ROOT}/lomo_globcolor"]
    + [f"{CACHE_ROOT}/lomo_globcolor_r{r}" for r in (2, 3, 4)],
    # Single-head arms (no ensemble).
    "ctrl1": [LOMO],
    "concat1": [f"{CACHE_ROOT}/lomo_globcolor"],
    # 3-seed ensemble = the production deployment shape (learned_head.json
    # ships head_files for seeds 42/43/44).
    "ctrl3": [LOMO] + [f"{CACHE_ROOT}/lomo_v26_r{r}" for r in (2, 3)],
}
W3 = [0.0, 0.35, 0.5, 0.7]
WG = [0.0, 0.35, 0.5]
BV = [0.5, 0.6, 0.7]
# Overridable from the CLI (--arms/--w3/--wg/--bvals). NOTE: workers re-import
# this module under multiprocessing spawn, so CONFIGS must be rebuilt from the
# parsed args inside main() and passed to score_dataset explicitly — and the
# script must be run as a real file, never via `python - <<heredoc` (spawn
# can't re-import a <stdin> main module and crashloops).
CONFIGS = [(arm, w3, wg, b) for arm in ARMS for w3 in W3 for wg in WG for b in BV]
DEPLOYED = ("ctrl", 0.7, 0.0, 0.6)


def cfg_name(c):
    arm, w3, wg, b = c
    return f"{arm}_x{w3:g}_g{wg:g}_b{b:g}"


def load_global(tgt):
    fns = json.load(open(f"{LOMO}/{tgt}/{tgt}_filenames.json"))
    cache = f"{BASE}/ClusteringBenchmark{NAMES[tgt]}/.reorder-cache"
    ch = json.load(open(f"{cache}/content_hashes.json"))
    sc = np.load(f"{cache}/global_color_cache.npz", allow_pickle=False)
    h2r = {h: i for i, h in enumerate(sc["hashes"].tolist())}
    idx = np.array([h2r[ch[f]] for f in fns])
    return l2(sc["color_global"][idx].astype(np.float32))


def ward_ari(S, true, N):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = np.clip(0.5 * (D + D.T), 0.0, None)
    Z = linkage(squareform(D, checks=False), method="ward")
    g = true >= 0
    return adjusted_rand_score(true[g], fcluster(Z, t=N, criterion="maxclust")[g])


def score_dataset(job):
    # job = (tgt, configs): configs ride along in the map payload because spawn
    # workers re-import this module and would otherwise see the default CONFIGS.
    tgt, configs = job
    t0 = time.time()
    peg, col, _, true, N = load_fold(tgt)
    glo = load_global(tgt)
    Speg, Sgrid, Sglob = peg @ peg.T, col @ col.T, glo @ glo.T

    fns0 = json.load(open(f"{LOMO}/{tgt}/{tgt}_filenames.json"))
    proj_ens = {}
    for arm in {c[0] for c in configs}:
        sims = []
        for root in ARMS[arm]:
            fns = json.load(open(f"{root}/{tgt}/{tgt}_filenames.json"))
            assert fns == fns0, f"{tgt}: filename order differs in {root}"
            p = l2(np.load(f"{root}/{tgt}/{tgt}_proj.npy").astype(np.float32))
            sims.append(p @ p.T)
        proj_ens[arm] = np.mean(sims, axis=0)

    res = {}
    for c in configs:
        arm, w3, wg, b = c
        Gzs = (Speg + w3 * w3 * Sgrid + wg * wg * Sglob) / (1 + w3 * w3 + wg * wg)
        res[cfg_name(c)] = ward_ari((1 - b) * Gzs + b * proj_ens[arm], true, N)
    print(f"{tgt}: done ({len(true)} imgs, {time.time()-t0:.0f}s)", flush=True)
    return tgt, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--w3", nargs="*", type=float, default=W3)
    ap.add_argument("--wg", nargs="*", type=float, default=WG)
    ap.add_argument("--bvals", nargs="*", type=float, default=BV)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "blend_combo_sweep.tsv"))
    args = ap.parse_args()
    targets = args.targets or ALL
    configs = [(arm, w3, wg, b) for arm in args.arms for w3 in args.w3
               for wg in args.wg for b in args.bvals]
    if DEPLOYED not in configs and "ctrl" in args.arms:
        configs.insert(0, DEPLOYED)   # always anchor the Δ column

    print(f"(grid, global, blend) combo sweep — {len(configs)} configs "
          f"({len(args.arms)} arms x {len(args.w3)}x{len(args.wg)}x{len(args.bvals)}), "
          f"{len(targets)} datasets")
    with Pool(args.workers) as pool:
        rows = dict(pool.map(score_dataset, [(t, configs) for t in targets]))

    def mean(name, subset):
        return float(np.mean([rows[t][name] for t in subset if t in rows]))

    ev = [t for t in targets if t in EVAL_SET and t in rows]
    anchor = DEPLOYED if DEPLOYED in configs else configs[0]
    base_full = mean(cfg_name(anchor), rows)
    base_ev = mean(cfg_name(anchor), ev)
    print(f"\n[anchor] {cfg_name(anchor)}: full={base_full:.4f}  eval={base_ev:.4f}")

    print(f"\n================ ALL CONFIGS — mean ARI ({len(rows)} full / {len(ev)} eval) ================")
    print(f"{'config':<24}{'full':>9}{'Δ':>9}   {'eval':>9}{'Δ':>9}  wins(full)")
    ranked = sorted(configs, key=lambda c: -mean(cfg_name(c), rows))
    for c in ranked:
        n = cfg_name(c)
        wins = sum(1 for t in rows if rows[t][n] > rows[t][cfg_name(anchor)])
        print(f"{n:<24}{mean(n, rows):>9.4f}{mean(n, rows)-base_full:>+9.4f}   "
              f"{mean(n, ev):>9.4f}{mean(n, ev)-base_ev:>+9.4f}  {wins}/{len(rows)}")

    best = ranked[0]
    print(f"\n================ PER-DATASET — anchor vs best ({cfg_name(best)}) ================")
    print(f"{'tgt':<6}{'anchor':>10}{'best':>10}{'Δ':>9}")
    for t in targets:
        if t not in rows:
            continue
        d0, b0 = rows[t][cfg_name(anchor)], rows[t][cfg_name(best)]
        print(f"{t:<6}{d0:>10.3f}{b0:>10.3f}{b0-d0:>+9.3f}")

    with open(args.out, "w") as f:
        names = [cfg_name(c) for c in configs]
        f.write("dataset\t" + "\t".join(names) + "\n")
        for t in targets:
            if t in rows:
                f.write(f"{t}\t" + "\t".join(f"{rows[t][n]:.4f}" for n in names) + "\n")
    print(f"\nTSV: {args.out}")


if __name__ == "__main__":
    main()
