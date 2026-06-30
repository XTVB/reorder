#!/usr/bin/env python3
"""LOMO evaluation of unsupervised color-weight selectors (Phase 3).

Consumes the oracle ARI matrix (colorw_oracle.py) + the unsupervised descriptors
(colorw_features.py). For each held-out dataset, a selector may look only at:
  - the held-out dataset's own descriptors (computable in production), and
  - the other datasets' (descriptors, ARI curves) as calibration data
    (this is what we'd ship: a small calibration table next to the head).
It picks a grid w; we score by lookup into the held-out oracle row.

Selectors:
  fixed@W      pick W always (0.7 = deployed baseline)
  oracle       per-dataset argmax (ceiling)
  knn<K>[feat] kernel selector: z-score the feature(s) over training sets, take
               the K nearest training datasets, average their gain curves
               dARI(w) = ARI(w)-ARI(0.7), pick argmax w; fall back to 0.7 unless
               predicted gain > --min-gain.
  lin[feat]    linear regression feat -> soft-best-w (softmax-weighted centroid
               of the training curves), snap to grid.

Reports eval-set mean ARI, delta vs deployed, wins/losses (|delta|>0.002), and
worst single-dataset regression — the deployability criteria.

Usage:
  python colorw_select.py                       # all single features, both selectors
  python colorw_select.py --features cmargin_10 knn_cp_10   # feature pairs too
"""
from __future__ import annotations
import argparse
import itertools

import numpy as np

from lomo_common import EVAL_SET

GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 2.0]
DEPLOYED = 0.7


def read_tsv(path):
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        rows = {}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            rows[parts[0]] = {h: v for h, v in zip(header[1:], parts[1:])}
    return rows


def load_data(oracle_path, feat_path):
    o = read_tsv(oracle_path)
    f = read_tsv(feat_path)
    ds = [d for d in o if d in f]
    curves = {d: np.array([float(o[d][f"w{w}"]) for w in GRID]) for d in ds}
    fkeys = list(next(iter(f.values())))
    feats = {d: {k: float(f[d][k]) for k in fkeys} for d in ds}
    return ds, curves, feats, fkeys


def soft_best_w(curve, tau=0.005):
    z = (curve - curve.max()) / tau
    p = np.exp(z)
    p /= p.sum()
    return float(p @ np.array(GRID))


def knn_select(d, train, curves, feats, fnames, k, min_gain):
    X = np.array([[feats[t][f] for f in fnames] for t in train])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xz = (X - mu) / sd
    xq = (np.array([feats[d][f] for f in fnames]) - mu) / sd
    dist = np.linalg.norm(Xz - xq, axis=1)
    nn = np.argsort(dist)[:k]
    wts = 1.0 / (dist[nn] + 1e-6)
    i07 = GRID.index(DEPLOYED)
    gain = np.zeros(len(GRID))
    for t_i, w_i in zip(nn, wts):
        c = curves[train[t_i]]
        gain += w_i * (c - c[i07])
    gain /= wts.sum()
    j = int(gain.argmax())
    return GRID[j] if gain[j] > min_gain else DEPLOYED


def lin_select(d, train, curves, feats, fnames):
    X = np.array([[feats[t][f] for f in fnames] for t in train])
    y = np.array([soft_best_w(curves[t]) for t in train])
    A = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = float(np.array([feats[d][f] for f in fnames] + [1.0]) @ coef)
    pred = min(max(pred, GRID[0]), GRID[-1])
    return min(GRID, key=lambda w: abs(w - pred))


def evaluate(selector_fn, ds, curves, eval_set):
    """LOMO: for each dataset, calibrate on the others, pick w, look up ARI."""
    i07 = GRID.index(DEPLOYED)
    picks, aris, deltas = {}, {}, {}
    for d in ds:
        train = [t for t in ds if t != d]
        w = selector_fn(d, train)
        picks[d] = w
        aris[d] = curves[d][GRID.index(w)]
        deltas[d] = aris[d] - curves[d][i07]
    ev = [d for d in ds if d in eval_set]
    mean_ari = np.mean([aris[d] for d in ev])
    mean_d = np.mean([deltas[d] for d in ev])
    wins = sum(1 for d in ev if deltas[d] > 0.002)
    losses = sum(1 for d in ev if deltas[d] < -0.002)
    worst = min(ev, key=lambda d: deltas[d])
    return dict(mean_ari=mean_ari, mean_delta=mean_d, wins=wins, losses=losses,
                worst=(worst, deltas[worst]), picks=picks, deltas=deltas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="colorw_oracle_ens3.tsv")
    ap.add_argument("--feat-file", default="colorw_features.tsv")
    ap.add_argument("--features", nargs="*", default=None,
                    help="evaluate exactly this feature set (else: every single feature)")
    ap.add_argument("--pairs", action="store_true", help="also try all feature pairs")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--min-gain", type=float, default=0.002)
    ap.add_argument("--exclude-m7-train", action="store_true")
    ap.add_argument("--detail", nargs="*", default=[], help="print per-dataset picks for these configs")
    args = ap.parse_args()

    ds, curves, feats, fkeys = load_data(args.oracle, args.feat_file)
    if args.exclude_m7_train:
        base_ds = [d for d in ds if d != "M7"]
    else:
        base_ds = ds
    ev = [d for d in ds if d in EVAL_SET]
    i07 = GRID.index(DEPLOYED)
    base = np.mean([curves[d][i07] for d in ev])
    orc = np.mean([curves[d].max() for d in ev])
    print(f"eval n={len(ev)}  fixed@0.7={base:.4f}  oracle={orc:.4f}  ceiling={orc - base:+.4f}\n")

    configs = []
    if args.features:
        fsets = [tuple(args.features)]
    else:
        fsets = [(f,) for f in fkeys]
        if args.pairs:
            fsets += list(itertools.combinations(fkeys, 2))
    for fs in fsets:
        configs.append((f"knn{args.k}[{','.join(fs)}]",
                        lambda d, tr, fs=fs: knn_select(d, tr, curves, feats, list(fs),
                                                        args.k, args.min_gain)))
        configs.append((f"lin[{','.join(fs)}]",
                        lambda d, tr, fs=fs: lin_select(d, tr, curves, feats, list(fs))))

    print(f"{'selector':<42}{'evalARI':>9}{'delta':>9}{'win':>5}{'loss':>5}  worst")
    results = []
    for name, fn in configs:
        r = evaluate(lambda d, tr, fn=fn: fn(d, [t for t in tr if t in base_ds] or tr),
                     ds, curves, EVAL_SET)
        results.append((name, r))
        w_d, w_v = r["worst"]
        print(f"{name:<42}{r['mean_ari']:>9.4f}{r['mean_delta']:>+9.4f}"
              f"{r['wins']:>5}{r['losses']:>5}  {w_d} {w_v:+.4f}")
        if name in args.detail:
            for d in ds:
                print(f"    {d}: pick={r['picks'][d]}  d={r['deltas'][d]:+.4f}")

    results.sort(key=lambda x: -x[1]["mean_ari"])
    print("\ntop 5:")
    for name, r in results[:5]:
        print(f"  {name}: {r['mean_ari']:.4f} ({r['mean_delta']:+.4f}, "
              f"{r['wins']}W/{r['losses']}L)")


if __name__ == "__main__":
    main()
