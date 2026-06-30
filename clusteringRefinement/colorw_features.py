#!/usr/bin/env python3
"""Unsupervised per-dataset descriptors for auto color-weight selection (Phase 2).

Everything here is computable in production: no labels, no group count, no
filenames — just the cached embeddings (peg, color) and the shipped learned-head
ensemble projection. Each descriptor is one scalar per dataset; Phase 3 maps
descriptor(s) -> color_w via a calibration fitted LOMO-style on the benchmarks.

Descriptor families:
  knn_cp_k / knn_cg_k   top-k neighbor-set overlap between color sim and
                        proj-ensemble / PE-G sim ("does color structure agree
                        with semantic structure?")
  cmargin_k             z-scored color affinity of each image's semantic (proj)
                        top-k neighbors vs random pairs ("are semantically-near
                        images color-similar?")
  pmargin_k             same with PE-G as the probe of proj neighbors (control)
  paircorr_cp           Spearman corr of (color sim, proj sim) on sampled pairs
  col_std / col_gap     dispersion + local tightness of the color space itself
  col_erank             effective rank of the color embedding (diversity dim.)
  log_n                 control

Usage: python colorw_features.py [--targets ...] [--out colorw_features.tsv]
"""
from __future__ import annotations
import argparse
import sys
import warnings

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
from lomo_common import ALL, load_fold  # noqa: E402
from colorw_oracle import load_ens_sim  # noqa: E402

KS = (10, 30)
N_PAIRS = 100_000
RNG_SEED = 0


def topk_idx(S, k):
    """Row-wise top-k neighbor indices, self excluded."""
    Sx = S.copy()
    np.fill_diagonal(Sx, -np.inf)
    return np.argpartition(-Sx, k - 1, axis=1)[:, :k]


def knn_overlap(Sa, Sb, k):
    ia, ib = topk_idx(Sa, k), topk_idx(Sb, k)
    n = Sa.shape[0]
    hits = 0
    for r in range(n):
        hits += len(np.intersect1d(ia[r], ib[r], assume_unique=False))
    return hits / (n * k)


def sample_pairs(n, rng, m=N_PAIRS):
    i = rng.integers(0, n, m)
    j = rng.integers(0, n, m)
    keep = i != j
    return i[keep], j[keep]


def eff_rank(X, cap=2000, rng=None):
    if X.shape[0] > cap:
        X = X[rng.choice(X.shape[0], cap, replace=False)]
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = (s * s) / (s * s).sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def features_for(tgt):
    peg, col, _, _, _ = load_fold(tgt)
    Sens = load_ens_sim(tgt)
    Speg = peg @ peg.T
    Scol = col @ col.T
    n = Scol.shape[0]
    rng = np.random.default_rng(RNG_SEED)
    pi, pj = sample_pairs(n, rng)
    col_rand = Scol[pi, pj]
    f = {"log_n": float(np.log10(n))}

    for k in KS:
        kk = min(k, n - 1)
        f[f"knn_cp_{k}"] = knn_overlap(Scol, Sens, kk)
        f[f"knn_cg_{k}"] = knn_overlap(Scol, Speg, kk)
        # color affinity of proj top-k neighbors, z-scored against random pairs
        nb = topk_idx(Sens, kk)
        col_nb = np.take_along_axis(Scol, nb, axis=1).mean(axis=1)
        f[f"cmargin_{k}"] = float((col_nb.mean() - col_rand.mean()) / (col_rand.std() + 1e-12))
        peg_rand = Speg[pi, pj]
        peg_nb = np.take_along_axis(Speg, nb, axis=1).mean(axis=1)
        f[f"pmargin_{k}"] = float((peg_nb.mean() - peg_rand.mean()) / (peg_rand.std() + 1e-12))
        # local tightness of color space itself
        cnb = topk_idx(Scol, kk)
        f[f"col_gap_{k}"] = float(np.take_along_axis(Scol, cnb, axis=1).mean() - col_rand.mean())

    f["paircorr_cp"] = float(spearmanr(col_rand, Sens[pi, pj]).statistic)
    f["paircorr_cg"] = float(spearmanr(col_rand, Speg[pi, pj]).statistic)
    f["knn_pg_10"] = knn_overlap(Sens, Speg, min(10, n - 1))
    f["col_std"] = float(col_rand.std())
    f["col_erank"] = eff_rank(col, rng=rng)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--out", default="colorw_features.tsv")
    args = ap.parse_args()
    targets = args.targets or ALL

    rows = []
    for tgt in targets:
        try:
            f = features_for(tgt)
        except FileNotFoundError as e:
            print(f"{tgt}: skipping ({e})", file=sys.stderr)
            continue
        rows.append((tgt, f))
        print(f"{tgt}: " + "  ".join(f"{k}={v:.4f}" for k, v in f.items()), flush=True)

    if not rows:
        sys.exit("no folds scored")
    keys = list(rows[0][1])
    with open(args.out, "w") as fh:
        fh.write("dataset\t" + "\t".join(keys) + "\n")
        for tgt, f in rows:
            fh.write(tgt + "\t" + "\t".join(f"{f[k]:.6f}" for k in keys) + "\n")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
