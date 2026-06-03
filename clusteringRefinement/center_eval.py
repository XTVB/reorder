#!/usr/bin/env python3
"""
Transductive mean-centering eval across the 20-dataset LOMO (idea #1 in LEARNED_HEAD.md).

Subtract the per-shoot mean embedding from each modality BEFORE the cosine blend.
The mean is computed over ALL images in the fold (grouped + ungrouped) — fully
unsupervised, the production analog (no labels, every image participates). It
strips the shoot-wide shared component (same person / lighting / aesthetic) that
pulls every pairwise cosine toward 1 and compresses the between-group gap Ward
cuts on.

Deterministic, NO training: re-scores the cached deployed folds in
/tmp/lomo_postaug (the same post-pixel-aug projections behind the 0.7717 row).
Reuses the deployed scoring path exactly — lomo_common.load_fold + the production
rescaleLearnedProjWeight blend + Ward at oracle-N, ARI on the grouped subset
(same protocol as benchmark_clustering.ts / rerank_eval.py).

Centering is center-then-renorm: each modality is already unit-normed by
load_fold, so we subtract its per-fold mean and re-normalize to unit — this keeps
the deployed per-modality weights (PE-G 1.0, color 0.7, learned lw) meaningful.

Variants (which modalities get transformed):
  none : deployed baseline (no centering)            <- self-check ≈ known 0.769 @ b=0.60
  peg  : PE-G only
  zs   : PE-G + color (zero-shot only; head untouched)   <- doc's recommended first test
  all  : PE-G + color + learned proj

Scored at two blends: b=0.00 (zero-shot base, no head) and b=0.60 (deployed head).

  python clusteringRefinement/center_eval.py              # pure centering
  python clusteringRefinement/center_eval.py --whiten     # shrinkage-PCA whitening instead
"""
from __future__ import annotations
import argparse, os, time, warnings
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import ALL, COLOR_W, OUTLIERS, PEG_W, EVAL_SET, l2, load_fold  # noqa: E402

VARIANTS = {"none": set(), "peg": {"peg"}, "zs": {"peg", "col"}, "all": {"peg", "col", "proj"}}
BVALS = [0.0, 0.60]


def center(x):
    """Transductive mean-centering, then re-normalize to unit (retrieval-standard)."""
    return l2(x - x.mean(axis=0, keepdims=True))


def whiten(x, shrink):
    """Center + shrinkage ZCA-whitening. Σ'=(1−a)Σ+a·(trΣ/d)I keeps it rank-safe for N<d."""
    xc = x - x.mean(axis=0, keepdims=True)
    n, d = xc.shape
    cov = (xc.T @ xc) / max(n - 1, 1)
    cov = (1 - shrink) * cov + shrink * (np.trace(cov) / d) * np.eye(d, dtype=cov.dtype)
    evals, evecs = np.linalg.eigh(cov)
    W = (evecs / np.sqrt(np.clip(evals, 1e-12, None))) @ evecs.T   # Σ^(−1/2)
    return l2(xc @ W)


def build_feat(peg, col, proj, b):
    """L2-normalized weighted concat; b = learned signal fraction (production rescale)."""
    parts = [PEG_W * peg, COLOR_W * col]
    if b > 0:
        S = PEG_W**2 + COLOR_W**2
        parts.append(np.sqrt(b * S / (1 - b)) * proj)
    return l2(np.concatenate(parts, axis=1))


def clust_ari(feat, N, true, grouped):
    d = 1.0 - (feat @ feat.T)
    np.fill_diagonal(d, 0.0)
    d = np.clip(0.5 * (d + d.T), 0.0, None)
    Z = linkage(squareform(d, checks=False), method="ward")
    pred = fcluster(Z, t=N, criterion="maxclust")
    return adjusted_rand_score(true[grouped], pred[grouped])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whiten", action="store_true", help="shrinkage-PCA whitening instead of plain centering")
    ap.add_argument("--shrink", type=float, default=0.1, help="whitening shrinkage α (only with --whiten)")
    ap.add_argument("--bsweep", action="store_true", help="sweep the learned-blend fraction (is there a b where centering wins?)")
    args = ap.parse_args()
    global BVALS
    if args.bsweep:
        BVALS = [0.0, 0.30, 0.45, 0.60, 0.75, 0.90]
    tf = (lambda v: whiten(v, args.shrink)) if args.whiten else center
    label = f"whiten(α={args.shrink})" if args.whiten else "center"

    # rows[tgt][b][variant] = ARI
    rows = {}
    print(f"Transductive {label} eval — Ward @ oracle-N, ARI on grouped subset")
    print(f"{'tgt':<5}{'n':>6}{'N':>5}   " + "  ".join(f"{v:>6}" for v in VARIANTS) + "   (b=0.60)")
    for tgt in ALL:
        t0 = time.time()
        peg, col, proj, true, N = load_fold(tgt)
        grouped = true >= 0
        # Pre-transform each modality once (reused across b-values).
        tp = {"peg": tf(peg), "col": tf(col), "proj": tf(proj)}
        rows[tgt] = {}
        for b in BVALS:
            rows[tgt][b] = {}
            for var, mods in VARIANTS.items():
                pe = tp["peg"] if "peg" in mods else peg
                co = tp["col"] if "col" in mods else col
                pr = tp["proj"] if "proj" in mods else proj
                rows[tgt][b][var] = clust_ari(build_feat(pe, co, pr, b), N, true, grouped)
        r60 = rows[tgt][0.60]
        print(f"{tgt:<5}{len(true):>6}{N:>5}   "
              + "  ".join(f"{r60[v]:>6.3f}" for v in VARIANTS)
              + f"   ({time.time()-t0:.0f}s)", flush=True)

    def mean(b, var, keys): return float(np.mean([rows[t][b][var] for t in keys]))

    for b in BVALS:
        tag = "ZERO-SHOT base (no head)" if b == 0 else "DEPLOYED head (b=0.60)"
        print(f"\n================ {tag} — mean ARI ({label}) ================")
        print(f"{'variant':<8}{'full-20':>10}{'Δ vs none':>11}   {'17-set':>10}{'Δ vs none':>11}")
        base20, base17 = mean(b, "none", ALL), mean(b, "none", EVAL_SET)
        for var in VARIANTS:
            m20, m17 = mean(b, var, ALL), mean(b, var, EVAL_SET)
            print(f"{var:<8}{m20:>10.4f}{m20-base20:>+11.4f}   {m17:>10.4f}{m17-base17:>+11.4f}")

    # Per-dataset deltas for the headline comparison (deployed head, all vs none).
    print(f"\n================ PER-DATASET  (deployed b=0.60: none → all) ================")
    print(f"{'tgt':<6}{'none':>8}{'zs':>8}{'all':>8}{'Δall':>9}")
    for t in ALL:
        r = rows[t][0.60]
        flag = "  *outlier" if t in OUTLIERS else ""
        print(f"{t:<6}{r['none']:>8.3f}{r['zs']:>8.3f}{r['all']:>8.3f}{r['all']-r['none']:>+9.3f}{flag}")

    out = os.path.join(os.path.dirname(__file__), "center_eval.tsv")
    with open(out, "w") as f:
        cols = [f"b{b}_{v}" for b in BVALS for v in VARIANTS]
        f.write("dataset\t" + "\t".join(cols) + "\n")
        for t in ALL:
            f.write(t + "\t" + "\t".join(f"{rows[t][b][v]:.4f}" for b in BVALS for v in VARIANTS) + "\n")
    print(f"\nTSV: {out}")


if __name__ == "__main__":
    main()
