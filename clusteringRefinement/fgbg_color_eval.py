#!/usr/bin/env python3
"""
Foreground/background color-split blend eval — 17-set LOMO (excl. M7/M14/M15).

Tests replacing the single color modality with separate foreground/background color
histograms (extract_fgbg_color.py). Deterministic, NO training: re-scores the cached
/tmp/lomo_postaug folds, same Ward@oracle-N + ARI-on-grouped protocol as
center_eval.py / rerank_eval.py.

Exact blend identity used (each modality is unit-norm, so concat→L2→cosine reduces to
a convex combination): blended_cosine = (1−b)·Gzs + b·Gproj, with
Gzs = (Σ wₘ²·Gramₘ)/(Σ wₘ²). So per-modality Grams are precomputed once per dataset
and every weight combo is a cheap scalar-weighted sum — only Ward re-runs.

Two families vs the deployed baseline (PE-G 1.0 + Color 0.7):
  replace : PE-G 1.0 + ColorFG x1 + ColorBG x2     (the user's framing)
  add     : PE-G 1.0 + Color 0.7 + ColorBG x2      (keep full color, add background)
Scored at b=0.00 (zero-shot) and b=0.60 (deployed head).
"""
from __future__ import annotations
import json, os, time, warnings
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import BASE, NAMES, LOMO, EVAL_SET, l2, load_fold  # noqa: E402

BVALS = [0.0, 0.60]

FGW = [0.0, 0.35, 0.5, 0.7]
BGW = [0.0, 0.35, 0.5, 0.7]
CONFIGS = [("baseline", {"peg": 1.0, "color": 0.7})]
for x1 in FGW:
    for x2 in BGW:
        if x1 == 0 and x2 == 0:
            continue
        CONFIGS.append((f"rep_fg{x1}_bg{x2}", {"peg": 1.0, "fg": x1, "bg": x2}))
for x2 in [0.35, 0.5, 0.7]:
    CONFIGS.append((f"add_bg{x2}", {"peg": 1.0, "color": 0.7, "bg": x2}))


def load_fgbg(tgt):
    """color_fg, color_bg, bg_coverage in the same fold order as load_fold (L2-normed)."""
    fns = json.load(open(f"{LOMO}/{tgt}/{tgt}_filenames.json"))
    ch = json.load(open(f"{BASE}/ClusteringBenchmark{NAMES[tgt]}/.reorder-cache/content_hashes.json"))
    sc = np.load(f"{BASE}/ClusteringBenchmark{NAMES[tgt]}/.reorder-cache/fgbg_color_cache.npz", allow_pickle=False)
    h2r = {h: i for i, h in enumerate(sc["hashes"].tolist())}
    idx = np.array([h2r[ch[f]] for f in fns])
    return l2(sc["color_fg"][idx].astype(np.float32)), l2(sc["color_bg"][idx].astype(np.float32)), sc["bg_coverage"][idx]


def gzs(grams, weights):
    num = sum(w * w * grams[m] for m, w in weights.items() if w > 0)
    den = sum(w * w for w in weights.values() if w > 0)
    return num / den


def ari_for(grams, Gproj, weights, b, N, true, grouped):
    sim = gzs(grams, weights) if b == 0 else (1 - b) * gzs(grams, weights) + b * Gproj
    d = 1.0 - sim
    np.fill_diagonal(d, 0.0)
    d = np.clip(0.5 * (d + d.T), 0.0, None)
    Z = linkage(squareform(d, checks=False), method="ward")
    pred = fcluster(Z, t=N, criterion="maxclust")
    return adjusted_rand_score(true[grouped], pred[grouped])


def main():
    rows = {}      # rows[tgt][cfgname][b]
    cov = {}
    print(f"FG/BG color-split eval — 17 datasets, Ward@oracle-N, ARI on grouped subset")
    print(f"{'tgt':<5}{'n':>6}{'N':>5}{'bg_cov':>8}   ({len(CONFIGS)} configs × {len(BVALS)} blends)")
    for tgt in EVAL_SET:
        t0 = time.time()
        peg, col, proj, true, N = load_fold(tgt)
        fg, bg, c = load_fgbg(tgt)
        grouped = true >= 0
        grams = {"peg": peg @ peg.T, "color": col @ col.T, "fg": fg @ fg.T, "bg": bg @ bg.T}
        Gproj = proj @ proj.T
        cov[tgt] = float(c.mean())
        rows[tgt] = {}
        for name, w in CONFIGS:
            rows[tgt][name] = {b: ari_for(grams, Gproj, w, b, N, true, grouped) for b in BVALS}
        print(f"{tgt:<5}{len(true):>6}{N:>5}{cov[tgt]:>8.2f}   ({time.time()-t0:.0f}s)", flush=True)

    def mean(name, b): return float(np.mean([rows[t][name][b] for t in EVAL_SET]))

    base = {b: mean("baseline", b) for b in BVALS}
    # Self-check: baseline @ 0.60 must reproduce the deployed 17-set number (~0.8046).
    print(f"\n[self-check] baseline @ b=0.60 17-set = {base[0.60]:.4f}  (expect ≈ 0.8046)")

    print(f"\n================ AVERAGE OVERVIEW — 17-set mean ARI ================")
    print(f"{'config':<16}{'b=0.00':>9}{'Δ':>9}   {'b=0.60':>9}{'Δ':>9}")
    for name, _ in CONFIGS:
        m0, m6 = mean(name, 0.0), mean(name, 0.60)
        print(f"{name:<16}{m0:>9.4f}{m0-base[0.0]:>+9.4f}   {m6:>9.4f}{m6-base[0.60]:>+9.4f}")

    # Winners by 17-set mean @ deployed b=0.60.
    rep = max((c for c in CONFIGS if c[0].startswith("rep_")), key=lambda c: mean(c[0], 0.60))[0]
    add = max((c for c in CONFIGS if c[0].startswith("add_")), key=lambda c: mean(c[0], 0.60))[0]

    print(f"\n================ FULL PER-DATASET @ b=0.60  (best replace={rep}, best add={add}) ================")
    print(f"{'tgt':<6}{'bg_cov':>7}{'baseline':>10}{'  '+rep:>16}{'Δrep':>8}{'  '+add:>14}{'Δadd':>8}")
    for t in EVAL_SET:
        b0, r0, a0 = rows[t]["baseline"][0.60], rows[t][rep][0.60], rows[t][add][0.60]
        print(f"{t:<6}{cov[t]:>7.2f}{b0:>10.3f}{r0:>16.3f}{r0-b0:>+8.3f}{a0:>14.3f}{a0-b0:>+8.3f}")
    print("-" * 70)
    print(f"{'MEAN':<6}{np.mean([cov[t] for t in EVAL_SET]):>7.2f}"
          f"{base[0.60]:>10.4f}{mean(rep,0.60):>16.4f}{mean(rep,0.60)-base[0.60]:>+8.4f}"
          f"{mean(add,0.60):>14.4f}{mean(add,0.60)-base[0.60]:>+8.4f}")

    out = os.path.join(os.path.dirname(__file__), "fgbg_color_eval.tsv")
    with open(out, "w") as f:
        cols = [f"{n}@{b}" for n, _ in CONFIGS for b in BVALS]
        f.write("dataset\tbg_cov\t" + "\t".join(cols) + "\n")
        for t in EVAL_SET:
            f.write(f"{t}\t{cov[t]:.3f}\t" + "\t".join(f"{rows[t][n][b]:.4f}" for n, _ in CONFIGS for b in BVALS) + "\n")
    print(f"\nTSV: {out}")


if __name__ == "__main__":
    main()
