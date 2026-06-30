#!/usr/bin/env python3
"""Internal-validation color-w selectors (Phase 4 of the auto-color-w study).

Idea: instead of predicting w from dataset descriptors (Phase 3 — null result),
cluster the dataset at every w and pick the clustering a held-out signal likes
best. DINOv3 CLS is the natural validator: it is in every extraction cache but
carries ZERO weight in the deployed blend (and the head never sees it), so it
has no built-in preference along the w axis.

Criteria per (dataset, w), computed on the Phase-1 cached Ward labels:
  <mod>_sep      mean within-cluster sim - mean between-cluster sim (sampled pairs)
  <mod>_knn      fraction of each image's top-10 <mod> neighbors in its cluster
  <mod>_sil      mean silhouette under 1 - <mod> sim
for mod in dino (held-out), peg / proj (bias controls).

Selection: argmax_w criterion, with optional shrink-to-0.7 margin (in units of
the criterion's per-dataset range). Scored by lookup into the oracle matrix.

Usage: python colorw_internal.py [--margin 0.0] [--out colorw_internal.tsv]
"""
from __future__ import annotations
import argparse
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")  # Accelerate sets spurious FP flags in matmul

from lomo_common import BASE, EVAL_SET, LOMO, NAMES, ALL, l2
from colorw_select import GRID, DEPLOYED, read_tsv

OUTDIR = __import__("os").path.expanduser("~/.cache/reorder/colorw_auto")
N_PAIRS = 200_000
KNN_K = 10


def load_modality(tgt, key):
    """Embedding `key` reindexed to the fold's filename order, L2-normed."""
    d = f"{BASE}/ClusteringBenchmark{NAMES[tgt]}"
    fns = json.load(open(f"{LOMO}/{tgt}/{tgt}_filenames.json"))
    npz = np.load(f"{d}/.reorder-cache/embeddings_hash_cache.npz")
    if key not in npz.files:
        return None
    ch = json.load(open(f"{d}/.reorder-cache/content_hashes.json"))
    hrow = {h: i for i, h in enumerate(npz["hashes"])}
    idx = np.array([hrow[ch[f]] for f in fns])
    return l2(npz[key][idx].astype(np.float32))


def load_ens_proj(tgt):
    import os
    projs = []
    for root in [LOMO] + [os.path.expanduser(f"~/.cache/reorder/lomo_v26_r{r}") for r in (2, 3)]:
        projs.append(l2(np.load(f"{root}/{tgt}/{tgt}_proj.npy").astype(np.float32)))
    return l2(np.hstack(projs) / np.sqrt(len(projs)))


def criteria_for(tgt, mods):
    z = np.load(f"{OUTDIR}/{tgt}_labels.npz")
    labels = {w: z[f"w{w}"] for w in GRID}
    n = len(z["true"])
    rng = np.random.default_rng(0)
    pi = rng.integers(0, n, N_PAIRS)
    pj = rng.integers(0, n, N_PAIRS)
    keep = pi != pj
    pi, pj = pi[keep], pj[keep]

    out = {}
    for mod, X in mods.items():
        S = X @ X.T
        psim = S[pi, pj]
        Sx = S.copy()
        np.fill_diagonal(Sx, -np.inf)
        k = min(KNN_K, n - 1)
        nbr = np.argpartition(-Sx, k - 1, axis=1)[:, :k]
        D = 1.0 - S
        for w in GRID:
            lab = labels[w]
            same = lab[pi] == lab[pj]
            if same.sum() == 0 or (~same).sum() == 0:
                sep = 0.0
            else:
                sep = float(psim[same].mean() - psim[~same].mean())
            knn = float((lab[nbr] == lab[:, None]).mean())
            out.setdefault(f"{mod}_sep", {})[w] = sep
            out.setdefault(f"{mod}_knn", {})[w] = knn
        # silhouette only for the held-out modality (it's the expensive one)
        if mod == "dino":
            from sklearn.metrics import silhouette_score
            sub = rng.choice(n, min(n, 3000), replace=False)
            Ds = np.clip(0.5 * (D[np.ix_(sub, sub)] + D[np.ix_(sub, sub)].T), 0, None)
            np.fill_diagonal(Ds, 0.0)
            for w in GRID:
                lab = labels[w][sub]
                try:
                    out.setdefault("dino_sil", {})[w] = float(
                        silhouette_score(Ds.astype(np.float64), lab, metric="precomputed"))
                except ValueError:
                    out.setdefault("dino_sil", {})[w] = 0.0
    return out


def pick(curve, margin_frac):
    """argmax with shrink-to-deployed: deviate only if the criterion gain over
    w=0.7 exceeds margin_frac of the criterion's range across the grid."""
    vals = np.array([curve[w] for w in GRID])
    rng_ = vals.max() - vals.min()
    j = int(vals.argmax())
    if vals[j] - curve[DEPLOYED] <= margin_frac * rng_:
        return DEPLOYED
    return GRID[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="colorw_oracle_ens3.tsv")
    ap.add_argument("--margins", nargs="*", type=float, default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--out", default="colorw_internal.tsv")
    args = ap.parse_args()

    o = read_tsv(args.oracle)
    targets = [t for t in (args.targets or ALL) if t in o]
    curves_ari = {d: np.array([float(o[d][f"w{w}"]) for w in GRID]) for d in targets}

    crit_rows = {}
    for tgt in targets:
        mods = {"dino": load_modality(tgt, "dinov3"),
                "peg": load_modality(tgt, "pecore_g"),
                "proj": load_ens_proj(tgt)}
        missing = [m for m, X in mods.items() if X is None]
        if missing:
            print(f"{tgt}: missing modality {missing} — skipped for those criteria")
            mods = {m: X for m, X in mods.items() if X is not None}
        crit_rows[tgt] = criteria_for(tgt, mods)
        best = {c: pick(v, 0.0) for c, v in crit_rows[tgt].items()}
        print(f"{tgt}: " + "  ".join(f"{c}->{w}" for c, w in best.items()), flush=True)

    crits = sorted({c for r in crit_rows.values() for c in r})
    i07 = GRID.index(DEPLOYED)
    ev = [d for d in targets if d in EVAL_SET]
    base = np.mean([curves_ari[d][i07] for d in ev])
    print(f"\neval n={len(ev)}  fixed@0.7={base:.4f}  "
          f"oracle={np.mean([curves_ari[d].max() for d in ev]):.4f}")
    print(f"{'criterion':<12}{'margin':>7}{'evalARI':>9}{'delta':>9}{'win':>5}{'loss':>5}  worst")
    for c in crits:
        for m in args.margins:
            deltas = {}
            for d in ev:
                if c not in crit_rows[d]:
                    continue  # modality missing for this dataset
                w = pick(crit_rows[d][c], m)
                deltas[d] = curves_ari[d][GRID.index(w)] - curves_ari[d][i07]
            md = float(np.mean(list(deltas.values())))
            wins = sum(1 for v in deltas.values() if v > 0.002)
            losses = sum(1 for v in deltas.values() if v < -0.002)
            worst = min(deltas, key=deltas.get)
            print(f"{c:<12}{m:>7.2f}{base + md:>9.4f}{md:>+9.4f}{wins:>5}{losses:>5}"
                  f"  {worst} {deltas[worst]:+.4f}")

    with open(args.out, "w") as fh:
        fh.write("dataset\tcriterion\t" + "\t".join(f"w{w}" for w in GRID) + "\n")
        for d in targets:
            for c in crits:
                if c in crit_rows[d]:
                    fh.write(d + "\t" + c + "\t"
                             + "\t".join(f"{crit_rows[d][c][w]:.6f}" for w in GRID) + "\n")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
