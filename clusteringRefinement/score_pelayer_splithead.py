#!/usr/bin/env python3
"""Score the PE-layer split-head sweep in the 3-head (joint/peg/color) prod blend.

Baselines (reused, 3-seed each):
  joint = lomo_jointmlx_s{42,43,44}   peg = lomo_pegmlx_s{42,43,44}
  color = lomo_col512mlx_s{42,43,44}
Augmented (this sweep, batched-seed layout $AUGROOT/<plc>/<L>_<pool>/<tgt>/s<seed>/):
  swap the joint and/or peg component for its layer-augmented version.

Blend = (zp·Szs_peg + zc·Szs_col + wb·S_joint + wp·S_peg + wc·S_color)/Σ, Ward@N,
ARI on labeled images. Default ratio = deployed b.55 p.30 c.15 (learned share)
on top of the 0.4 zero-shot split (zp,zc) — i.e. the production composition.

For each layer×pooling config reports Δ vs baseline for three placements:
  joint-aug (layer→joint head), peg-aug (layer→peg head), both-aug.

Usage: AUGROOT=... python score_pelayer_splithead.py [--jobs 6] [--out tsv]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import BASE, NAMES, l2  # noqa: E402

# Only the two partial-label datasets are excluded from eval (they're never
# folds anyway). M7 is a normal eval target now; M27 is included.
OUTLIERS = {"M14", "M15"}

CACHE = os.path.expanduser("~/.cache/reorder")
AUGROOT = os.environ.get("AUGROOT", f"{CACHE}/pelayer_splithead12")
SEEDS = [42, 43, 44]
# Baselines regenerated at 12/cosine inside this sweep (batched layout).
JOINT_BASE = f"{AUGROOT}/base/joint"
PEG_BASE = f"{AUGROOT}/base/peg"
COL_BASE = f"{AUGROOT}/base/color"
LAYERS_POOLS = [f"{L}:{p}" for L in (42, 44, 46, 47) for p in ("mean", "gem3", "attnpool")]

# Deployed blend (listStore.ts): the three learned dials are fractions of the
# final cosine SIGNAL and sum to 1.0 (joint .30, peg .55, color .15) → b=1, so
# zero-shot is fully zeroed and S_blend = Σ frac_k · S_k (unit sub-vectors →
# each head's cosine contribution equals its fraction). No zero-shot term.
ZP, ZC = 0.0, 0.0
WB, WP, WC = 0.30, 0.55, 0.15


def ens_sim(roots_or_dirs, tgt, batched=False):
    """Mean of per-seed L2-normed proj-sims. batched=False: roots are per-seed
    fold dirs {root}/{tgt}/{tgt}_proj.npy. batched=True: one dir with seed subdirs
    {dir}/s{seed}/{tgt}_proj.npy."""
    S = None
    n = 0
    if batched:
        d = roots_or_dirs
        for s in SEEDS:
            p = f"{d}/s{s}/{tgt}_proj.npy"
            if not os.path.exists(p):
                return None
            pr = l2(np.load(p).astype(np.float32))
            S = pr @ pr.T if S is None else S + pr @ pr.T
            n += 1
    else:
        for root in roots_or_dirs:
            p = f"{root}/{tgt}/{tgt}_proj.npy"
            if not os.path.exists(p):
                return None
            pr = l2(np.load(p).astype(np.float32))
            S = pr @ pr.T if S is None else S + pr @ pr.T
            n += 1
    return S / n


def ward_ari(S, true, N):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(np.clip(0.5 * (D + D.T), 0, None), checks=False), "ward")
    return adjusted_rand_score(true, fcluster(Z, t=N, criterion="maxclust"))


def load_zs_and_labels(tgt, fns):
    """peg, col (L2-normed, in fns order) + int group label per row (-1 ungrouped).
    Self-contained — no dependency on any external fold root (works for M27)."""
    d = f"{BASE}/ClusteringBenchmark{NAMES[tgt]}"
    z = np.load(f"{d}/.reorder-cache/embeddings_hash_cache.npz")
    ch = json.load(open(f"{d}/.reorder-cache/content_hashes.json"))
    hrow = {h: i for i, h in enumerate(z["hashes"])}
    idx = np.array([hrow[ch[f]] for f in fns])
    peg = l2(z["pecore_g"][idx].astype(np.float32))
    col = l2(z["color"][idx].astype(np.float32))
    graw = json.load(open(f"{d}/.reorder-groups.json"))
    glist = graw if isinstance(graw, list) else graw.get("groups", [])
    fn2lab = {fn: gi for gi, g in enumerate(glist) for fn in g["images"]}
    true = np.array([fn2lab.get(f, -1) for f in fns])
    return peg, col, true


def score_one(tgt):
    """Returns {config_variant: ari} for this fold, grouped-only."""
    # Filename order from the baseline joint fold (proj rows are aligned to it).
    fn_path = f"{JOINT_BASE}/{tgt}/s{SEEDS[0]}/{tgt}_filenames.json"
    if not os.path.exists(fn_path):
        return tgt, None
    fns = json.load(open(fn_path))
    peg, col, true = load_zs_and_labels(tgt, fns)
    g = true >= 0
    peg_g, col_g, true_g = peg[g], col[g], true[g]
    N = len(np.unique(true_g))
    Szs_peg = peg_g @ peg_g.T
    Szs_col = col_g @ col_g.T

    def comp(s_joint, s_peg, s_col):
        S = (ZP * Szs_peg + ZC * Szs_col + WB * s_joint + WP * s_peg + WC * s_col) / (ZP + ZC + WB + WP + WC)
        return ward_ari(S, true_g, N)

    def gmask(S):
        return None if S is None else S[np.ix_(g, g)]

    Sj = gmask(ens_sim(f"{JOINT_BASE}/{tgt}", tgt, batched=True))
    Sp = gmask(ens_sim(f"{PEG_BASE}/{tgt}", tgt, batched=True))
    Sc = gmask(ens_sim(f"{COL_BASE}/{tgt}", tgt, batched=True))
    if Sj is None or Sp is None or Sc is None:
        return tgt, None
    out = {"base": comp(Sj, Sp, Sc)}
    for lp in LAYERS_POOLS:
        L, pool = lp.split(":")
        key = f"L{L}_{pool}"
        Saj = gmask(ens_sim(f"{AUGROOT}/joint/{L}_{pool}/{tgt}", tgt, batched=True))
        Sap = gmask(ens_sim(f"{AUGROOT}/peg/{L}_{pool}/{tgt}", tgt, batched=True))
        if Saj is not None:
            out[f"{key}|joint"] = comp(Saj, Sp, Sc)
        if Sap is not None:
            out[f"{key}|peg"] = comp(Sj, Sap, Sc)
        if Saj is not None and Sap is not None:
            out[f"{key}|both"] = comp(Saj, Sap, Sc)
    return tgt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()
    # Default targets = whatever baseline-joint folds exist, minus partials.
    if args.targets:
        targets = args.targets
    else:
        base_dir = JOINT_BASE
        targets = sorted([t for t in os.listdir(base_dir)
                          if os.path.isdir(f"{base_dir}/{t}")],
                         key=lambda s: int(s[1:])) if os.path.isdir(base_dir) else []
    if not targets:
        sys.exit("no baseline folds found — run the sweep (./pelayer_sweep start) first")

    rows = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(score_one, t): t for t in targets}
        for f in as_completed(futs):
            t, res = f.result()
            if res:
                rows[t] = res
                print(f"  scored {t} ({len(res)} variants)", file=sys.stderr, flush=True)

    if not rows:
        sys.exit("no folds scored — has the sweep produced any folds yet?")
    ds_all = sorted(rows, key=lambda s: int(s[1:]))
    ds = [t for t in ds_all if t not in OUTLIERS]
    excl = [t for t in ds_all if t not in ds]
    base = np.mean([rows[t]["base"] for t in ds])
    print(f"\n=== {len(ds)} folds (grouped-only), deployed blend zp{ZP:.2f} zc{ZC:.2f} "
          f"b{WB} p{WP} c{WC} ==="
          + (f"   [excluded from mean: {' '.join(excl)}]" if excl else ""))
    print(f"baseline (no layer): {base:.4f}\n")
    print(f"{'config':<16}{'joint Δ':>9}{'peg Δ':>9}{'both Δ':>9}{'best':>9}")
    summary = []
    for lp in LAYERS_POOLS:
        L, pool = lp.split(":")
        key = f"L{L}_{pool}"
        cell = {}
        for plc in ("joint", "peg", "both"):
            vals = [rows[t].get(f"{key}|{plc}") for t in ds]
            if all(v is not None for v in vals):
                cell[plc] = np.mean(vals) - base
        if not cell:
            print(f"{key:<16}{'(no folds yet)':>27}")
            continue
        best = max(cell.values())
        summary.append((key, cell, best))
        fmt = lambda k: f"{cell[k]:+.4f}" if k in cell else "   --   "
        star = " *" if best > 0.0062 else ""
        print(f"{key:<16}{fmt('joint'):>9}{fmt('peg'):>9}{fmt('both'):>9}{best:>+9.4f}{star}")
    if summary:
        bk, bc, bb = max(summary, key=lambda x: x[2])
        bp = max(bc, key=bc.get)
        print(f"\nBEST: {bk} via {bp}-placement  Δ={bb:+.4f}  (baseline {base:.4f} → {base + bb:.4f})")
        print("(* = clears the ±0.0062 95% LOMO noise band)")

    if args.out:
        cols = ["base"] + [f"{k}|{p}" for k in [f"L{L}_{pl}" for L in (42, 44, 46, 47)
                for pl in ("mean", "gem3", "attnpool")] for p in ("joint", "peg", "both")]
        with open(args.out, "w") as fh:
            fh.write("dataset\t" + "\t".join(cols) + "\n")
            for t in ds_all:  # save every scored fold (incl. outliers) to the tsv
                fh.write(t + "\t" + "\t".join(f"{rows[t].get(c, '')}" for c in cols) + "\n")
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
