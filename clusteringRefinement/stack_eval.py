#!/usr/bin/env python3
"""Final stack eval: seed-ensemble × TTA × blend re-tune, on the LOMO folds.

Per fold and per replica head r: proj_r^tta = L2(mean(proj_r, head_r(view_k)))
over valid pixel-aug views (falls back to proj_r where views are missing).
Ensemble similarity = mean_r(proj_r^tta @ proj_r^tta.T), blended with zero-shot
at each w in the grid, Ward@N, labeled-ARI.

Configs reported: solo1 @0.60 (deployed control), ens3 @0.60, ens3+tta @0.60,
ens3+tta @ each grid blend.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import COLOR_W, EVAL_SET, PEG_W, ALL, LOMO, l2, load_fold  # noqa: E402
from train_projection_head import ProjectionHead  # noqa: E402
from tta_eval import load_views  # noqa: E402

ROOTS = [LOMO] + [os.path.expanduser(f"~/.cache/reorder/lomo_v26_r{r}") for r in (2, 3)]
BLENDS = [0.55, 0.60, 0.65, 0.70]


def ward_ari(S, true, N):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = np.clip(0.5 * (D + D.T), 0.0, None)
    Z = linkage(squareform(D, checks=False), method="ward")
    g = true >= 0
    return adjusted_rand_score(true[g], fcluster(Z, t=N, criterion="maxclust")[g])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for tgt in args.targets or ALL:
        try:
            peg, col, proj1, true, N = load_fold(tgt)
        except FileNotFoundError as e:
            print(f"{tgt}: skipping ({e})", file=sys.stderr)
            continue
        v = load_views(tgt)

        Gzs = (PEG_W**2 * (peg @ peg.T) + COLOR_W**2 * (col @ col.T)) / (PEG_W**2 + COLOR_W**2)

        def blend(sim_proj, w):
            return (1 - w) * Gzs + w * sim_proj

        tta_sims, plain_sims = [], []
        for root in ROOTS:
            pr = l2(np.load(f"{root}/{tgt}/{tgt}_proj.npy").astype(np.float32))
            plain_sims.append(pr @ pr.T)
            if v is None:
                tta_sims.append(plain_sims[-1])
                continue
            peg_v, col_v, valid = v
            head = ProjectionHead(in_dim=peg.shape[1] + col.shape[1], hidden=1024, out_dim=512)
            head.load_state_dict(torch.load(f"{root}/{tgt}/proj_head.pt",
                                            map_location="cpu", weights_only=True))
            head.eval()
            acc, cnt = pr.copy(), np.ones(len(pr), dtype=np.float32)
            for k in range(peg_v.shape[1]):
                feats = np.concatenate([l2(peg_v[valid, k]), col_v[valid, k]], axis=1)
                with torch.no_grad():
                    out = l2(head(torch.from_numpy(feats)).numpy())
                acc[valid] += out
                cnt[valid] += 1
            pt = l2(acc / cnt[:, None])
            tta_sims.append(pt @ pt.T)

        res = {"dataset": tgt,
               "solo1": ward_ari(blend(plain_sims[0], 0.60), true, N),
               "ens3": ward_ari(blend(np.mean(plain_sims, axis=0), 0.60), true, N)}
        ens_tta = np.mean(tta_sims, axis=0)
        for w in BLENDS:
            res[f"ens3tta_b{w:g}"] = ward_ari(blend(ens_tta, w), true, N)
        rows.append(res)
        print(f"{tgt}: " + "  ".join(f"{k}={v_:.4f}" for k, v_ in res.items() if k != "dataset"),
              flush=True)

    if not rows:
        sys.exit("no folds scored")
    keys = [k for k in rows[0] if k != "dataset"]
    ev = [r for r in rows if r["dataset"] in EVAL_SET]
    print(f"\n=== means ({len(rows)} scored, {len(ev)} in EVAL_SET) ===")
    base = np.mean([r["solo1"] for r in ev])
    for k in keys:
        e = np.mean([r[k] for r in ev])
        wins = sum(1 for r in ev if r[k] > r["solo1"])
        print(f"{k:<14} {e:.4f}  Δ={e - base:+.4f}  wins {wins}/{len(ev)}")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\t".join(["dataset"] + keys) + "\n")
            for r in rows:
                fh.write("\t".join([r["dataset"]] + [f"{r[k]:.6f}" for k in keys]) + "\n")
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
