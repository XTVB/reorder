#!/usr/bin/env python3
"""Test-time augmentation A/B (LOMO folds).

The pixel-aug views (RandomResizedCrop+flip+jitter → PE-G + color, K=3/image)
already exist in each benchmark's .reorder-cache for training. This evaluates
using them at *inference*: average each image's representation over base + valid
views, on either side of the blend:

  tta_proj : push each view through the fold's trained head, mean the L2-normed
             projections with the base projection, renormalize
  tta_zs   : mean the L2-normed PE-G (and color) embeddings over base + views
  tta_both : both at once

Images without valid views fall back to their base representation. Datasets
without view files are skipped (M7/M22/M23/M24/M26 + partials).

Same Ward@N / labeled-ARI scoring as the other evals.
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
from lomo_common import BASE, COLOR_W, EVAL_SET, NAMES, PEG_W, ALL, LOMO, l2, load_fold  # noqa: E402
from train_projection_head import ProjectionHead  # noqa: E402

B = 0.60


def ward_ari(S, true, N):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = np.clip(0.5 * (D + D.T), 0.0, None)
    Z = linkage(squareform(D, checks=False), method="ward")
    g = true >= 0
    return adjusted_rand_score(true[g], fcluster(Z, t=N, criterion="maxclust")[g])


def load_views(tgt):
    cache = f"{BASE}/ClusteringBenchmark{NAMES[tgt]}/.reorder-cache"
    paths = [f"{cache}/pecore_g_views.npy", f"{cache}/color_views.npy", f"{cache}/views_meta.json"]
    if not all(os.path.exists(p) for p in paths):
        return None
    peg_v = np.load(paths[0]).astype(np.float32)  # (N, K, 1280)
    col_v = np.load(paths[1]).astype(np.float32)  # (N, K, 693)
    vm = json.load(open(paths[2]))
    n = peg_v.shape[0]
    valid = np.zeros(n, dtype=bool)
    if vm.get("view_indices") is not None:
        valid[np.asarray(vm["view_indices"], dtype=int)] = True
    else:
        valid[: int(vm.get("completed_through", 0))] = True
    # Defensive: drop rows with non-finite or all-zero views.
    finite = np.isfinite(peg_v).all(axis=(1, 2)) & np.isfinite(col_v).all(axis=(1, 2))
    nonzero = (np.abs(peg_v).sum(axis=(1, 2)) > 0)
    return peg_v, col_v, valid & finite & nonzero


def head_proj(head, peg, col):
    feats = np.concatenate([l2(peg), col], axis=1)
    with torch.no_grad():
        out = head(torch.from_numpy(feats)).numpy()
    return l2(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for tgt in args.targets or ALL:
        try:
            peg, col, proj, true, N = load_fold(tgt)
        except FileNotFoundError as e:
            print(f"{tgt}: skipping ({e})", file=sys.stderr)
            continue
        v = load_views(tgt)
        if v is None:
            print(f"{tgt}: no views, skipping", file=sys.stderr)
            continue
        peg_v, col_v, valid = v
        K = peg_v.shape[1]
        head = ProjectionHead(in_dim=peg.shape[1] + col.shape[1], hidden=1024, out_dim=512)
        head.load_state_dict(torch.load(f"{LOMO}/{tgt}/proj_head.pt",
                                        map_location="cpu", weights_only=True))
        head.eval()

        # --- head-side TTA: mean of L2-normed projections (base + valid views)
        acc = proj.copy()
        cnt = np.ones(len(proj), dtype=np.float32)
        for k in range(K):
            pv = head_proj(head, peg_v[valid, k], col_v[valid, k])
            acc[valid] += pv
            cnt[valid] += 1
        proj_tta = l2(acc / cnt[:, None])

        # --- zero-shot-side TTA: mean of L2-normed embeddings (base + views)
        def tta_embed(base_e, views):
            a = base_e.copy()
            c = np.ones(len(base_e), dtype=np.float32)
            for k in range(K):
                a[valid] += l2(views[valid, k])
                c[valid] += 1
            return l2(a / c[:, None])

        peg_tta, col_tta = tta_embed(peg, peg_v), tta_embed(col, col_v)

        def blend(p, c, pr):
            Gzs = (PEG_W**2 * (p @ p.T) + COLOR_W**2 * (c @ c.T)) / (PEG_W**2 + COLOR_W**2)
            return (1 - B) * Gzs + B * (pr @ pr.T)

        res = {"dataset": tgt, "n_valid": int(valid.sum()),
               "base": ward_ari(blend(peg, col, proj), true, N),
               "tta_proj": ward_ari(blend(peg, col, proj_tta), true, N),
               "tta_zs": ward_ari(blend(peg_tta, col_tta, proj), true, N),
               "tta_both": ward_ari(blend(peg_tta, col_tta, proj_tta), true, N)}
        rows.append(res)
        print(f"{tgt}: base={res['base']:.4f}  proj={res['tta_proj'] - res['base']:+.4f}  "
              f"zs={res['tta_zs'] - res['base']:+.4f}  both={res['tta_both'] - res['base']:+.4f}  "
              f"(views on {res['n_valid']})", flush=True)

    if not rows:
        sys.exit("no folds scored")
    keys = ["base", "tta_proj", "tta_zs", "tta_both"]
    ev = [r for r in rows if r["dataset"] in EVAL_SET]
    print(f"\n=== means ({len(rows)} scored, {len(ev)} in EVAL_SET) ===")
    be = np.mean([r["base"] for r in ev])
    for k in keys:
        e = np.mean([r[k] for r in ev])
        wins = sum(1 for r in ev if r[k] > r["base"])
        extra = "" if k == "base" else f"  Δ={e - be:+.4f}  wins {wins}/{len(ev)}"
        print(f"{k:<10} {e:.4f}{extra}")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\t".join(["dataset"] + keys) + "\n")
            for r in rows:
                fh.write("\t".join([r["dataset"]] + [f"{r[k]:.6f}" for k in keys]) + "\n")
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
