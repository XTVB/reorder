#!/usr/bin/env python3
"""
Production-accurate (re-rank ON) re-evaluation across the 20-dataset LOMO.

Everything earlier in this effort ran Ward on plain/learned-blended cosine, with
NO k-reciprocal re-ranking — but production at the time defaulted to re-rank ON +
average linkage (this study is what dropped that default; see LEARNED_HEAD.md).
This harness reuses the *exact* production re-rank code
(scripts/precompute_rerank_distance.compute_rerank_distance) and replicates the
production blend (0.7·rerank + 0.3·cosine), validated to match the Rust pipeline.

It builds the feature space three ways and clusters at oracle-N (ARI on the
grouped subset, same protocol as benchmark_clustering.ts):
  - BASE   : peg + 0.7·color                         (no head)
  - HEAD   : peg + 0.7·color + learned·w(b)          (b = learned signal fraction)
HEAD uses each fold's held-out projection from /tmp/lomo_postaug (true LOMO),
weighted by the production rescale  w = sqrt(b·S/(1−b)),  S = peg²+color².

Configs answer:
  1. re-rank's real lift            : base rerank+avg  vs  base cosine+ward (old benchmark style)
  2. ward vs average WITH re-rank   : base rerank+ward vs base rerank+avg
  3. head's value WITH re-rank      : head rerank+avg  vs  base rerank+avg
  4. best learned fraction w/ rerank: sweep b ∈ {0,.45,.6,.75} on rerank+avg
"""
from __future__ import annotations
import sys, os, json, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import numpy as np
from precompute_rerank_distance import compute_rerank_distance
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import ALL, COLOR_W, PEG_W, l2, load_fold  # noqa: E402

RERANK_BLEND, K1, K2 = 0.7, 65, 4


def build_feat(peg, col, proj, b):
    """L2-normalized weighted concat. b = learned signal fraction (0 = no head)."""
    parts = [PEG_W * peg, COLOR_W * col]
    if b > 0:
        S = PEG_W**2 + COLOR_W**2
        lw = np.sqrt(b * S / (1 - b))     # production rescaleLearnedProjWeight
        parts.append(lw * proj)
    return l2(np.concatenate(parts, axis=1))


def production_dist(feat):
    d_rr = compute_rerank_distance(feat, K1, K2)
    d_cos = 1.0 - (feat @ feat.T)
    d = RERANK_BLEND * d_rr + (1 - RERANK_BLEND) * d_cos
    np.fill_diagonal(d, 0.0)
    return np.clip(0.5 * (d + d.T), 0.0, None)


def cosine_dist(feat):
    d = 1.0 - (feat @ feat.T)
    np.fill_diagonal(d, 0.0)
    return np.clip(0.5 * (d + d.T), 0.0, None)


def clust_ari(d, method, N, true, grouped):
    Z = linkage(squareform(d, checks=False), method=method)
    pred = fcluster(Z, t=N, criterion="maxclust")
    return adjusted_rand_score(true[grouped], pred[grouped])


def main():
    rows = {}
    bsweep = {}
    BVALS = [0.45, 0.60, 0.75]
    for tgt in ALL:
        t0 = time.time()
        peg, col, proj, true, N = load_fold(tgt)
        grouped = true >= 0
        f_base = build_feat(peg, col, proj, 0)
        f_head = build_feat(peg, col, proj, 0.60)
        d_base_cos = cosine_dist(f_base)
        d_base_rr = production_dist(f_base)
        d_head_cos = cosine_dist(f_head)
        d_head_rr = production_dist(f_head)
        r = {
            "b_cos_ward": clust_ari(d_base_cos, "ward", N, true, grouped),     # old benchmark style
            "b_cos_avg":  clust_ari(d_base_cos, "average", N, true, grouped),
            "b_rr_ward":  clust_ari(d_base_rr, "ward", N, true, grouped),
            "b_rr_avg":   clust_ari(d_base_rr, "average", N, true, grouped),   # production base
            "h_cos_ward": clust_ari(d_head_cos, "ward", N, true, grouped),     # old head-LOMO style
            "h_rr_avg":   clust_ari(d_head_rr, "average", N, true, grouped),   # production + head
            "h_rr_ward":  clust_ari(d_head_rr, "ward", N, true, grouped),
        }
        rows[tgt] = r
        # learned-fraction sweep under rerank+avg. b=0 == b_rr_avg, and b=0.60 is
        # exactly h_rr_avg already computed above — reuse both (one rerank/dataset).
        bs = {0.0: r["b_rr_avg"], 0.60: r["h_rr_avg"]}
        for b in BVALS:
            if b in bs:
                continue
            fb = build_feat(peg, col, proj, b)
            bs[b] = clust_ari(production_dist(fb), "average", N, true, grouped)
        bsweep[tgt] = bs
        print(f"{tgt:<4} n={len(true):<5} N={N:<4} "
              + " ".join(f"{k}={v:.3f}" for k, v in r.items())
              + f"  ({time.time()-t0:.0f}s)", flush=True)

    def mean(key): return np.mean([rows[t][key] for t in ALL])

    print("\n================ BASE PIPELINE (no head), mean ARI over 20 ================")
    print(f"  cosine + ward   (old benchmark style) : {mean('b_cos_ward'):.4f}")
    print(f"  cosine + average                      : {mean('b_cos_avg'):.4f}")
    print(f"  rerank + ward                         : {mean('b_rr_ward'):.4f}")
    print(f"  rerank + average  (PRODUCTION base)   : {mean('b_rr_avg'):.4f}")
    print(f"  → re-rank lift (rr+avg − cos+ward)    : {mean('b_rr_avg')-mean('b_cos_ward'):+.4f}")
    print(f"  → ward vs avg WITH rerank (ward−avg)  : {mean('b_rr_ward')-mean('b_rr_avg'):+.4f}")

    print("\n================ HEAD (learned b=0.60), mean ARI over 20 ================")
    print(f"  head cosine + ward (old head-LOMO)    : {mean('h_cos_ward'):.4f}")
    print(f"  head rerank + average (PRODUCTION)    : {mean('h_rr_avg'):.4f}")
    print(f"  → head lift WITH rerank (h_rr_avg − b_rr_avg): {mean('h_rr_avg')-mean('b_rr_avg'):+.4f}")
    print(f"  → head lift NO rerank   (h_cos_ward − b_cos_ward): {mean('h_cos_ward')-mean('b_cos_ward'):+.4f}")

    print("\n================ LEARNED-FRACTION SWEEP under rerank+avg, mean ARI ================")
    for b in [0.0] + BVALS:
        print(f"  b={b:.2f} : {np.mean([bsweep[t][b] for t in ALL]):.4f}")

    print("\n================ PER-DATASET (rerank+avg base | +head | old cos+ward base) ================")
    print(f"{'tgt':<5}{'b_cos_ward':>11}{'b_rr_avg':>10}{'h_rr_avg':>10}{'Δrerank':>9}{'Δhead':>8}")
    for t in ALL:
        r = rows[t]
        print(f"{t:<5}{r['b_cos_ward']:>11.3f}{r['b_rr_avg']:>10.3f}{r['h_rr_avg']:>10.3f}"
              f"{r['b_rr_avg']-r['b_cos_ward']:>+9.3f}{r['h_rr_avg']-r['b_rr_avg']:>+8.3f}")

    out = os.path.join(os.path.dirname(__file__), "rerank_eval.tsv")
    with open(out, "w") as f:
        keys = list(rows[ALL[0]].keys())
        f.write("dataset\t" + "\t".join(keys) + "\t" + "\t".join(f"b{b}" for b in [0.0]+BVALS) + "\n")
        for t in ALL:
            f.write(t + "\t" + "\t".join(f"{rows[t][k]:.4f}" for k in keys)
                    + "\t" + "\t".join(f"{bsweep[t][b]:.4f}" for b in [0.0]+BVALS) + "\n")
    print(f"\nTSV: {out}")


if __name__ == "__main__":
    main()
