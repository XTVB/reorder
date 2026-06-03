#!/usr/bin/env python3
"""
Oracle-ceiling analysis for a boundary-pair verifier — 17-set (excl M7/M14/M15).

Question: before building a cross-encoder "same-set?" verifier, what is the MAXIMUM
ARI a perfect one could buy, and how good would a real one need to be?

Method (deterministic, no training; backbone-agnostic — only distances + labels):
  1. D = deployed blended distance (PE-G 1.0 + Color 0.7, head b=0.60) — what Ward cuts.
  2. Candidate edges = each image's k nearest neighbors under D (the pairs the real
     verifier would be asked to judge).
  3. Perfect oracle: overwrite ONLY those edges — same-set→0, different-set→2.0
     (edges touching an ungrouped image are left as-is). Re-run Ward@oracle-N → ARI.
  4. Headroom curve: sweep k. Also report same-set-pair RECALL at each k — if the
     ceiling keeps rising with k, the limit is candidate RECALL (a verifier can't fix
     pairs it never sees); if it plateaus, the limit is VERIFICATION (what a
     cross-encoder is for).
  5. Accuracy requirement: at a fixed budget, corrupt the oracle to p% correct on the
     gray-zone edges and sweep p — how accurate must the real verifier be to net +0.05?
"""
from __future__ import annotations
import os, time, warnings
from collections import Counter
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import PEG_W, COLOR_W, EVAL_SET, l2, load_fold  # noqa: E402

B = 0.60
KS = [5, 10, 20, 40, 80]
K_ACC = 40
PVALS = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70]
FAR = 2.0


def deployed_D(peg, col, proj):
    Gzs = (PEG_W**2 * (peg @ peg.T) + COLOR_W**2 * (col @ col.T)) / (PEG_W**2 + COLOR_W**2)
    sim = (1 - B) * Gzs + B * (proj @ proj.T)
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    return np.clip(0.5 * (D + D.T), 0.0, None)


def ward_ari(D, N, true, grouped):
    Z = linkage(squareform(D, checks=False), method="ward")
    pred = fcluster(Z, t=N, criterion="maxclust")
    return adjusted_rand_score(true[grouped], pred[grouped])


def knn_edges(D, k):
    n = len(D)
    k = min(k, n - 2)
    nn = np.argpartition(D, kth=k, axis=1)[:, :k + 1]   # k nearest incl. self (unordered)
    edges = set()
    for i in range(n):
        for j in nn[i]:
            if j != i:
                edges.add((i, j) if i < j else (j, i))
    return edges


def oracle_D(D, edges, true, p, rng):
    Dc = D.copy()
    for i, j in edges:
        if true[i] < 0 or true[j] < 0:
            continue
        same = true[i] == true[j]
        if p < 1.0 and rng.random() > p:      # verifier is wrong on this edge
            same = not same
        Dc[i, j] = Dc[j, i] = 0.0 if same else FAR
    return Dc


def oracle_D_soft(D, edges, true, p, rng, alpha):
    """Realistic integration: blend the verifier answer into the bi-encoder distance
    (target 0/1, strength alpha) instead of hard-welding to 0/2.0."""
    Dc = D.copy()
    for i, j in edges:
        if true[i] < 0 or true[j] < 0:
            continue
        same = true[i] == true[j]
        if p < 1.0 and rng.random() > p:
            same = not same
        target = 0.0 if same else 1.0
        Dc[i, j] = Dc[j, i] = (1 - alpha) * D[i, j] + alpha * target
    return Dc


def same_pair_recall(edges, true):
    cnt = Counter(true[true >= 0].tolist())
    total = sum(c * (c - 1) // 2 for c in cnt.values())
    captured = sum(1 for i, j in edges if true[i] >= 0 and true[j] >= 0 and true[i] == true[j])
    return captured, total


def main():
    base, ceil_k, recall_k, acc, acc_soft = {}, {}, {}, {}, {}
    ALPHAS = [0.3, 0.6, 1.0]
    print("Oracle-ceiling — 17 datasets, deployed blend (b=0.60), Ward@oracle-N")
    print(f"{'tgt':<5}{'n':>6}{'N':>5}{'base':>8}   ceiling@k → " + " ".join(f"k{k}" for k in KS))
    for tgt in EVAL_SET:
        t0 = time.time()
        peg, col, proj, true, N = load_fold(tgt)
        grouped = true >= 0
        D = deployed_D(peg, col, proj)
        base[tgt] = ward_ari(D, N, true, grouped)
        ceil_k[tgt], recall_k[tgt] = {}, {}
        edges_by_k = {}
        for k in KS:
            e = knn_edges(D, k)
            edges_by_k[k] = e
            ceil_k[tgt][k] = ward_ari(oracle_D(D, e, true, 1.0, None), N, true, grouped)
            cap, tot = same_pair_recall(e, true)
            recall_k[tgt][k] = cap / max(tot, 1)
        # accuracy requirement at fixed budget K_ACC — hard weld + soft blend
        acc[tgt], acc_soft[tgt] = {}, {}
        for p in PVALS:
            acc[tgt][p] = ward_ari(oracle_D(D, edges_by_k[K_ACC], true, p, np.random.default_rng(int(p * 100))), N, true, grouped)
            for a in ALPHAS:
                acc_soft[tgt][(a, p)] = ward_ari(
                    oracle_D_soft(D, edges_by_k[K_ACC], true, p, np.random.default_rng(int(p * 100) + int(a * 7)), a),
                    N, true, grouped)
        print(f"{tgt:<5}{len(true):>6}{N:>5}{base[tgt]:>8.3f}   "
              + " ".join(f"{ceil_k[tgt][k]:.3f}" for k in KS) + f"   ({time.time()-t0:.0f}s)", flush=True)

    def m(d, *keys):
        for k in keys:
            d = {t: d[t][k] for t in EVAL_SET}
        return float(np.mean([d[t] for t in EVAL_SET]))

    mbase = float(np.mean([base[t] for t in EVAL_SET]))
    print(f"\n[self-check] baseline 17-set mean = {mbase:.4f}  (expect ≈ 0.8046)")

    print(f"\n================ HEADROOM vs verification budget k (perfect oracle) ================")
    print(f"{'k':>5}{'ceiling':>10}{'Δ vs base':>11}{'same-pair recall':>18}")
    for k in KS:
        c = float(np.mean([ceil_k[t][k] for t in EVAL_SET]))
        r = float(np.mean([recall_k[t][k] for t in EVAL_SET]))
        print(f"{k:>5}{c:>10.4f}{c-mbase:>+11.4f}{r:>17.1%}")

    print(f"\n================ ACCURACY REQUIRED (budget k={K_ACC}) ================")
    print(f"{'verifier acc':>13}{'ARI':>9}{'Δ vs base':>11}")
    for p in PVALS:
        a = float(np.mean([acc[t][p] for t in EVAL_SET]))
        print(f"{p:>12.0%}{a:>10.4f}{a-mbase:>+11.4f}")

    print(f"\n================ ACCURACY REQUIRED — SOFT integration (budget k={K_ACC}) ================")
    print("blend = (1-α)·bi-encoder dist + α·verifier(0/1);  α=1.0 ≈ trust verifier fully")
    print(f"{'verifier acc':>13}" + "".join(f"{'α='+str(a):>11}" for a in ALPHAS))
    for p in PVALS:
        row = "".join(f"{float(np.mean([acc_soft[t][(a,p)] for t in EVAL_SET]))-mbase:>+11.4f}" for a in ALPHAS)
        print(f"{p:>12.0%}{row}")
    print(f"(values are Δ vs baseline {mbase:.4f})")

    print(f"\n================ PER-DATASET (base | ceiling@k={K_ACC} | recall@k={K_ACC}) ================")
    print(f"{'tgt':<6}{'base':>8}{'ceil':>8}{'Δ':>8}{'recall':>9}")
    for t in EVAL_SET:
        print(f"{t:<6}{base[t]:>8.3f}{ceil_k[t][K_ACC]:>8.3f}{ceil_k[t][K_ACC]-base[t]:>+8.3f}{recall_k[t][K_ACC]:>9.1%}")

    out = os.path.join(os.path.dirname(__file__), "oracle_ceiling.tsv")
    with open(out, "w") as f:
        f.write("dataset\tbase\t" + "\t".join(f"ceil_k{k}" for k in KS)
                + "\t" + "\t".join(f"recall_k{k}" for k in KS)
                + "\t" + "\t".join(f"acc_p{int(p*100)}" for p in PVALS) + "\n")
        for t in EVAL_SET:
            f.write(f"{t}\t{base[t]:.4f}\t" + "\t".join(f"{ceil_k[t][k]:.4f}" for k in KS)
                    + "\t" + "\t".join(f"{recall_k[t][k]:.4f}" for k in KS)
                    + "\t" + "\t".join(f"{acc[t][p]:.4f}" for p in PVALS) + "\n")
    print(f"\nTSV: {out}")


if __name__ == "__main__":
    main()
