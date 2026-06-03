#!/usr/bin/env python3
"""
"Would a more accurate verifier help?" — synthetic verifier accuracy sweep (17-set).

The pooled verifier failed at 87% not because 87% is low, but because its errors were
(a) CORRELATED with the bi-encoder (wrong on the same hard pairs → adds no new info) and
(b) false-merge-biased (Ward amplifies bad merges). This isolates accuracy from error
structure: build synthetic verifiers at a target accuracy with three error models, blend
into the deployed distance (α=0.3), and measure real end-to-end ARI.

  random     : errors on random edges (independent of bi-encoder) — the optimistic case
  correlated : errors concentrated where the bi-encoder is wrong/ambiguous (realistic)
  corr+merge : correlated AND biased toward false merges (different→same) — worst, what we saw

Anchor check: corr+merge @ ~87% should land near the observed real-verifier ARI (≈0.735).
"""
from __future__ import annotations
import os, warnings
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import PEG_W, COLOR_W, EVAL_SET, load_fold  # noqa: E402

B, K, ALPHA = 0.60, 40, 0.30
ACCS = [0.87, 0.90, 0.93, 0.95, 0.97, 1.00]
MODELS = ["random", "correlated", "corr+merge"]


def deployed_D(peg, col, proj):
    Gzs = (PEG_W**2 * (peg @ peg.T) + COLOR_W**2 * (col @ col.T)) / (PEG_W**2 + COLOR_W**2)
    sim = (1 - B) * Gzs + B * (proj @ proj.T)
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    return np.clip(0.5 * (D + D.T), 0.0, None)


def knn_edges(D, true, k):
    n = len(D); k = min(k, n - 2)
    nn_idx = np.argpartition(D, kth=k, axis=1)[:, :k + 1]
    seen, ii, jj = set(), [], []
    for i in range(n):
        if true[i] < 0:
            continue
        for j in nn_idx[i]:
            if j == i or true[j] < 0:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b)); ii.append(a); jj.append(b)
    ii, jj = np.array(ii), np.array(jj)
    return ii, jj, D[ii, jj], (true[ii] == true[jj]).astype(np.float32)


def ward_ari(D, N, true, grouped):
    Z = linkage(squareform(D, checks=False), method="ward")
    return adjusted_rand_score(true[grouped], fcluster(Z, t=N, criterion="maxclust")[grouped])


def err_idx(model, n_err, label, dist, thr, rng):
    M = len(label)
    n_err = min(n_err, M)
    if model == "random":
        return rng.choice(M, n_err, replace=False)
    bi_wrong = (dist < thr) != label.astype(bool)
    order = np.lexsort((np.abs(dist - thr), ~bi_wrong))   # bi-wrong first, then most ambiguous
    if model == "correlated":
        return order[:n_err]
    diff = order[label[order] == 0]                        # bias errors onto different-set edges
    same = order[label[order] == 1]                        # → false merges when flipped
    return np.concatenate([diff, same])[:n_err]


def best_thr(dist, label):
    cand = np.quantile(dist, np.linspace(0.02, 0.98, 80))
    accs = [((dist < t) == label.astype(bool)).mean() for t in cand]
    return cand[int(np.argmax(accs))]


def main():
    out = {(a, m): [] for a in ACCS for m in MODELS}
    base = []
    rng = np.random.default_rng(0)
    for tgt in EVAL_SET:
        peg, col, proj, true, N = load_fold(tgt)
        grouped = true >= 0
        D = deployed_D(peg, col, proj)
        base.append(ward_ari(D, N, true, grouped))
        ii, jj, dist, label = knn_edges(D, true, K)
        thr = best_thr(dist, label)
        M = len(label)
        for a in ACCS:
            n_err = round((1 - a) * M)
            for model in MODELS:
                ver_same = label.astype(bool).copy()
                if n_err > 0:
                    e = err_idx(model, n_err, label, dist, thr, rng)
                    ver_same[e] = ~ver_same[e]
                target = np.where(ver_same, 0.0, 1.0)
                Dv = D.copy()
                blended = (1 - ALPHA) * D[ii, jj] + ALPHA * target
                Dv[ii, jj] = blended; Dv[jj, ii] = blended
                out[(a, model)].append(ward_ari(Dv, N, true, grouped))
        print(f"{tgt} done", flush=True)

    mbase = float(np.mean(base))
    print(f"\nbaseline ARI = {mbase:.4f}   (real pooled verifier landed at 0.7354 = {0.7354-mbase:+.4f})")
    print(f"\n{'acc':>6}" + "".join(f"{m:>14}" for m in MODELS) + "      (Δ ARI vs baseline)")
    for a in ACCS:
        row = "".join(f"{float(np.mean(out[(a,m)]))-mbase:>+14.4f}" for m in MODELS)
        print(f"{a:>6.0%}{row}")
    print(f"\n(α={ALPHA} soft blend, k={K}, errors symmetric except corr+merge which biases to false-merge)")

    tsv = os.path.join(os.path.dirname(__file__), "verifier_accuracy_sweep.tsv")
    with open(tsv, "w") as f:
        f.write("acc\t" + "\t".join(MODELS) + "\n")
        for a in ACCS:
            f.write(f"{a}\t" + "\t".join(f"{float(np.mean(out[(a,m)])):.4f}" for m in MODELS) + "\n")
    print(f"TSV: {tsv}")


if __name__ == "__main__":
    main()
