#!/usr/bin/env python3
"""
Clustering-algorithm bake-off on the deployed full-aug LOMO projections.

For every held-out dataset we reuse the deployed head's projection (z) from
/tmp/lomo_postaug/<tgt>/, rebuild the exact deployed blended representation
  x = [sqrt(w)·ẑ, sqrt(1-w)·b̂],   w = 0.6
(so ||x_i - x_j||² = 2·(w·d_learned + (1-w)·d_baseline) = 2·d_blend), then run a
menu of clustering algorithms and score ARI at oracle-N (N = #ground-truth
groups) on the grouped subset — same protocol as benchmark_clustering.ts.

Distance-based methods get the condensed cosine-blend distance; coordinate
methods get x; graph methods get a cosine kNN graph on x. Sweepable density/
graph methods (HDBSCAN/DBSCAN/Leiden) pick the parameter whose cluster count is
CLOSEST TO N (a label-free choice, comparable to cut-at-N) — never best-ARI.

Output: per-dataset ARI per method + a ranked summary (mean ARI over the shared
dataset set, and win/tie/loss vs ward-cosine).
"""
from __future__ import annotations
import json, os, sys, time, warnings
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import (AgglomerativeClustering, DBSCAN, KMeans, Birch,
                             SpectralClustering)
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import hdbscan
import leidenalg, igraph as ig

warnings.filterwarnings("ignore")
from lomo_common import ALL, COLOR_W, PEG_W, l2, load_fold  # noqa: E402

W = 0.60    # learned-head blend fraction (deployed)
KNN = 15    # kNN graph degree for graph / connectivity methods


def load_dataset(tgt):
    """Returns (x, Dcond, Dsquare, true_labels(int, -1 if ungrouped), N)."""
    peg, col, z, true, n_groups = load_fold(tgt, np.float64)
    # Deployed blended representation: x = [√W·z, √(1−W)·b],  b = unit(peg ⊕ COLOR_W·col),
    # so ‖x_i − x_j‖² = 2·(W·d_learned + (1−W)·d_baseline).
    b = l2(np.concatenate([PEG_W * peg, COLOR_W * col], axis=1))
    x = np.concatenate([np.sqrt(W) * z, np.sqrt(1 - W) * b], axis=1)
    # Blended cosine distance via Gram matrices (cheap, exact).
    Dsq = W * (1 - z @ z.T) + (1 - W) * (1 - b @ b.T)
    np.fill_diagonal(Dsq, 0.0)
    Dsq = np.clip(Dsq, 0.0, None)
    Dsq = 0.5 * (Dsq + Dsq.T)  # symmetrize numerical drift
    return x, squareform(Dsq, checks=False), Dsq, true, n_groups


def ari(true, pred, grouped):
    return adjusted_rand_score(true[grouped], pred[grouped])


def relabel_noise(pred):
    """Give each HDBSCAN/DBSCAN noise point (-1) its own singleton label."""
    pred = pred.copy()
    nxt = pred.max() + 1
    for i in np.where(pred == -1)[0]:
        pred[i] = nxt; nxt += 1
    return pred


def knn_graph(x):
    return kneighbors_graph(x, n_neighbors=min(KNN, len(x) - 1),
                            metric="cosine", mode="connectivity",
                            include_self=False)


def leiden_or_louvain(x, N, louvain=False):
    """kNN cosine graph → community detection; resolution chosen to hit ~N."""
    A = kneighbors_graph(x, n_neighbors=min(KNN, len(x) - 1), metric="cosine",
                         mode="distance", include_self=False)
    A = A.maximum(A.T)  # symmetric mutual graph
    coo = A.tocoo()
    edges = [(int(i), int(j)) for i, j in zip(coo.row, coo.col) if i < j]
    w = [1.0 - float(v) for i, j, v in zip(coo.row, coo.col, coo.data) if i < j]  # sim
    g = ig.Graph(n=len(x), edges=edges)
    g.es["weight"] = w
    if louvain:
        part = g.community_multilevel(weights="weight")
        return np.array(part.membership)
    best, bestgap = None, None
    for res in [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0]:
        p = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                     weights="weight", resolution_parameter=res, seed=42)
        m = np.array(p.membership)
        gap = abs(len(set(m)) - N)
        if bestgap is None or gap < bestgap:
            best, bestgap = m, gap
    return best


def sweep_closest_N(fn, params, N):
    """Run fn(param) → labels; keep the one whose #clusters is closest to N."""
    best, bestgap = None, None
    for p in params:
        try:
            lab = fn(p)
        except Exception:
            continue
        k = len(set(lab[lab >= 0])) if (lab < 0).any() else len(set(lab))
        gap = abs(k - N)
        if bestgap is None or gap < bestgap:
            best, bestgap = lab, gap
    return best


def run_methods(x, Dc, Ds, N):
    """Return {method: labels}. Each wrapped by caller in try/except."""
    n = len(x)
    out = {}
    def link(method, data, condensed):
        Z = linkage(data, method=method)
        return fcluster(Z, t=N, criterion="maxclust")

    out["ward_cosine"]    = lambda: link("ward", Dc, True)        # current pipeline
    out["ward_euclid"]    = lambda: link("ward", x, False)        # true-Euclidean Ward
    out["average"]        = lambda: link("average", Dc, True)
    out["complete"]       = lambda: link("complete", Dc, True)
    out["weighted"]       = lambda: link("weighted", Dc, True)
    out["single"]         = lambda: link("single", Dc, True)
    out["centroid"]       = lambda: link("centroid", x, False)
    out["median"]         = lambda: link("median", x, False)
    out["hdbscan"]        = lambda: relabel_noise(sweep_closest_N(
        lambda m: hdbscan.HDBSCAN(min_cluster_size=m, metric="precomputed")
                  .fit_predict(Ds), [3,5,8,12,20,30,50], N))
    out["dbscan"]         = lambda: relabel_noise(sweep_closest_N(
        lambda e: DBSCAN(eps=e, min_samples=3, metric="precomputed").fit_predict(Ds),
        [0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5], N))
    out["leiden"]         = lambda: leiden_or_louvain(x, N, louvain=False)
    out["louvain"]        = lambda: leiden_or_louvain(x, N, louvain=True)
    out["spherical_kmeans"] = lambda: KMeans(n_clusters=N, n_init=4, random_state=42).fit_predict(l2(x))
    out["agg_ward_knn"]   = lambda: AgglomerativeClustering(
        n_clusters=N, linkage="ward", connectivity=knn_graph(x)).fit_predict(x)
    out["agg_avg_knn"]    = lambda: AgglomerativeClustering(
        n_clusters=N, linkage="average", metric="euclidean", connectivity=knn_graph(x)).fit_predict(x)
    out["birch"]          = lambda: Birch(n_clusters=N).fit_predict(x)
    # Whitened-100 PCA is identical for ward & average — fit once, reuse.
    _pca = {}
    def whitened_pca():
        if "v" not in _pca:
            _pca["v"] = PCA(n_components=min(100, x.shape[1], n-1), whiten=True, random_state=42).fit_transform(x)
        return _pca["v"]
    out["pca_ward"]       = lambda: link("ward", whitened_pca(), False)
    out["pca_average"]    = lambda: fcluster(linkage(whitened_pca(), method="average"), t=N, criterion="maxclust")
    # Heavy — guarded by scale (see caller).
    out["gmm_diag"]       = lambda: GaussianMixture(n_components=N, covariance_type="diag", max_iter=100, random_state=42).fit_predict(PCA(n_components=min(50, x.shape[1], n-1), random_state=42).fit_transform(x))
    out["spectral"]       = lambda: SpectralClustering(n_clusters=N, affinity="nearest_neighbors", n_neighbors=KNN, assign_labels="kmeans", random_state=42).fit_predict(x)
    return out


# Skip heavy methods past these scales to keep the batch bounded (logged, not silent).
HEAVY_GUARD = {"gmm_diag": lambda n, N: N > 200, "spectral": lambda n, N: n > 4000 or N > 200}


def main():
    methods = None
    results = {}   # method -> {tgt: ari}
    skipped = {}   # method -> [tgt...]
    for tgt in ALL:
        t0 = time.time()
        x, Dc, Ds, true, N = load_dataset(tgt)
        grouped = true >= 0
        m = run_methods(x, Dc, Ds, N)
        if methods is None:
            methods = list(m.keys())
            results = {k: {} for k in methods}
            skipped = {k: [] for k in methods}
        line = [f"{tgt:<4} n={len(x):<5} N={N:<4}"]
        for name in methods:
            guard = HEAVY_GUARD.get(name)
            if guard and guard(len(x), N):
                skipped[name].append(tgt); line.append(f"{name[:4]}=skip"); continue
            try:
                lab = m[name]()
                a = ari(true, np.asarray(lab), grouped)
                results[name][tgt] = a
                line.append(f"{name[:4]}={a:.3f}")
            except Exception as e:
                skipped[name].append(tgt)
                line.append(f"{name[:4]}=ERR")
                print(f"   ! {tgt}/{name}: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
        print(f"{' '.join(line)}   ({time.time()-t0:.0f}s)", flush=True)

    # ── Summary ──
    base = "ward_cosine"
    shared = set(ALL)
    for name in methods:
        shared &= set(results[name].keys())
    shared = sorted(shared, key=lambda t: int(t[1:]))
    print(f"\n# Shared datasets (all methods ran): {len(shared)}/20")

    rows = []
    for name in methods:
        got = results[name]
        mean_all = np.mean(list(got.values())) if got else float("nan")
        mean_shared = np.mean([got[t] for t in shared]) if shared else float("nan")
        wins = ties = losses = 0
        for t in shared:
            d = got[t] - results[base][t]
            if d > 0.0005: wins += 1
            elif d < -0.0005: losses += 1
            else: ties += 1
        lift = mean_shared - np.mean([results[base][t] for t in shared]) if shared else 0
        rows.append((name, mean_shared, mean_all, len(got), lift, wins, ties, losses))

    rows.sort(key=lambda r: -r[1])
    print(f"\n{'method':<18}{'ARI(shared)':>12}{'ARI(all)':>10}{'cov':>5}{'Δ vs ward':>11}{'W/T/L vs ward':>15}")
    print("-" * 73)
    for name, ms, ma, cov, lift, w, t, l in rows:
        tag = "  <- current" if name == base else ""
        print(f"{name:<18}{ms:>12.4f}{ma:>10.4f}{cov:>5}{lift:>+11.4f}{f'{w}/{t}/{l}':>15}{tag}")
    sk = {k: v for k, v in skipped.items() if v}
    if sk:
        print("\nskipped/errored:", {k: len(v) for k, v in sk.items()})

    # Machine-readable
    out_tsv = os.path.join(os.path.dirname(__file__), "algo_comparison.tsv")
    with open(out_tsv, "w") as f:
        f.write("method\t" + "\t".join(ALL) + "\tmean_shared\tmean_all\n")
        for name in methods:
            vals = [f"{results[name].get(t, float('nan')):.4f}" for t in ALL]
            ms = np.mean([results[name][t] for t in shared]) if shared else float("nan")
            ma = np.mean(list(results[name].values())) if results[name] else float("nan")
            f.write(f"{name}\t" + "\t".join(vals) + f"\t{ms:.4f}\t{ma:.4f}\n")
    print(f"\nTSV: {out_tsv}")


if __name__ == "__main__":
    main()
