# Clustering — overview & current state

This folder records what we learned while improving the clustering quality of
the reorder app. The goal is to keep future iterations from re-running the same
experiments.

- **[research-log.md](research-log.md)** — chronological log of what was tried,
  what worked, what didn't, and concrete numbers.

## TL;DR

The clustering pipeline now uses **k-reciprocal re-ranking** as a default
post-processing step on top of cosine distance. Combined with **average
linkage**, it improves ARI on the Evie Lee labeled benchmark from **0.724 →
0.845** (+16.7% relative) without any new feature extraction. The implementation
is small, transparent, and feeds the result through the existing `--dist-matrix`
hook in the Rust cluster-tool.

## Key learnings

1. **The signal is in the kNN graph, not in better features.** Cosine distance
   only ever sees a single pair's view of similarity. The re-rank algorithm
   uses *global* graph structure: two images are similar only if their full
   neighbourhoods agree, AND if the members of those neighbourhoods vouch for
   each other. That filters most of the "bridge" pairs that confuse plain
   cosine.

2. **Adding more features barely helps.** DINOv3 (CLS or patches), spatial
   colour, and foreground masking gave at best +0.5% ARI for double the
   compute. The "we need more features" intuition was wrong — the missing
   signal was in how we *use* the features we already have.

3. **Single global thresholds cannot satisfy mixed-scale datasets.** ~30-50%
   of real-world photoshoot groups have intra-group spread larger than the gap
   to the nearest neighbouring group (e.g. Alice cosplay's intra_p90=0.19 vs
   nearest-group distance=0.04 → ratio 5.06). For these, no fixed
   `n_clusters`, distance threshold, or HDBSCAN `min_cluster_size` works
   universally. Re-ranking helps because it changes the *shape* of the
   distance distribution rather than picking a different cut on the same
   shape.

4. **Average linkage > Ward when working with re-rank distance.** Re-rank
   distances are more bimodal than cosine (clear "in-group" vs "out-group"
   modes), so Ward's variance-minimising heuristic fights the structure;
   average linkage flows with it. With raw cosine, Ward is still the right
   default.

5. **Iterative re-ranking does not help.** Running the algorithm a second time
   on its own output amplifies noise and reduces ARI. Stop at one iteration.

## Default parameter set

These are baked into the codebase. The benchmark sweep showed they're robust:
ARI ≥ 0.825 across the entire window k1∈[60,80], k2∈[4,8], λ∈[0.4,0.7].

| Parameter | Value | Where |
|---|---|---|
| `k1` (kNN size for k-reciprocal sets) | 65 | `src/cluster.ts` const |
| `k2` (local query expansion) | 4 | `src/cluster.ts` const |
| `λ` blend (rerank vs cosine) | 0.7 | UI slider, server default |
| Linkage method (re-rank on) | `average` | Auto-selected in `runLinkage` |
| Linkage method (re-rank off) | `ward` | Preserves prior behaviour |
| `useRerank` | `true` | UI toggle, default ON |

## Where the code lives

- `scripts/precompute_rerank_distance.py` — Python precompute (k-reciprocal R*
  + LQE → distance matrix .bin).
- `src/cluster.ts` — `ensureRerankDistMatrix`, `runLinkage` with
  `LinkageOptions`, threading through `runFullCluster` /
  `runLinkageOnly`.
- `rust/cluster-tool/src/main.rs` — `Linkage` enum + `lance_williams()`
  helper, `--linkage` CLI flag, default `average`.
- `src/client/stores/clusterStore.ts` — `useRerank`, `rerankBlend` state.
- `src/client/components/ClusterView/ClusterToolbar.tsx` — toggle + slider.

The re-rank distance is cached at
`.reorder-cache/rerank_dist_matrix.bin` (~30 bytes per pair, scales as n²) with
a sidecar `.meta.json` storing a single content-derived `signature` field.

Cache invalidation uses **`computeCacheSignature` from `src/cache-utils.ts`** —
the signature is a SHA-1 of:
- sorted `(filename, content_hash)` pairs from `content_hashes.json`
- `_v_<model>` version strings from the NPZ for each model with non-zero
  weight (re-extracting an active model invalidates the cache)
- the weights themselves
- algorithm version + params (`k1`, `k2`, rerank version)

No mtime checks anywhere — the same primitives drive `ensurePatchDistMatrix`
too. Renames preserve the cache (content hashes unchanged); image
add/remove/re-extract invalidates it. The cache hits sub-second on no-op
re-runs even though `extract_features.py` rewrites `content_hashes.json`
unconditionally.

## Benchmark dataset

Evie Lee Mikomin-backup (label-rich, 7608 images, 212 ground-truth groups,
all images grouped). Lily Lou is also referenced as a smaller secondary
benchmark (2224 images, partial labels for the 0001-0126 Christmas-shoot set).

The experimental scripts under `scripts/experiments/` are not shipped with
the app; they exist to reproduce the numbers in
[research-log.md](research-log.md). Re-running any of them requires the venv
described in `CLAUDE.md`.

## What's still on the table

These were considered but not adopted. Each is worth revisiting if a future
benchmark shows the current pipeline regressing:

- **DINOv3 separate-rerank blend** (+~0.5% ARI, 2× compute) — implementation
  in `scripts/experiments/rerank_dual.py`.
- **Per-cluster expand-with-density-budget** — UI affordance, not algorithmic.
  Useful for the long-tail of failures the algorithm legitimately can't
  resolve (mutually-overlapping lingerie shoots).
- **Hierarchical UI navigation** of the linkage tree — same purpose as above
  but more general.
- **Approximate kNN (faiss/hnswlib)** — only relevant past ~30k images, where
  the dense `n × n` similarity matrix becomes prohibitive. Not a concern at
  current dataset sizes.

Anything not on this list (mutual-reachability HDBSCAN, mixed-granularity
adaptive merge, foreground masking, EXIF metadata, spatial colour, iterative
re-ranking) was tested and found to underperform — see the research log.
