# Clustering research log

Detailed account of experiments run while improving clustering quality, in
roughly chronological order. Numbers are reproducible via the scripts in
`scripts/experiments/` (not shipped, run from the venv at
`~/.venvs/imgcluster-env`).

For the high-level summary and current configuration, see
[README.md](README.md).

---

## Starting point

Default cluster settings before this work:

- Distance: cosine on weighted-concat PE-G (1.0) + colour (0.5).
- Linkage: Ward (in `wards_linkage_cosine`, Rust cluster-tool).
- Cut: fixed `n_clusters = 200`, or HDBSCAN-style adaptive at
  `min_cluster_size = 5`.

Symptom report: large photoshoot sets (e.g. Lily Lou's 0001-0126 Christmas
set, 126 images) never appeared as a single suggested cluster. Lowering N
made the rest of the dataset blob into mixed clusters.

---

## Initial diagnostic — Lily Lou 0001-0068 (the wrong ground truth)

**Question:** is the issue in the features or the cut?

First-pass distance analysis using indices 0001-0068 as the assumed ground
truth set:

- INSIDE pairwise PE-G distance: mean 0.12, max 0.32.
- 10/68 targets had their nearest neighbour OUTSIDE the assumed set.
- Closest "outside" image: 0119 at d=0.0092 to 0068.

This led to an incorrect early conclusion that the features were inadequate.

The user then noted the actual related set was **0001-0126**, not 0001-0068.
Re-running the diagnostic with the correct ground truth flipped the picture
entirely:

- 126/126 targets have their nearest neighbour INSIDE the set.
- Closest truly-outside image (0083 ↔ 1971) was at d=0.13.
- **Ward's at n_clusters=30 unifies all 126 with zero contamination.**
- At the default n_clusters=200, the 126-set fragments into exactly 5 pure
  subclusters of sizes [56, 24, 22, 19, 5].

**Lesson:** the features were fine. The default cut landed at the worst
spot. Always validate ground truth before chasing feature improvements.

---

## The Evie Lee benchmark

Switched to a properly-labelled dataset for systematic experiments:

- **Evie Lee Mikomin-backup** (7608 images, 212 groups, every image labelled).
- Group sizes: median 33, max 139, range 1-139.
- Has PE-G + colour + DINOv3 + CLIP all extracted.

Metrics used:

- **ARI** (Adjusted Rand Index) — standard partition similarity.
- **Weighted group recall** = Σ_g (max cluster ∩ g) weighted by |g|.
- **Weighted cluster purity** = Σ_c (max GT-group ∩ c) weighted by |c|.
- **F1** = harmonic mean of weighted recall and purity.

Cosine baseline (Ward, fixed N):

| N | ARI | weighted recall | weighted purity | F1 |
|---:|---:|---:|---:|---:|
| 100 | 0.526 | 0.912 | 0.612 | 0.733 |
| 150 | 0.687 | 0.881 | 0.755 | 0.813 |
| **200** | **0.724** | 0.826 | 0.845 | 0.835 |
| 300 | 0.698 | 0.721 | 0.931 | 0.812 |
| 500 | 0.593 | 0.598 | 0.972 | 0.741 |

HDBSCAN-style stability extraction:

| min_cluster_size | n_clusters | ARI | recall | purity |
|---:|---:|---:|---:|---:|
| 5 | 1794 | 0.706 | 0.655 | 0.988 |
| **10** | 2389 | **0.710** | 0.640 | 0.983 |
| 20 | 3827 | 0.593 | 0.508 | 0.945 |

Both top out around ARI 0.71-0.72. That was the bar to beat.

---

## Why the dataset is hard

Per-group structural analysis (first 30 groups):

| Group | size | intra_p90 | nearest other group | ratio |
|---|---:|---:|---:|---:|
| Red Riding Hood | 37 | 0.094 | 0.123 | 0.77 |
| Alice (Wonderland) | 37 | 0.191 | 0.038 | **5.06** |
| Dorothy Gale | 51 | 0.145 | 0.038 | **3.84** |
| Wednesday Addams | 122 | 0.279 | 0.081 | **3.43** |
| ... (most others < 2.0) | | | | |

When the ratio exceeds 1, a typical group member is *farther* from its own
group's outliers than from the nearest neighbouring group. **No global
distance threshold can keep these groups intact without bleeding into
adjacent groups.** Across the dataset the ratio ranges 0.7-5.1, so any
single threshold mis-fits some part of the data.

Looking at concrete failure cases at fixed-N=200:

- **Cluster 72** (size 53, purity 0.13): mixed Yor Forger + Raphtalia +
  Misato Bunny + Hinata + Rizu-Kyun. All dark-haired anime cosplay.
  Closest cross-group pair was at d=0.149 — *below* both groups' p90 intra.
- **Cluster 174** (size 149, purity 0.21): mixed pink-aesthetic lingerie
  shoots (Cheeky Delights, Pink Heart Fishnets, Rainbow Crochet, Pink In
  Paradise, Pink Black Seduction). Colour saturates the signal.
- **Cluster 161** (size 101, purity 0.27): mixed black-lace shoots
  (Secretary, Nun, Green Forest Lace).

These are the cases that matter for "automated clustering quality".

---

## What didn't work

### Mixed-granularity adaptive merge (approach #5 from initial planning)

`scripts/experiments/adaptive_merge.py`

Walk a linkage tree in distance order, accept each merge only when
`d ≤ β × max(intra_density)`. Idea: each cluster carries its own scale, so
tight clusters get tight thresholds and loose clusters get loose ones.

Tried with multiple density proxies and tree sources:

| Variant | Best ARI |
|---|---:|
| Ward tree, raw merge dist as density | 0.585 |
| Ward tree, centroid distance | 0.0 (over-merges to 1 cluster) |
| Complete-linkage tree, max-pairwise | 0.628 |
| Average-linkage tree | 0.473 |

All worse than fixed-N=200 (ARI 0.724).

**Why it failed:** ~30-50% of groups have intra-spread > nearest-group
distance. The β rule says "merge if compatible with cluster's own scale";
when scale is loose, compatible merges include adjacent groups. The rule
degenerates to a global threshold whenever the density floor dominates.

**Caveat — does work on Lily Lou specifically**: complete-linkage tree, β=1.30,
floor=0.15 recovered 121/126 of the Christmas set with 0 extras at total
n=73. The Christmas set is aesthetically distinctive, so the loose-density
failure mode doesn't apply. Adaptive merge is therefore a *valid alternative*
mode for distinctive-but-spread photoshoots, just not a universal default.

### DINOv3 alternatives

Tested whether DINOv3 features provide signal that PE-G+colour misses.
User priors confirmed by experiment: DINOv3 globally is roughly equivalent
to or slightly worse than PE-G+colour.

**DINOv3 patch matching** (max / bidirectional / top-K-bidirectional
aggregations of 49×49 patch cosine sims) on the actual failure clusters:

| Discriminator | within median | across median | overlap on failures |
|---|---:|---:|---:|
| Cosine PE-G+colour | 0.064-0.11 | 0.10-0.21 | 46-64% confused |
| DINOv3 patch_max | similar | similar | 67-76% confused |
| DINOv3 patch_bidir | similar | similar | 67-78% confused |

Patches barely move the needle on the failure cases. They're slightly
better at near-duplicate matching (which is why the existing Merge
Suggestions feature uses them) but don't help cluster discrimination.

**Why no help:** the failure groups (Yor / Raphtalia / Hinata) genuinely
look similar at every spatial scale to a vision model trained on natural
images. The discriminating signal is character identity (face, costume
specifics), not spatial composition.

### Mutual reachability distance (HDBSCAN-style core distance)

`mr(a, b) = max(d(a,b), core_k(a), core_k(b))` where core_k is distance to
k-th NN. Designed to penalise pairs where one is in a sparse region.

Best result: ARI 0.750 at α=0.5 (modest improvement, similar to vanilla SNN).
Worse than k-reciprocal SNN (0.766). Not competitive with re-rank (0.845).

### Two-pass / iterative re-ranking

Run re-rank, then run re-rank again on the rerank distance.

Best iter-2 result: ARI 0.823 (down from 0.845). Iterating amplifies noise.
**Stop at one iteration.**

### Consensus re-rank (intersect kNN graphs from PE-G and DINOv3)

Only count NN edges that are mutual in *both* feature spaces. Intuition:
filter out spurious neighbours that one space gets wrong.

Result: ARI 0.819 — actively worse than re-rank on PE-G+colour alone
(0.845). The two spaces disagree often enough that intersection drops too
many true edges.

---

## What worked

### Vanilla shared-NN (SNN) Jaccard distance

`d_snn(a, b) = 1 − |kNN(a) ∩ kNN(b)| / |kNN(a) ∪ kNN(b)|`

First test on the three actual failure clusters (anime cosplay / pink
lingerie / black lace) showed the median gap between within-group and
across-group SNN was **5-10× wider** than the same gap for cosine. Strong
discriminator.

Full benchmark with `d = (1−α)·cosine + α·d_snn`:

| k | α | ARI |
|---:|---:|---:|
| 30 | 0 (cosine) | 0.724 |
| 50 | 0.3 | 0.750 |
| **75** | **0.5** | **0.757** |
| 100 | 0.3 | 0.734 |

Sweet spot k=50-75. Larger k dilutes (too many bridge images in NN list);
smaller k loses signal. **+4.6% ARI over cosine baseline.**

Lily Lou tangent: at k=100 (matched to the 126-group size), N=50 gave
125/126 of the Christmas set as one cluster with zero extras. But this
required knowing the group size in advance — we don't ship k=100 as default.

### k-reciprocal SNN (mutual edges only)

Filter the NN graph to keep only edges where i is in NN(j) AND j is in
NN(i). Removes "hub" effects where a popular image gets in everyone's NN
list one-way.

Best: ARI **0.766** at k1=100, α=0.3, N=200. **+5.8% over cosine baseline.**

Standard fix in person re-identification literature; works reliably here
too.

### Re-ranking with R* expansion + LQE (Zhong et al. 2017)

Implementation: `scripts/precompute_rerank_distance.py`. Algorithm:

1. R(p, k1) = k-reciprocal NN set.
2. R*(p, k1) = R(p, k1) ∪ R(q, k1/2) for each q where overlap is ≥ 2/3.
3. Local query expansion: average each indicator over its top-k2 cosine NNs.
4. Build weighted indicator vectors V_p; compute cosine kernel V·Vᵀ.
5. d_rerank(p, g) = 1 − normalized kernel similarity.
6. Final blend: `d = (1−λ) · d_cosine + λ · d_rerank`.

Linkage method matters a lot here:

| Linkage | k1 | k2 | λ | N | ARI | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Ward | 75 | 6 | 0.5 | 200 | 0.827 | 0.886 |
| Ward | 75 | 6 | 0.7 | 200 | 0.825 | 0.883 |
| Ward | **65** | **4** | **0.7** | 200 | **0.839** | **0.891** |
| **Average** | **65** | **4** | **0.7** | **300** | **0.845** | **0.895** |
| Average | 65 | 4 | 0.5 | 300 | 0.843 | 0.895 |
| Average (pure rerank) | 65 | 4 | 1.0 | 300 | 0.843 | 0.894 |

**Total improvement vs cosine baseline: +16.7% relative ARI, +7.2% F1.**

Why average linkage wins: re-rank distances are more bimodal than cosine
(clusters of identical-content images have d_rerank ≈ 0; everything else is
pushed toward 1.0). Ward's variance-minimisation fights this; UPGMA's
size-weighted mean follows it.

Robustness sweep showed ARI ≥ 0.825 across the entire window k1∈[60,80],
k2∈[4,8], λ∈[0.4,0.7] — defaults are not knife-edge tunings.

### DINOv3 separate-rerank blend (rejected from default)

Compute re-rank distance separately on (PE-G+colour) and on (DINOv3 CLS),
linearly blend the two distance matrices.

Best: ARI **0.850** with `0.3·cos + 0.5·rerank_pec + 0.2·rerank_dino`,
average N=300. **+0.5% ARI over single-rerank.**

Trade-off: 2× the precompute cost (compute rerank twice on different
feature spaces, ~6s instead of ~3s on 7608 images), 2× the cached matrix
storage. Not worth shipping by default. Implementation lives at
`scripts/experiments/rerank_dual.py` for future reference.

---

## Final benchmark summary (Evie Lee, 7608 images, 212 groups)

| Method | n_clusters | ARI | F1 | Δ ARI |
|---|---:|---:|---:|---:|
| Cosine + Ward (old default) | 200 | 0.724 | 0.835 | — |
| HDBSCAN mcs=10 | 2389 | 0.710 | — | -1.9% |
| Adaptive merge (best) | 253 | 0.628 | — | -13.3% |
| SNN k=75 α=0.5 (Ward) | 200 | 0.757 | 0.846 | +4.6% |
| k-reciprocal SNN k=100 α=0.3 (Ward) | 200 | 0.766 | 0.855 | +5.8% |
| Re-rank R* k1=65 k2=4 λ=0.7 (Ward) | 200 | 0.839 | 0.891 | +15.9% |
| **Re-rank R* k1=65 k2=4 λ=0.7 (Average)** | **300** | **0.845** | **0.895** | **+16.7%** |
| Re-rank + DINOv3 blend (Average) | 300 | 0.850 | 0.900 | +17.4% |

The shipped default is the bolded row.

---

## Lessons / heuristics for future work

- **Validate ground truth before iterating.** The first half-day was wasted
  on Lily Lou's wrong group boundaries. Re-running the diagnostic with
  proper labels flipped multiple conclusions.
- **Test discriminators on the actual failure pairs.** Aggregate metrics
  (ARI) hide which kinds of mistakes a method makes. Looking at the
  confusion in cluster 72 (anime cosplay) directly told us which signals
  helped and which didn't.
- **Re-ranking is well-studied in person re-id.** When stuck, look for prior
  art in adjacent fields. The Zhong 2017 algorithm was published in a
  context (cross-camera person matching) with the same fundamental problem
  (bridge images in kNN graphs).
- **Linkage method matters as much as distance.** Average vs Ward made the
  same distance matrix differ by ~0.6% ARI. Worth testing both whenever a
  new distance is introduced.
- **Don't trust "average" performance for parameter selection.** The
  relevant question is whether the params are *robust* across reasonable
  values, not whether one specific tuning beats another. Wide robustness
  windows (as we got for k1, k2, λ) mean the default works on unseen
  datasets too.
- **Negative results matter as much as positive ones.** Mutual reachability,
  iterative re-ranking, consensus re-rank, mixed-granularity adaptive merge,
  spatial colour, foreground masking — every one of these had a plausible
  story but didn't beat the simpler approach. Documenting them prevents
  re-doing the same experiments.

---

## Open questions for future work

1. **Does re-rank generalise to scoped clustering?** Currently disabled
   there because the Rust tool's `--filenames` and `--dist-matrix` flags
   are mutually exclusive. Worth checking whether reranking on the subset
   improves quality enough to motivate fixing that.
2. **Approximate kNN past 30k images.** The dense `n × n` similarity matrix
   becomes the bottleneck. faiss/hnswlib for the kNN graph, then standard
   re-rank from there.
3. **Auto-tuning k1.** The robustness window is wide, but extreme datasets
   (hundreds of single-image groups, or one giant group) might benefit from
   `k1 ≈ 2 × median(confirmed group size)`. Cheap to implement, low
   priority.
4. **Per-cluster expand-with-density-budget UI.** Algorithmic ceiling on
   global metrics is approached at ARI ~0.85-0.87 because some groups are
   genuinely inseparable from this signal. The remaining gap should be
   closed in UI rather than algorithm — give the user a button to expand
   any cluster by its own density budget.
5. **Watch for regressions on Lily Lou and other large-group folders.** The
   benchmark improvements were verified on Evie Lee. Lily Lou recovered
   126/126 of its Christmas set with 3 extras at N=50 in our smoke test,
   but the rest of that folder is unlabelled — extended use should
   surface any regressions on dataset shapes we haven't covered.
