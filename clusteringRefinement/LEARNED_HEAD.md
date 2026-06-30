# Learned Projection Head

A small MLP trained on labeled photoshoot data that maps PE-G (1280d) + color (693d) → 512d, deployed as a **3-seed ensemble**. Blended with zero-shot PE-G + color distances at clustering time. Improves cluster ARI by roughly +0.11–0.13 on held-out models compared to baseline (single head), plus ~+0.010 from the ensemble.

## How it's used at runtime

The head lives at `~/.cache/reorder/learned_head.pt` (+ `learned_head_s43.pt`, `learned_head_s44.pt` for the other ensemble seeds) + `learned_head.json` (with a content-derived version string covering all heads, and a `head_files` list). Three places integrate it:

1. **`scripts/extract_features.py`** runs `_maybe_update_learned_proj` at the end of every extraction. It pushes the (PE-G, color) features through **every** head in `head_files`; the per-head L2-normed blocks are concatenated and scaled by 1/√n_heads, so rows stay unit-norm and their dot product equals the ensemble-MEAN cosine — the downstream blend is unchanged. The result is the (N, 1536) `learned_proj` in `.reorder-cache/embeddings_hash_cache.npz` with a `_v_learned_proj` version key. If the head's version differs from what's stored, the projection is recomputed. If the head isn't installed, the step is silently skipped. (The current deployed head is a **3-group split ensemble** — joint `learned_proj` + `learned_proj_peg` + `learned_proj_color`, each 3 seeds — and the joint group has an optional **L44 attn-pool layer-augmented variant** used when the layer is fully extracted; see "Split single-modality heads" and "Production integration" below.)

2. **`rust/cluster-tool`** exposes `--learned-proj-weight`. It reads the `learned_proj` array from the NPZ alongside the other model embeddings and blends them as one more weighted component in the combined feature vector.

3. **Cluster UI** (`ClusterToolbar.tsx`) has a "Learned head" slider in the weight panel, range 0–1. The slider value is the **target fraction of the final cosine signal**, not a raw concat weight — the server rescales it (see `rescaleLearnedProjWeight` in `src/cluster/pipeline.ts`) so that "0.60" means "learned head contributes 60% of the clustering distance, regardless of how PE-G and color are weighted." Default is **0.60**, the optimum for the deployed 3-seed ensemble (blend curve peaks at 0.60 = 0.8172 eval-23 vs 0.8150 at 0.65; same optimum as the single 512-dim head). The earlier 256-dim head peaked at 0.45. Set to 0 to disable the head entirely. Set to 1 to use only the head (PE-G and color contributions zeroed).

## Retraining (e.g. when you add a new labeled dataset)

The training script is `clusteringRefinement/train_final_head.py`. It trains on all configured datasets (no LOMO holdout) with the winning hyperparams.

Pick whichever fits your workflow:

```sh
# 1. Default — uses the hardcoded dataset list
python clusteringRefinement/train_final_head.py

# 2. Per-run override
python clusteringRefinement/train_final_head.py \
    --dataset M1:/path/to/Bench1 \
    --dataset M2:/path/to/Bench2 \
    ...

# 3. From a config file (one path per line, # for comments)
echo /Users/me/PicsStaging/Bench1 >> ~/.config/reorder/training_datasets.txt
echo /Users/me/PicsStaging/Bench2 >> ~/.config/reorder/training_datasets.txt
python clusteringRefinement/train_final_head.py
```

Prerequisites for any training dataset:
- `.reorder-groups.json` with labeled groups
- `.reorder-cache/embeddings_hash_cache.npz` containing `pecore_g` and `color`
- `.reorder-cache/pecore_g_views.npy` + `color_views.npy` (pixel-aug views) — produced by `clusteringRefinement/pixel_aug start`. Optional but recommended; the head trains fine without them, just slightly less well.

After retraining the version string in `learned_head.json` changes. Next time `extract_features.py` runs on any dataset, it detects the mismatch and refreshes that dataset's `learned_proj` slice — automatically, no manual cache wipe.

To run a deeper LOMO sweep (rather than just training one final head) use `clusteringRefinement/train_projection_head.py` directly with the `--train` flag set to a subset.

## Results summary (the journey)

Each line is "training-set size → best fixed-blend avg Δ-ARI across LOMO test datasets":

| stage | training data | augmentation | best blend | avg Δ-ARI | datasets ≥+0.10 |
|---|---|---|---|---|---|
| N=2 (initial) | M1+M2 | none | 50% | +0.045 | 0/2 |
| N=6 | M1-M6 | none | 30% | +0.026 | 1/6 |
| N=8 | +M11, M12, P=32 | none | 30% | +0.035 | 1/8 |
| N=12, ICOMB | +M3-M10 | mixup+drop-color+cross-mixup+hard-neg | 40% | +0.096 | 6/11 |
| N=12, +pixel-aug | all + 3 aug views/img | +pixel-aug | 40-50% | +0.099 | 5/11 |
| **N=20, high-capacity** | **+M13-M20** | **+pixel-aug** | **60%** | **+0.113** | — |

The N=20 row is a full 20-dataset LOMO snapshot taken **before the last 6 augmented datasets** (avg ARI 0.7681 at blend 0.60 vs zero-shot 0.6556 = +0.113); the fully-deployed config with all 17 augmented datasets is 0.7717 = +0.116 (see pixel-aug section below). It folds in two changes: more data (M13-M20) and the high-capacity head (out_dim=512, temperature=0.07, lr=1e-4 + grad-clip). The hyperparameter change alone, holding data fixed, was +0.0084 avg ARI over the prior 256/0.1/3e-4 head and won 16/20 datasets — but **only with the blend raised to 0.60**; at the old 0.45 blend the gain shrinks to ~+0.002.

Final deployed config (currently saved): **P=32, K=12, ep=12 + cosine LR decay, out_dim=512, temperature=0.07, lr=1e-4, grad-clip=5.0, ICOMB augmentations, pixel-aug enabled, blend at 0.60**. (ep=12+cosine replaced ep=15+constant in 2026-06: paired 26-fold LOMO Δ +0.0061 @ seed 42 / +0.0005 @ seed 43 — never worse, 20% less training compute. Cosine @ 10 ep was neutral, +0.0008; cosine @ 15 ep +0.0023.)

### Pixel-aug: isolated contribution (controlled A/B)

Measured directly on the deployed config by toggling `--use-augmented-views` on/off for the *same* 20-dataset LOMO, same seed, same disk state (so the only variable is pixel-aug; the ICOMB feature-space augmentations stay active in **both** arms). Scored at blend 0.60.

| condition | avg ARI @0.60 | Δ vs zero-shot |
|---|---|---|
| zero-shot (no head) | 0.6556 | — |
| no pixel-aug (views ignored) | 0.7613 | +0.1057 |
| full pixel-aug (17 datasets w/ views) | 0.7717 | +0.1161 |

**Pixel-aug's isolated contribution: +0.0104 avg ARI** (12/20 datasets improve; comfortably above the 20-dataset averaging noise floor). This decomposes cleanly — the three states sum:

| step | avg ARI @0.60 | Δ |
|---|---|---|
| no pixel-aug | 0.7613 | — |
| + views on the original 11 datasets (M1-6, M8-12) | 0.7681 | +0.0068 |
| + views on 6 more (M13, M16-20) | 0.7717 | +0.0036 |

Notes: this is the *standalone* value of pixel-level augmented views on top of an already-ICOMB-augmented head — consistent with the historical N=12 table (+pixel-aug added ~+0.003 over ICOMB alone; with 17 augmented datasets it is now ~+0.010). Per-dataset deltas are single-seed-noisy (training-seed swings reach ±0.01-0.02 on MPS); only the 20-dataset average is reliable. Diminishing returns: each added batch of augmented datasets contributes less. Reproduce with `clusteringRefinement/run_deployed_lomo.sh` (`USE_AUG=0` for the no-aug arm); result snapshots in `lomo_noaug.tsv` / `lomo_post_pixelaug.tsv`.

### Per-dataset color preference (PE-G / color ratio)

The color weight that gave each dataset its best ARI (learned blend fixed at 0.60, PE-G weight fixed at 1.0; `color_w=0` is PE-G-only, larger is more color). Deterministic re-scoring, no training. Global best average is `color_w≈0.8`; deployed default is 0.7.

| dataset | best color_w |
|---|---|
| M2 sarah | 2.0 |
| M17 dusha | 1.5 |
| M16 zoe | 1.25 |
| M20 salome | 0.8 |
| M10 alina | 0.8 |
| M5 lily | 0.7 |
| M6 sabrina | 0.7 |
| M1 austin | 0.6 |
| M4 mia | 0.4 |
| M8 evie | 0.4 |
| M19 andreea | 0.4 |
| M3 eva | 0.3 |
| M9 darshelle | 0.2 |
| M12 anna | 0.2 |
| M11 amanda | 0.0 |
| M13 hunny | 0.0 |
| M18 railey | 0.0 |

Quick guide for manual slider tuning per shoot:
- **Color up (≥1.0):** sarah, zoe, dusha
- **Color off (PE-G only, 0.0):** amanda, hunny, railey
- **Light color (0.2–0.4):** eva, mia, evie, darshelle, anna, andreea
- **Near the 0.7 default:** austin, lily, sabrina, alina, salome

autumn and the two partial-label sets (vixen, verity) were excluded. Sweep: `clusteringRefinement/run_pegcolor_sweep.sh`; full matrix in `/tmp/pegcolor_sweep/per_dataset.tsv` (purged; superseded by `colorw_oracle_ens3.tsv`, see next section).

**Caveat (added with the auto-color-w study below): this table is roughly half noise.** Re-deriving it on the v26 folds against a different head seed moves most of the "best color_w" entries — per-dataset gain curves correlate only 0.28 (mean) across training seeds. The only preferences that survive cross-seed validation are **sarah (M2), dusha (M17), andreea (M19): color up** and weakly hunny (M13): color down. Treat the rest of the table as within-noise.

### Auto color-weight selection from images alone — falsified

Goal: pick the per-dataset color weight automatically in production (no labels, no
known group count, images only). Benchmarked on the 24 v26 LOMO folds against the
deployed config (3-seed ensemble, blend 0.60, Ward, oracle-N scoring); selectors
were always given the *oracle-N generosity* they wouldn't have in production, so
these numbers are upper bounds. Eval-23 baseline fixed\@0.7 = 0.8172.

**The ceiling itself is mostly seed noise.** Per-dataset oracle over the 12-point
grid = 0.8262 (**+0.0091**, the apparent prize; a 3-way {0.2, 0.7, 1.25} oracle
keeps +0.0053 of it). But rebuilding the oracle matrix against a held-out head
seed (r4) shows the target is unstable: mean per-dataset Δ-curve correlation
across seeds **0.28** (median 0.45), soft-best-w spearman 0.50. Oracle picks
taken from ens3 curves and scored on r4 capture only **+0.0046 of r4's +0.0142**
ceiling; picks taken from r4 and scored on the deployed ens3 get **−0.0013** —
i.e. labels-on-another-replica already loses to fixed 0.7. Even the *global*
optimum moves with seed (0.7 on ens3, 0.3 on r4). An image-based selector is
bounded by the seed-stable component, ≈ +0.005 at best for a perfect one.
Per-dataset extremes flip too: M7's headline +0.070 at color-off becomes
"wants 0.8" under r4.

Four selector families, all LOMO-honest (calibration on the other 23 datasets,
predict on the held-out one), all scored by lookup into the oracle matrix:

| family | best eval-23 Δ vs 0.7 | notes |
|---|---|---|
| dataset descriptors → w (`colorw_features.py` + `colorw_select.py`) | ≈ +0.002 | 16 unsupervised features (color↔proj/PE-G kNN agreement, color margin of semantic neighbors, dispersion, eff-rank…), kNN-over-gain-curves + linear maps, singles and pairs. Best of 60+ configs +0.002 = selection noise; best single-feature spearman vs best-w only 0.38 |
| held-out-modality internal validation (`colorw_internal.py`) | −0.0003 | DINOv3 CLS carries zero weight in the deployed blend → unbiased judge; score each w's Ward clustering by DINOv3 kNN-purity / pair-separation / silhouette. dino_knn −0.0015→−0.0003 across shrink margins; never positive |
| bootstrap stability (`colorw_stability.py`) | −0.0016 | Ben-Hur/Lange resampling: ARI agreement of Ward on overlapping 80% subsamples per w. −0.0051 at margin 0, 4W/14L; stability curves nearly flat |
| ridge over all criterion curves | +0.0015 | all internal criteria + stability + descriptors jointly, predict ΔARI(w); positive only after threshold tuning |

**Is the color channel signal or noise?** Signal — but the head has already
eaten most of it. Zero-shot (no head), color\@0.7 vs PE-G-only is **+0.0330**
eval-23 (18W/3L; up to +0.12 on M23/M21, but −0.11 on M26 and −0.06 on M5).
Under the deployed head+blend, the *residual* zero-shot color channel is worth
**+0.0062** on ens3 (16W/6L; +0.0036 at r4's own optimum w=0.3) — the head,
trained on PE-G⊕color, re-encodes ~80% of the zero-shot color value into the
projection. So: keep the channel on at 0.7 (consistently positive on average),
but per-dataset tuning of it is fighting over the small residual, which is why
the selection problem above is unwinnable.

Conclusion: **keep the fixed 0.7 default — it is exactly the global optimum on the
deployed ensemble.** The per-shoot preference signal that looks predictable in any
single replica's sweep table is two-thirds head-seed noise; what's stable is
concentrated in 2–3 shoots (sarah/dusha-style uniform-backdrop sets wanting more
color) and is served by the existing UI slider. This is the same lesson as the
shoot-context head, one level up: the adaptation signal is real but tiny, and
*the target itself* dissolves under reseeding before any unsupervised predictor
can reach it.

Artifacts: `colorw_oracle.py` (grid ARI matrix + cached Ward labels/trees per w;
`COLORW_ROOTS`/`COLORW_OUTDIR` env to score other seeds), `colorw_features.py`,
`colorw_select.py`, `colorw_internal.py`, `colorw_stability.py`; results in
`colorw_oracle_ens3.tsv`, `colorw_oracle_r4.tsv`, `colorw_features.tsv`,
`colorw_internal.tsv`, `colorw_stability.tsv`; per-w labels under
`~/.cache/reorder/colorw_auto*/`.

### Big-set training ablation & seed-noise floor

Big-set ablation: remove dataset(s) from the full pool, true LOMO over the 22 non-partial targets, deployed config @ blend 0.60 (`run_bigset_ablation.sh`; 10 arms / 203 folds, single seed 42). Avg Δ-ARI = ablation − full-pool baseline (negative = removing it lowered ARI):

| arm (removed, imgs) | avg Δ | | arm | avg Δ |
|---|---|---|---|---|
| −M3 eva (8023) | −0.0010 | | −M3,M8 | −0.0029 |
| −M8 evie (7345) | −0.0004 | | −M10,M9 | −0.0041 |
| −M2 sarah (5548) | +0.0030 | | −M2,M6 | −0.0046 |
| −M10 alina (5349) | −0.0006 | | −M3,M8,M2,M10,M9 | −0.0126 |
| −M9 darshelle (4955) | −0.0036 | | | |
| −M6 sabrina (4580) | −0.0033 | | | |

Seed-noise floor — same full-pool folds retrained at 5 seeds (42–46), `run_seed_noise.sh`:

| target | mean ARI | seed σ | seed range |
|---|---|---|---|
| M4 mia | 0.632 | 0.0101 | 0.028 |
| M19 andreea | 0.736 | 0.0120 | 0.033 |
| M22 alexis | 0.786 | 0.0073 | 0.020 |
| M10 alina | 0.874 | 0.0074 | 0.023 |
| M11 amanda | 0.902 | 0.0062 | 0.017 |
| M17 dusha | 0.846 | 0.0188 | 0.053 |

Single-fold seed σ ≈ 0.006–0.019 (mean 0.010). Propagated to a paired LOMO Δ averaged over ~20 datasets: 1σ ≈ ±0.0032, 95% ≈ ±0.0062.

Observations:
- Every single- and pair-removal avg Δ falls inside the ±0.0062 (95%) noise band; only removing the 5 biggest together (−0.0126) is outside it.
- The largest single dataset (eva, 8023 imgs) is among the near-zero effects; removal Δ shows no ordering by image count in this run.
- −M2 (sarah) removal was positive in two independent runs (+0.0030 here, +0.0031 in the earlier M2 LOMO).
- Reseeding the −M9 arm: avg Δ over 6 targets is +0.0020 ± 0.0029 across seeds 42–46 (single-seed value was −0.0036) — straddles zero.
- To resolve a ~0.003/dataset effect above this floor, average each fold over ~5 seeds (Δ-noise → ±0.0014), ~5× the compute.

## What worked

- **Augmentations that simulate within-shoot diversity** — drop-color (analog of "background masking" — color histograms encode backdrop), cross-mixup with soft SupCon labels (smooths between-group boundaries), and pixel-augmented views during training.
- **More positives per anchor (K=12 over K=8)** — K=12 was the sweet spot at our data scale; K=20 broke because too many groups have <20 images.
- **More training data**, with diminishing returns per added dataset.
- **Heavier blend at inference for the high-capacity head (60%)** — the 256-dim head peaked at 45% and regressed hard at 100%; the 512-dim head peaks at 60% and its blend curve is far flatter on the right, so its optimum sits at a lighter zero-shot weight.
- **High-capacity head — out_dim=512 + temperature=0.07 + lr=1e-4 together.** None of the three helps alone; it's an interaction. The bigger, sharper head needs the gentler LR (and grad-clip) to avoid overfitting/divergence, and at N≥20 there's finally enough data for the extra capacity to pay off.

## What didn't work

- **DINOv3 features as head input** — net regression vs PE-G + color alone. Zero-shot DINOv3 CLS is fine; the head can't use the 7×7 patch tensor productively.
- **More than 15 epochs** — pretty much always made things worse (overfit) even with augmentations.
- **Larger output dim (512)** — overfits at N≤8. *(Update: at N≥20 it became part of the winning config — see "What worked". Capacity needs both data scale and the lower LR to pay off.)*
- **ArcFace loss** — neutral or slightly worse at our data scale.
- **Excluding M7 from training** — initially looked promising on M3+M4 but evened out across all 11 datasets. M7-style datasets (very large groups) are outliers the head can neither help nor be hurt by much.
- **Blend-aware loss** (`--blend-aware`, default off) — train the SupCon loss on the *blended* (learned + zero-shot) similarity the pipeline scores on, with train-time blend weight = inference blend (0.60), so the head optimises the residual rather than re-learning zero-shot structure. Regressed; kept as an opt-in knob in `train_projection_head.py` for future variants (learnable `w`, residual-only-on-hard-pairs, smaller blend-weight).

  What we ran: full 20-dataset LOMO, deployed config (512/0.07/1e-4 + grad-clip, full pixel-aug), same seed as the control, scored across a blend grid. Numbers (avg ARI over 20 datasets):

  | blend | deployed (control) | blend-aware |
  |---|---|---|
  | 45% | 0.7679 | 0.7597 |
  | 50% | 0.7711 | 0.7627 |
  | 60% (deployed) | **0.7717** | 0.7606 |
  | 65% | 0.7696 | **0.7636** |
  | 100% (pure head) | 0.7583 | 0.7416 |

  At blend 0.60 the per-dataset diff was 7 up / 13 down, avg **−0.0111**. Each config at its own best blend: control 60%→0.7717 vs blend-aware 65%→0.7636 (**−0.0081**), so the regression isn't just a wrong-blend artifact. Note the dev set (M2/M11/M19) gave **+0.0041** — opposite sign from the full LOMO, since 2/3 of those happen to be sets where it helps; a reminder to confirm on the full LOMO. Results in `lomo_blendaware_full.tsv`; reproduce with `EXTRA_ARGS="--blend-aware --blend-weight 0.6" run_deployed_lomo.sh` + `run_blend_curve.sh`. We did not investigate why it regressed.

- **De-duplicating self-pairs in the SupCon loss** (`--dedup-self-pairs`, default off) — when a group has fewer than K images the `PKSampler` draws with replacement, so the same image lands at multiple batch positions. The diagonal self-mask only zeros the literal `i==i` entry, so those off-diagonal duplicates survive as perfect (cos=1) positives that also dominate the softmax denominator (`exp(1/τ)=exp(14.3)≈1.6M` at τ=0.07). The flag masks **all** same-source-image pairs from both the numerator and the denominator. The pathology is real and frequent — **59% of eligible groups have <K=12 images** — but it doesn't actually destabilize training (no divergence even with `--grad-clip 0`), so fixing it changed nothing measurable.

  What we ran: full 20-dataset LOMO, deployed config (512/0.07/1e-4 + grad-clip, full pixel-aug), same seed as the control, scored @ blend 0.60. Avg ARI:

  | metric | control | dedup-self-pairs | Δ |
  |---|---|---|---|
  | full-20 | 0.7702 | 0.7678 | **−0.0025** |
  | 17-set (excl. M7/M14/M15) | 0.8052 | 0.8066 | **+0.0014** |

  Within noise (±0.01–0.02 single-seed floor), 8 up / 11 down / 1 tie; the full-20 spread is dominated by the partial-label / large-group outliers (M14/M15/M7). A grad-clip × dedup 2×2 (full LOMO, aug) confirms the two are **weakly-redundant minor stabilizers**, not safety nets: dedup's full-20 Δ flips from −0.0024 (clip on) to **+0.0044** (clip off), and dropping grad-clip costs −0.0062 with dedup off but +0.0007 with dedup on — each partially substitutes for the other, but no arm beats the deployed clip-on baseline (0.7702). The outlier swings are noise, not mechanism: dedup *hurts* M14/M15 under clip-on yet *helps* them (M14 +0.067) under clip-off. Kept as an opt-in knob. Results in `dedup_ab_full_{baseline,dedup}_aug.tsv`; reproduce with `EXTRA_ARGS="--dedup-self-pairs" USE_AUG=1 run_deployed_lomo.sh` (add `--grad-clip 0` for the no-clip arm).

- **Transductive mean-centering** — subtract each shoot's own per-modality mean embedding before the cosine blend (`center_eval.py`, 17-set). Helps zero-shot (+0.029 full-20) but not the deployed head+ward config: 17-set Δ −0.014 (center peg+color) to −0.005 (center all three); full-20 +0.001 (outlier-driven). Shrinkage PCA-whitening strictly worse (−0.06 @ b=0.60). `center_eval.tsv`.

- **Foreground/background color split** — person-mask each image (torchvision DeepLabV3), compute the 693-d color histogram over fg and bg pixels separately, blend as extra modalities (`extract_fgbg_color.py`, `fgbg_color_eval.py`). Best variant +0.0007 (replace color with fg+bg) / +0.0009 (add bg as 3rd channel) on 17-set @ b=0.60 — within seed noise. Sweep zeroed the fg-color weight; bg-only color ≈ full color. `fgbg_color_eval.tsv`; sidecar caches at `.reorder-cache/fgbg_color_cache.npz`.

- **Pairwise boundary-pair verifier** — learned MLP judges "same set?" for each image's k=40 nearest-neighbor edges, soft-blended into the distance (α=0.3) before Ward; LOMO 17-set (`pairwise_verifier_gate.py`). Pooled peg/color/proj features: 87.2% edge accuracy (+0.9% over the bi-encoder distance threshold on the same edges), end-to-end ARI 0.8046→0.735 (−0.069). Stronger training (richer `[cos,|a−b|,a*b]` features, 1024-wide net, 3-seed ensemble, val early-stopping; `improved_verifier.py`): 88.0% accuracy (+2.0% margin), ARI −0.076. `pairwise_verifier_gate.tsv`.

  Bounds for this approach (`oracle_ceiling.py`, `verifier_accuracy_sweep.py`): a *perfect* verifier on each image's k nearest edges ceilings 17-set ARI at +0.05 (k=5) → +0.14 (k=40) → +0.16 (k=80). Synthetic verifiers (k=40, α=0.3): ~95% edge accuracy needed to break even and ~97% for +0.05 when errors fall on the bi-encoder's hardest edges; 87% already gives +0.07 if errors fall on random edges.

## 26-dataset era: v26 LOMO baseline & the post-processing dead ends

A fresh 24-fold LOMO (all registered datasets minus the M14/M15 partials, which
stay in every training pool) at the deployed config lives in **persistent**
fold roots — `~/.cache/reorder/lomo_v26{,_r2,_r3,_r4}` (seeds 42/43/44/45;
`/tmp` roots get purged by macOS, which is how the old `/tmp/lomo_postaug` died;
`lomo_common.py` now reads `$LOMO_ROOT`, defaulting to the v26 root). Baseline:
**eval-23 mean 0.8128** (full-24 0.8027 incl. M7) for the seed-42 replica;
expected single-seed head over 4 replicas = **0.8073** (seed 42 was lucky).
`lomo_v26_deployed.tsv`.

What got tried on top (all scored on these folds, python Ward@oracle-N):

- **Over-cluster → merge-back** (`overcluster_merge_eval.py`): cut the ward tree
  at m·N, greedily re-merge to N by a robust pairwise statistic (median/q25/q75/
  mean/max/density-normalized median — the merge-suggestions-style score).
  **Falsified hard**: best config −0.085 eval-23, 1–2/23 wins, every m/crit
  negative. Only M5 (+0.19) and M7 (+0.15) improve — the two lowest-baseline
  sets — so it's at most a per-dataset fallback. Ward's variance objective
  orders late merges better than any same-signal pairwise statistic.
  `overcluster_merge_v26.tsv`.
- **Boundary polish** (`polish_eval.py`): keep Ward@N, reassign images to their
  best-fit cluster (top-k-neighbor score, margin-gated, few iterations). Null:
  best −0.0007, losses (small-group sets) outweigh wins. Full convergence
  drifts toward spherical-kmeans (much worse than ward) as predicted by
  `algo_comparison.tsv`. `polish_v26.tsv`.
- **Shoot-context head** (`--shoot-context` in the trainer: dataset-mean feature
  vector appended to every input row, per-sample `--ctx-dropout 0.3`): −0.0040
  avg, huge per-dataset variance (M17 +0.052, M12 −0.050). The per-shoot
  adaptation signal the color-weight table shows is real, but a mean-vector
  context doesn't capture it. Kept as an opt-in knob. `lomo_v26_ctx.tsv`.

Lesson (rhymes with the verifier ceiling work): **re-processing the same
distance matrix doesn't pay — only new signal does.**

### What does pay: seed-ensemble + TTA (deployable)

- **Seed-ensemble** (`seed_ensemble_eval.py`): average the proj similarity over
  heads trained at different seeds. ens3 = **0.8172 eval-23, +0.010 vs the
  expected single-seed head** (+0.004 vs the lucky seed-42 one). ens4 ≈ ens3.
  Production shape: train the final head at 3 seeds, cache 3 projections,
  mean the similarities. `seed_ensemble_v26.tsv`.
- **TTA over the pixel-aug views** (`tta_eval.py`, `stack_eval.py`) — measured
  but **NOT deployed**: at inference, mean each image's L2-normed projection
  over base + the K=3 cached views (per head). +0.0047 alone (12/19 wins);
  stacked on ens3 at blend 0.65 → 0.8216 eval-23 (+0.014 vs expected-seed).
  Why not deployed: TTA needs views on the **clustered** folder, and views are
  only ever extracted for benchmarks — real folders would silently get
  ensemble-only while benchmarks scored ~+0.005 higher, making every future
  benchmark number overestimate production. Pixel-aug stays a *training-time*
  technique. `tta_v26.tsv` / `stack_v26.tsv`; the ens-only blend optimum is
  **0.60** (0.8172; 0.65 → 0.8150), which is the shipped default.

### Intermediate PE-G layers → DEPLOYED (L44 attn-pool into the joint head)

Resolution below; bottom line: **L44 attn-pool fed to the joint head, +0.006 ARI
in the deployed 3-head blend, shipped with a toggle + automatic fallback** (see
"Definitive sweep" and "Production integration").

Probe (`probe_pe_layers3.py`, L30–49 × mean/max/gem3/attnpool, fp32 attn-pool,
4 datasets, features cached in `~/.cache/reorder/pe_probe3/`): solo zero-shot
peaks at **L47/L48-attnpool (0.619/0.628 vs final-proj 0.608)** and L44–46
mean/gem3. As a **zero-shot blend component** on the 4 datasets with clean
cached extractions (M1/M3/M5/M6, `pe_layers_zs_eval.py`):
**L47-attnpool @ w=0.5 = +0.0095 under the deployed head+ward config, 4/4
wins** (every layer/weight combo tested won 4/4). The layer features carry
signal the head doesn't extract.

Paired mini-LOMO on those 4 sets (`run_pelayer_minilomo.sh` +
`score_pelayer_minilomo.py`, 2 seeds, 3-set train pools): layer-as-head-input
(`--pe-layer 47:attnpool`) +0.0041 vs control, zs-side blend +0.0020,
both-at-once ≈ 0 (they double-count). Small-pool numbers — the decisive test
is the full 24-fold LOMO with `--pe-layer`, gated on extraction.

**Extraction state**: DONE on all 27 datasets (a 27th, M27 mvngokitty, was
added). The extractor only encodes **grouped** images (the head trains on those
only), so ungrouped images are legitimately zero-filled — `nonzero rows ==
grouped count` is the correctness check, NOT zero-row count. M1/M2/M3/M5/M6/M7
were done by the older all-images extractor (superset, also fine). Verify with
the grouped-vs-nonzero audit in this session's notes, not a naive zero-row
threshold.

### Full-scale verdict (the clean-4 preview did NOT hold)

**zs-blend, FALSIFIED** (`pelayer_lomo_eval.py`, `pelayer_lomo_zsblend.tsv`):
reuse the v26 folds, add L47-attnpool to the zero-shot side, score grouped-only
(ungrouped images have no layer features — clustering them with a zero block is
a benchmark artifact absent in production, so grouped-only is the faithful
test). Over **22 datasets** (M20 dropped — its v26 fold went stale when the
dataset was renamed/re-extracted): best weight w=1 → **+0.0029 mean, 13/22
wins, median +0.0034** — inside the ±0.0062 95% noise band. Several datasets
clearly hurt (M24 −0.043, M23 −0.028, M10 −0.013). Oracle per-shoot weight
ceilings at +0.0058. The clean-4 (M1/M3/M5/M6) +0.0095 was regression to the
mean — those four happen to be favorable, fully-grouped sets.

**head-input — REAL, modest** (`run_pelayer_headinput_lomo.sh` +
`score_pelayer_headinput.py`, `pelayer_headinput.tsv`): paired ctrl-vs-`--pe-layer
47:attnpool` full LOMO, 27-roster, seed 42, scored grouped-only. **mean Δ
+0.0088, median +0.0079, 16/24 wins**; bootstrap 95% CI **[+0.0005, +0.0172]**
(excludes 0). Robust to dropping the big winner (M27 +0.074 → mean +0.0059,
median +0.0079) or the big loser (M24 −0.051 → mean +0.0114); 7 datasets up
>0.015 vs only 1 down. So the head-INPUT integration works where the zs-blend
didn't — the head *learns to use* the layer (different mechanism), worth
~**+0.006–0.009 ARI** (single-seed; confirm multi-seed before shipping).

**Production cost — the real gate**: the layer comes from the SAME PE-G forward
pass that already runs in `extract_features.py`, so capturing the layer in the
deployed MLX path (`scripts/mlx_pe_core`, `forward_capture`) is near-zero extra
compute — BUT the deployed pe_layers *extractor* currently encodes grouped
images only; a fresh unlabeled folder has no groups → extracts nothing. Shipping
needs all-image extraction + a head retrained at 3509-d input (1973 + 1536).

### Definitive sweep: layer × pooling × placement in the 3-head prod blend

The head-input result above was vs the *single-joint-head* baseline at one seed.
The deployment question is whether it survives against the real **3-head ensemble
blend** (joint .30 / peg .55 / color .15, zero-shot zeroed) at the deployed
12-epoch/cosine config. Full sweep (`run_pelayer_splithead_sweep.sh` + the
`pelayer_sweep` controller; `score_pelayer_splithead.py`,
`pelayer_splithead12.tsv`): **regenerated** all baselines at 12/cosine + every
layer-augmented variant, batched 3-seed MLX, 25 folds (incl. M27; only the two
partials excluded), grouped-only scoring. **675 trained folds = 27 jobs × 25.**

Baseline (regenerated 3-head blend): **0.8104**. Δ vs baseline by config:

| layer | pool | joint | peg | both |
|---|---|---|---|---|
| L42 | attnpool | **+0.0070** | +0.0040 | +0.0063 |
| L44 | attnpool | +0.0064 | +0.0041 | **+0.0070** |
| L46 | attnpool | +0.0040 | **+0.0069** | +0.0049 |
| L47 | attnpool | +0.0032 | +0.0042 | +0.0063 |

Pooling averaged over the 4 layers (placement): **attnpool** joint/peg/both =
+0.0052/+0.0048/+0.0061; mean = +0.0026/+0.0037/+0.0038; gem3 ≈ 0. **attnpool
wins at every layer.** Robustness (bootstrap 95% CI of the mean Δ over 25 folds):
only these clear zero — ranked by win-rate:

| config | meanΔ | wins | 95% CI |
|---|---|---|---|
| **L44_attnpool → joint** | +0.0064 | **20/25** | [+0.0029, +0.0101] |
| L42_attnpool → joint | +0.0070 | 18/25 | [+0.0031, +0.0113] |
| L46_mean → peg | +0.0048 | 17/25 | [+0.0007, +0.0095] |
| L46_attnpool → peg | +0.0069 | 16/25 | [+0.0007, +0.0143] |

The `both`-placement / L44_attnpool|both configs have higher raw means but CIs
that **include 0** — injecting into two heads doubles the variance without
reliable gain. Single-head **joint** placement is the safe pick. The gain
concentrates on weak shoots: L44_attnpool|joint averages **+0.0133 on the 7
low-baseline (<0.78) sets** (M27 +0.073, M4 +0.056, M16 +0.038, M23 +0.036) vs
**+0.0038 on the 18 stronger ones** — and unlike `both`, it doesn't hurt the
high-baseline sets (M6/M10/M18 stay flat-to-positive).

**Deployed: L44 attn-pool → joint head.** +0.0064 ARI, 20/25 wins, CI excludes 0.

### Production integration (toggle + automatic fallback)

Wired through end-to-end:
- **`train_final_head.py`** trains a parallel layer-augmented joint head
  (`learned_head_layer{,_s43,_s44}.pt`, input 3509 = peg 1280 ⊕ color 693 ⊕
  L44-attnpool 1536) alongside the plain joint head, and records `pe_layer`,
  `pe_layer_input_dim`, `pe_layer_head_files` on the joint group + a top-level
  `use_pe_layer` toggle. `PE_LAYER = "44:attnpool"`; `--no-pe-layer` disables.
- **`extract_features.py`** (`_compute_learned_proj`) uses the layer head **only
  when the layer is extracted FULLY for the folder** (`_load_full_pe_layer`:
  array present, all N rows, no zero rows) AND the toggle is on; otherwise it
  transparently falls back to the plain joint head. Output is the same (N, 1536)
  `learned_proj` either way, so the Rust blend is unchanged. The layer decision
  is encoded in `_v_learned_proj` (`…+peL44:attnpool` marker) so a toggle flip or
  newly-complete extraction re-projects the cache; idempotent otherwise.
- **Toggle**: config `use_pe_layer` (default true) overridable by env
  `REORDER_USE_PE_LAYER=0/1`.
- **In-pass capture (production, free)** — `extract_features.py` captures the
  layer during the SAME MLX PE-G forward it already runs: `PECoreBigG.
  forward_and_layer` returns the final embedding + the L44 attn-pool pooled
  features in one pass (no recompute), via the same `_pool_tokens` the head
  trained on (verified **bit-identical** to the offline features, cosine
  1.00000). It's stored in `embeddings_hash_cache.npz` as `pe_layer` + a
  `_v_pe_layer` spec key, riding the **content-hash cache** like the models — a
  pre-extraction snapshot survives the intermediate checkpoint rewrites, so
  incremental re-extracts keep full coverage. The layer is captured only when
  the installed head declares one (`_deployed_pe_layer`) and the MLX backend is
  used; the torch path captures nothing → plain-head fallback. **Fresh
  production folders get the layer automatically — no extra step.**
- **`extract_pe_layers.py --all-images`** is the offline/benchmark fallback
  (default is grouped-only, for training sets); `_compute_learned_proj` reads
  the npz `pe_layer` first, then this `.npy`. Mode is tracked in
  `pe_layers_meta.json` so a grouped→all switch restarts cleanly.

Verified end-to-end: fresh pecore_g extract → in-pass capture (111/111),
bit-identical to offline, layer used (`+peL44:attnpool`); cache-valid re-run →
idempotent no-op, preserved; incremental other-model re-extract → `pe_layer`
carried through the snapshot (111/111); grouped-only / partial → fallback;
`REORDER_USE_PE_LAYER=0` → fallback; Rust linkage reads the npz with the extra
keys cleanly (ARI unchanged). The deployed head was retrained with the layer
(version `…c9829bbaa16db0ed`); prior heads backed up under
`~/.cache/reorder/backup_pre_layer_*`.

Caveat — benchmark scoring is now *mixed*: fully-grouped benchmark folders use
the layer, partially-grouped ones fall back (grouped-only extraction leaves zero
rows). That's a benchmark artifact; production with `--all-images` uses the layer
uniformly. The honest deployed-blend gain is **+0.006 ARI**, concentrated on
hard/weak shoots — small but real (CI excludes 0), and free at inference once the
layer is captured in the existing PE-G pass.

## Split single-modality heads — the first clear win since the seed-ensemble

Follow-up to the auto-color-w study's "the joint head eats ~80% of the zero-shot
color signal": train **separate heads per modality** (`--input-mods peg` /
`--input-mods color` in `train_projection_head.py`; same deployed recipe, except
the color head drops `--drop-color-prob` — it would zero its entire input) and
blend the head *similarities* with free weights. v26-style 24-fold LOMO, seeds
42/43/44 per modality, fold roots `~/.cache/reorder/lomo_{pegonly,coloronly}{,_s43,_s44}`.

Single-seed sweep over (zs_peg, zs_col, joint, peg-head, color-head) weight
vectors (`split_head_blend_eval.py`; configs overridable via `SPLIT_CONFIGS_JSON`,
ensemble roots via `SPLIT_{PEG,COL,BOTH}_ROOTS`):

- **The peg-only head alone (0.8208) beats the whole deployed blend (0.8128)**;
  the joint head alone is worse than that (0.8061). Trained jointly, the head
  trades PE-G fidelity for color encoding.
- The color head is worthless solo (0.49) but adds ~+0.01 as a **10–20% channel**;
  beyond ~25% share it degrades fast.
- Independent zs_peg × zs_col factorial on top of the best pair: tiny dashes
  (≤4% raw peg, ≤2% raw color) are within-noise positive; anything more dilutes.
  The two heads absorb essentially all raw-signal value. `split_head_zsgrid.tsv`.
- The whole winning region is a flat plateau (pair ratio 92:8 → 80:20, with or
  without joint head / tiny zs, all 0.830–0.833 single-seed). `split_head_blend.tsv`,
  `split_head_refine.tsv`, `split_head_ratio.tsv`.

**Ensemble-vs-ensemble validation** (every head a 3-seed sim-mean, vs the
deployed ens3 baseline 0.8172; `split_head_ens3.tsv`):

| config (ens3 each) | eval-23 | Δ | W/L |
|---|---|---|---|
| deployed: zs 0.4 + joint 0.6 | 0.8172 | — | — |
| peg-head only | 0.8288 | +0.0116 | 15/7 |
| peg .85 + color .15 | 0.8353 | +0.0181 | 15/6 |
| **joint .55 + peg .30 + color .15** | **0.8366** | **+0.0194** | **19/3** |

The 3-head mix is the deployment shape: same mean as the pair but a far safer
tail (worst regression M11 −0.009, vs the pair's M26 −0.019; only 3 sets lose
> 0.002). Full-24 +0.0194. Biggest wins: M21 +0.072, M24 +0.052, M23 +0.051,
M17 +0.042. M7 (excluded outlier) +0.078 under peg-only — the peg-dominant
blend serves its color-off preference automatically.

**Hyperparams transfer as-is; smaller color head falsified.** Dev-set sweep of
10 variants (`sweep_heads_score.py`, `sweep_heads_dev.tsv`): the peg head is
already at its optimum (temp 0.1 / hidden 2048 cost −0.004; out_dim 256/768
flat-to-negative); color-head capacity barely matters at all (solo ARI moves
only 0.52–0.55 across a 64→512-d range). The nominal dev winner (color
128/512, +0.0024) was then full-LOMO-validated: 3 MLX seeds per arm vs a 3-seed
MLX-trained 512/1024 control, joint+peg components held fixed — **−0.0011 to
−0.0017 in both blend shapes**, i.e. the dev delta was noise. Keep 512/1024
everywhere (`split_head_ens3_col{128,512mlx}.tsv`; fold roots
`~/.cache/reorder/lomo_col{128,512mlx}_s{42,43,44}`). Incidental findings:
MLX ≈ torch on quality (3-head 0.8359 vs 0.8366) at ~20× the speed for
color-only heads; and with MLX color ensembles the pure pair (0.8395) nudges
past the 3-head (0.8359) **and loses its tail risk** (worst −0.007 — the torch
arm's M26 −0.019 above was seed luck). Both shapes sit on the same ~0.004
plateau, but re-check 55:30:15 vs 85:15 on the final production heads before
freezing the default dials.

**Final all-MLX dial sweep — default moves to 30:55:15.** With every component
MLX-trained (joint `lomo_jointmlx_s{42,43,44}`, peg `lomo_pegmlx_s{42,43,44}`,
color `lomo_col512mlx_s{42,43,44}`; 20 configs, `split_head_mlx_final.tsv`),
the optimum shifts to a **lighter joint / heavier peg** mix: b.30 p.55 c.15 =
0.8357 eval-23 (+0.0225 vs the zs+joint deployed shape, 21W/1L). Paired against
the previously-wired 55:30:15 default it is **+0.0056, bootstrap CI95
[+0.0030, +0.0085], 12W/0L, worst dataset −0.0013** — outside both the CI and
the ±0.004 seed band, with no meaningful regression anywhere, so the default
dials should be **joint .30 / peg .55 / color .15**. Runners-up on the same
plateau: +tiny-zs (zp.04 zc.02) adds +0.0011 mean but brings a −0.006 tail
(M17) — leave zs at 0 by default; b.30p.60c.10 and b.15p.70c.15 trail with
worse tails. The b-axis curve (b at p+c≈.70, c=.15): b.55 0.8302 → b.45 0.8331
→ **b.30 0.8357** → b.20 0.8343 → b.15 0.8344 → pure pair 0.8340. Note the
MLX joint ens is slightly weaker than the torch one in the old deployed shape
(0.8132 vs 0.8172), which is partly why the sweep leans further off the joint
head than the torch-era tables above.

The durable lesson here (the user's observation, and every sweep arm agrees):
**the exact coordinates don't generalise — the shape does.** You want some
combination of the joint head and the peg head, each covering for what the
other misses, plus a tiny bit of the color head. Any point on the
b.15–.45 × c.10–.20 plateau is within ~0.003 of any other, and the optimum
slid from 55:30:15 to 30:55:15 just from the torch→MLX retrain — so treat the
default as "the zero-regression point on the plateau as of the current heads",
re-find it after any retrain, and rely on the dials (not the defaults) for the
rest.

**Deployment (wired, pending final training).** `train_final_head.py` now
trains 3 modality groups × 3 seeds (9 head files; `--mods` to retrain a subset;
the color group drops drop-color). `learned_head.json` gains a `heads` list
(legacy fields kept for the joint group); `extract_features.py` emits
`learned_proj` + `learned_proj_peg` + `learned_proj_color`, all covered by one
`_v_learned_proj` version key. Both Rust binaries take
`--learned-proj-{peg,color}-weight`. The UI exposes **three learned dials**
(Learned joint / Learned PE-G / Learned color), each a target fraction of the
final signal, defaulting to the winning **.55/.30/.15** (zero-shot contributes
nothing when they sum to 1). `rescaleLearnedProjWeight` converts fractions to
raw concat weights; a dial whose array is missing from the cache contributes
nothing (zero-shot absorbs the remainder, so a joint-only cache lands near the
old deployed blend); headless caches fall back to zero-shot. To ship: run
`python clusteringRefinement/train_final_head.py`, then any extraction
refreshes every cache's arrays automatically. This also dissolves the
per-dataset color-w question: the fixed 3-head blend (+0.019) is ~2× the
*oracle* ceiling of per-dataset zs-color tuning (+0.009), and the knob it tuned
no longer exists.

## Clustering defaults: linkage & re-rank

A full 20-dataset LOMO (oracle-N, `clusteringRefinement/rerank_eval.py`, reusing the
production `compute_rerank_distance` — validated to match the Rust pipeline exactly)
compared linkage methods and the k-reciprocal re-rank blend. Mean ARI (17 datasets,
excluding the M7/M14/M15 outliers; full-20 in parentheses):

| config | mean ARI |
|---|---|
| **learned head + ward, NO re-rank** | **0.8046 (0.7692)** |
| learned head + ward + re-rank | 0.7816 (0.7435) |
| learned head + average + re-rank (the *old* default) | 0.7361 (0.6888) |
| base (no head) + ward, no re-rank | 0.6891 (0.6623) |
| base + average + re-rank | 0.6533 (0.6126) |

Findings:
- **Re-rank cost ARI in every pairing** at fixed oracle-N — even its best pairing (head+ward) lost ~0.023–0.026. It helped on a minority of datasets (eva, lily, sabrina, evie, darshelle, amanda) and hurt the small/few-per-group sets badly (k1=65 ≫ group size smears across groups).
- **Ward beat average in every pairing**, including with re-rank on (+0.04) — so the old "re-rank ⇒ average" auto-heuristic was miscalibrated.
- Head still helped under re-rank (+0.08) but less than without it (+0.11).

So the deployed cluster **defaults** (`listStore.ts`) are pinned to the winner: **learned head + ward + re-rank OFF** (`useRerank: false`, `linkage: "ward"`, `learned_proj: 0.6`). The Re-rank toggle and linkage dropdown remain for per-dataset overrides (re-rank helps on the handful of sets above; average wins on a few uneven-size sets — see the linkage tooltip).

Caveat: this is fixed-oracle-N. Re-rank reshapes the distance distribution and could still help the unknown-N **adaptive/threshold cut** — untested. Re-tuning k1 to group size might also rescue it on the small sets.

## MLX training backend — ~1.7× faster per head

`train_projection_head.py --backend mlx` runs the identical training recipe on
MLX instead of torch/MPS. The torch step is kernel-dispatch bound: a single
Python thread enqueues ~100 Metal kernels per step and the GPU drains them
faster than they arrive (CPU-enqueue 2.35ms vs 2.41ms wall per step, measured).
The escape hatches inside torch are dead ends on MPS — `torch.compile` is
*slower* than eager (2.7ms vs 1.2ms for fwd+loss+bwd), the foreach API is
unsupported, and fused AdamW (now enabled on the torch path; trajectory matches
the single-tensor path to 4 decimals) doesn't move the epoch time because
dispatch elsewhere still dominates. MLX's lazy graphs + `mx.compile` cut the
step to near the GPU floor (~6.6 GFLOP/step against the M4 Max's ~10.5 fp32
TFLOPS; bf16 is only 1.13× on this chip, so not worth the numerics risk).

Per-head wall time, deployed joint-head config (26 datasets, 15 ep × 400
batches, P×K=32×12, M4 Max, warm page cache):

|                       | torch (before) | torch (now) | mlx       |
|-----------------------|----------------|-------------|-----------|
| training loop         | 16.6s (2.75ms/step) | 16.6s  | 10.8s (1.78ms/step) |
| artifacts (26 ds)     | 2.9s           | skipped (`--eval none`) | skipped |
| **total**             | **21.7s**      | **19.1s**   | **13.1s** |

`train_final_head.py` now defaults to `--backend mlx` and passes `--eval none`
(the per-seed runs only consume `proj_head.pt`; the 26 dist matrices it used to
write into the throwaway tmpdir cost ~3s + ~1GB per run). `--jobs N` trains the
seeds of a modality group as concurrent processes (default 3 = all seeds of a
group in flight) — a modest win now that MLX training is GPU-bound rather than
dispatch-bound: the 3-seed joint group takes 39s / 33s / 31s at jobs 1 / 2 / 3
(15 ep; **25s** with the deployed ep-12 cosine recipe).

Validation (same `--seed` draws the same PKSampler batches in both backends;
init distributions, loss math and AdamW (bias-corrected) match op-for-op; the
device RNG streams necessarily differ, so runs are statistically — not bit —
equivalent):

- Loss trajectory, deployed config: mlx 4.1471 / 3.4862 vs torch 4.1453 / 3.4852
  (epochs 1–2); final epoch-15 loss 3.1216 vs 3.1264.
- `proj_head.pt` round-trip: the MLX-trained state_dict loads into the torch
  `ProjectionHead` with max forward deviation 2e-7 — downstream consumers
  (extract_features.py) are unaffected.
- Holdout pair-AUC (3-dataset, ArcFace config): within ±0.01 of torch per epoch.
- Mini-LOMO ARI @ blend 0.60 over {M4, M12, M16, M21, M23}: torch 0.7431 vs
  mlx 0.7428 (M23, the one single-seed outlier at −0.024, averaged over seeds
  42–44 lands at torch 0.7338 vs mlx 0.7421 — inside the single-fold seed σ of
  0.006–0.019 documented above).

Notes:
- Beta(α,α) mixup λs are drawn on host with numpy (MLX has no Beta sampler);
  all other augmentation randomness is in-graph from MLX's seeded RNG.
- ArcFace heads get one optimizer each, stepped only on their dataset's batches
  — mirrors torch, where inactive heads have `grad=None` and are skipped.
- `--backend mlx` is the default everywhere (with an automatic fallback to
  torch if `mlx` isn't importable); pass `--backend torch` to use the reference
  implementation. LOMO harnesses should also pass `--eval $HELDOUT` — the
  default writes projection artifacts for *all* datasets, of which a LOMO fold
  uses exactly one.

### Single-process LOMO (`--lomo`) + the pipelined sweep harness

`train_projection_head.py --lomo` runs a whole leave-one-model-out sweep in one
process: `--train` is the full pool; each fold trains on the pool minus one
dataset and writes its artifacts to `<output-dir>/<name>/` (`--lomo-folds`
restricts to a subset). Datasets load + upload once, and the hard-neg pools are
computed once and shared (they depend only on (dataset, seed, split), not the
fold). Each fold re-seeds every RNG it owns, so fold outputs are **byte-
identical** to standalone single-fold runs — verified on the dist matrices.

`run_deployed_lomo.sh` builds on it: ONE `--lomo` training process streams
folds while up to `$JOBS` (default 4) CPU scorers (blend + 2× `ari`) consume
each fold the moment its `summary.json` lands — GPU training and CPU Rust
linkage fully overlap. The full 26-fold deployed-config LOMO now takes
**287s (~4.8 min)** end-to-end vs ~15 min as serial per-fold processes
(avg blend-0.60 ARI 0.7762, Δ +0.1157 vs zero-shot — matches the documented
deployed numbers). Cached-fold reuse, `TARGETS=`, `USE_AUG=`, `EXTRA_ARGS=`
behave as before; the per-fold seed-retry ladder is replaced by a `SEED=` env
(MLX training is deterministic and hasn't diverged; rerun a fold with
`SEED=43` if it ever does).

### Batched multi-seed training (`--seeds`)

`train_projection_head.py --seeds 42,43,44` trains all ensemble seeds
simultaneously in one MLX graph: parameters stacked into (S, …) tensors, every
step a batched GEMM, per-lane grad clip, one AdamW. Each lane consumes exactly
the batches and mixup λs its standalone `--seed` run would (per-lane PKSampler
+ Beta rng), lanes never mix, and batches pad to a fixed B_max with pad columns
masked out of both the softmax and the positives — the padded loss is exact.
Writes `proj_head_s<seed>.pt` per seed plus per-seed artifacts under `s<seed>/`.

Validation (5-fold mini-LOMO, deployed config, blend 0.60): batched ens3
0.7486 vs independently-trained ens3 0.7483 (Δ +0.0003) — equivalent, with
per-seed spreads fully overlapping.

Speed verdict — honest: **no win over parallel processes.** Batched epochs run
2.2s for 3 lanes = 0.73s/lane, identical to a single-seed run's 0.72s — the
B=384 GEMMs were already at the M4 Max's efficiency plateau, so tripling the
GEMM batch bought nothing, and 3-seed group time is 35.2s vs 31.2s for
`--jobs 3` (whose host phases overlap the GPU). `train_final_head.py
--seed-mode batched` therefore stays opt-in: quality-equivalent, ~13% slower,
1/3 the memory (one ~3GB process instead of three) — use it when RAM is tight
or only one process may own the GPU. Unsupported there: ArcFace,
shoot-context, blend-aware, within-holdout (single-seed paths keep full
coverage).

## How to run more sweeps

`clusteringRefinement/run_augmentation_sweep.sh` is the template — 9 configs, each a full LOMO across all datasets. About an hour on MPS at 12 datasets. Modify the config list at the top to test new ideas.

The benchmark tool `clusteringRefinement/benchmark_clustering.ts` evaluates a single clustering on ground-truth labels and produces ARI / NMI / pair accuracy. It accepts either zero-shot weights or a `--dist-matrix` (e.g. the learned head's projected distances, optionally blended via `clusteringRefinement/blend_dist_matrix.py`).

For hyperparam exploration, the cheapest signal usually comes from a 4-dataset "dev set" LOMO (`{M2, M5, M9, M11}` is a reasonable choice — one big winner, one regressor, one medium, one high-baseline) — gives ~80% of the ranking signal at ~30% the compute, then validate the winner on the full 12-dataset LOMO.

### Sweep harness & efficiency

The sweep scripts share `common.sh` (dataset registry + `ari()`/`npy_bad()`/`pforeach()`) and `lomo_common.py` (the `load_fold()` LOMO loader). Keep the M-id→path map in **one** place — it lived in 11 copies and drifted into a path-typo bug once. Notes for keeping future runs fast:

- **Don't recompute what you can reuse.** The dominant cost is GPU training, not scoring. The pure-scoring sweeps (`run_blend_grid`, `run_blend_curve`, `run_pegcolor_sweep`) re-blend cached `*_dist_matrix.bin` from a trained LOMO root — they never retrain. When adding a sweep, point it at an existing `OUTROOT`/`LOMO_ROOT` of trained folds rather than training fresh. Within a script, blend the whole weight grid in **one** `blend_dist_matrix.py` call (it computes the zero-shot cosine once), not per-weight. If a config you want is already scored elsewhere in the run, alias it instead of recomputing (e.g. `rerank_eval` reuses its `h_rr_avg` matrix for the `b=0.60` sweep point — one re-rank per dataset, not two).
- **The scoring grids are embarrassingly parallel.** Each (dataset, weight) `ari()` call is independent and CPU-only. The three pure-scoring sweeps fan out across datasets via `pforeach "$JOBS"` (default 4; raise with `JOBS=8`), writing per-dataset fragment files that are aggregated **after** the barrier — bash subshells can't write back to parent state, so accumulate from files, not shared associative arrays. Training sweeps stay serial (GPU-bound); only their post-training scoring passes are parallelizable.
- **`benchmark_clustering.ts` reruns the full Rust Ward linkage every call** and nothing caches it across invocations — so a grid of N weights on one dataset is N linkage runs. That's not redundant (each blend is a different matrix) but it *is* the per-grid-point cost; use a coarse grid first, then refine around the peak.
