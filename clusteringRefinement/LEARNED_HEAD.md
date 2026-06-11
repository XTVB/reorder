# Learned Projection Head

A small MLP trained on labeled photoshoot data that maps PE-G (1280d) + color (693d) → 512d, deployed as a **3-seed ensemble**. Blended with zero-shot PE-G + color distances at clustering time. Improves cluster ARI by roughly +0.11–0.13 on held-out models compared to baseline (single head), plus ~+0.010 from the ensemble.

## How it's used at runtime

The head lives at `~/.cache/reorder/learned_head.pt` (+ `learned_head_s43.pt`, `learned_head_s44.pt` for the other ensemble seeds) + `learned_head.json` (with a content-derived version string covering all heads, and a `head_files` list). Three places integrate it:

1. **`scripts/extract_features.py`** runs `_maybe_update_learned_proj` at the end of every extraction. It pushes the (PE-G, color) features through **every** head in `head_files`; the per-head L2-normed blocks are concatenated and scaled by 1/√n_heads, so rows stay unit-norm and their dot product equals the ensemble-MEAN cosine — the downstream blend is unchanged. The result is the (N, 1536) `learned_proj` in `.reorder-cache/embeddings_hash_cache.npz` with a `_v_learned_proj` version key. If the head's version differs from what's stored, the projection is recomputed. If the head isn't installed, the step is silently skipped.

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

Final deployed config (currently saved): **P=32, K=12, ep=15, out_dim=512, temperature=0.07, lr=1e-4, grad-clip=5.0, ICOMB augmentations, pixel-aug enabled, blend at 0.60**.

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

autumn and the two partial-label sets (vixen, verity) were excluded. Sweep: `clusteringRefinement/run_pegcolor_sweep.sh`; full matrix in `/tmp/pegcolor_sweep/per_dataset.tsv`.

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

### Intermediate PE-G layers: the open lead

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

**Extraction state** (`run_pe_layer_extraction.sh`, now defaults to all 26
registered datasets): M1/M3/M5/M6 done+clean (L42/44/46/47 × mean/gem3/attnpool),
M2/M4/M7 stale (npz changed since; the extractor auto-detects via
`pe_layers_meta.json` n_images mismatch and restarts them fresh), the other 19
never extracted. Remaining ≈ 56k images ≈ 8–9h on MLX — run it when the
machine is idle, then: full LOMO with `EXTRA_ARGS="--pe-layer 47:attnpool"`
vs the v26 baseline, and the zs-blend arm via `pe_layers_zs_eval.py` logic.

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

## How to run more sweeps

`clusteringRefinement/run_augmentation_sweep.sh` is the template — 9 configs, each a full LOMO across all datasets. About an hour on MPS at 12 datasets. Modify the config list at the top to test new ideas.

The benchmark tool `clusteringRefinement/benchmark_clustering.ts` evaluates a single clustering on ground-truth labels and produces ARI / NMI / pair accuracy. It accepts either zero-shot weights or a `--dist-matrix` (e.g. the learned head's projected distances, optionally blended via `clusteringRefinement/blend_dist_matrix.py`).

For hyperparam exploration, the cheapest signal usually comes from a 4-dataset "dev set" LOMO (`{M2, M5, M9, M11}` is a reasonable choice — one big winner, one regressor, one medium, one high-baseline) — gives ~80% of the ranking signal at ~30% the compute, then validate the winner on the full 12-dataset LOMO.

### Sweep harness & efficiency

The sweep scripts share `common.sh` (dataset registry + `ari()`/`npy_bad()`/`pforeach()`) and `lomo_common.py` (the `load_fold()` LOMO loader). Keep the M-id→path map in **one** place — it lived in 11 copies and drifted into a path-typo bug once. Notes for keeping future runs fast:

- **Don't recompute what you can reuse.** The dominant cost is GPU training, not scoring. The pure-scoring sweeps (`run_blend_grid`, `run_blend_curve`, `run_pegcolor_sweep`) re-blend cached `*_dist_matrix.bin` from a trained LOMO root — they never retrain. When adding a sweep, point it at an existing `OUTROOT`/`LOMO_ROOT` of trained folds rather than training fresh. Within a script, blend the whole weight grid in **one** `blend_dist_matrix.py` call (it computes the zero-shot cosine once), not per-weight. If a config you want is already scored elsewhere in the run, alias it instead of recomputing (e.g. `rerank_eval` reuses its `h_rr_avg` matrix for the `b=0.60` sweep point — one re-rank per dataset, not two).
- **The scoring grids are embarrassingly parallel.** Each (dataset, weight) `ari()` call is independent and CPU-only. The three pure-scoring sweeps fan out across datasets via `pforeach "$JOBS"` (default 4; raise with `JOBS=8`), writing per-dataset fragment files that are aggregated **after** the barrier — bash subshells can't write back to parent state, so accumulate from files, not shared associative arrays. Training sweeps stay serial (GPU-bound); only their post-training scoring passes are parallelizable.
- **`benchmark_clustering.ts` reruns the full Rust Ward linkage every call** and nothing caches it across invocations — so a grid of N weights on one dataset is N linkage runs. That's not redundant (each blend is a different matrix) but it *is* the per-grid-point cost; use a coarse grid first, then refine around the peak.
