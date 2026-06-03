# Learned Projection Head

A small MLP trained on labeled photoshoot data that maps PE-G (1280d) + color (693d) → 256d. Blended with zero-shot PE-G + color distances at clustering time. Improves cluster ARI by roughly +0.10 to +0.12 on held-out models compared to baseline.

## How it's used at runtime

The head lives at `~/.cache/reorder/learned_head.pt` + `learned_head.json` (with a content-derived version string). Three places integrate it:

1. **`scripts/extract_features.py`** runs `_maybe_update_learned_proj` at the end of every extraction. It loads the head, pushes the (PE-G, color) features through it, and stores the (N, 256) projection as `learned_proj` in `.reorder-cache/embeddings_hash_cache.npz` with a `_v_learned_proj` version key. If the head's version differs from what's stored, the projection is recomputed. If the head isn't installed, the step is silently skipped.

2. **`rust/cluster-tool`** exposes `--learned-proj-weight`. It reads the `learned_proj` array from the NPZ alongside the other model embeddings and blends them as one more weighted component in the combined feature vector.

3. **Cluster UI** (`ClusterToolbar.tsx`) has a "Learned head" slider in the weight panel, range 0–1. The slider value is the **target fraction of the final cosine signal**, not a raw concat weight — the server rescales it (see `rescaleLearnedProjWeight` in `src/cluster/pipeline.ts`) so that "0.60" means "learned head contributes 60% of the clustering distance, regardless of how PE-G and color are weighted." Default is **0.60**, the best fixed-blend value in our full 20-dataset LOMO for the current high-capacity (512-dim) head — the earlier 256-dim head peaked at 0.45. Set to 0 to disable the head entirely. Set to 1 to use only the head (PE-G and color contributions zeroed).

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

## Paths to improve further

In rough order of expected impact:

1. **More diverse training datasets.** The marginal Δ-ARI per added dataset has been roughly:
   - **N=2 → N=6 (+4 datasets): +0.001 / dataset** (saturated early because the first datasets were similar)
   - **N=6 → N=8 (+2 datasets): +0.005 / dataset**
   - **N=8 → N=12 (+4 datasets, with augmentation breakthrough): +0.015 / dataset** (most of this was the augmentation lift; pure data ~+0.003/dataset)

   So **roughly +0.003 to +0.008 ARI per non-redundant new dataset** at current N=12. Adding 5 more diverse datasets (different shoot styles, group sizes, photography aesthetics) should push the avg from ~+0.10 toward ~+0.12-0.14. Adding more *similar* datasets (same model type as existing ones) hits diminishing returns fast — probably <+0.001 each.

2. **Per-dataset blend weight at inference**, instead of a single global 0.60. Datasets with low baseline (e.g. M4 at 0.50) want lighter blends (~40%); high-baseline datasets (M11 at 0.81) want heavier (~50%). A simple heuristic ("blend ∝ 1 − baseline_estimate") could capture the variance we see. Not big but easy.

3. **Better label quality on training data.** When you relabeled M1/M2/M3/M5/M6/M12, baselines moved by up to +0.10 and the head's contribution stayed proportionally similar — but if the head trains on cleaner labels it builds a sharper notion of "same shoot". Probably +0.01-0.02 average if labels improve across the board.

4. **Augmentation hyperparameter tuning.** mixup_alpha, drop_color_prob, cross_mixup_prob — all swept at single values. Modest gains likely (+0.005).

5. **LoRA on PE-G itself.** Higher capacity, but our diagnostics suggest the bottleneck is data diversity, not method capacity. Unlikely to help at N=12 without much more data; defer.
