# Learned Projection Head

A small MLP trained on labeled photoshoot data that maps PE-G (1280d) + color (693d) → 256d. Blended with zero-shot PE-G + color distances at clustering time. Improves cluster ARI by roughly +0.07 to +0.10 on held-out models compared to baseline.

## How it's used at runtime

The head lives at `~/.cache/reorder/learned_head.pt` + `learned_head.json` (with a content-derived version string). Three places integrate it:

1. **`scripts/extract_features.py`** runs `_maybe_update_learned_proj` at the end of every extraction. It loads the head, pushes the (PE-G, color) features through it, and stores the (N, 256) projection as `learned_proj` in `.reorder-cache/embeddings_hash_cache.npz` with a `_v_learned_proj` version key. If the head's version differs from what's stored, the projection is recomputed. If the head isn't installed, the step is silently skipped.

2. **`rust/cluster-tool`** exposes `--learned-proj-weight`. It reads the `learned_proj` array from the NPZ alongside the other model embeddings and blends them as one more weighted component in the combined feature vector.

3. **Cluster UI** (`ClusterToolbar.tsx`) has a "Learned head" slider in the weight panel, range 0–1. The slider value is the **target fraction of the final cosine signal**, not a raw concat weight — the server rescales it (see `rescaleLearnedProjWeight` in `src/cluster/pipeline.ts`) so that "0.45" means "learned head contributes 45% of the clustering distance, regardless of how PE-G and color are weighted." Default is **0.45**, the best fixed blend value in our LOMO sweeps. Set to 0 to disable the head entirely. Set to 1 to use only the head (PE-G and color contributions zeroed).

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
| **N=12, +pixel-aug** | **all + 3 aug views/img** | **+pixel-aug** | **40-50%** | **+0.099** | **5/11** |

Final deployed config (currently saved): **P=32, K=12, ep=15, batch=384, ICOMB augmentations, pixel-aug enabled, blend at 0.45**.

## What worked

- **Augmentations that simulate within-shoot diversity** — drop-color (analog of "background masking" — color histograms encode backdrop), cross-mixup with soft SupCon labels (smooths between-group boundaries), and pixel-augmented views during training.
- **More positives per anchor (K=12 over K=8)** — K=12 was the sweet spot at our data scale; K=20 broke because too many groups have <20 images.
- **More training data**, with diminishing returns per added dataset.
- **30-50% blend at inference** — pure learned (100%) regresses, head needs zero-shot ballast.

## What didn't work

- **DINOv3 features as head input** — net regression vs PE-G + color alone. Zero-shot DINOv3 CLS is fine; the head can't use the 7×7 patch tensor productively.
- **More than 15 epochs** — pretty much always made things worse (overfit) even with augmentations.
- **Larger output dim (512)** — overfits at N≤8; might be useful at much larger N.
- **ArcFace loss** — neutral or slightly worse at our data scale.
- **Excluding M7 from training** — initially looked promising on M3+M4 but evened out across all 11 datasets. M7-style datasets (very large groups) are outliers the head can neither help nor be hurt by much.

## How to run more sweeps

`clusteringRefinement/run_augmentation_sweep.sh` is the template — 9 configs, each a full LOMO across all datasets. About an hour on MPS at 12 datasets. Modify the config list at the top to test new ideas.

The benchmark tool `clusteringRefinement/benchmark_clustering.ts` evaluates a single clustering on ground-truth labels and produces ARI / NMI / pair accuracy. It accepts either zero-shot weights or a `--dist-matrix` (e.g. the learned head's projected distances, optionally blended via `clusteringRefinement/blend_dist_matrix.py`).

For hyperparam exploration, the cheapest signal usually comes from a 4-dataset "dev set" LOMO (`{M2, M5, M9, M11}` is a reasonable choice — one big winner, one regressor, one medium, one high-baseline) — gives ~80% of the ranking signal at ~30% the compute, then validate the winner on the full 12-dataset LOMO.

## Paths to improve further

In rough order of expected impact:

1. **More diverse training datasets.** The marginal Δ-ARI per added dataset has been roughly:
   - **N=2 → N=6 (+4 datasets): +0.001 / dataset** (saturated early because the first datasets were similar)
   - **N=6 → N=8 (+2 datasets): +0.005 / dataset**
   - **N=8 → N=12 (+4 datasets, with augmentation breakthrough): +0.015 / dataset** (most of this was the augmentation lift; pure data ~+0.003/dataset)

   So **roughly +0.003 to +0.008 ARI per non-redundant new dataset** at current N=12. Adding 5 more diverse datasets (different shoot styles, group sizes, photography aesthetics) should push the avg from ~+0.10 toward ~+0.12-0.14. Adding more *similar* datasets (same model type as existing ones) hits diminishing returns fast — probably <+0.001 each.

2. **Per-dataset blend weight at inference**, instead of a single global 0.45. Datasets with low baseline (e.g. M4 at 0.50) want lighter blends (~40%); high-baseline datasets (M11 at 0.81) want heavier (~50%). A simple heuristic ("blend ∝ 1 − baseline_estimate") could capture the variance we see. Not big but easy.

3. **Better label quality on training data.** When you relabeled M1/M2/M3/M5/M6/M12, baselines moved by up to +0.10 and the head's contribution stayed proportionally similar — but if the head trains on cleaner labels it builds a sharper notion of "same shoot". Probably +0.01-0.02 average if labels improve across the board.

4. **Augmentation hyperparameter tuning.** mixup_alpha, drop_color_prob, cross_mixup_prob — all swept at single values. Modest gains likely (+0.005).

5. **LoRA on PE-G itself.** Higher capacity, but our diagnostics suggest the bottleneck is data diversity, not method capacity. Unlikely to help at N=12 without much more data; defer.
