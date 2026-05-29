#!/usr/bin/env bash
# Full-config LOMO baseline for the 5 eval targets, BEFORE running more pixel-aug.
#
# For each TARGET: train the projection head on all OTHER datasets using the
# deployed winning config (P=32 K=12 ep=15 + ICOMB augs + pixel-aug views where
# they currently exist), hold the target out, then score the target's clustering
# ARI. Whatever pixel-aug views exist right now get used — so re-running this
# after extracting more views measures the benefit of the additional pixel-aug.
#
# Matches clusteringRefinement/train_final_head.py DEFAULT_HYPERPARAMS exactly,
# minus the LOMO holdout (--train excludes the target).
#
#   ./run_lomo_baseline.sh            # run all 5 targets
#
# Results table + machine-readable TSV written under $OUTROOT.
set -euo pipefail

HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
OUTROOT=${OUTROOT:-/tmp/lomo_baseline}
TARGETS=(M16 M19 M2 M11 M3)

mkdir -p "$OUTROOT"

# Pre-flight
for m in "${ALL[@]}"; do
  d=$(dir "$m")
  [[ -f "$d/.reorder-cache/embeddings_hash_cache.npz" && -f "$d/.reorder-cache/content_hashes.json" && -f "$d/.reorder-groups.json" ]] \
    || { echo "ERROR: $m missing caches at $d" >&2; exit 1; }
done
echo "Pre-flight OK: ${#ALL[@]} datasets, ${#TARGETS[@]} LOMO targets."

# ── Train one LOMO fold per target ────────────────────────────────────────────
for TARGET in "${TARGETS[@]}"; do
  echo ""
  echo "================ LOMO fold: holding out $TARGET ($(basename "$(dir "$TARGET")")) ================"
  TRAIN_NAMES=""
  DATASET_ARGS=()
  for m in "${ALL[@]}"; do
    DATASET_ARGS+=(--dataset "$m:$(dir "$m")")
    if [[ "$m" != "$TARGET" ]]; then
      [[ -z "$TRAIN_NAMES" ]] && TRAIN_NAMES="$m" || TRAIN_NAMES="$TRAIN_NAMES,$m"
    fi
  done
  OUT=$OUTROOT/$TARGET
  mkdir -p "$OUT"
  # Winning hyperparams from train_final_head.py DEFAULT_HYPERPARAMS.
  "$PY" "$TRAIN" "${DATASET_ARGS[@]}" \
    --train "$TRAIN_NAMES" --eval "$TARGET" --within-holdout-frac 0 \
    --epochs 15 --batches-per-epoch 400 --p-groups 32 --k-images 12 \
    --out-dim 256 --hidden 1024 --dropout 0.1 --temperature 0.1 \
    --lr 3e-4 --weight-decay 1e-4 \
    --mixup-alpha 0.4 --drop-color-prob 0.5 \
    --cross-mixup-prob 0.3 --cross-mixup-alpha 0.4 \
    --hard-neg-frac 0.5 --hard-neg-pool-k 20 \
    --arcface-weight 0 \
    --use-augmented-views --use-singleton-negatives \
    --output-dir "$OUT" 2>&1 | tee "$OUT/train.log" | tail -2
  # Blends of the learned dist matrix with the zero-shot baseline.
  "$PY" "$BLEND" "$(dir "$TARGET")" \
    --learned "$OUT/${TARGET}_dist_matrix.bin" \
    --weights 0.3,0.5,0.7 \
    --output-pattern "$OUT/${TARGET}_blend_{w}.bin" >/dev/null 2>&1
  echo "  [$TARGET] training + blends done"
done

# ── Score ─────────────────────────────────────────────────────────────────────
TSV=$OUTROOT/results.tsv
printf "target\tbaseline_ari\tpure_ari\tb30_ari\tb50_ari\tb70_ari\tbest_ari\tdelta_vs_base\tbest_blend\n" > "$TSV"

echo ""
echo "==================== LOMO BASELINE RESULTS (current pixel-aug state) ===================="
printf "%-7s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n" \
  "target" "base" "learned" "30%" "50%" "70%" "best" "Δ vs base"
echo "--------------------------------------------------------------------------------------------"
sum_delta=0
for TARGET in "${TARGETS[@]}"; do
  d=$(dir "$TARGET")
  OUT=$OUTROOT/$TARGET
  base=$(ari "$d")
  pure=$(ari "$d" --dist-matrix "$OUT/${TARGET}_dist_matrix.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
  b30=$(ari "$d" --dist-matrix "$OUT/${TARGET}_blend_0.3.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
  b50=$(ari "$d" --dist-matrix "$OUT/${TARGET}_blend_0.5.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
  b70=$(ari "$d" --dist-matrix "$OUT/${TARGET}_blend_0.7.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
  # best of {pure,b30,b50,b70} and which blend won
  best=$pure; bestlbl="100%"
  for pair in "$b30:30%" "$b50:50%" "$b70:70%"; do
    v=${pair%%:*}; lbl=${pair##*:}
    awk "BEGIN{exit !($v>$best)}" && { best=$v; bestlbl=$lbl; }
  done
  delta=$(awk "BEGIN{printf \"%.4f\", $best-$base}")
  sum_delta=$(awk "BEGIN{printf \"%.4f\", $sum_delta+$delta}")
  sdelta=$(awk "BEGIN{printf \"%+.4f\", $delta}")
  printf "%-7s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s (%s)\n" \
    "$TARGET" "$base" "$pure" "$b30" "$b50" "$b70" "$best" "$sdelta" "$bestlbl"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$TARGET" "$base" "$pure" "$b30" "$b50" "$b70" "$best" "$delta" "$bestlbl" >> "$TSV"
done
avg=$(awk "BEGIN{printf \"%+.4f\", $sum_delta/${#TARGETS[@]}}")
echo "--------------------------------------------------------------------------------------------"
printf "AVG Δ ARI (best blend vs zero-shot baseline): %s\n" "$avg"
echo ""
echo "TSV: $TSV"
