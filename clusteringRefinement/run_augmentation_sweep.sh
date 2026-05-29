#!/usr/bin/env bash
# Runs the augmentation ablation sweep once feature extraction is complete.
# Each config is an N=8 LOMO sweep over (M1, M2, M3, M4, M5, M6, M11, M12).
#
#   Config A: mixup-only           — same-group MixUp
#   Config B: drop-color-only      — break the color-as-backdrop shortcut
#   Config C: feature-dropout-only — generic robustness
#   Config D: feature-noise-only   — smoother decision boundary
#   Config E: drop-peg-only        — CONTROL: should HURT if PE-G is doing the work
#   Config F: mixup + drop-color   — top two combined
#   Config G: cross-mixup-only     — smooth between-group boundaries (soft SupCon)
#   Config H: hard-neg-mining-only — over-sample confusable groups per batch
#   Config I: all the good ones    — mixup + drop-color + cross-mixup + hard-neg
#
# After each config completes, generate blends and compute average ARI gain
# vs baseline (best of 0%, 30%, 50%, 70% learned blend). Compare to the
# un-augmented baseline of +0.035 from N=8 P=32 LOMO.
#
# Usage:
#   ./run_augmentation_sweep.sh         (default: N=12 — all M1-M12)
#   ./run_augmentation_sweep.sh n8      (only M1-M6, M11, M12)
set -e

N_MODE="${1:-n12}"
HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
S=$TRAIN   # train_projection_head.py

# This sweep predates the M13-M20 datasets; SET is its own N=8/N=12 subset
# (do NOT reuse common.sh's full 20-entry ALL).
if [[ "$N_MODE" == "n12" ]]; then
  SET=(M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12)
else
  SET=(M1 M2 M3 M4 M5 M6 M11 M12)
fi

# Pre-flight check: required caches must exist
for m in "${SET[@]}"; do
  D=$(dir "$m")
  if [[ ! -f "$D/.reorder-cache/content_hashes.json" || ! -f "$D/.reorder-cache/embeddings_hash_cache.npz" ]]; then
    echo "ERROR: $m missing required cache at $D/.reorder-cache/" >&2
    exit 1
  fi
done
echo "Pre-flight OK: ${#SET[@]} datasets ready."

run_lomo() {
  local TAG=$1; shift
  local EXTRA=("$@")
  echo ""
  echo "================ Config '$TAG' — args: ${EXTRA[*]} ================"
  for HELDOUT in "${SET[@]}"; do
    TRAIN_NAMES=""
    DATASET_ARGS=()
    for m in "${SET[@]}"; do
      DATASET_ARGS+=(--dataset "$m:$(dir "$m")")
      if [[ "$m" != "$HELDOUT" ]]; then
        [[ -z "$TRAIN_NAMES" ]] && TRAIN_NAMES="$m" || TRAIN_NAMES="$TRAIN_NAMES,$m"
      fi
    done
    "$PY" "$S" "${DATASET_ARGS[@]}" \
      --train "$TRAIN_NAMES" --within-holdout-frac 0 \
      --epochs 15 --batches-per-epoch 400 --p-groups 32 --k-images 8 \
      --out-dim 256 --arcface-weight 0 \
      "${EXTRA[@]}" \
      --output-dir /tmp/aug_${TAG}_${HELDOUT} 2>&1 | tail -1
    echo "  [$TAG] $HELDOUT done"
  done

  # Generate blends + score
  mkdir -p /tmp/aug_${TAG}_blends
  for m in "${SET[@]}"; do
    "$PY" "$BLEND" "$(dir "$m")" \
      --learned /tmp/aug_${TAG}_${m}/${m}_dist_matrix.bin \
      --weights 0.3,0.5,0.7 \
      --output-pattern /tmp/aug_${TAG}_blends/${m}_blend_{w}.bin > /dev/null 2>&1
  done
}

eval_config() {
  local TAG=$1
  echo ""
  echo "===== Eval config '$TAG' ====="
  printf "%-6s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n" "cold" "baseline" "100%" "30%blend" "50%blend" "70%blend" "Δ best"
  local total=0
  for m in "${SET[@]}"; do
    local d=$(dir "$m")
    local base=$(ari "$d")
    local pure=$(ari "$d" --dist-matrix /tmp/aug_${TAG}_${m}/${m}_dist_matrix.bin --dist-matrix-weight 1.0 --weights pecore_g=0)
    local b30=$(ari "$d" --dist-matrix /tmp/aug_${TAG}_blends/${m}_blend_0.3.bin --dist-matrix-weight 1.0 --weights pecore_g=0)
    local b50=$(ari "$d" --dist-matrix /tmp/aug_${TAG}_blends/${m}_blend_0.5.bin --dist-matrix-weight 1.0 --weights pecore_g=0)
    local b70=$(ari "$d" --dist-matrix /tmp/aug_${TAG}_blends/${m}_blend_0.7.bin --dist-matrix-weight 1.0 --weights pecore_g=0)
    local best=$(echo -e "$pure\n$b30\n$b50\n$b70" | sort -g | tail -1)
    local delta=$(echo "$best - $base" | bc -l)
    total=$(echo "$total + $delta" | bc -l)
    local sign=$(echo "$delta" | awk '{ if ($1 > 0) printf "+"; printf "%.4f", $1 }')
    printf "%-6s | %s   | %s   | %s   | %s   | %s   | %s\n" "$m" "$base" "$pure" "$b30" "$b50" "$b70" "$sign"
  done
  local avg=$(echo "$total / ${#SET[@]}" | bc -l)
  printf "AVG Δ ARI: %+.4f\n" "$avg"
}

# Run + eval each config
run_lomo "A_mixup"        --mixup-alpha 0.4
eval_config "A_mixup"

run_lomo "B_dropcolor"    --drop-color-prob 0.5
eval_config "B_dropcolor"

run_lomo "C_featdrop"     --feature-dropout 0.2
eval_config "C_featdrop"

run_lomo "D_featnoise"    --feature-noise 0.02
eval_config "D_featnoise"

run_lomo "E_droppeg"      --drop-peg-prob 0.3
eval_config "E_droppeg"

run_lomo "F_mixup_dropcolor"  --mixup-alpha 0.4 --drop-color-prob 0.5
eval_config "F_mixup_dropcolor"

run_lomo "G_crossmixup"   --cross-mixup-prob 0.3 --cross-mixup-alpha 0.4
eval_config "G_crossmixup"

run_lomo "H_hardneg"      --hard-neg-frac 0.5 --hard-neg-pool-k 20
eval_config "H_hardneg"

run_lomo "I_combined"     --mixup-alpha 0.4 --drop-color-prob 0.5 --cross-mixup-prob 0.3 --hard-neg-frac 0.5
eval_config "I_combined"

echo ""
echo "==================== SUMMARY ===================="
echo "Reference: N=8 P=32 un-augmented baseline avg Δ = +0.035"
echo "Run 'grep ^AVG /tmp/aug_sweep_log.txt' or inspect /tmp/aug_*/ artifacts for per-config details."
