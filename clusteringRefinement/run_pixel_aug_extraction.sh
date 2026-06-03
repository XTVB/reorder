#!/usr/bin/env bash
# Overnight pixel-aug pre-extraction across all 12 benchmark datasets.
#
# Per dataset: K=3 augmented views per image → pecore_g_views.npy + color_views.npy
# in the dataset's .reorder-cache/. Each view gets RandomResizedCrop + HorizontalFlip
# + ColorJitter applied at the pixel level before PE-G.
#
# Runtime estimate (PE-G at ~2 img/s on MPS, K=3 views per image, M7 skipped):
#   total images across 11 datasets ≈ 41,900
#   total view-extractions = 41,900 × 3 = ~125,700
#   wall time ≈ 125,700 / 2 = ~17.5 hours
#
# Resumable: extract_augmented_views.py checkpoints every 200 images and skips
# already-done indices on re-run. Safe to Ctrl-C and restart.
#
# Usage:
#   ./run_pixel_aug_extraction.sh             # default: K=3 views, MLX backend
#   ./run_pixel_aug_extraction.sh 4           # K=4 views
#   ./run_pixel_aug_extraction.sh 3 pytorch   # K=3 views, fallback pytorch backend
set -e

K="${1:-3}"
BACKEND="${2:-mlx}"
HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
S=$HERE/extract_augmented_views.py

# Pixel-aug extraction set = datasets.txt minus the no-pixel-aug ones (M7 is
# already ~149 imgs/group so pixel-aug is redundant; M14/M15 are partial-label).
MS=("${PIXEL_AUG[@]}")

echo "Pixel-aug pre-extraction starting"
echo "  K=$K views/image, backend=$BACKEND"
echo "  ${#MS[@]} datasets"
echo ""

START_TIME=$(date +%s)
for m in "${MS[@]}"; do
  D=$(dir "$m"); name=$(basename "$D")
  echo "==== $name ===="
  if [[ ! -f "$D/.reorder-cache/content_hashes.json" ]]; then
    echo "  SKIP: content_hashes.json not found (run extract_features.py first)"
    continue
  fi
  "$PY" "$S" "$D" --n-views "$K" --pecore-g-backend "$BACKEND"
done

ELAPSED=$(( $(date +%s) - START_TIME ))
printf "\nALL DONE in %dh %dm\n" $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60))
