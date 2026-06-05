#!/usr/bin/env bash
# Overnight pixel-aug pre-extraction across all 12 benchmark datasets.
#
# Per dataset: K=3 augmented views per image → pecore_g_views.npy + color_views.npy
# in the dataset's .reorder-cache/. Each view gets RandomResizedCrop + HorizontalFlip
# + ColorJitter applied at the pixel level before PE-G.
#
# Only images in real (≥2-member) groups are extracted — singletons/ungrouped
# images are never sampled as views by the trainer, so PE-G is skipped for them.
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

# Pixel-aug extraction set = datasets.txt minus the no-pixel-aug ones
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
