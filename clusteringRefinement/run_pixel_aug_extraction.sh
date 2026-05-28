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
PY=~/.venvs/imgcluster-env/bin/python3
S=/Users/abdudh/dev/utilities/reorder/clusteringRefinement/extract_augmented_views.py

# M7 deliberately skipped: ~149 imgs/group already (4x denser than the next
# largest), so pixel-aug is mostly redundant signal there. Skipping saves ~3.5h.
declare -a DIRS=(
  /Users/abdudh/Downloads/PicsStaging/ClusteringBenchmark1-austin
  /Users/abdudh/Downloads/PicsStaging/ClusteringBenchmark2-sarah
  /Users/abdudh/Downloads/PicsStaging/ClusteringBenchmark3-eva
  /Users/abdudh/Downloads/PicsStaging/ClusteringBenchmark4-mia
  /Users/abdudh/Downloads/PicsStaging/ClusteringBenchmark5-lily
  /Users/abdudh/Downloads/PicsStaging/ClusteringBenchmark6-sabrina
  /Users/abdudh/Downloads/PicsStaging/ClusteringBenchmark8-evie
  /Users/abdudh/Downloads/PicsStaging/ClusteringBenchmark9-darshelle
  /Users/abdudh/Downloads/PicsStaging/ClusteringBenchmark10-alina
  /Users/abdudh/Downloads/PicsStaging/ClusteringBenchmark11-amanda
  /Users/abdudh/Downloads/PicsStaging/ClusteringBenchmark12-anna
)

echo "Pixel-aug pre-extraction starting"
echo "  K=$K views/image, backend=$BACKEND"
echo "  ${#DIRS[@]} datasets"
echo "  estimate: ~21 hours total at 2 img/s (MLX)"
echo ""

START_TIME=$(date +%s)
for D in "${DIRS[@]}"; do
  name=$(basename "$D")
  echo "==== $name ===="
  if [[ ! -f "$D/.reorder-cache/content_hashes.json" ]]; then
    echo "  SKIP: content_hashes.json not found (run extract_features.py first)"
    continue
  fi
  $PY $S "$D" --n-views "$K" --pecore-g-backend "$BACKEND"
done

ELAPSED=$(( $(date +%s) - START_TIME ))
printf "\nALL DONE in %dh %dm\n" $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60))
