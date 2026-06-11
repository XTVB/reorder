#!/usr/bin/env bash
# Overnight intermediate PE-G layer extraction across the established LOMO datasets.
#
# Per dataset: features from candidate transformer blocks (default L42/44/46/47),
# pooled three ways (mean/gem3/attnpool) -> pe_layers_L<NN>_<pool>.npy in the
# dataset's .reorder-cache/, row-aligned to the cached pecore_g (for the
# learned-head experiment; the post-extraction search picks pooling x layer x w).
# Only images in .reorder-groups.json groups are encoded (the head trains on
# grouped images only); ungrouped rows are left zero-filled.
#
# Resumable: extract_pe_layers.py checkpoints every 200 images and skips
# already-done rows on re-run. Safe to Ctrl-C / pause and restart.
#
# MLX backend by default (~2 img/s, fp32 — same regime as the deployed pecore_g).
# Grouped images only, so the wall-clock is proportional to grouped counts, not
# directory sizes. Override the set with MS_OVERRIDE="M1 M2 ...", the layers
# with LAYERS=..., the backend with BACKEND=pytorch.
#
# Usage:
#   ./run_pe_layer_extraction.sh                 # default layers + dataset set, MLX
#   LAYERS="44,46" ./run_pe_layer_extraction.sh  # fewer layers
#   BACKEND=pytorch ./run_pe_layer_extraction.sh # fallback backend
#   MS_OVERRIDE="M1 M3 M5" ./run_pe_layer_extraction.sh
set -e

HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
S=$HERE/extract_pe_layers.py
LAYERS="${LAYERS:-42,44,46,47}"
BACKEND="${BACKEND:-mlx}"

# Default = every registered dataset (from datasets.txt), partials included —
# if the layer features become head input, every TRAINING dataset needs them,
# and M14/M15 sit in every training pool. Already-extracted datasets are
# skipped via pe_layers_meta.json; stale ones (dataset changed since — npz row
# count mismatch) are auto-detected by extract_pe_layers.py and restart fresh.
# State as of 2026-06-10: M1/M3/M5/M6 done+clean (all rows — extracted before
# the grouped-only filter), M2/M4/M7 stale (auto-redo), rest never extracted.
DEFAULT_MS=("${ALL[@]}")
if [[ -n "${MS_OVERRIDE:-}" ]]; then read -ra MS <<< "$MS_OVERRIDE"; else MS=("${DEFAULT_MS[@]}"); fi

echo "PE-G intermediate-layer extraction"
echo "  layers=$LAYERS   backend=$BACKEND   ${#MS[@]} datasets: ${MS[*]}"
echo ""
START=$(date +%s)
for m in "${MS[@]}"; do
  D=$(dir "$m"); name=$(basename "$D")
  echo "==== $name ===="
  if [[ ! -f "$D/.reorder-cache/embeddings_hash_cache.npz" ]]; then
    echo "  SKIP: embeddings_hash_cache.npz not found (run extract_features.py first)"
    continue
  fi
  "$PY" "$S" "$D" --layers "$LAYERS" --backend "$BACKEND"
done
ELAPSED=$(( $(date +%s) - START ))
printf "\nALL DONE in %dh %dm\n" $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60))
