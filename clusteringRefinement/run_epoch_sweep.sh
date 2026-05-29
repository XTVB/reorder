#!/usr/bin/env bash
# Epoch sweep on the deployed config (out_dim=512, temp=0.07, lr=1e-4, grad-clip,
# ICOMB + pixel-aug), full 20-dataset LOMO scored at the deployed blend (0.60).
#
# Motivation: the deployed lr (1e-4) is 3x lower than the old 3e-4, so 15 epochs
# may be under-converged — more epochs could now help where they didn't before.
# ep15 reuses the cached post-pixel-aug folds (/tmp/lomo_postaug); other epoch
# values train fresh into their own dirs (the fold cache doesn't key on epochs,
# so each value MUST get a distinct OUTROOT).
#
#   ./run_epoch_sweep.sh
set -euo pipefail

HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
RUN=$HERE/run_deployed_lomo.sh
SWEEP=/tmp/epoch_sweep
mkdir -p "$SWEEP"

EPOCHS_LIST=(15 20 25 30)
declare -A OUTROOTS=( [15]=/tmp/lomo_postaug )   # reuse cached ep15 folds

SUMMARY=$SWEEP/summary.tsv
printf "epochs\tavg_ari_blend0.60\tdelta_vs_zeroshot\n" > "$SUMMARY"
echo "Epoch sweep — deployed config, full 20-dataset LOMO @ blend 0.60"
echo ""

for E in "${EPOCHS_LIST[@]}"; do
  OUT=${OUTROOTS[$E]:-/tmp/lomo_ep$E}
  RES=$SWEEP/ep$E.tsv
  echo "==== epochs=$E  (OUTROOT=$OUT) ===="
  EPOCHS=$E USE_AUG=1 BLEND_W=0.60 OUTROOT="$OUT" RESULT="$RES" \
    bash "$RUN" 2>&1 | tee "$SWEEP/ep$E.log" | grep -E "AVG over 20" || true
  # Parse avg ARI + delta from the run's final line.
  line=$(grep "AVG over 20" "$SWEEP/ep$E.log" | tail -1)
  avg=$(echo "$line" | sed -nE 's/.*blend0\.60=([0-9.]+).*/\1/p')
  dlt=$(echo "$line" | sed -nE 's/.*Δ vs zero-shot ([+-][0-9.]+).*/\1/p')
  printf "%s\t%s\t%s\n" "$E" "$avg" "$dlt" >> "$SUMMARY"
  echo ""
done

echo "==================== EPOCH SWEEP SUMMARY (sorted by avg ARI) ===================="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k2 -gr; } | column -t -s$'\t'
echo ""
echo "Reference: ep15 is the current deployed setting. Lift = (avg_ari at E) − (avg_ari at 15)."
e15=$(awk -F'\t' '$1==15{print $2}' "$SUMMARY")
echo ""
printf "%-8s %-12s %-12s\n" "epochs" "avg_ari" "lift_vs_ep15"
tail -n +2 "$SUMMARY" | sort -t$'\t' -k1 -n | while IFS=$'\t' read -r e a d; do
  lift=$(awk "BEGIN{printf \"%+.4f\", $a-$e15}")
  printf "%-8s %-12s %-12s\n" "$e" "$a" "$lift"
done
echo ""
echo "Full TSVs in $SWEEP/"
