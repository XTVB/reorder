#!/usr/bin/env bash
# Big-set training ablation: measure each big dataset's marginal contribution by
# removing it (and small combos) from the FULL training pool and re-running true
# leave-one-out over the 22 non-partial test targets. Deployed config, blend 0.60.
#
# Efficiency vs naive:
#   - baseline (full pool) is FREE: it's the existing WITH arm (lomo_m25m26_with.tsv),
#     same from-scratch config + seed 42 — every Delta is measured against it.
#   - zero-shot ARI is head-independent → reused from the WITH tsv, never recomputed.
#   - GPU folds run at concurrency 2 (measured sweet spot: ~1.36x; C=4 regresses).
#   - each fold is a genuine from-scratch train (never sees the removed set) — no
#     warm-start contamination.
#
# Each fold: train head on (full registry \ removed \ target), project+score target.
set -uo pipefail

HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
ROOT=${ROOT:-/tmp/lomo_bigset}
BLEND_W=${BLEND_W:-0.60}
CONC=${CONC:-2}
BASELINE_TSV=${BASELINE_TSV:-$HERE/lomo_m25m26_with.tsv}   # full-pool reference (free)
mkdir -p "$ROOT"

# 22 non-partial test targets (same set the WITH arm scored, so baseline is free).
TARGETS=(M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13 M16 M17 M18 M19 M20 M21 M22 M23 M24)

# Arms: armid → comma-separated datasets removed from the training pool.
ARM_ORDER=(A_M3 A_M8 A_M2 A_M10 A_M9 A_M6 P_M3M8 P_M10M9 P_M2M6 U5)
declare -A ARMS=(
  [A_M3]="M3" [A_M8]="M8" [A_M2]="M2" [A_M10]="M10" [A_M9]="M9" [A_M6]="M6"
  [P_M3M8]="M3,M8" [P_M10M9]="M10,M9" [P_M2M6]="M2,M6"
  [U5]="M3,M8,M2,M10,M9"
)

DEPLOYED_ARGS=(
  --within-holdout-frac 0
  --epochs 15 --batches-per-epoch 400 --p-groups 32 --k-images 12
  --out-dim 512 --hidden 1024 --dropout 0.1 --temperature 0.07
  --lr 1e-4 --weight-decay 1e-4
  --mixup-alpha 0.4 --drop-color-prob 0.5
  --cross-mixup-prob 0.3 --cross-mixup-alpha 0.4
  --hard-neg-frac 0.5 --hard-neg-pool-k 20
  --arcface-weight 0 --use-singleton-negatives --use-augmented-views
)
DA=(); for m in "${ALL[@]}"; do DA+=(--dataset "$m:$(dir "$m")"); done

# do_fold <armid> <removedCSV> <target>: train (skip if cached) + blend + score.
do_fold() {
  local armid=$1 removed=$2 tgt=$3
  local out=$ROOT/$armid/$tgt; mkdir -p "$out"
  local d; d=$(dir "$tgt")
  if [[ -f $out/score.txt ]]; then return 0; fi
  # Build train list = ALL minus removed minus target.
  local -A rm=(); local x
  for x in ${removed//,/ } "$tgt"; do rm[$x]=1; done
  local TR=""
  for m in "${ALL[@]}"; do [[ -n ${rm[$m]:-} ]] && continue; TR=${TR:+$TR,}$m; done
  # Train (seed 42, retry only on divergence to match the baseline's seed policy).
  if [[ ! -f $out/${tgt}_proj.npy ]] || [[ $(npy_bad "$out/${tgt}_proj.npy") == 1 ]]; then
    local ok=0 seed
    for seed in 42 43 44; do
      rm -f "$out/${tgt}_proj.npy" "$out/${tgt}_dist_matrix.bin"
      "$PY" "$TRAIN" "${DA[@]}" --train "$TR" --eval "$tgt" "${DEPLOYED_ARGS[@]}" \
        --seed "$seed" --output-dir "$out" >"$out/train.seed${seed}.log" 2>&1 || true
      [[ $(npy_bad "$out/${tgt}_proj.npy") == 0 ]] && { ok=1; break; }
    done
    [[ $ok == 1 ]] || { echo "NA" > "$out/score.txt"; echo "  [$armid/$tgt] DIVERGED"; return 0; }
  fi
  # Blend at deployed weight + score (CPU; overlaps with other folds' GPU work).
  "$PY" "$BLEND" "$d" --learned "$out/${tgt}_dist_matrix.bin" \
    --weights "$BLEND_W" --output-pattern "$out/${tgt}_blend_{w}.bin" >/dev/null 2>&1
  local lv
  lv=$(ari "$d" --dist-matrix "$out/${tgt}_blend_${BLEND_W}.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
  echo "${lv:-NA}" > "$out/score.txt"
  echo "  [$armid/$tgt] ARI@${BLEND_W}=${lv:-NA}"
}

# Build the task list (skip target when it IS one of the removed sets — that fold
# is identical to the baseline, so Delta=0 by construction).
TASKS=()
for armid in "${ARM_ORDER[@]}"; do
  removed=${ARMS[$armid]}
  declare -A isrm=(); for x in ${removed//,/ }; do isrm[$x]=1; done
  for t in "${TARGETS[@]}"; do
    [[ -n ${isrm[$t]:-} ]] && continue
    TASKS+=("$armid|$removed|$t")
  done
  unset isrm
done
echo "Big-set ablation: ${#ARM_ORDER[@]} arms, ${#TASKS[@]} folds, concurrency $CONC, blend $BLEND_W"
echo "ROOT=$ROOT  baseline=$BASELINE_TSV"

# Run with GPU concurrency CONC.
sem=0
for task in "${TASKS[@]}"; do
  IFS='|' read -r armid removed tgt <<< "$task"
  do_fold "$armid" "$removed" "$tgt" &
  if (( ++sem >= CONC )); then wait -n; ((sem--)); fi
done
wait
echo "ALL FOLDS DONE."

# Aggregate into one flat results file (armid, target, learned ARI).
RES=$ROOT/results.tsv
printf "arm\ttarget\tlearned_ari\n" > "$RES"
for armid in "${ARM_ORDER[@]}"; do
  for t in "${TARGETS[@]}"; do
    s=$ROOT/$armid/$t/score.txt
    [[ -f $s ]] && printf "%s\t%s\t%s\n" "$armid" "$t" "$(cat "$s")" >> "$RES"
  done
done
echo "Wrote $RES"
