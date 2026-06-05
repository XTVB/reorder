#!/usr/bin/env bash
# Quantify single-seed ARI noise: retrain the same folds across several seeds and
# measure the spread. Two arms on the same 6 targets/seeds:
#   base : full pool (ALL \ target)          → raw per-fold seed sigma
#   noM9 : pool minus M9 (ALL \ M9 \ target) → is the -M9 effect real or noise?
set -uo pipefail
HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
ROOT=${ROOT:-/tmp/lomo_noise}; BLEND_W=0.60; CONC=${CONC:-2}
SEEDS=(42 43 44 45 46)
TGTS=(M4 M19 M22 M10 M11 M17)   # span low→high baseline ARI
mkdir -p "$ROOT"
DEPLOYED_ARGS=(--within-holdout-frac 0 --epochs 15 --batches-per-epoch 400 --p-groups 32 --k-images 12
  --out-dim 512 --hidden 1024 --dropout 0.1 --temperature 0.07 --lr 1e-4 --weight-decay 1e-4
  --mixup-alpha 0.4 --drop-color-prob 0.5 --cross-mixup-prob 0.3 --cross-mixup-alpha 0.4
  --hard-neg-frac 0.5 --hard-neg-pool-k 20 --arcface-weight 0 --use-singleton-negatives --use-augmented-views)
DA=(); for m in "${ALL[@]}"; do DA+=(--dataset "$m:$(dir "$m")"); done

do_fold() {  # <kind> <removeCSV-or-none> <target> <seed>
  local kind=$1 remove=$2 tgt=$3 seed=$4
  local out=$ROOT/$kind/${tgt}_s${seed}; mkdir -p "$out"
  [[ -f $out/score.txt ]] && return 0
  local d; d=$(dir "$tgt")
  local -A rm=(); local x
  [[ $remove != none ]] && for x in ${remove//,/ }; do rm[$x]=1; done
  rm[$tgt]=1
  local TR="" m
  for m in "${ALL[@]}"; do [[ -n ${rm[$m]:-} ]] && continue; TR=${TR:+$TR,}$m; done
  "$PY" "$TRAIN" "${DA[@]}" --train "$TR" --eval "$tgt" "${DEPLOYED_ARGS[@]}" \
    --seed "$seed" --output-dir "$out" >"$out/train.log" 2>&1 || true
  if [[ $(npy_bad "$out/${tgt}_proj.npy") == 1 ]]; then echo NA > "$out/score.txt"; return 0; fi
  "$PY" "$BLEND" "$d" --learned "$out/${tgt}_dist_matrix.bin" --weights "$BLEND_W" \
    --output-pattern "$out/b_{w}.bin" >/dev/null 2>&1
  local lv; lv=$(ari "$d" --dist-matrix "$out/b_${BLEND_W}.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
  echo "${lv:-NA}" > "$out/score.txt"
  echo "  [$kind $tgt seed$seed] $lv"
}

TASKS=()
for s in "${SEEDS[@]}"; do for t in "${TGTS[@]}"; do
  TASKS+=("base|none|$t|$s"); TASKS+=("noM9|M9|$t|$s")
done; done
echo "Seed-noise: ${#TASKS[@]} folds (2 arms × ${#TGTS[@]} targets × ${#SEEDS[@]} seeds), conc $CONC"
sem=0
for task in "${TASKS[@]}"; do
  IFS='|' read -r k r t s <<< "$task"
  do_fold "$k" "$r" "$t" "$s" &
  if (( ++sem >= CONC )); then wait -n; ((sem--)); fi
done
wait
RES=$ROOT/results.tsv; printf "kind\ttarget\tseed\tari\n" > "$RES"
for k in base noM9; do for t in "${TGTS[@]}"; do for s in "${SEEDS[@]}"; do
  f=$ROOT/$k/${t}_s${s}/score.txt; [[ -f $f ]] && printf "%s\t%s\t%s\t%s\n" "$k" "$t" "$s" "$(cat "$f")" >> "$RES"
done; done; done
echo "DONE → $RES"
