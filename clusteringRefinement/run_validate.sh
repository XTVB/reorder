#!/usr/bin/env bash
# Validate the sweep's top candidates on the FULL 5-target set (not just the
# 3-dataset dev set), and test whether the two trusted orthogonal winners
# (temp=0.07, out-dim=512) stack. Compares each to the 5-target baseline.
#
#   ./run_validate.sh
set -euo pipefail

HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
OUTROOT=${OUTROOT:-/tmp/hparam_validate}
BASELINE_TSV=${BASELINE_TSV:-/tmp/lomo_baseline/results.tsv}
TARGETS=(M16 M19 M2 M11 M3)

BASE_ARGS=(
  --within-holdout-frac 0
  --epochs 15 --batches-per-epoch 400 --p-groups 32 --k-images 12
  --out-dim 256 --hidden 1024 --dropout 0.1 --temperature 0.1
  --lr 3e-4 --weight-decay 1e-4
  --mixup-alpha 0.4 --drop-color-prob 0.5
  --cross-mixup-prob 0.3 --cross-mixup-alpha 0.4
  --hard-neg-frac 0.5 --hard-neg-pool-k 20
  --arcface-weight 0
  --use-augmented-views --use-singleton-negatives
)

# Validation candidates: the two trusted winners, their combo, and the combo
# plus the lower-LR axis (which also helped on dev).
CONFIGS=(
  "temp07|--temperature 0.07"
  "dim512|--out-dim 512"
  "temp07_dim512|--temperature 0.07 --out-dim 512"
  "temp07_dim512_lr1e4|--temperature 0.07 --out-dim 512 --lr 1e-4"
)
# Override the candidate list via env: CONFIGS_ENV="label|flags;label2|flags2"
if [[ -n "${CONFIGS_ENV:-}" ]]; then
  IFS=';' read -ra CONFIGS <<< "$CONFIGS_ENV"
fi
# Seeds to try per fold (first that trains clean wins). Override via SEEDS env.
read -ra SEED_LIST <<< "${SEEDS:-42 43}"

mkdir -p "$OUTROOT"

declare -A BASE_ZS BASE_BEST
while IFS=$'\t' read -r tgt zs _pure _b30 _b50 _b70 best _delta _blend; do
  [[ $tgt == target ]] && continue
  BASE_ZS[$tgt]=$zs; BASE_BEST[$tgt]=$best
done < "$BASELINE_TSV"

# Trains a fold, echoes "best_ari" (absolute, not delta), or "NAN".
run_fold() {  # $1=config-outdir $2=target
  local cdir=$1 tgt=$2 d; d=$(dir "$tgt")
  local out=$cdir/$tgt; mkdir -p "$out"
  local DA=() TR=""
  for m in "${ALL[@]}"; do
    DA+=(--dataset "$m:$(dir "$m")")
    [[ $m != "$tgt" ]] && { [[ -z $TR ]] && TR=$m || TR=$TR,$m; }
  done
  if [[ ! -f "$out/${tgt}_proj.npy" ]] || [[ $(npy_bad "$out/${tgt}_proj.npy") == 1 ]] || [[ ! -f "$out/${tgt}_blend_0.5.bin" ]]; then
    local ok=0
    for seed in "${SEED_LIST[@]}"; do
      rm -f "$out/${tgt}_proj.npy" "$out/${tgt}_dist_matrix.bin" "$out/${tgt}_blend_"*.bin
      "$PY" "$TRAIN" "${DA[@]}" --train "$TR" --eval "$tgt" \
        "${BASE_ARGS[@]}" $OVERRIDE --seed "$seed" --output-dir "$out" >"$out/train.seed${seed}.log" 2>&1 || true
      if [[ $(npy_bad "$out/${tgt}_proj.npy") == 0 ]]; then ok=1; break; fi
    done
    [[ $ok == 1 ]] || { echo "NAN"; return; }
    "$PY" "$BLEND" "$d" --learned "$out/${tgt}_dist_matrix.bin" \
      --weights 0.3,0.5,0.7 --output-pattern "$out/${tgt}_blend_{w}.bin" >/dev/null 2>&1
  fi
  local best="" v
  for f in "${tgt}_dist_matrix" "${tgt}_blend_0.3" "${tgt}_blend_0.5" "${tgt}_blend_0.7"; do
    v=$(ari "$d" --dist-matrix "$out/$f.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
    [[ -z $v ]] && continue
    [[ -z $best ]] && { best=$v; continue; }
    awk "BEGIN{exit !($v>$best)}" && best=$v
  done
  [[ -z $best ]] && echo "NAN" || echo "$best"
}

# Baseline reference (5-target avg Δ) from the TSV.
bsum=0
for m in "${TARGETS[@]}"; do bsum=$(awk "BEGIN{print $bsum+(${BASE_BEST[$m]}-${BASE_ZS[$m]})}"); done
BASE_AVG_DELTA=$(awk "BEGIN{printf \"%.4f\", $bsum/${#TARGETS[@]}}")

echo "Full 5-target validation. Baseline avg Δ-ARI = $BASE_AVG_DELTA"
echo "Targets: ${TARGETS[*]}"
printf "%-20s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n" \
  "config" "${TARGETS[@]}" "avgΔ" "lift"
echo "------------------------------------------------------------------------------------------------------------"
# Baseline row (best ARIs from TSV)
printf "%-20s |" "BASELINE"
for m in "${TARGETS[@]}"; do printf " %-8s |" "${BASE_BEST[$m]}"; done
printf " %-8s | %-8s\n" "$BASE_AVG_DELTA" "+0.0000"

for entry in "${CONFIGS[@]}"; do
  label=${entry%%|*}; OVERRIDE=${entry#*|}
  cdir=$OUTROOT/$label
  declare -A B=(); sum=0; ndone=0
  for m in "${TARGETS[@]}"; do
    B[$m]=$(run_fold "$cdir" "$m")
    if [[ ${B[$m]} != NAN ]]; then
      sum=$(awk "BEGIN{print $sum+(${B[$m]}-${BASE_ZS[$m]})}"); ndone=$((ndone+1))
    fi
  done
  if [[ $ndone -eq 0 ]]; then avg=NAN; lift=NAN; else
    avg=$(awk "BEGIN{printf \"%.4f\", $sum/$ndone}")
    lift=$(awk "BEGIN{printf \"%+.4f\", $avg-$BASE_AVG_DELTA}")
  fi
  printf "%-20s |" "$label"
  for m in "${TARGETS[@]}"; do printf " %-8s |" "${B[$m]}"; done
  printf " %-8s | %-8s\n" "$avg" "$lift"
done
echo "------------------------------------------------------------------------------------------------------------"
echo "(cells are best-blend ARI per target; avgΔ vs zero-shot; lift vs deployed baseline)"
