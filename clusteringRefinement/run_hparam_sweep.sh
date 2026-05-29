#!/usr/bin/env bash
# Coordinate-descent hyperparameter / augmentation sweep on a 3-dataset dev set.
#
# Dev set spans the range (per LEARNED_HEAD.md cheap-signal recipe):
#   M2  big winner   (baseline Δ +0.173)
#   M11 high-baseline (baseline Δ +0.085)
#   M19 weak/regressor(baseline Δ +0.038)
#
# Each config = the deployed winning config (train_final_head DEFAULT_HYPERPARAMS)
# with ONE axis perturbed (argparse last-wins, so override flags are appended).
# For each dev target: LOMO train on all-other-19 datasets, blend, take best ARI
# over {pure,30%,50%,70%}. Reports avg Δ-ARI vs zero-shot, and the lift over the
# unperturbed baseline (reused from /tmp/lomo_baseline/results.tsv).
#
#   ./run_hparam_sweep.sh
set -euo pipefail

HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
OUTROOT=${OUTROOT:-/tmp/hparam_sweep}
BASELINE_TSV=${BASELINE_TSV:-/tmp/lomo_baseline/results.tsv}
DEV=(M2 M11 M19)

# Deployed winning config (DEFAULT_HYPERPARAMS).
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

# Configs: "label|override flags". One axis each (+ a few combos at the end).
CONFIGS=(
  "k8|--k-images 8"
  "k16|--k-images 16"
  "k20|--k-images 20"
  "temp07|--temperature 0.07"
  "temp15|--temperature 0.15"
  "dim384|--out-dim 384"
  "dim512|--out-dim 512"
  "ep20|--epochs 20"
  "ep25|--epochs 25"
  "dropcol3|--drop-color-prob 0.3"
  "dropcol7|--drop-color-prob 0.7"
  "mixup2|--mixup-alpha 0.2"
  "mixup6|--mixup-alpha 0.6"
  "crossmix5|--cross-mixup-prob 0.5"
  "hardneg7|--hard-neg-frac 0.7"
  "dropout2|--dropout 0.2"
  "hidden1536|--hidden 1536"
  "lr1e3|--lr 1e-3"
  "lr1e4|--lr 1e-4"
  "wd1e3|--weight-decay 1e-3"
)

mkdir -p "$OUTROOT"

# Read baseline (zero-shot base + best-blend delta) per dev target from the
# baseline TSV: target  base  pure  b30  b50  b70  best  delta  blend
declare -A BASE_ZS BASE_DELTA
while IFS=$'\t' read -r tgt zs _pure _b30 _b50 _b70 _best delta _blend; do
  [[ $tgt == target ]] && continue
  BASE_ZS[$tgt]=$zs; BASE_DELTA[$tgt]=$delta
done < "$BASELINE_TSV"
base_ref_sum=0
for m in "${DEV[@]}"; do base_ref_sum=$(awk "BEGIN{print $base_ref_sum+${BASE_DELTA[$m]}}"); done
BASE_REF_AVG=$(awk "BEGIN{printf \"%.4f\", $base_ref_sum/${#DEV[@]}}")

# Trains + blends one LOMO fold, echoes best Δ-ARI vs zero-shot, or "NAN" if the
# head diverged (retries once with a different seed before giving up).
run_fold() {  # $1=config-outdir $2=target
  local cdir=$1 tgt=$2 d; d=$(dir "$tgt")
  local out=$cdir/$tgt; mkdir -p "$out"
  local DA=() TR=""
  for m in "${ALL[@]}"; do
    DA+=(--dataset "$m:$(dir "$m")")
    [[ $m != "$tgt" ]] && { [[ -z $TR ]] && TR=$m || TR=$TR,$m; }
  done
  # (Re)train unless a clean (non-NaN) projection + blends already exist.
  if [[ ! -f "$out/${tgt}_proj.npy" ]] || [[ $(npy_bad "$out/${tgt}_proj.npy") == 1 ]] || [[ ! -f "$out/${tgt}_blend_0.5.bin" ]]; then
    local ok=0
    for seed in 42 43; do
      rm -f "$out/${tgt}_proj.npy" "$out/${tgt}_dist_matrix.bin" "$out/${tgt}_blend_"*.bin
      "$PY" "$TRAIN" "${DA[@]}" --train "$TR" --eval "$tgt" \
        "${BASE_ARGS[@]}" $OVERRIDE --seed "$seed" --output-dir "$out" >"$out/train.seed${seed}.log" 2>&1 || true
      if [[ $(npy_bad "$out/${tgt}_proj.npy") == 0 ]]; then ok=1; cp "$out/train.seed${seed}.log" "$out/train.log"; break; fi
    done
    [[ $ok == 1 ]] || { echo "NAN"; return; }
    "$PY" "$BLEND" "$d" --learned "$out/${tgt}_dist_matrix.bin" \
      --weights 0.3,0.5,0.7 --output-pattern "$out/${tgt}_blend_{w}.bin" >/dev/null 2>&1
  fi
  local pure b30 b50 b70 best
  pure=$(ari "$d" --dist-matrix "$out/${tgt}_dist_matrix.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
  b30=$(ari "$d" --dist-matrix "$out/${tgt}_blend_0.3.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
  b50=$(ari "$d" --dist-matrix "$out/${tgt}_blend_0.5.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
  b70=$(ari "$d" --dist-matrix "$out/${tgt}_blend_0.7.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
  best=""
  for v in "$pure" "$b30" "$b50" "$b70"; do
    [[ -z $v ]] && continue
    [[ -z $best ]] && { best=$v; continue; }
    awk "BEGIN{exit !($v>$best)}" && best=$v
  done
  [[ -z $best ]] && { echo "NAN"; return; }
  awk "BEGIN{printf \"%.4f\", $best-${BASE_ZS[$tgt]}}"   # Δ vs zero-shot
}

SUMMARY=$OUTROOT/summary.tsv
printf "config\tM2_d\tM11_d\tM19_d\tavg_d\tlift_vs_baseline\t_sortkey\n" > "$SUMMARY"
echo "Dev set: ${DEV[*]}   baseline avg Δ = $BASE_REF_AVG  (M2=${BASE_DELTA[M2]} M11=${BASE_DELTA[M11]} M19=${BASE_DELTA[M19]})"
echo "Sweeping ${#CONFIGS[@]} configs (each axis perturbed from the deployed winner)."

# Baseline row first (reuse precomputed numbers, no recompute)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "BASELINE" "${BASE_DELTA[M2]}" "${BASE_DELTA[M11]}" "${BASE_DELTA[M19]}" "$BASE_REF_AVG" "+0.0000" "$BASE_REF_AVG" >> "$SUMMARY"

for entry in "${CONFIGS[@]}"; do
  label=${entry%%|*}; OVERRIDE=${entry#*|}
  cdir=$OUTROOT/$label
  echo ""
  echo "==== config '$label'   ($OVERRIDE) ===="
  declare -A D=()
  sum=0; ndone=0; nnan=0
  for m in "${DEV[@]}"; do
    D[$m]=$(run_fold "$cdir" "$m")
    if [[ ${D[$m]} == NAN ]]; then
      nnan=$((nnan+1)); echo "   $m  Δ=NAN (diverged)"
    else
      sum=$(awk "BEGIN{print $sum+${D[$m]}}"); ndone=$((ndone+1))
      echo "   $m  Δ=${D[$m]}"
    fi
  done
  if [[ $ndone -eq 0 ]]; then
    avg=NAN; lift=NAN; sortkey=-9
    echo "   avg Δ=NAN (all folds diverged)"
  else
    avg=$(awk "BEGIN{printf \"%.4f\", $sum/$ndone}")
    lift=$(awk "BEGIN{printf \"%+.4f\", $avg-$BASE_REF_AVG}")
    sortkey=$avg
    note=""; [[ $nnan -gt 0 ]] && note="  ($nnan/${#DEV[@]} folds diverged — avg over completed only)"
    echo "   avg Δ=$avg   lift vs baseline=$lift$note"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$label" "${D[M2]}" "${D[M11]}" "${D[M19]}" "$avg" "$lift" "$sortkey" >> "$SUMMARY"
done

echo ""
echo "==================== SWEEP SUMMARY (sorted by avg Δ-ARI) ===================="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k7 -gr; } | cut -f1-6 | column -t -s$'\t'
echo ""
echo "Full TSV: $SUMMARY"
echo "Baseline dev avg Δ-ARI = $BASE_REF_AVG. Positive 'lift_vs_baseline' = improvement."
