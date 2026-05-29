#!/usr/bin/env bash
# Final head-to-head: FULL 20-dataset LOMO of the deployed baseline config vs the
# sweep winner (temperature=0.07, out-dim=512, lr=1e-4).
#
# For each of the 20 labeled datasets: hold it out, train the head on the other
# 19, score best-blend ARI (over {100%,30%,50%,70%}). Reports per-dataset and
# average Δ-ARI vs zero-shot, for both configs, plus how often the winner wins.
#
#   ./run_full_lomo.sh
set -euo pipefail

HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
OUTROOT=${OUTROOT:-/tmp/full_lomo}

BASE_ARGS=(
  --train _SET_ --within-holdout-frac 0
  --epochs 15 --batches-per-epoch 400 --p-groups 32 --k-images 12
  --out-dim 256 --hidden 1024 --dropout 0.1 --temperature 0.1
  --lr 3e-4 --weight-decay 1e-4
  --mixup-alpha 0.4 --drop-color-prob 0.5
  --cross-mixup-prob 0.3 --cross-mixup-alpha 0.4
  --hard-neg-frac 0.5 --hard-neg-pool-k 20
  --arcface-weight 0
  --use-augmented-views --use-singleton-negatives
)
WINNER_OVERRIDE="--temperature 0.07 --out-dim 512 --lr 1e-4"

mkdir -p "$OUTROOT"

# Trains one fold for a config; echoes best-blend ARI or NAN.
run_fold() {  # $1=config-outdir $2=target $3=override
  local cdir=$1 tgt=$2 ovr=$3 d; d=$(dir "$tgt")
  local out=$cdir/$tgt; mkdir -p "$out"
  local DA=() TR=""
  for m in "${ALL[@]}"; do
    DA+=(--dataset "$m:$(dir "$m")")
    [[ $m != "$tgt" ]] && { [[ -z $TR ]] && TR=$m || TR=$TR,$m; }
  done
  local ARGS=("${BASE_ARGS[@]}"); ARGS[1]=$TR   # fill in --train SET
  if [[ ! -f "$out/${tgt}_proj.npy" ]] || [[ $(npy_bad "$out/${tgt}_proj.npy") == 1 ]] || [[ ! -f "$out/${tgt}_blend_0.5.bin" ]]; then
    local ok=0
    for seed in 42 43 44; do
      rm -f "$out/${tgt}_proj.npy" "$out/${tgt}_dist_matrix.bin" "$out/${tgt}_blend_"*.bin
      "$PY" "$TRAIN" "${DA[@]}" --eval "$tgt" "${ARGS[@]}" $ovr --seed "$seed" \
        --output-dir "$out" >"$out/train.seed${seed}.log" 2>&1 || true
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

TSV=$OUTROOT/full_lomo.tsv
printf "target\tname\tzeroshot\tbase_best\twin_best\tbase_delta\twin_delta\twin_minus_base\n" > "$TSV"
echo "FULL 20-dataset LOMO: deployed baseline vs winner (temp=0.07 dim=512 lr=1e-4)"
printf "%-6s %-18s | %-8s | %-8s | %-8s | %-9s | %-9s | %-9s\n" \
  "tgt" "name" "zeroshot" "base" "winner" "Δbase" "Δwin" "win-base"
echo "--------------------------------------------------------------------------------------------------------"

bsum=0; wsum=0; zsum=0; nwin=0; ntie=0; nloss=0; ndone=0; nan_note=""
for tgt in "${ALL[@]}"; do
  d=$(dir "$tgt")
  zs=$(ari "$d")                                        # zero-shot (default weights)
  bb=$(run_fold "$OUTROOT/baseline" "$tgt" "")
  wb=$(run_fold "$OUTROOT/winner"   "$tgt" "$WINNER_OVERRIDE")
  if [[ $bb == NAN || $wb == NAN || -z $zs ]]; then
    printf "%-6s %-18s | %-8s | %-8s | %-8s | %-9s | %-9s | %-9s\n" "$tgt" "${P[$tgt]}" "${zs:-NA}" "$bb" "$wb" "NA" "NA" "NA"
    nan_note+=" $tgt"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tgt" "${P[$tgt]}" "${zs:-NA}" "$bb" "$wb" "NA" "NA" "NA" >> "$TSV"
    continue
  fi
  bd=$(awk "BEGIN{printf \"%.4f\", $bb-$zs}")
  wd=$(awk "BEGIN{printf \"%.4f\", $wb-$zs}")
  diff=$(awk "BEGIN{printf \"%+.4f\", $wb-$bb}")
  bsum=$(awk "BEGIN{print $bsum+$bd}"); wsum=$(awk "BEGIN{print $wsum+$wd}"); zsum=$(awk "BEGIN{print $zsum+$zs}")
  ndone=$((ndone+1))
  cmp=$(awk "BEGIN{d=$wb-$bb; print (d>0.0005)?\"W\":((d<-0.0005)?\"L\":\"T\")}")
  case $cmp in W) nwin=$((nwin+1));; L) nloss=$((nloss+1));; T) ntie=$((ntie+1));; esac
  printf "%-6s %-18s | %-8s | %-8s | %-8s | %-9s | %-9s | %-9s\n" "$tgt" "${P[$tgt]}" "$zs" "$bb" "$wb" "$bd" "$wd" "$diff"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tgt" "${P[$tgt]}" "$zs" "$bb" "$wb" "$bd" "$wd" "$diff" >> "$TSV"
done

echo "--------------------------------------------------------------------------------------------------------"
bavg=$(awk "BEGIN{printf \"%+.4f\", $bsum/$ndone}")
wavg=$(awk "BEGIN{printf \"%+.4f\", $wsum/$ndone}")
liftw=$(awk "BEGIN{printf \"%+.4f\", ($wsum-$bsum)/$ndone}")
zavg=$(awk "BEGIN{printf \"%.4f\", $zsum/$ndone}")
printf "AVG over %d datasets   zero-shot=%s   Δbase=%s   Δwin=%s   winner lift=%s\n" "$ndone" "$zavg" "$bavg" "$wavg" "$liftw"
printf "Winner vs baseline per-dataset:  %d win / %d tie / %d loss\n" "$nwin" "$ntie" "$nloss"
[[ -n $nan_note ]] && echo "NOTE: diverged/skipped targets:$nan_note"
echo "TSV: $TSV"
