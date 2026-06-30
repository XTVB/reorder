#!/usr/bin/env bash
# Unified baseline + layer-augmented sweep for the 3-head (joint/peg/color) prod
# blend, at the DEPLOYED head config (epochs=12, cosine LR) so numbers transfer
# straight to production. Batched 3-seed MLX per fold.
#
# Jobs (output subdir under $AUGROOT, all batched --seeds → s<seed>/ inside):
#   base/joint   mods=peg,color  drop-color 0.5   (no layer)   } regenerated
#   base/peg     mods=peg        drop-color 0.5   (no layer)   } baselines at
#   base/color   mods=color      drop-color 0.0   (no layer)   } 12/cosine
#   joint/<L>_<pool>  mods=peg,color drop-color 0.5  +pe-layer  (layer→joint)
#   peg/<L>_<pool>    mods=peg       drop-color 0.5  +pe-layer  (layer→peg)
# (color head never gets the PE-G-derived layer.)
#
# Roster = all 27 datasets incl. partials (M14/M15 stay in training pools, as in
# the deployed head) and M27. Eval folds = 25 (all minus the two partials).
# Output: $AUGROOT/<job>/<tgt>/s<seed>/<tgt>_proj.npy. Resumable: a (job,fold)
# whose 3 seed projs all exist is skipped.
#
#   LAYERS_POOLS="47:attnpool" PLACEMENTS=joint TARGETS=M5 ./run_pelayer_splithead_sweep.sh
set -euo pipefail
HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
AUGROOT=${AUGROOT:-$HOME/.cache/reorder/pelayer_splithead12}
SEEDS=${SEEDS:-42,43,44}
EPOCHS=${EPOCHS:-12}
LAYERS_POOLS=${LAYERS_POOLS:-"42:mean 42:gem3 42:attnpool 44:mean 44:gem3 44:attnpool 46:mean 46:gem3 46:attnpool 47:mean 47:gem3 47:attnpool"}
PLACEMENTS=${PLACEMENTS:-"joint peg"}
DO_BASELINES=${DO_BASELINES:-1}
# Eval folds: all registered minus the two partials (M14/M15 train-only).
if [[ -n "${TARGETS:-}" ]]; then read -ra TGTS <<< "$TARGETS"
else TGTS=(); for m in "${ALL[@]}"; do [[ $m =~ ^M(14|15)$ ]] || TGTS+=("$m"); done; fi

# Shared config = deployed head recipe minus the per-job bits (mods, drop-color,
# pe-layer). epochs=12 + cosine LR matches learned_head.json exactly.
COMMON=(
  --within-holdout-frac 0 --epochs "$EPOCHS" --lr-schedule cosine
  --batches-per-epoch 400 --p-groups 32 --k-images 12
  --out-dim 512 --hidden 1024 --dropout 0.1 --temperature 0.07 --lr 1e-4 --weight-decay 1e-4
  --grad-clip 5.0 --mixup-alpha 0.4 --cross-mixup-prob 0.3 --cross-mixup-alpha 0.4
  --hard-neg-frac 0.5 --hard-neg-pool-k 20 --arcface-weight 0
  --use-singleton-negatives --use-augmented-views --backend mlx --seeds "$SEEDS"
)
DA=(); for m in "${ALL[@]}"; do DA+=(--dataset "$m:$(dir "$m")"); done
IFS=',' read -ra SEEDLIST <<< "$SEEDS"

# Build job list: "<subdir>|<mods>|<drop_color>|<pe_layer or '-'>"
JOBS=()
if [[ $DO_BASELINES == 1 ]]; then
  JOBS+=("base/joint|peg,color|0.5|-" "base/peg|peg|0.5|-" "base/color|color|0.0|-")
fi
for lp in $LAYERS_POOLS; do L=${lp%%:*}; pool=${lp##*:}
  for plc in $PLACEMENTS; do
    if [[ $plc == joint ]]; then JOBS+=("joint/${L}_${pool}|peg,color|0.5|${L}:${pool}")
    else JOBS+=("peg/${L}_${pool}|peg|0.5|${L}:${pool}"); fi
  done
done

total=$(( ${#JOBS[@]} * ${#TGTS[@]} )); done=0; trained=0
echo "Split-head sweep @ 12/cosine: ${#JOBS[@]} jobs × ${#TGTS[@]} folds = $total batched runs -> $AUGROOT"
for job in "${JOBS[@]}"; do
  IFS='|' read -r sub mods dc layer <<< "$job"
  for tgt in "${TGTS[@]}"; do
    out=$AUGROOT/$sub/$tgt
    ok=1; for s in "${SEEDLIST[@]}"; do [[ -f "$out/s$s/${tgt}_proj.npy" && $(npy_bad "$out/s$s/${tgt}_proj.npy") == 0 ]] || ok=0; done
    if [[ $ok == 1 ]]; then done=$((done+1)); continue; fi
    TR=""; for m in "${ALL[@]}"; do [[ $m == "$tgt" ]] && continue; [[ -z $TR ]] && TR=$m || TR=$TR,$m; done
    mkdir -p "$out"
    layer_args=(); [[ $layer != "-" ]] && layer_args=(--pe-layer "$layer")
    echo "[$((done+trained+1))/$total] $sub eval=$tgt (mods=$mods dc=$dc ${layer/-/no-layer})"
    "$PY" "$TRAIN" "${DA[@]}" --train "$TR" --eval "$tgt" "${COMMON[@]}" \
      --input-mods "$mods" --drop-color-prob "$dc" "${layer_args[@]}" --output-dir "$out" \
      > "$out/train.log" 2>&1 || { echo "  FAILED (see $out/train.log)"; continue; }
    trained=$((trained+1))
  done
done
echo "SWEEP DONE: $trained trained, $done already-cached, of $total"
