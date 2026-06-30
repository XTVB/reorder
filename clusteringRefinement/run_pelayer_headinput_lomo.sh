#!/usr/bin/env bash
# Paired full LOMO for the PE-layer-as-HEAD-INPUT test, on the current 27-dataset
# roster. Two arms per held-out target, same seed:
#   ctrl  : deployed head (peg+color input)
#   layer : deployed head + --pe-layer 47:attnpool input (dim 1973 -> 3509)
# Writes <tgt>_proj.npy per fold/arm; score grouped-only with
# score_pelayer_headinput.py (paired Δ on the labeled images).
set -euo pipefail
HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
SEED=${SEED:-42}
PE_LAYER=${PE_LAYER:-47:attnpool}
OUTROOT=${OUTROOT:-$HOME/.cache/reorder/pelayer_hi_lomo}
# Eval over non-outlier, non-partial targets (M7 dense / M14,M15 partial excluded).
if [[ -n "${TARGETS:-}" ]]; then read -ra TGTS <<< "$TARGETS"
else TGTS=(); for m in "${ALL[@]}"; do [[ $m =~ ^M(7|14|15)$ ]] || TGTS+=("$m"); done; fi

DEPLOYED_ARGS=(
  --within-holdout-frac 0 --epochs 15 --batches-per-epoch 400 --p-groups 32 --k-images 12
  --out-dim 512 --hidden 1024 --dropout 0.1 --temperature 0.07 --lr 1e-4 --weight-decay 1e-4
  --mixup-alpha 0.4 --drop-color-prob 0.5 --cross-mixup-prob 0.3 --cross-mixup-alpha 0.4
  --hard-neg-frac 0.5 --hard-neg-pool-k 20 --arcface-weight 0
  --use-singleton-negatives --use-augmented-views
)
DA=(); for m in "${ALL[@]}"; do DA+=(--dataset "$m:$(dir "$m")"); done

echo "PE-layer head-input LOMO  seed=$SEED  layer=$PE_LAYER  ${#TGTS[@]} targets  -> $OUTROOT"
for tgt in "${TGTS[@]}"; do
  TR=""; for m in "${ALL[@]}"; do [[ $m != "$tgt" ]] && { [[ -z $TR ]] && TR=$m || TR=$TR,$m; }; done
  for arm in ctrl layer; do
    out=$OUTROOT/$arm/$tgt; mkdir -p "$out"
    [[ -f "$out/${tgt}_proj.npy" && $(npy_bad "$out/${tgt}_proj.npy") == 0 ]] && { echo "  $tgt/$arm cached"; continue; }
    EXTRA=(); [[ $arm == layer ]] && EXTRA=(--pe-layer "$PE_LAYER")
    echo "== $tgt arm=$arm"
    "$PY" "$TRAIN" "${DA[@]}" --train "$TR" --eval "$tgt" "${DEPLOYED_ARGS[@]}" "${EXTRA[@]}" \
      --seed "$SEED" --output-dir "$out" > "$out/train.log" 2>&1 \
      || { echo "  $tgt/$arm FAILED (see $out/train.log)"; }
  done
done
echo "PELAYER-HI LOMO DONE -> $OUTROOT"
