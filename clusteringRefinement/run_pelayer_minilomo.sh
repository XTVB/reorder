#!/usr/bin/env bash
# Paired mini-LOMO over the 4 datasets with clean pe_layers extractions
# (M1/M3/M5/M6): control head (peg+color) vs layer-augmented head
# (peg+color+L47-attnpool), same seeds, deployed hyperparams. Small train
# pools (3 sets) — only the PAIRED Δ is meaningful, not absolute ARI.
#
#   PE_LAYER="47:attnpool" SEEDS="42 43" ./run_pelayer_minilomo.sh
set -euo pipefail

HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
CLEAN=(M1 M3 M5 M6)
PE_LAYER=${PE_LAYER:-47:attnpool}
SEEDS=${SEEDS:-42 43}
OUTROOT=${OUTROOT:-$HOME/.cache/reorder/pelayer_minilomo}

DEPLOYED_ARGS=(
  --within-holdout-frac 0
  --epochs 15 --batches-per-epoch 400 --p-groups 32 --k-images 12
  --out-dim 512 --hidden 1024 --dropout 0.1 --temperature 0.07
  --lr 1e-4 --weight-decay 1e-4
  --mixup-alpha 0.4 --drop-color-prob 0.5
  --cross-mixup-prob 0.3 --cross-mixup-alpha 0.4
  --hard-neg-frac 0.5 --hard-neg-pool-k 20
  --arcface-weight 0
  --use-singleton-negatives --use-augmented-views
)

DA=()
for m in "${CLEAN[@]}"; do DA+=(--dataset "$m:$(dir "$m")"); done

for tgt in "${CLEAN[@]}"; do
  TR=""
  for m in "${CLEAN[@]}"; do
    [[ $m != "$tgt" ]] && { [[ -z $TR ]] && TR=$m || TR=$TR,$m; }
  done
  for arm in ctrl layer; do
    EXTRA=()
    [[ $arm == layer ]] && EXTRA=(--pe-layer "$PE_LAYER")
    for seed in $SEEDS; do
      out=$OUTROOT/$arm/$tgt/s$seed
      if [[ -f "$out/${tgt}_proj.npy" && $(npy_bad "$out/${tgt}_proj.npy") == 0 ]]; then
        echo "  $tgt/$arm/s$seed cached"; continue
      fi
      mkdir -p "$out"
      echo "== $tgt arm=$arm seed=$seed (train $TR)"
      "$PY" "$TRAIN" "${DA[@]}" --train "$TR" --eval "$tgt" \
        "${DEPLOYED_ARGS[@]}" "${EXTRA[@]}" --seed "$seed" --output-dir "$out" \
        > "$out/train.log" 2>&1
    done
  done
done
echo "MINI-LOMO TRAINING DONE → $OUTROOT"
