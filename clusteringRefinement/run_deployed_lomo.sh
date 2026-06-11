#!/usr/bin/env bash
# Deployed-config 20-dataset LOMO scored at the deployed inference blend (0.60).
# Used to measure the impact of pixel-aug: run once BEFORE extracting more views
# (snapshot saved durably), then again AFTER, and diff.
#
# Uses the exact deployed config (train_final_head.py DEFAULT_HYPERPARAMS:
# out_dim=512, temperature=0.07, lr=1e-4, grad-clip=5.0, K=12, ep=15, ICOMB augs,
# pixel-aug views). Reuses cached folds in OUTROOT if present (so pointing it at
# an existing winner-folds dir just re-scores; a fresh OUTROOT retrains).
#
#   # Snapshot the pre-pixel-aug baseline (reuses the cached winner folds):
#   OUTROOT=/tmp/full_lomo/winner RESULT=clusteringRefinement/baseline_pre_pixelaug.tsv ./run_deployed_lomo.sh
#
#   # After extracting more pixel-aug views, train fresh and diff vs the snapshot:
#   OUTROOT=/tmp/lomo_postaug RESULT=clusteringRefinement/lomo_post_pixelaug.tsv \
#     BASELINE=clusteringRefinement/baseline_pre_pixelaug.tsv ./run_deployed_lomo.sh
set -euo pipefail

HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
OUTROOT=${OUTROOT:-/tmp/lomo_deployed}
RESULT=${RESULT:-$HERE/lomo_deployed.tsv}
BASELINE=${BASELINE:-}          # optional prior TSV to diff against
BLEND_W=${BLEND_W:-0.60}        # deployed inference blend
USE_AUG=${USE_AUG:-1}           # 1 = use pixel-aug views; 0 = ignore them entirely
EPOCHS=${EPOCHS:-15}            # deployed default is 15
# Which datasets to hold out & score (training pool is always the other 19).
# Override with TARGETS="M2 M11 M19" for a quick dev-set A/B.
if [[ -n "${TARGETS:-}" ]]; then read -ra SCORE_TARGETS <<< "$TARGETS"; else SCORE_TARGETS=("${ALL[@]}"); fi

# Deployed config = DEFAULT_HYPERPARAMS. grad-clip defaults to 5.0 in the trainer.
DEPLOYED_ARGS=(
  --within-holdout-frac 0
  --epochs "$EPOCHS" --batches-per-epoch 400 --p-groups 32 --k-images 12
  --out-dim 512 --hidden 1024 --dropout 0.1 --temperature 0.07
  --lr 1e-4 --weight-decay 1e-4
  --mixup-alpha 0.4 --drop-color-prob 0.5
  --cross-mixup-prob 0.3 --cross-mixup-alpha 0.4
  --hard-neg-frac 0.5 --hard-neg-pool-k 20
  --arcface-weight 0
  --use-singleton-negatives
)
# Pixel-aug views are opt-in here so we can run a no-aug control (USE_AUG=0).
[[ $USE_AUG == 1 ]] && DEPLOYED_ARGS+=(--use-augmented-views)
# Append arbitrary extra flags (e.g. EXTRA_ARGS="--blend-aware --blend-weight 0.6").
[[ -n "${EXTRA_ARGS:-}" ]] && DEPLOYED_ARGS+=($EXTRA_ARGS)
echo "USE_AUG=$USE_AUG  EXTRA_ARGS='${EXTRA_ARGS:-}'  (pixel-aug $([[ $USE_AUG == 1 ]] && echo ENABLED || echo DISABLED))"

mkdir -p "$OUTROOT" "$(dirname "$RESULT")"
printf "target\tname\tzeroshot\tblend%s\n" "$BLEND_W" > "$RESULT"
echo "Deployed-config LOMO @ blend $BLEND_W   (OUTROOT=$OUTROOT)"

zsum=0; lsum=0; n=0
for tgt in "${SCORE_TARGETS[@]}"; do
  d=$(dir "$tgt"); out=$OUTROOT/$tgt; mkdir -p "$out"
  # Train fold unless a clean cached one exists.
  if [[ ! -f "$out/${tgt}_proj.npy" ]] || [[ $(npy_bad "$out/${tgt}_proj.npy") == 1 ]]; then
    DA=(); TR=""
    for m in "${ALL[@]}"; do
      DA+=(--dataset "$m:$(dir "$m")")
      [[ $m != "$tgt" ]] && { [[ -z $TR ]] && TR=$m || TR=$TR,$m; }
    done
    ok=0
    # Retry ladder; override (e.g. SEEDS="43 143 243") to train an independent
    # replica for seed-ensembling without colliding with another replica's seeds.
    for seed in ${SEEDS:-42 43 44}; do
      rm -f "$out/${tgt}_proj.npy" "$out/${tgt}_dist_matrix.bin"
      "$PY" "$TRAIN" "${DA[@]}" --train "$TR" --eval "$tgt" "${DEPLOYED_ARGS[@]}" \
        --seed "$seed" --output-dir "$out" >"$out/train.seed${seed}.log" 2>&1 || true
      [[ $(npy_bad "$out/${tgt}_proj.npy") == 0 ]] && { ok=1; break; }
    done
    [[ $ok == 1 ]] || { echo "  $tgt: DIVERGED"; printf "%s\t%s\tNA\tNA\n" "$tgt" "${P[$tgt]}" >> "$RESULT"; continue; }
  fi
  # Score at the deployed blend.
  "$PY" "$BLEND" "$d" --learned "$out/${tgt}_dist_matrix.bin" \
    --weights "$BLEND_W" --output-pattern "$out/${tgt}_blend_{w}.bin" >/dev/null 2>&1
  zs=$(ari "$d")
  lv=$(ari "$d" --dist-matrix "$out/${tgt}_blend_${BLEND_W}.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
  printf "%-6s %-18s zero-shot=%s  blend%s=%s\n" "$tgt" "${P[$tgt]}" "$zs" "$BLEND_W" "$lv"
  printf "%s\t%s\t%s\t%s\n" "$tgt" "${P[$tgt]}" "$zs" "$lv" >> "$RESULT"
  zsum=$(awk "BEGIN{print $zsum+$zs}"); lsum=$(awk "BEGIN{print $lsum+$lv}"); n=$((n+1))
done

echo "------------------------------------------------------------"
printf "AVG over %d datasets:  zero-shot=%.4f  blend%s=%.4f  (Δ vs zero-shot %+.4f)\n" \
  "$n" "$(awk "BEGIN{print $zsum/$n}")" "$BLEND_W" "$(awk "BEGIN{print $lsum/$n}")" \
  "$(awk "BEGIN{print ($lsum-$zsum)/$n}")"
echo "Saved: $RESULT"

# Optional diff vs a prior snapshot (e.g. the pre-pixel-aug baseline).
if [[ -n $BASELINE && -f $BASELINE ]]; then
  echo ""
  echo "===== Δ vs $BASELINE (post − pre) ====="
  "$PY" - "$BASELINE" "$RESULT" "$BLEND_W" <<'PY'
import sys, csv
base, cur, w = sys.argv[1], sys.argv[2], sys.argv[3]
def load(p):
    d={}
    with open(p) as f:
        r=csv.reader(f, delimiter='\t'); next(r)
        for row in r:
            if len(row)>=4 and row[3] not in ("NA",""):
                d[row[0]]=(row[1], float(row[3]))
    return d
b, c = load(base), load(cur)
keys=[k for k in c if k in b]
print(f"{'target':<6} {'name':<18} {'pre':>8} {'post':>8} {'Δ':>9}")
tot=0.0
for k in keys:
    pre=b[k][1]; post=c[k][1]; dlt=post-pre; tot+=dlt
    print(f"{k:<6} {c[k][0]:<18} {pre:>8.4f} {post:>8.4f} {dlt:>+9.4f}")
print("-"*52)
print(f"AVG Δ (cur − baseline) over {len(keys)} datasets: {tot/len(keys):+.4f}")
PY
fi
