#!/usr/bin/env bash
# Fixed-blend-grid analysis over the full-LOMO trained heads.
#
# Deployment ships ONE global blend weight (the learned-head fraction, default
# 0.60), so the relevant question is which single weight maximises the AVERAGE
# ARI across all datasets — not which weight happens to win per-dataset.
#
# Reuses the dist matrices already trained by run_full_lomo.sh (no retraining):
# for each weight w, blends w*learned + (1-w)*zero-shot, scores every dataset,
# and averages. w=0.00 is pure zero-shot; w=1.00 is pure learned. Pure scoring
# (no GPU) — datasets run JOBS-at-a-time in parallel.
#
#   ./run_blend_grid.sh                 # uses /tmp/full_lomo
#   LOMO_ROOT=/path JOBS=8 ./run_blend_grid.sh
set -euo pipefail

HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
LOMO_ROOT=${LOMO_ROOT:-/tmp/full_lomo}
SCRATCH=${SCRATCH:-/tmp/blend_grid}
JOBS=${JOBS:-4}
CONFIGS=(baseline winner)
# Learned-head blend fractions to test. 0.60 is the current deployed default.
GRID=(0.20 0.30 0.40 0.45 0.50 0.60 0.70)

mkdir -p "$SCRATCH"

# Only score datasets where BOTH configs trained cleanly (comparable set).
USABLE=()
for tgt in "${ALL[@]}"; do
  ok=1
  for cfg in "${CONFIGS[@]}"; do
    dm=$LOMO_ROOT/$cfg/$tgt/${tgt}_dist_matrix.bin
    pj=$LOMO_ROOT/$cfg/$tgt/${tgt}_proj.npy
    [[ -f $dm && -f $pj && $(npy_bad "$pj") == 0 ]] || ok=0
  done
  [[ $ok == 1 ]] && USABLE+=("$tgt")
done
echo "Scoring fixed-blend grid over ${#USABLE[@]}/${#ALL[@]} datasets (both configs clean): ${USABLE[*]}"
echo "Grid = zero-shot(0.00), ${GRID[*]}, pure-learned(1.00).  Deployed default = 0.60.  JOBS=$JOBS"
echo ""

# Score one dataset; writes "zs"/"<cfg>" rows to its own fragment (parallel-safe).
score_dataset() {
  local tgt=$1 d zs cfg dm sd w v out
  d=$(dir "$tgt")
  out=$SCRATCH/frag_$tgt.tsv; : > "$out"
  zs=$(ari "$d")
  printf "zs\t\t%s\n" "${zs:-0}" >> "$out"      # zero-shot is config-independent
  for cfg in "${CONFIGS[@]}"; do
    dm=$LOMO_ROOT/$cfg/$tgt/${tgt}_dist_matrix.bin
    sd=$SCRATCH/$cfg/$tgt; mkdir -p "$sd"
    # Generate all grid blends for this (cfg,tgt) in one call (baseline cosine
    # computed once); pure-learned (1.0) is the dist matrix itself.
    "$PY" "$BLEND" "$d" --learned "$dm" \
      --weights "$(IFS=,; echo "${GRID[*]}")" \
      --output-pattern "$sd/{w}.bin" >/dev/null 2>&1
    for w in "${GRID[@]}"; do
      v=$(ari "$d" --dist-matrix "$sd/$w.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
      printf "%s\t%s\t%s\n" "$cfg" "$w" "${v:-0}" >> "$out"
    done
    v=$(ari "$d" --dist-matrix "$dm" --dist-matrix-weight 1.0 --weights pecore_g=0)   # 1.00
    printf "%s\t1.00\t%s\n" "$cfg" "${v:-0}" >> "$out"
  done
}

rm -f "$SCRATCH"/frag_*.tsv
pforeach "$JOBS" score_dataset "${USABLE[@]}"

# Accumulators: per config+weight sum of ARI. Keys: "<cfg>:<w>"; "zs" once.
declare -A SUM; SUM[zs]=0
while IFS=$'\t' read -r key w v; do
  if [[ $key == zs ]]; then SUM[zs]=$(awk "BEGIN{print ${SUM[zs]}+${v:-0}}")
  else SUM[$key:$w]=$(awk "BEGIN{print ${SUM[$key:$w]:-0}+${v:-0}}"); fi
done < <(cat "$SCRATCH"/frag_*.tsv)

n=${#USABLE[@]}
zavg=$(awk "BEGIN{printf \"%.4f\", ${SUM[zs]}/$n}")

echo "================ AVERAGE ARI BY FIXED GLOBAL BLEND (over $n datasets) ================"
printf "%-18s | %-10s | %-10s\n" "blend (learned %)" "baseline" "winner"
echo "-------------------------------------------------------------"
printf "%-18s | %-10s | %-10s\n" "0%  (zero-shot)" "$zavg" "$zavg"
best_b=$zavg; best_bw="0%"; best_w=$zavg; best_ww="0%"
for w in "${GRID[@]}" 1.00; do
  ba=$(awk "BEGIN{printf \"%.4f\", ${SUM[baseline:$w]:-0}/$n}")
  wa=$(awk "BEGIN{printf \"%.4f\", ${SUM[winner:$w]:-0}/$n}")
  label=$(awk "BEGIN{printf \"%d%%\", $w*100}")
  [[ $w == 0.60 ]] && label="$label (deployed)"
  awk "BEGIN{exit !($ba>$best_b)}" && { best_b=$ba; best_bw=$label; }
  awk "BEGIN{exit !($wa>$best_w)}" && { best_w=$wa; best_ww=$label; }
  printf "%-18s | %-10s | %-10s\n" "$label" "$ba" "$wa"
done
echo "-------------------------------------------------------------"
echo "Best fixed blend (avg ARI):"
echo "   baseline config : $best_bw  -> $best_b   (Δ vs zero-shot $(awk "BEGIN{printf \"%+.4f\", $best_b-$zavg}"))"
echo "   winner   config : $best_ww  -> $best_w   (Δ vs zero-shot $(awk "BEGIN{printf \"%+.4f\", $best_w-$zavg}"))"
echo "   winner best-blend lift over baseline best-blend: $(awk "BEGIN{printf \"%+.4f\", $best_w-$best_b}")"
