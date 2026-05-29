#!/usr/bin/env bash
# Blend-curve comparison: average ARI across all 20 datasets at a grid of
# inference blend weights, for two trained LOMO heads (flat <root>/<tgt>/ dirs).
# Used to check whether the blend-aware head peaks at a different blend than the
# deployed control (so a fixed-0.60 A/B doesn't under/over-state it).
# Pure scoring (no GPU) — datasets run JOBS-at-a-time in parallel.
#
#   ROOTS="deployed:/tmp/lomo_postaug blendaware:/tmp/lomo_blendaware_full" ./run_blend_curve.sh
set -euo pipefail

HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
SCRATCH=${SCRATCH:-/tmp/blend_curve}
JOBS=${JOBS:-4}
ROOTS="${ROOTS:-deployed:/tmp/lomo_postaug blendaware:/tmp/lomo_blendaware_full}"
GRID=(0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.80)

# Parse ROOTS into label list + dir map.
declare -A RDIR; LABELS=()
for spec in $ROOTS; do LABELS+=("${spec%%:*}"); RDIR[${spec%%:*}]="${spec#*:}"; done

# Usable = datasets where every root has a clean matrix.
USABLE=()
for tgt in "${ALL[@]}"; do
  ok=1
  for lb in "${LABELS[@]}"; do
    [[ -f "${RDIR[$lb]}/$tgt/${tgt}_dist_matrix.bin" && $(npy_bad "${RDIR[$lb]}/$tgt/${tgt}_proj.npy") == 0 ]] || ok=0
  done
  [[ $ok == 1 ]] && USABLE+=("$tgt")
done
echo "Blend curve over ${#USABLE[@]}/${#ALL[@]} datasets. Roots: ${LABELS[*]}  (JOBS=$JOBS)"
echo "Grid: ${GRID[*]} (+ pure-learned 1.00). Deployed blend = 0.60."
mkdir -p "$SCRATCH"

# Score one dataset across all roots; writes "zs"/"<label>" rows to a fragment.
score_dataset() {
  local tgt=$1 d lb dm sd w v out
  d=$(dir "$tgt")
  out=$SCRATCH/frag_$tgt.tsv; : > "$out"
  printf "zs\t\t%s\n" "$(ari "$d")" >> "$out"
  for lb in "${LABELS[@]}"; do
    dm="${RDIR[$lb]}/$tgt/${tgt}_dist_matrix.bin"
    sd=$SCRATCH/$lb/$tgt; mkdir -p "$sd"
    "$PY" "$BLEND" "$d" --learned "$dm" --weights "$(IFS=,; echo "${GRID[*]}")" \
      --output-pattern "$sd/{w}.bin" >/dev/null 2>&1
    for w in "${GRID[@]}"; do
      v=$(ari "$d" --dist-matrix "$sd/$w.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
      printf "%s\t%s\t%s\n" "$lb" "$w" "${v:-0}" >> "$out"
    done
    v=$(ari "$d" --dist-matrix "$dm" --dist-matrix-weight 1.0 --weights pecore_g=0)
    printf "%s\t1.00\t%s\n" "$lb" "${v:-0}" >> "$out"
  done
}

rm -f "$SCRATCH"/frag_*.tsv
pforeach "$JOBS" score_dataset "${USABLE[@]}"

declare -A SUM; SUM[zs]=0
while IFS=$'\t' read -r key w v; do
  if [[ $key == zs ]]; then SUM[zs]=$(awk "BEGIN{print ${SUM[zs]}+${v:-0}}")
  else SUM[$key:$w]=$(awk "BEGIN{print ${SUM[$key:$w]:-0}+${v:-0}}"); fi
done < <(cat "$SCRATCH"/frag_*.tsv)

n=${#USABLE[@]}
echo ""
printf "%-8s" "blend"; for lb in "${LABELS[@]}"; do printf " | %-12s" "$lb"; done; echo ""
echo "------------------------------------------------------"
printf "%-8s" "0% (zs)"; z=$(awk "BEGIN{printf \"%.4f\", ${SUM[zs]}/$n}"); for lb in "${LABELS[@]}"; do printf " | %-12s" "$z"; done; echo ""
declare -A BESTV BESTW
for lb in "${LABELS[@]}"; do BESTV[$lb]=$z; BESTW[$lb]="0%"; done
for w in "${GRID[@]}" 1.00; do
  lbl=$(awk "BEGIN{printf \"%d%%\", $w*100}"); [[ $w == 0.60 ]] && lbl="$lbl*"
  printf "%-8s" "$lbl"
  for lb in "${LABELS[@]}"; do
    a=$(awk "BEGIN{printf \"%.4f\", ${SUM[$lb:$w]:-0}/$n}")
    awk "BEGIN{exit !($a>${BESTV[$lb]})}" && { BESTV[$lb]=$a; BESTW[$lb]=$lbl; }
    printf " | %-12s" "$a"
  done
  echo ""
done
echo "------------------------------------------------------ (* = deployed blend)"
for lb in "${LABELS[@]}"; do
  echo "  $lb: peaks at ${BESTW[$lb]} -> ${BESTV[$lb]}"
done
