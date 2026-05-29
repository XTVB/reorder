#!/usr/bin/env bash
# PE-G / color ratio sweep at fixed learned blend (0.60). No training — re-blends
# the deployed-config learned matrices (/tmp/lomo_postaug) with a zero-shot
# baseline built at varying color weights, and scores ARI. PE-G weight is fixed
# at 1.0; cosine normalizes the concat, so only the color:PE-G ratio matters
# (color_weight=0 → PE-G only; larger → more color emphasis).
#
# Excludes M7 (no pixel-aug, outlier) and the two partial-label datasets (M14, M15).
# Pure scoring (no GPU) — datasets run JOBS-at-a-time in parallel.
#
#   ./run_pegcolor_sweep.sh            # JOBS=4 by default
#   JOBS=8 ./run_pegcolor_sweep.sh
set -euo pipefail

HERE=/Users/abdudh/dev/utilities/reorder/clusteringRefinement
source "$HERE/common.sh"
LEARNED_ROOT=${LEARNED_ROOT:-/tmp/lomo_postaug}     # deployed-config learned dist matrices
TMP=${TMP:-/tmp/pegcolor_sweep}
JOBS=${JOBS:-4}
W=0.60                                              # fixed learned blend fraction

# 17 datasets: all 20 minus M7 (no views) and M14/M15 (partial labels).
DATASETS=(M1 M2 M3 M4 M5 M6 M8 M9 M10 M11 M12 M13 M16 M17 M18 M19 M20)
# color weight grid (PE-G fixed at 1.0). 0.7 = deployed UI; 0.8 = prior baseline composition.
CW=(0.0 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.25 1.5 2.0)

mkdir -p "$TMP"

# Score one dataset across the full color-weight grid; writes "<tgt>\t<cw>\t<ari>"
# rows to its own fragment file (parallel-safe — no shared state).
score_dataset() {
  local tgt=$1 d dm cw v line out
  d=$(dir "$tgt"); dm=$LEARNED_ROOT/$tgt/${tgt}_dist_matrix.bin
  out=$TMP/perds_$tgt.tsv; : > "$out"
  if [[ ! -f $dm ]]; then echo "  WARN: missing learned matrix for $tgt ($dm) — skipping"; return; fi
  line="  $tgt:"
  for cw in "${CW[@]}"; do
    "$PY" "$BLEND" "$d" --learned "$dm" --weights "$W" \
      --peg-weight 1.0 --color-weight "$cw" \
      --output-pattern "$TMP/${tgt}_cw${cw}_{w}.bin" >/dev/null 2>&1
    v=$(ari "$d" --dist-matrix "$TMP/${tgt}_cw${cw}_${W}.bin" --dist-matrix-weight 1.0 --weights pecore_g=0)
    [[ -z $v ]] && v=NA
    printf "%s\t%s\t%s\n" "$tgt" "$cw" "$v" >> "$out"
    line+=" $cw=$v"
    rm -f "$TMP/${tgt}_cw${cw}_${W}.bin"
  done
  echo "$line"
}

echo "PE-G/color ratio sweep @ learned blend $W   (${#DATASETS[@]} datasets, PE-G weight fixed 1.0, JOBS=$JOBS)"
echo "color weights: ${CW[*]}"
rm -f "$TMP"/perds_*.tsv
pforeach "$JOBS" score_dataset "${DATASETS[@]}"

# Combine fragments → per-dataset TSV, then aggregate SUM/CNT (order-independent).
PERDS=$TMP/per_dataset.tsv
printf "dataset\tcolor_w\tari\n" > "$PERDS"
cat "$TMP"/perds_*.tsv >> "$PERDS"
declare -A SUM CNT
while IFS=$'\t' read -r _tgt cw v; do
  [[ $v == NA || -z $v ]] && continue
  SUM[$cw]=$(awk "BEGIN{print ${SUM[$cw]:-0}+$v}"); CNT[$cw]=$(( ${CNT[$cw]:-0} + 1 ))
done < <(cat "$TMP"/perds_*.tsv)

SUMMARY=$TMP/summary.tsv
printf "color_w\tpeg_color_ratio\tavg_ari\tn\n" > "$SUMMARY"
echo ""
echo "============== AVG ARI BY PE-G/COLOR RATIO (learned blend $W, over ${#DATASETS[@]} datasets) =============="
printf "%-9s | %-14s | %-9s\n" "color_w" "PE-G:color" "avg ARI"
echo "-------------------------------------------------"
best_cw=""; best_avg=-1
for cw in "${CW[@]}"; do
  n=${CNT[$cw]:-0}; [[ $n -eq 0 ]] && continue
  avg=$(awk "BEGIN{printf \"%.4f\", ${SUM[$cw]}/$n}")
  ratio=$(awk "BEGIN{ if ($cw==0) printf \"1:0 (PE-G only)\"; else printf \"1:%.2f\", $cw }")
  tag=""; [[ $cw == 0.7 ]] && tag="  ← deployed UI"; [[ $cw == 0.8 ]] && tag="  ← prior baseline composition"
  printf "%-9s | %-14s | %-9s%s\n" "$cw" "$ratio" "$avg" "$tag"
  printf "%s\t%s\t%s\t%s\n" "$cw" "$ratio" "$avg" "$n" >> "$SUMMARY"
  awk "BEGIN{exit !($avg>$best_avg)}" && { best_avg=$avg; best_cw=$cw; }
done
echo "-------------------------------------------------"
echo "Best avg ARI: color_w=$best_cw  ->  $best_avg"
dep=$(awk -F'\t' '$1=="0.7"{print $3}' "$SUMMARY")
echo "Deployed (color_w=0.7): $dep    |    lift of best over deployed: $(awk "BEGIN{printf \"%+.4f\", $best_avg-$dep}")"
echo ""
echo "Per-dataset: $PERDS    Summary: $SUMMARY"
