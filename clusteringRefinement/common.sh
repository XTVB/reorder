# Shared config + helpers for the clusteringRefinement LOMO / eval sweep scripts.
# Single source of truth for the M-id → ClusterBenchmark dataset registry — it
# used to be copy-pasted into every script and drifted once into a path-typo bug
# (ClusterBenchmarksClusteringBenchmark…). Also holds the ari()/npy_bad() helpers
# every scorer shares.
#
# Callers set HERE to this directory, then `source "$HERE/common.sh"`.
# Override BASE/PY before sourcing if ever needed; everything else is derived.

PY=~/.venvs/imgcluster-env/bin/python3
HERE=${HERE:-/Users/abdudh/dev/utilities/reorder/clusteringRefinement}
BASE=${BASE:-/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarks}
TRAIN=$HERE/train_projection_head.py
BLEND=$HERE/blend_dist_matrix.py
BENCH=$HERE/benchmark_clustering.ts
BENCH_CWD=/Users/abdudh/dev/utilities/reorder

# M-id → "<N>-<name>" suffix; dir() expands to the full dataset path.
declare -A P=(
  [M1]=1-austin [M2]=2-sarah [M3]=3-eva [M4]=4-mia [M5]=5-lily [M6]=6-sabrina
  [M7]=7-autumn [M8]=8-evie [M9]=9-darshelle [M10]=10-alina [M11]=11-amanda
  [M12]=12-anna [M13]=13-hunny [M14]=14-vixen-partial [M15]=15-verity-partial
  [M16]=16-zoe [M17]=17-dusha [M18]=18-railey [M19]=19-andreea [M20]=20-salome
)
ALL=(M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13 M14 M15 M16 M17 M18 M19 M20)
dir() { echo "$BASE/ClusteringBenchmark${P[$1]}"; }

# ARI of a benchmark run (optionally with extra benchmark_clustering.ts args).
ari() { ( cd "$BENCH_CWD" && bun "$BENCH" "$1" "${@:2}" 2>/dev/null | awk '/^  ARI:/{print $2}' ); }

# Print 1 if the .npy is missing or contains NaN/Inf, else 0.
npy_bad() { "$PY" - "$1" <<'PY'
import sys, os, numpy as np
p = sys.argv[1]
if not os.path.exists(p): print(1); raise SystemExit
a = np.load(p); print(1 if (np.isnan(a).any() or np.isinf(a).any()) else 0)
PY
}

# Bounded-concurrency runner: pforeach <max-jobs> <fn> <arg>...
# Runs `fn arg` for every arg, at most <max-jobs> at a time. `fn` must write its
# own output to a per-arg file (NOT to shared state) — children are subshells and
# cannot write back to the parent. Aggregate from those files after pforeach.
pforeach() {
  local max=$1 fn=$2; shift 2
  local running=0
  for arg in "$@"; do
    "$fn" "$arg" &
    if (( ++running >= max )); then wait -n; ((running--)); fi
  done
  wait
}
