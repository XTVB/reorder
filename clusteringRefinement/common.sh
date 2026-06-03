# Shared config + helpers for the clusteringRefinement LOMO / eval sweep scripts.
# Loads the dataset registry from datasets.txt (the single source of truth — add a
# dataset by appending one line there). Also holds the ari()/npy_bad() helpers
# every scorer shares.
#
# Callers set HERE to this directory, then `source "$HERE/common.sh"`.
# Override BASE/PY/REGISTRY before sourcing if ever needed; everything else is derived.

PY=~/.venvs/imgcluster-env/bin/python3
HERE=${HERE:-/Users/abdudh/dev/utilities/reorder/clusteringRefinement}
BASE=${BASE:-/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarks}
REGISTRY=${REGISTRY:-$HERE/datasets.txt}
TRAIN=$HERE/train_projection_head.py
BLEND=$HERE/blend_dist_matrix.py
BENCH=$HERE/benchmark_clustering.ts
BENCH_CWD=/Users/abdudh/dev/utilities/reorder

# Parse datasets.txt (lines: "<n> <name> [flags]") into the registry arrays:
#   P[M-id]   → "<n>-<name>" suffix     (dir() expands to the full dataset path)
#   ALL       → every M-id, in file order
#   PIXEL_AUG → M-ids without the no-pixel-aug flag (the pixel-aug extraction set)
declare -A P=()
ALL=()
PIXEL_AUG=()
while read -r _n _name _flags; do
  [[ -z $_n || $_n == \#* ]] && continue
  P[M$_n]="$_n-$_name"
  ALL+=("M$_n")
  [[ " $_flags " != *" no-pixel-aug "* ]] && PIXEL_AUG+=("M$_n")
done < "$REGISTRY"
unset _n _name _flags
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
