"""Shared dataset registry + fold loader for the LOMO clustering-eval harnesses
(compare_algorithms.py, rerank_eval.py, and the verifier/center/oracle evals).

Reads the M-id → ClusterBenchmark mapping from datasets.txt — the single source
of truth shared with common.sh and train_final_head.py (add a dataset by
appending one line there). Also holds the held-out fold loader and the eval
subset constants. Blend constants match the deployed cluster blend (color 0.7).
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np

BASE = "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarks"
# Trained-fold root. /tmp roots get purged by macOS (we lost /tmp/lomo_postaug
# that way) — keep folds under ~/.cache/reorder/ and override per-run via $LOMO_ROOT.
LOMO = os.environ.get("LOMO_ROOT", os.path.expanduser("~/.cache/reorder/lomo_v26"))
REGISTRY = Path(__file__).resolve().parent / "datasets.txt"

# Zero-shot baseline composition: peg ⊕ COLOR_W·color, then unit-norm → cosine.
PEG_W, COLOR_W = 1.0, 0.7

def _load_registry(path: Path = REGISTRY) -> dict[str, str]:
    """Parse datasets.txt → {M-id: "<n>-<name>" suffix}, in file order.
    Each line is "<n> <name> [flags]". Mirrors the bash parser in common.sh and
    load_dataset_registry in train_final_head.py."""
    names: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        n, name = line.split()[:2]
        names[f"M{n}"] = f"{n}-{name}"
    return names


# M-id → "<N>-<name>" suffix for every registered dataset.
NAMES = _load_registry()

# The sweep runs on whichever registered datasets have a held-out fold built
# under LOMO/ — derived, so a new dataset joins automatically once its fold
# exists (and is skipped, not crashed on, until then).
ALL = [m for m in NAMES if os.path.isdir(f"{LOMO}/{m}")]
_no_fold = [m for m in NAMES if m not in ALL]
if _no_fold:
    print(f"[lomo_common] no LOMO fold under {LOMO}, skipping: {' '.join(_no_fold)}",
          file=sys.stderr)

# Held out of the reported mean: M7 is ~4x denser per group than any other shoot;
# M14/M15 are partial-label sets whose ARI isn't comparable. EVAL_SET is the rest.
OUTLIERS = {"M7", "M14", "M15"}
EVAL_SET = [m for m in ALL if m not in OUTLIERS]


def l2(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def load_fold(tgt, dtype=np.float32):
    """Load a held-out fold, reindexed to the projection's filename order.
    Returns (peg, col, proj, true, n_groups): peg/col/proj are L2-normalized;
    `true` is the int group label per image (-1 if ungrouped)."""
    d = f"{BASE}/ClusteringBenchmark{NAMES[tgt]}"
    fns = json.load(open(f"{LOMO}/{tgt}/{tgt}_filenames.json"))
    cache = f"{d}/.reorder-cache"
    npz = np.load(f"{cache}/embeddings_hash_cache.npz", allow_pickle=False)
    ch = json.load(open(f"{cache}/content_hashes.json"))
    hrow = {h: i for i, h in enumerate(npz["hashes"])}
    idx = np.array([hrow[ch[f]] for f in fns])
    peg = l2(npz["pecore_g"][idx].astype(dtype))
    col = l2(npz["color"][idx].astype(dtype))
    proj = l2(np.load(f"{LOMO}/{tgt}/{tgt}_proj.npy").astype(dtype))
    groups = json.load(open(f"{d}/.reorder-groups.json"))
    fn2lab = {fn: gi for gi, g in enumerate(groups) for fn in g["images"]}
    true = np.array([fn2lab.get(f, -1) for f in fns])
    return peg, col, proj, true, len(groups)
