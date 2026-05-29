"""Shared dataset registry + fold loader for the LOMO clustering-eval harnesses
(compare_algorithms.py, rerank_eval.py).

Single source of truth for the M-id → ClusterBenchmark mapping and the held-out
fold loader, which were copy-pasted between the two scripts. Constants match the
deployed cluster blend (color leaned to 0.7).
"""
from __future__ import annotations
import json
import numpy as np

BASE = "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarks"
LOMO = "/tmp/lomo_postaug"

# Zero-shot baseline composition: peg ⊕ COLOR_W·color, then unit-norm → cosine.
PEG_W, COLOR_W = 1.0, 0.7

NAMES = {f"M{i}": n for i, n in enumerate(
    ["1-austin", "2-sarah", "3-eva", "4-mia", "5-lily", "6-sabrina", "7-autumn",
     "8-evie", "9-darshelle", "10-alina", "11-amanda", "12-anna", "13-hunny",
     "14-vixen-partial", "15-verity-partial", "16-zoe", "17-dusha", "18-railey",
     "19-andreea", "20-salome"], 1)}
ALL = [f"M{i}" for i in range(1, 21)]


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
