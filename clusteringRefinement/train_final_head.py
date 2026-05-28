#!/usr/bin/env python3
"""
Train the deployable projection head on all configured datasets (no LOMO).
Saves the head + config to ~/.cache/reorder/ for the rest of the pipeline to load.

Usage:
  # Default: train on the configured dataset list
  python clusteringRefinement/train_final_head.py

  # Override or extend the dataset list (one PATH per --dataset, format NAME:PATH)
  python clusteringRefinement/train_final_head.py \\
      --dataset M1:/path/to/Bench1 \\
      --dataset M2:/path/to/Bench2 \\
      ...

  # Use a config file (one PATH per line — name derived from basename)
  python clusteringRefinement/train_final_head.py --datasets-from ~/.config/reorder/training_datasets.txt

Outputs:
  ~/.cache/reorder/learned_head.pt        — head state_dict
  ~/.cache/reorder/learned_head.json      — config (input/output dims, hyperparams, version)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_projection_head.py"

HEAD_DIR = Path(os.environ.get("REORDER_HEAD_DIR", os.path.expanduser("~/.cache/reorder")))
HEAD_PT = HEAD_DIR / "learned_head.pt"
HEAD_CONFIG = HEAD_DIR / "learned_head.json"
DEFAULT_CONFIG_PATH = Path(os.path.expanduser("~/.config/reorder/training_datasets.txt"))

# Winning hyperparameters from the LOMO sweep series (P=32, K=12, ep=15,
# pixel-aug, ICOMB augmentations + cross-mixup + hard-neg).
DEFAULT_HYPERPARAMS = {
    "epochs": 15,
    "batches_per_epoch": 400,
    "p_groups": 32,
    "k_images": 12,
    "out_dim": 256,
    "hidden": 1024,
    "dropout": 0.1,
    "temperature": 0.1,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "mixup_alpha": 0.4,
    "drop_color_prob": 0.5,
    "cross_mixup_prob": 0.3,
    "cross_mixup_alpha": 0.4,
    "hard_neg_frac": 0.5,
    "hard_neg_pool_k": 20,
    "use_singleton_negatives": True,
    "use_augmented_views": True,
    "arcface_weight": 0.0,
}

# Fallback dataset list (used when no --dataset, --datasets-from, or default config).
DEFAULT_TRAINING_DATASETS = [
    ("M1", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark1-austin"),
    ("M2", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark2-sarah"),
    ("M3", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark3-eva"),
    ("M4", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark4-mia"),
    ("M5", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark5-lily"),
    ("M6", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark6-sabrina"),
    ("M7", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark7-autumn"),
    ("M8", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark8-evie"),
    ("M9", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark9-darshelle"),
    ("M10", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark10-alina"),
    ("M11", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark11-amanda"),
    ("M12", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark12-anna"),
    ("M13", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark13-hunny"),
    ("M14", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark14-vixen-partial"),
    ("M15", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark15-verity-partial"),
    ("M16", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark16-zoe"),
    ("M17", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark17-dusha"),
    ("M18", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark18-railey"),
    ("M19", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark19-andreea"),
    ("M20", "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarksClusteringBenchmark20-salome"),
]


def parse_dataset_arg(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"--dataset wants NAME:PATH, got {s!r}")
    name, path = s.split(":", 1)
    return name, os.path.abspath(path)


def load_datasets_from_file(path: Path) -> list[tuple[str, str]]:
    """Read one path per line. Name is the basename of the dir.
    Lines starting with # are comments. Blank lines ignored."""
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = os.path.abspath(os.path.expanduser(line))
        name = os.path.basename(p)
        out.append((name, p))
    return out


def verify_dataset(name: str, path: str) -> bool:
    """Check that the dataset has the caches the training script needs."""
    cache = os.path.join(path, ".reorder-cache")
    required = [
        os.path.join(cache, "embeddings_hash_cache.npz"),
        os.path.join(cache, "content_hashes.json"),
        os.path.join(path, ".reorder-groups.json"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print(f"  [{name}] WARN: missing required files: {missing}", file=sys.stderr)
        return False
    return True


def compute_head_version(pt_path: Path, datasets: list[tuple[str, str]]) -> str:
    """Stable version string derived from the head weights + training dataset list.
    Used to invalidate cached learned_proj features when the head is retrained."""
    h = hashlib.blake2b(digest_size=8)
    h.update(pt_path.read_bytes())
    for name, path in sorted(datasets):
        h.update(f"{name}:{path}".encode())
    return f"learned-head-{h.hexdigest()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", type=parse_dataset_arg, default=None,
                    help="NAME:PATH — can repeat. Overrides default + config file.")
    ap.add_argument("--datasets-from", type=Path, default=None,
                    help="File with one dataset path per line.")
    ap.add_argument("--output-dir", type=Path, default=HEAD_DIR,
                    help=f"Where to save head + config (default {HEAD_DIR})")
    ap.add_argument("--epochs", type=int, default=DEFAULT_HYPERPARAMS["epochs"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the training command without running.")
    args = ap.parse_args()

    # Resolve dataset list
    if args.dataset:
        datasets = args.dataset
    elif args.datasets_from:
        datasets = load_datasets_from_file(args.datasets_from)
    elif DEFAULT_CONFIG_PATH.exists():
        print(f"Reading dataset list from {DEFAULT_CONFIG_PATH}", file=sys.stderr)
        datasets = load_datasets_from_file(DEFAULT_CONFIG_PATH)
    else:
        datasets = DEFAULT_TRAINING_DATASETS

    print(f"Training datasets ({len(datasets)}):", file=sys.stderr)
    valid = []
    for name, path in datasets:
        ok = verify_dataset(name, path)
        flag = "✓" if ok else "✗"
        print(f"  {flag} {name:<6}  {path}", file=sys.stderr)
        if ok:
            valid.append((name, path))
    if not valid:
        sys.exit("No valid datasets; aborting.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build the training command. Train on ALL valid datasets, no holdout.
    hp = DEFAULT_HYPERPARAMS
    train_names = ",".join(n for n, _ in valid)
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            *sum([["--dataset", f"{n}:{p}"] for n, p in valid], []),
            "--train", train_names,
            "--within-holdout-frac", "0",
            "--epochs", str(args.epochs),
            "--batches-per-epoch", str(hp["batches_per_epoch"]),
            "--p-groups", str(hp["p_groups"]),
            "--k-images", str(hp["k_images"]),
            "--out-dim", str(hp["out_dim"]),
            "--hidden", str(hp["hidden"]),
            "--dropout", str(hp["dropout"]),
            "--temperature", str(hp["temperature"]),
            "--lr", str(hp["lr"]),
            "--weight-decay", str(hp["weight_decay"]),
            "--mixup-alpha", str(hp["mixup_alpha"]),
            "--drop-color-prob", str(hp["drop_color_prob"]),
            "--cross-mixup-prob", str(hp["cross_mixup_prob"]),
            "--cross-mixup-alpha", str(hp["cross_mixup_alpha"]),
            "--hard-neg-frac", str(hp["hard_neg_frac"]),
            "--hard-neg-pool-k", str(hp["hard_neg_pool_k"]),
            "--arcface-weight", str(hp["arcface_weight"]),
            "--output-dir", tmpdir,
        ]
        if hp["use_augmented_views"]:
            cmd.append("--use-augmented-views")
        if hp["use_singleton_negatives"]:
            cmd.append("--use-singleton-negatives")

        if args.dry_run:
            print("Would run:")
            print("  " + " ".join(repr(c) if " " in c else c for c in cmd))
            return

        print(f"\nRunning training (this may take ~30-60s)...\n", file=sys.stderr)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"Training failed with exit code {result.returncode}")

        # Promote the trained head to the canonical location
        src_pt = Path(tmpdir) / "proj_head.pt"
        if not src_pt.exists():
            sys.exit(f"Training did not produce {src_pt}")
        shutil.copy(src_pt, HEAD_PT)
        print(f"\nHead saved → {HEAD_PT}", file=sys.stderr)

        # Save config (includes version derived from weights + dataset list)
        version = compute_head_version(HEAD_PT, valid)
        config = {
            "version": version,
            "input_dim_peg": 1280,
            "input_dim_color": 693,
            "input_dim_total": 1973,
            "out_dim": hp["out_dim"],
            "hidden": hp["hidden"],
            "dropout": hp["dropout"],
            "training_datasets": [{"name": n, "path": p} for n, p in valid],
            "hyperparams": hp,
        }
        HEAD_CONFIG.write_text(json.dumps(config, indent=2))
        print(f"Config saved → {HEAD_CONFIG}", file=sys.stderr)
        print(f"Head version: {version}", file=sys.stderr)


if __name__ == "__main__":
    main()
