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
  ~/.cache/reorder/learned_head.pt        — head state_dict (seed 1 of the ensemble)
  ~/.cache/reorder/learned_head_s<N>.pt   — remaining ensemble heads
  ~/.cache/reorder/learned_head.json      — config (dims, hyperparams, version, head_files)

The deployed head is a 3-seed ENSEMBLE (see LEARNED_HEAD.md "seed-ensemble"):
extract_features.py projects through every head and concatenates the L2-normed
projections scaled by 1/√n_heads, so the existing cosine blend computes the
ensemble-mean similarity with no downstream changes.
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

# Winning hyperparameters from the LOMO sweep series. See LEARNED_HEAD.md for
# the rationale, results, and the matching inference blend (0.60).
# epochs 12 + cosine decay replaced epochs 15 + constant LR (2026-06): paired
# 26-fold LOMO Δ = +0.0061 (seed 42) / +0.0005 (seed 43) — never worse, 20%
# less training compute.
DEFAULT_HYPERPARAMS = {
    "epochs": 12,
    "lr_schedule": "cosine",
    "batches_per_epoch": 400,
    "p_groups": 32,
    "k_images": 12,
    "out_dim": 512,
    "hidden": 1024,
    "dropout": 0.1,
    "temperature": 0.07,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
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

# Ensemble seeds. 3 heads ≈ +0.010 ARI over the expected single-seed head in
# the 24-fold LOMO (seed_ensemble_v26.tsv); ens4 added nothing over ens3.
ENSEMBLE_SEEDS = [42, 43, 44]

# Modality groups for the split-head blend (LEARNED_HEAD.md "Split single-
# modality heads": joint .55 + peg .30 + color .15 beats the joint-only blend
# by +0.019 eval-23). Each group trains one head per ENSEMBLE_SEED.
# The color head disables drop-color — that augmentation would zero its entire
# input half the time.
MODALITY_GROUPS = [
    {"mod": "both", "npz_key": "learned_proj", "input_mods": "peg,color",
     "input_dim": 1973, "hp_overrides": {}},
    {"mod": "peg", "npz_key": "learned_proj_peg", "input_mods": "peg",
     "input_dim": 1280, "hp_overrides": {}},
    {"mod": "color", "npz_key": "learned_proj_color", "input_mods": "color",
     "input_dim": 693, "hp_overrides": {"drop_color_prob": 0.0}},
]

# Intermediate PE-G layer fed as EXTRA joint-head input (LEARNED_HEAD.md
# "PE-layer head input"). The 12-config × 2-placement sweep picked L44 attn-pool
# into the JOINT head: +0.0064 ARI, 20/25 wins, 95% CI excludes 0. A parallel
# layer-augmented joint head is trained alongside the plain one; extraction uses
# it only when the layer is extracted fully (else falls back to plain) and the
# `use_pe_layer` toggle is on. None disables the whole thing.
PE_LAYER = "44:attnpool"
PE_LAYER_DIM = 1536          # attn-pool block width (pe_layers_L44_attnpool.npy)
PE_LAYER_GROUP = "both"      # which modality group the layer augments
def pe_layer_head_filename(seed: int, seed_index: int) -> str:
    return "learned_head_layer.pt" if seed_index == 0 else f"learned_head_layer_s{seed}.pt"


def pe_layer_npy(path: str, pe_layer: str) -> str:
    """Path to a dataset's extracted PE-layer features for '<layer>:<pool>'.
    Mirrors train_projection_head.py's naming (pe_layers_L<NN>_<pool>.npy)."""
    lno, pool = pe_layer.split(":")
    return os.path.join(path, ".reorder-cache", f"pe_layers_L{int(lno):02d}_{pool}.npy")


def head_filename(mod: str, seed: int, seed_index: int) -> str:
    """The joint head keeps the legacy names (learned_head.pt + _s<N>.pt) so
    older consumers keep working; split heads are suffixed by modality."""
    if mod == "both":
        return "learned_head.pt" if seed_index == 0 else f"learned_head_s{seed}.pt"
    return f"learned_head_{mod}_s{seed}.pt"

# Dataset registry: single source of truth shared with common.sh and the
# pixel-aug scripts. Add a dataset by appending one line to datasets.txt.
DATASET_REGISTRY = SCRIPT_DIR / "datasets.txt"
DATASET_BASE = Path("/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarks")


def load_dataset_registry(registry: Path = DATASET_REGISTRY,
                          base: Path = DATASET_BASE) -> list[tuple[str, str]]:
    """Parse datasets.txt → [(M-id, full_path)]. Each line is "<n> <name> [flags]".
    Comments (#) and blank lines ignored. Mirrors the bash parser in common.sh."""
    out = []
    for line in registry.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        n, name = line.split()[:2]
        out.append((f"M{n}", str(base / f"ClusteringBenchmark{n}-{name}")))
    return out


# Fallback dataset list (used when no --dataset, --datasets-from, or default config).
DEFAULT_TRAINING_DATASETS = load_dataset_registry()


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


def run_seed_jobs(jobs: list[tuple[int, Path, Path, list[str]]], max_parallel: int, mod: str):
    """Run per-seed training subprocesses, at most max_parallel at once.
    Sequential runs stream output through; parallel runs log to train.log in
    their output dir (tail printed on failure). Waits FIFO — seeds of a group
    are equal-length runs, so out-of-order completion costs nothing."""
    pending = list(jobs)
    running: list[tuple[subprocess.Popen, int, Path, Path, object]] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            seed, dst, out, cmd = pending.pop(0)
            print(f"\nTraining {mod} head seed={seed}...\n", file=sys.stderr)
            if max_parallel == 1:
                proc, logf = subprocess.Popen(cmd), None
            else:
                logf = open(out / "train.log", "w")
                proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            running.append((proc, seed, dst, out, logf))
        proc, seed, dst, out, logf = running.pop(0)
        rc = proc.wait()
        if logf:
            logf.close()
        if rc != 0:
            for p, *_rest, lf in running:
                p.terminate()
                if lf:
                    lf.close()
            if logf:
                sys.stderr.write((out / "train.log").read_text()[-4000:])
            sys.exit(f"Training ({mod}, seed {seed}) failed with exit code {rc}")
        src_pt = out / "proj_head.pt"
        if not src_pt.exists():
            sys.exit(f"Training did not produce {src_pt}")
        shutil.copy(src_pt, dst)
        print(f"Head ({mod}, seed {seed}) saved → {dst}", file=sys.stderr)


def compute_head_version(pt_paths: list[Path], datasets: list[tuple[str, str]]) -> str:
    """Stable version string derived from every head's weights + training dataset
    list. Used to invalidate cached learned_proj features when the head is retrained."""
    h = hashlib.blake2b(digest_size=8)
    for p in pt_paths:
        h.update(p.read_bytes())
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
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx",
                    help="training framework passed to train_projection_head.py "
                         "(default mlx — ~2x faster per head on Apple Silicon, "
                         "statistically equivalent; see LEARNED_HEAD.md)")
    ap.add_argument("--jobs", type=int, default=3,
                    help="(--seed-mode jobs only) train up to N ensemble seeds as "
                         "concurrent processes (~2-3GB RAM each). 3-seed group: "
                         "39s/33s/31s at jobs 1/2/3. 1 = sequential, streaming output.")
    ap.add_argument("--seed-mode", choices=["jobs", "batched"], default="jobs",
                    help="jobs (default): one process per seed, parallelized per "
                         "--jobs — fastest (host phases overlap the GPU). batched "
                         "(mlx only): all seeds in ONE process via --seeds (stacked "
                         "weights, one compiled graph) — quality-equivalent, ~13% "
                         "slower, but 1/3 the memory; use when RAM is tight or only "
                         "one process may own the GPU.")
    ap.add_argument("--mods", default="both,peg,color",
                    help="Comma list of modality groups to (re)train: both,peg,color. "
                         "Untrained groups must already have head files on disk — the "
                         "config always describes all three.")
    ap.add_argument("--pe-layer", default=PE_LAYER,
                    help=f"intermediate PE-G layer fed to the joint head as "
                         f"'<layer>:<pool>' (default {PE_LAYER}). Trains a parallel "
                         f"layer-augmented joint head used at inference when the layer "
                         f"is fully extracted (else falls back to the plain head).")
    ap.add_argument("--no-pe-layer", dest="pe_layer", action="store_const", const=None,
                    help="don't train the layer head, and set the deployed toggle off "
                         "(extraction always uses the plain joint head).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the training command without running.")
    args = ap.parse_args()
    train_mods = set(args.mods.split(","))
    if not train_mods <= {g["mod"] for g in MODALITY_GROUPS}:
        sys.exit(f"--mods must be a subset of both,peg,color (got {args.mods!r})")

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

    # Build the training commands. Train on ALL valid datasets, no holdout.
    # One run per (modality group, ensemble seed); each head gets its own file.
    train_names = ",".join(n for n, _ in valid)

    def build_cmd(hp: dict, input_mods: str, pe_layer: str | None = None,
                  ds: list[tuple[str, str]] | None = None) -> list[str]:
        ds = valid if ds is None else ds
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            *sum([["--dataset", f"{n}:{p}"] for n, p in ds], []),
            "--train", ",".join(n for n, _ in ds),
            "--within-holdout-frac", "0",
            "--input-mods", input_mods,
            *(["--pe-layer", pe_layer] if pe_layer else []),
            "--epochs", str(args.epochs),
            "--batches-per-epoch", str(hp["batches_per_epoch"]),
            "--p-groups", str(hp["p_groups"]),
            "--k-images", str(hp["k_images"]),
            "--out-dim", str(hp["out_dim"]),
            "--hidden", str(hp["hidden"]),
            "--dropout", str(hp["dropout"]),
            "--temperature", str(hp["temperature"]),
            "--lr", str(hp["lr"]),
            "--lr-schedule", hp["lr_schedule"],
            "--weight-decay", str(hp["weight_decay"]),
            "--grad-clip", str(hp["grad_clip"]),
            "--mixup-alpha", str(hp["mixup_alpha"]),
            "--drop-color-prob", str(hp["drop_color_prob"]),
            "--cross-mixup-prob", str(hp["cross_mixup_prob"]),
            "--cross-mixup-alpha", str(hp["cross_mixup_alpha"]),
            "--hard-neg-frac", str(hp["hard_neg_frac"]),
            "--hard-neg-pool-k", str(hp["hard_neg_pool_k"]),
            "--arcface-weight", str(hp["arcface_weight"]),
            # Only proj_head.pt is consumed — skip per-dataset projection artifacts.
            "--eval", "none",
            "--backend", args.backend,
        ]
        if hp["use_augmented_views"]:
            cmd.append("--use-augmented-views")
        if hp["use_singleton_negatives"]:
            cmd.append("--use-singleton-negatives")
        return cmd

    groups_out = []  # (group, head_paths) for the config
    with tempfile.TemporaryDirectory() as tmpdir:
        def train_heads(label: str, base_cmd: list[str], head_paths: list[Path]):
            """Train one head group (all ensemble seeds) → head_paths. Honors
            --seed-mode (batched/jobs) and --dry-run."""
            if args.dry_run:
                seeds_arg = ",".join(str(s) for s in ENSEMBLE_SEEDS)
                if args.seed_mode == "batched" and args.backend == "mlx":
                    print(f"[{label}] would run ONCE, batched seeds {ENSEMBLE_SEEDS}:")
                    tail = f" --seeds {seeds_arg} --output-dir <tmp>"
                else:
                    print(f"[{label}] would run (once per seed {ENSEMBLE_SEEDS}):")
                    tail = " --seed <s> --output-dir <tmp>"
                print("  " + " ".join(repr(c) if " " in c else c for c in base_cmd) + tail)
                return
            if args.seed_mode == "batched" and args.backend == "mlx":
                out = Path(tmpdir) / label
                out.mkdir()
                seeds_arg = ",".join(str(s) for s in ENSEMBLE_SEEDS)
                print(f"\nTraining {label} heads, batched seeds {seeds_arg}...\n", file=sys.stderr)
                result = subprocess.run(base_cmd + ["--seeds", seeds_arg, "--output-dir", str(out)])
                if result.returncode != 0:
                    sys.exit(f"Training ({label}, seeds {seeds_arg}) failed "
                             f"with exit code {result.returncode}")
                for seed, dst in zip(ENSEMBLE_SEEDS, head_paths):
                    src_pt = out / f"proj_head_s{seed}.pt"
                    if not src_pt.exists():
                        sys.exit(f"Training did not produce {src_pt}")
                    shutil.copy(src_pt, dst)
                    print(f"Head ({label}, seed {seed}) saved → {dst}", file=sys.stderr)
            else:
                jobs = []
                for seed, dst in zip(ENSEMBLE_SEEDS, head_paths):
                    out = Path(tmpdir) / f"{label}_s{seed}"
                    out.mkdir()
                    jobs.append((seed, dst, out,
                                 base_cmd + ["--seed", str(seed), "--output-dir", str(out)]))
                run_seed_jobs(jobs, max_parallel=max(1, args.jobs), mod=label)

        for group in MODALITY_GROUPS:
            hp = {**DEFAULT_HYPERPARAMS, **group["hp_overrides"]}
            head_paths = [args.output_dir / head_filename(group["mod"], s, i)
                          for i, s in enumerate(ENSEMBLE_SEEDS)]
            groups_out.append((group, head_paths, hp))
            if group["mod"] not in train_mods:
                missing = [p for p in head_paths if not p.exists()]
                if missing:
                    sys.exit(f"--mods skips '{group['mod']}' but {missing} missing — "
                             f"train it or remove it from the config expectations")
                print(f"\nSkipping '{group['mod']}' (not in --mods; reusing existing heads)",
                      file=sys.stderr)
                continue
            train_heads(group["mod"], build_cmd(hp, group["input_mods"]), head_paths)

        # Layer-augmented joint head (parallel to the plain "both" head). Trained
        # only when the layer is requested and the joint group is being (re)trained.
        layer_paths = None
        if args.pe_layer and PE_LAYER_GROUP in train_mods:
            # The layer head can only train on datasets that have the PE-layer
            # extracted; silently drop the rest rather than fail the whole run.
            layer_ds = [(n, p) for n, p in valid
                        if os.path.exists(pe_layer_npy(p, args.pe_layer))]
            skipped = [n for n, p in valid if (n, p) not in layer_ds]
            if skipped:
                print(f"\n[{PE_LAYER_GROUP}+layer] skipping {len(skipped)} dataset(s) "
                      f"without {args.pe_layer} extracted: {', '.join(skipped)} "
                      f"(run run_pe_layer_extraction.sh to include them)", file=sys.stderr)
            if not layer_ds:
                print(f"[{PE_LAYER_GROUP}+layer] no datasets have {args.pe_layer} "
                      f"extracted — skipping the layer head entirely", file=sys.stderr)
            else:
                hp = {**DEFAULT_HYPERPARAMS,
                      **next(g["hp_overrides"] for g in MODALITY_GROUPS if g["mod"] == PE_LAYER_GROUP)}
                mods = next(g["input_mods"] for g in MODALITY_GROUPS if g["mod"] == PE_LAYER_GROUP)
                layer_paths = [args.output_dir / pe_layer_head_filename(s, i)
                               for i, s in enumerate(ENSEMBLE_SEEDS)]
                train_heads(f"{PE_LAYER_GROUP}+layer",
                            build_cmd(hp, mods, pe_layer=args.pe_layer, ds=layer_ds), layer_paths)
        elif args.pe_layer:
            # Joint group skipped but layer requested: reuse existing layer heads.
            cand = [args.output_dir / pe_layer_head_filename(s, i)
                    for i, s in enumerate(ENSEMBLE_SEEDS)]
            layer_paths = cand if all(p.exists() for p in cand) else None

        if args.dry_run:
            return

        # Save config (version derived from ALL groups' weights + dataset list —
        # retraining any modality refreshes every cached learned_proj* array).
        hp = DEFAULT_HYPERPARAMS
        all_paths = [p for _, paths, _ in groups_out for p in paths]
        if layer_paths:
            all_paths = all_paths + layer_paths
        version = compute_head_version(all_paths, valid)

        def head_entry(g, paths, ghp):
            entry = {
                "npz_key": g["npz_key"],
                "input_mods": g["input_mods"],
                "input_dim": g["input_dim"],
                "head_files": [p.name for p in paths],
                "hyperparams": ghp,
            }
            # Joint group gains a parallel layer-augmented head: used at inference
            # only when the PE-layer is fully extracted (else falls back to the
            # plain head above) and the use_pe_layer toggle is on.
            if g["mod"] == PE_LAYER_GROUP and layer_paths:
                entry["pe_layer"] = args.pe_layer
                entry["pe_layer_dim"] = PE_LAYER_DIM
                entry["pe_layer_input_dim"] = g["input_dim"] + PE_LAYER_DIM
                entry["pe_layer_head_files"] = [p.name for p in layer_paths]
            return entry

        config = {
            "version": version,
            # The deployed toggle: extraction uses the layer head when present &
            # fully extracted. Flip to false (or env REORDER_USE_PE_LAYER=0) to
            # force the plain joint head everywhere.
            "use_pe_layer": bool(args.pe_layer),
            # Modality groups (consumed by extract_features._compute_learned_proj):
            # per group, project through every head file, concat L2-normed
            # blocks / sqrt(n). Joint group keeps the legacy fields below too.
            "heads": [head_entry(g, paths, ghp) for g, paths, ghp in groups_out],
            # Legacy fields — pre-split consumers read these (joint head only).
            "input_dim_peg": 1280,
            "input_dim_color": 693,
            "input_dim_total": 1973,
            "out_dim": hp["out_dim"],
            "hidden": hp["hidden"],
            "dropout": hp["dropout"],
            "head_files": next(
                [p.name for p in paths] for g, paths, _ in groups_out if g["mod"] == "both"
            ),
            "ensemble_seeds": ENSEMBLE_SEEDS,
            "learned_proj_dim": hp["out_dim"] * len(ENSEMBLE_SEEDS),
            "training_datasets": [{"name": n, "path": p} for n, p in valid],
            "hyperparams": hp,
        }
        config_path = args.output_dir / "learned_head.json"
        config_path.write_text(json.dumps(config, indent=2))
        print(f"Config saved → {config_path}", file=sys.stderr)
        print(f"Head version: {version}  "
              f"({len(MODALITY_GROUPS)} groups × {len(ENSEMBLE_SEEDS)} seeds)", file=sys.stderr)


if __name__ == "__main__":
    main()
