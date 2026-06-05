#!/usr/bin/env python3
"""
Train a small projection head on top of cached PE-G + color-histogram features
using supervised contrastive loss. Constrains every batch to come from a single
"model" (= dataset), so the head learns shoot-discriminative axes rather than
identity-discriminative ones. The face/identity feature is held constant
within a batch, so it carries no gradient.

Outputs, per --output-dir:
  proj_head.pt                          (state_dict)
  <DATASET>_proj.npy                    (projected vectors, sorted-filename order)
  <DATASET>_dist_matrix.bin             (condensed f64 distances for cluster-tool)
  <DATASET>_filenames.json              (the sorted filename list this dataset uses)
  <DATASET>_train_filenames.json        (only when DATASET was a training set)
  <DATASET>_holdout_filenames.json      (only when DATASET was a training set; held-out groups)
  summary.json                          (config + per-epoch metrics)

Usage:
  python scripts/train_projection_head.py \\
      --dataset M1:/abs/path/ClusteringBenchmark1-austin \\
      --dataset M2:/abs/path/ClusteringBenchmark2-sarah \\
      --train M1,M2 \\
      --eval M1,M2 \\
      --within-holdout-frac 0.2 \\
      --epochs 50 \\
      --output-dir /tmp/proj_head_run1

Splits:
  - Whole-model holdout: pass --train M1 (or --train M2). Datasets not listed
    in --train are evaluated as never-seen.
  - Within-model holdout: --within-holdout-frac > 0 partitions each TRAINING
    dataset's groups (NOT images) into train/eval sets. The held-out groups'
    filenames are written to *_holdout_filenames.json for filtered ARI eval.

Backbone: MPS by default on macOS, falls back to CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Data loading ─────────────────────────────────────────────────────────────


@dataclass
class Dataset:
    """One model's data: features, ground-truth groups, filename ordering."""
    name: str
    target_dir: str
    filenames: list[str]                  # sorted by filename, length = N
    features: torch.Tensor                # (N, in_dim) float32, raw concat of pe-g + color
    group_id: list[int | None]            # length N; None for ungrouped
    # Indices used by the sampler. Built once.
    train_group_to_idxs: dict[int, list[int]] = field(default_factory=dict)
    holdout_group_to_idxs: dict[int, list[int]] = field(default_factory=dict)
    # Images that are the sole member of an explicit (user-labeled) group.
    # Used only as background negatives — never anchors or positives.
    # Distinct from ungrouped images, which we drop entirely (could secretly
    # belong to an existing shoot).
    singleton_idxs: list[int] = field(default_factory=list)
    # ArcFace local class indices: original group_id → [0, n_train_groups)
    gid_to_cls: dict[int, int] = field(default_factory=dict)
    # Augmented views: (N, K, D) tensor. None if not loaded. views_valid_mask is a
    # (N,) bool marking rows with real augmented data; the rest fall back to original.
    views_features: torch.Tensor | None = None
    views_valid_mask: torch.Tensor | None = None


def load_dataset(name: str, target_dir: str, use_dinov3_patches: bool = False, use_augmented_views: bool = False) -> Dataset:
    target_dir = os.path.abspath(target_dir)
    cache = os.path.join(target_dir, ".reorder-cache")
    npz_path = os.path.join(cache, "embeddings_hash_cache.npz")
    ch_path = os.path.join(cache, "content_hashes.json")
    groups_path = os.path.join(target_dir, ".reorder-groups.json")

    with open(ch_path) as f:
        content_hashes: dict[str, str] = json.load(f)
    filenames = sorted(content_hashes.keys())
    fn_to_idx = {fn: i for i, fn in enumerate(filenames)}

    z = np.load(npz_path, allow_pickle=False)
    hashes = list(z["hashes"])
    hash_to_row = {h: i for i, h in enumerate(hashes)}
    if "pecore_g" not in z.files:
        sys.exit(f"[{name}] pecore_g missing from {npz_path}")
    if "color" not in z.files:
        sys.exit(f"[{name}] color missing from {npz_path}")
    peg_hash = z["pecore_g"]          # (M, 1280)
    col_hash = z["color"]             # (M, 693)

    # Reindex from hash order → sorted-filename order
    n = len(filenames)
    peg = np.empty((n, peg_hash.shape[1]), dtype=np.float32)
    col = np.empty((n, col_hash.shape[1]), dtype=np.float32)
    for i, fn in enumerate(filenames):
        row = hash_to_row[content_hashes[fn]]
        peg[i] = peg_hash[row]
        col[i] = col_hash[row]
    # Defensive L2-normalize PE-G (already is, but verify); color is left raw,
    # the input LayerNorm in the head handles its different scale.
    peg /= np.linalg.norm(peg, axis=1, keepdims=True).clip(min=1e-8)
    parts = [peg, col]

    if use_dinov3_patches:
        patches_path = os.path.join(cache, "dinov3_patches_hash_cache.npy")
        patches_hashes_path = os.path.join(cache, "dinov3_patches_hashes.json")
        if not os.path.exists(patches_path) or not os.path.exists(patches_hashes_path):
            sys.exit(f"[{name}] dinov3 patches missing at {patches_path}")
        with open(patches_hashes_path) as f:
            patch_hashes = json.load(f)
        patch_arr = np.load(patches_path, mmap_mode="r")  # (M_dino, 49, 768)
        patch_hash_to_row = {h: i for i, h in enumerate(patch_hashes)}
        # Flatten the 7×7 grid to 49*768 = 37632 per image, then L2-normalize so
        # it contributes on the same scale as PE-G under cosine distance. Images
        # whose patches are missing get zero-filled (defensive — should be rare).
        flat_dim = patch_arr.shape[1] * patch_arr.shape[2]
        dino = np.zeros((n, flat_dim), dtype=np.float32)
        missing = 0
        for i, fn in enumerate(filenames):
            h = content_hashes[fn]
            row = patch_hash_to_row.get(h)
            if row is None:
                missing += 1
                continue
            dino[i] = patch_arr[row].reshape(-1)
        if missing:
            print(f"  [{name}] WARN: {missing} images missing dinov3 patches (zero-filled)", file=sys.stderr)
        dino /= np.linalg.norm(dino, axis=1, keepdims=True).clip(min=1e-8)
        parts.append(dino)

    features = np.concatenate(parts, axis=1)

    with open(groups_path) as f:
        groups_raw = json.load(f)
    groups = groups_raw if isinstance(groups_raw, list) else groups_raw.get("groups", [])
    group_id: list[int | None] = [None] * n
    for gi, g in enumerate(groups):
        for fn in g["images"]:
            if fn in fn_to_idx:
                group_id[fn_to_idx[fn]] = gi

    n_grouped = sum(1 for g in group_id if g is not None)
    print(f"  [{name}] loaded N={n} dim={features.shape[1]} grouped={n_grouped} groups={len(groups)}", file=sys.stderr)

    # Augmented views (pixel-aug pre-extraction). PE-G + color only (no DINOv3
    # patches in the augmented cache — would be excessive storage). Loaded as a
    # (N, K, D_peg+color) tensor; the train loop randomly picks view 0 (original)
    # or one of the K augmented views per sample.
    views_features = None
    views_valid_mask = None
    if use_augmented_views:
        peg_v_path = os.path.join(cache, "pecore_g_views.npy")
        col_v_path = os.path.join(cache, "color_views.npy")
        meta_path = os.path.join(cache, "views_meta.json")
        if all(os.path.exists(p) for p in [peg_v_path, col_v_path, meta_path]):
            with open(meta_path) as f:
                vm = json.load(f)
            views_complete = int(vm.get("completed_through", 0))
            # Rows with real data. Legacy caches lack view_indices (extracted all),
            # so the valid set is the [0, completed_through) prefix.
            view_indices = vm.get("view_indices")
            peg_v = np.load(peg_v_path)   # (N, K, 1280)
            col_v = np.load(col_v_path)   # (N, K, 693)
            if peg_v.shape[0] != n or col_v.shape[0] != n:
                print(f"  [{name}] WARN: views shape mismatch (N={n}, peg_v={peg_v.shape[0]}), ignoring views", file=sys.stderr)
            else:
                # The training-side feature vector is PE-G + color (+ optional dinov3).
                # Augmented views are PE-G + color only — for the dinov3 slot, reuse
                # the original (training will still see the augmented PE-G+color
                # combined with the un-augmented dinov3 if it's enabled).
                view_pegcol = np.concatenate([peg_v, col_v], axis=2)  # (N, K, 1973)
                if use_dinov3_patches:
                    # Stitch in the original dinov3 (shared across all K views).
                    # features has order: [peg(1280), color(693), dino(...)]
                    dino_dim = features.shape[1] - 1280 - 693
                    dino_per_image = features[:, 1280 + 693:].reshape(n, 1, dino_dim)
                    dino_tile = np.broadcast_to(dino_per_image, (n, peg_v.shape[1], dino_dim))
                    view_full = np.concatenate([view_pegcol, dino_tile], axis=2)
                else:
                    view_full = view_pegcol
                views_features = torch.from_numpy(view_full.astype(np.float32))
                # Valid = a target row that's been processed (< completed).
                mask = np.zeros(n, dtype=bool)
                if view_indices is None:
                    mask[:views_complete] = True
                else:
                    vi = np.asarray(view_indices, dtype=np.int64)
                    vi = vi[(vi >= 0) & (vi < views_complete)]
                    mask[vi] = True
                views_valid_mask = torch.from_numpy(mask)
                print(f"  [{name}] loaded {peg_v.shape[1]} augmented views "
                      f"({int(mask.sum())}/{n} rows valid)", file=sys.stderr)
        else:
            print(f"  [{name}] augmented views not found in {cache}", file=sys.stderr)

    return Dataset(
        name=name,
        target_dir=target_dir,
        filenames=filenames,
        features=torch.from_numpy(features),
        group_id=group_id,
        views_features=views_features,
        views_valid_mask=views_valid_mask,
    )


def split_groups(ds: Dataset, holdout_frac: float, seed: int):
    """Partition this dataset's GROUPS (not images) into train + holdout sets.
    Also assigns local class indices [0, n_train_groups) for ArcFace."""
    group_to_idxs: dict[int, list[int]] = defaultdict(list)
    for i, gid in enumerate(ds.group_id):
        if gid is not None:
            group_to_idxs[gid].append(i)
    eligible = [gid for gid, ix in group_to_idxs.items() if len(ix) >= 2]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    n_holdout = int(round(len(eligible) * holdout_frac))
    holdout_set = set(eligible[:n_holdout])
    ds.train_group_to_idxs = {gid: ix for gid, ix in group_to_idxs.items() if gid not in holdout_set and len(ix) >= 2}
    ds.holdout_group_to_idxs = {gid: ix for gid, ix in group_to_idxs.items() if gid in holdout_set}
    ds.singleton_idxs = [ix[0] for gid, ix in group_to_idxs.items() if len(ix) == 1]
    # Local class indices for ArcFace head (per-dataset = no cross-model gradient)
    ds.gid_to_cls = {gid: ci for ci, gid in enumerate(sorted(ds.train_group_to_idxs.keys()))}
    print(
        f"  [{ds.name}] split: train_groups={len(ds.train_group_to_idxs)} "
        f"holdout_groups={len(ds.holdout_group_to_idxs)} "
        f"singletons={len(ds.singleton_idxs)} (background negatives only)",
        file=sys.stderr,
    )


# ── Model ────────────────────────────────────────────────────────────────────


class ManualLayerNorm(nn.Module):
    """LayerNorm with the affine step done outside the fused MPS kernel.
    PyTorch ≤ 2.11's MPS fused LayerNorm has a broken backward when affine
    (weight/bias) is enabled — the weight gradient comes back NaN (or zero),
    silently NaN-ing every other parameter on the next AdamW step. The fused
    forward + backward without affine is correct, though, so we call it
    weight-less and apply our own scale+shift after."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(x, (self.dim,), None, None, self.eps)
        return normalized * self.weight + self.bias


class ProjectionHead(nn.Module):
    """LayerNorm → MLP → L2-normalize. Backbone-free; expects concat features."""

    def __init__(self, in_dim: int, hidden: int = 1024, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_norm = ManualLayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(self.in_norm(x))
        return F.normalize(z, dim=-1)


# ── ArcFace head ─────────────────────────────────────────────────────────────


class ArcFaceHead(nn.Module):
    """
    One learned prototype per class. Weights are L2-normalized at every forward
    so columns live on the unit sphere. Adds angular margin m to the true class
    logit (cos(θ + m) = cosθ·cosm − sinθ·sinm), scales by s, then standard CE.
    Uses the "easy margin" guard to avoid the cos(θ+m) blowup near θ=π early in
    training. One head per dataset → no gradient ever flows across model identity.
    """

    def __init__(self, in_dim: int, n_classes: int, margin: float = 0.3, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, in_dim) * 0.01)
        self.margin = margin
        self.scale = scale
        self._cos_m = float(np.cos(margin))
        self._sin_m = float(np.sin(margin))
        # threshold cosθ < th  → use cosφ - sinm (easy margin fallback)
        self._th = float(np.cos(np.pi - margin))
        self._mm = float(np.sin(np.pi - margin) * margin)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # z already L2-normalized by the projection head
        w = F.normalize(self.weight, dim=1)
        cos = z @ w.t()                                      # (B, C)
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin = torch.sqrt(1.0 - cos * cos)
        cos_phi = cos * self._cos_m - sin * self._sin_m      # cos(θ+m)
        # easy margin fallback when θ close to π → cos < th
        cos_phi = torch.where(cos > self._th, cos_phi, cos - self._mm)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        logits = one_hot * cos_phi + (1.0 - one_hot) * cos
        logits = logits * self.scale
        return F.cross_entropy(logits, labels)


# ── Loss ─────────────────────────────────────────────────────────────────────


# Per-(device, B) cache of the (1 - eye) valid-pair mask and the diagonal
# -1e9 self-pair mask. Batch shape is constant within a run, so caching these
# across batches avoids re-allocating them every step.
_MASK_VALID_CACHE: dict[tuple[str, int], torch.Tensor] = {}
_DIAG_NEGINF_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def _mask_valid(b: int, device) -> torch.Tensor:
    key = (str(device), b)
    m = _MASK_VALID_CACHE.get(key)
    if m is None:
        m = 1.0 - torch.eye(b, device=device)
        _MASK_VALID_CACHE[key] = m
    return m


def _diag_neginf(b: int, device) -> torch.Tensor:
    key = (str(device), b)
    m = _DIAG_NEGINF_CACHE.get(key)
    if m is None:
        m = torch.eye(b, device=device) * -1e9
        _DIAG_NEGINF_CACHE[key] = m
    return m


# Zero-shot baseline composition, matching blend_dist_matrix.py defaults:
# concat(peg · 1.0, L2(color) · 0.8) → unit-norm → cosine. PE-G is the first
# 1280 dims of the head input (already L2-normalized in load_dataset); color is
# the remainder (raw).
PEG_DIM = 1280
BASE_PEG_WEIGHT = 1.0
BASE_COLOR_WEIGHT = 0.8


def zeroshot_base_sim(batch_feats: torch.Tensor) -> torch.Tensor:
    """(B, B) detached zero-shot cosine-sim matrix for blend-aware training.
    `batch_feats` is the head input (peg ⊕ color concat). Reconstructs the same
    weighted-concat the clustering pipeline's baseline uses."""
    peg = batch_feats[:, :PEG_DIM] * BASE_PEG_WEIGHT
    col = F.normalize(batch_feats[:, PEG_DIM:], dim=1) * BASE_COLOR_WEIGHT
    b = F.normalize(torch.cat([peg, col], dim=1), dim=1)
    return (b @ b.t()).detach()


def sup_con_loss(
    z: torch.Tensor,
    pos_weights: torch.Tensor,
    temperature: float = 0.1,
    base_sim: torch.Tensor | None = None,
    blend_w: float = 1.0,
    self_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Soft-label supervised contrastive loss.
    z: (B, D) L2-normalized features.
    pos_weights: (B, B) where pos_weights[i, j] ∈ [0, 1] is how much sample j
                 counts as a positive for anchor i. For hard labels this is just
                 (label[i] == label[j]).float(). For mixed samples (cross-group
                 MixUp) it's the soft-label overlap. Self-pairs are masked.
    base_sim: optional (B, B) DETACHED zero-shot cosine-similarity matrix. When
              given, the contrastive logits use the blended similarity that the
              clustering pipeline actually scores on — blend_w·cos(z) +
              (1−blend_w)·base_sim — so the head learns to *complement* the
              zero-shot signal (residual learning) rather than re-learn it.
              Linearly blending cosines == linearly blending the deployed
              condensed distances (the constant offset cancels row-wise in the
              softmax). base_sim is constant w.r.t. the head, so gradients still
              flow only through z, scaled by blend_w.
    blend_w: learned-head fraction; should match the inference blend (0.60).
    self_idx: optional (B,) long tensor of the source IMAGE index per batch row.
              When given, ALL same-source-image pairs (not just the literal
              i==i diagonal) are treated as self-pairs and excluded from both
              the numerator and the softmax denominator. The PKSampler draws
              with replacement for groups with < K images, so the same image
              can land at multiple batch positions; those off-diagonal
              duplicates would otherwise be perfect (cos=1) positives that also
              dominate the denominator (exp(1/τ) ≫ everything). When None, falls
              back to the cached eye-based diagonal masking (legacy behavior).
    """
    device = z.device
    b = z.shape[0]
    if base_sim is not None:
        sim = (blend_w * (z @ z.t()) + (1.0 - blend_w) * base_sim) / temperature
    else:
        sim = z @ z.t() / temperature
    # Pin the self-pair entries to -1e9 so they contribute ~0 to the softmax
    # denominator; mask_valid still zeros W on those entries so the
    # (W * log_prob) product is finite (0 × big-negative). With self_idx the
    # self-pair set is "same source image" (subsumes the diagonal); content
    # varies per batch so these can't use the (device, b) caches.
    if self_idx is not None:
        same = (self_idx.unsqueeze(0) == self_idx.unsqueeze(1))
        neg_inf = same.float() * -1e9
        mask_valid = (~same).float()
    else:
        neg_inf = _diag_neginf(b, device)
        mask_valid = _mask_valid(b, device)
    log_prob = F.log_softmax(sim + neg_inf, dim=1)

    W = pos_weights * mask_valid
    pos_weight_sum = W.sum(dim=1)
    has_pos = (pos_weight_sum > 1e-6).float()
    # Sync-free reduction: weight each row's log-prob by has_pos and divide by
    # the count, rather than `mean_log_prob_pos[has_pos].mean()` — boolean
    # masked-select would force a CPU↔GPU sync mid-batch.
    mean_log_prob_pos = (W * log_prob).sum(dim=1) / pos_weight_sum.clamp(min=1e-12)
    return -(mean_log_prob_pos * has_pos).sum() / has_pos.sum().clamp(min=1.0)


def hard_label_weights(labels: torch.Tensor) -> torch.Tensor:
    """Convert (B,) int labels to (B, B) hard pos-weight matrix."""
    return (labels.unsqueeze(0) == labels.unsqueeze(1)).float()


def apply_cross_mixup(
    feats: torch.Tensor,
    labels: list[int],
    alpha: float,
    prob: float,
    device,
    cross_lam: torch.Tensor | None = None,
    partner: torch.Tensor | None = None,
    do_mix: torch.Tensor | None = None,
    label_col: torch.Tensor | None = None,
    n_classes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each sample with probability `prob`, pair with a random DIFFERENT-group
    sample in the batch and interpolate features with λ ~ Beta(α, α). Returns
    the (possibly modified) features and a (B, B) soft pos-weight matrix that
    accounts for the mixed soft labels.

    Vectorized: builds the diff-group candidate matrix once, samples partners
    with a single multinomial draw per sample. Was O(B²) python before.
    """
    b = feats.shape[0]
    # Caller may pass precomputed (label_col, n_classes) to skip the per-batch
    # Python densification + host→device copy. Keep the in-function fallback so
    # this function still works standalone.
    if label_col is None or n_classes is None:
        unique_gids = sorted(set(labels))
        gid_to_col = {g: c for c, g in enumerate(unique_gids)}
        label_col = torch.tensor([gid_to_col[g] for g in labels], dtype=torch.long, device=device)
        n_classes = len(unique_gids)

    # One-hot label table. F.one_hot is a single op vs zeros+scatter (two ops).
    L = F.one_hot(label_col, num_classes=n_classes).float()

    if prob <= 0 or alpha <= 0:
        return feats.clone(), L @ L.t()

    # Partner and do_mix can be precomputed at epoch start. do_mix is the set of
    # rows that actually get mixed, so it must already exclude rows with no
    # different-group partner (possible when a batch has < P distinct groups —
    # the sampler can draw groups with replacement). Precomputed do_mix passed in
    # by the caller carries the same guard; see the bulk precompute in main().
    if partner is None or do_mix is None:
        diff_mat = (label_col.unsqueeze(0) != label_col.unsqueeze(1)).float()
        if do_mix is None:
            do_mix = (torch.rand(b, device=device) < prob) & (diff_mat.sum(dim=1) > 0)
        if partner is None:
            rand_mat = torch.rand_like(diff_mat)
            partner = torch.where(diff_mat > 0, rand_mat, rand_mat.new_full((), -1.0)).argmax(dim=1)
    has_partner = do_mix

    # λ ~ Beta(α, α). Pin λ=1 for rows we don't mix → out_feats[i] == feats[i].
    if cross_lam is None:
        beta = torch.distributions.Beta(alpha, alpha)
        lam = beta.sample((b,)).to(device)
    else:
        lam = cross_lam
    lam = torch.where(has_partner, lam, torch.ones_like(lam))
    lam_v = lam.unsqueeze(1)

    out_feats = lam_v * feats + (1 - lam_v) * feats[partner]

    # Soft label = λ * one_hot(self) + (1-λ) * one_hot(partner).
    L = lam_v * L + (1 - lam_v) * L[partner]

    return out_feats, L @ L.t()


# ── Sampler ──────────────────────────────────────────────────────────────────


# ── Augmentations (feature-space) ────────────────────────────────────────────


# Same-group mixup partner index is fully determined by (mix_b, k, device) —
# cache once per dataset so the hot loop doesn't pay for arange + arithmetic.
_PARTNER_IDX_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}


def _partner_idx(mix_b: int, k: int, device) -> torch.Tensor:
    key = (str(device), mix_b, k)
    p = _PARTNER_IDX_CACHE.get(key)
    if p is None:
        pos = torch.arange(mix_b, device=device)
        block_start = (pos // k) * k
        p = block_start + (pos - block_start + 1) % k
        _PARTNER_IDX_CACHE[key] = p
    return p


def apply_augmentations(
    feats: torch.Tensor,
    k: int,
    *,
    mixup_alpha: float,
    drop_color_prob: float,
    drop_peg_prob: float,
    feature_dropout: float,
    feature_noise: float,
    peg_dim: int,
    color_dim: int,
    mixup_end: int | None = None,
    mixup_lam: torch.Tensor | None = None,
    drop_color_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    feats: (B, D) where rows are grouped in chunks of K (P groups × K images).
    Returns possibly-augmented copy of feats. Augmentations are training-only.
    mixup_end bounds same-group mixup to [0, mixup_end); the tail (singletons)
    still gets per-row augs.
    """
    b, _d = feats.shape
    out = feats.clone()
    mix_b = b if mixup_end is None else mixup_end

    # Same-group MixUp: within each K-block, rotate-pair each anchor with a
    # same-group partner and interpolate with λ ~ Beta(α, α). Label is unchanged
    # (both anchor and partner are from the same group). Strengthens the
    # "what does within-group variation look like" signal without changing the
    # contrastive structure.
    if mixup_alpha > 0 and k >= 2 and mix_b >= k:
        # Partner = next index within each K-block (last wraps to first). Cached
        # per (mix_b, k, device) — constant across batches for the same dataset.
        partner_idx = _partner_idx(mix_b, k, feats.device)
        if mixup_lam is None:
            beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
            lam = beta.sample((mix_b, 1)).to(feats.device)
        else:
            lam = mixup_lam
        out[:mix_b] = lam * out[:mix_b] + (1 - lam) * feats[partner_idx]

    # Drop color: with per-image prob, zero the color slice. Closest feature-
    # space analog to background-masking — color histograms heavily encode
    # backdrop appearance, so this attacks the most plausible shortcut.
    if drop_color_prob > 0 and color_dim > 0:
        if drop_color_mask is None:
            mask = (torch.rand(b, device=feats.device) < drop_color_prob).float().unsqueeze(1)
        else:
            mask = drop_color_mask
        out[:, peg_dim:peg_dim + color_dim] *= (1 - mask)

    # Drop PE-G: symmetric to drop-color. Useful as a sanity check / control.
    # If this hurts, PE-G carries most of the signal (expected). If it helps,
    # the head is over-relying on PE-G shortcuts.
    if drop_peg_prob > 0 and peg_dim > 0:
        mask = (torch.rand(b, device=feats.device) < drop_peg_prob).float().unsqueeze(1)
        out[:, :peg_dim] *= (1 - mask)

    # Feature dropout: per-dim Bernoulli on the whole input. Scaled to preserve
    # expectation. Forces head to be robust to missing feature dimensions.
    if feature_dropout > 0:
        keep = 1 - feature_dropout
        mask = (torch.rand_like(out) < keep).float() / keep
        out = out * mask

    # Gaussian feature noise: σ-scaled additive noise. Different shape than
    # dropout — smoothes the decision boundary rather than masking dims.
    # Note: PE-G is L2-normalized so unit-magnitude per row; color is raw.
    # Apply per-modality so the noise scale is proportionate.
    if feature_noise > 0:
        noise = torch.randn_like(out) * feature_noise
        out = out + noise

    return out


class PKSampler:
    """
    Each batch: P groups × K images, all from a single dataset. The dataset is
    chosen round-robin across the training set. This enforces the "same-model
    only" constraint, making identity invariant within a batch.

    With hard_neg_frac > 0, after picking one random anchor group, a fraction
    of the remaining P-1 groups is drawn from the anchor's nearest neighbors in
    raw PE-G centroid space. The rest are drawn uniformly at random. This
    over-represents confusable groups in the contrastive denominator.

    With use_singletons=True, every batch additionally appends ALL singleton-
    group images for that dataset. They get unique negative-int sentinel labels
    so they never match any real group (or each other) as positives —
    sup_con_loss's has_pos mask skips them as anchors, but they show up in
    every other anchor's denominator. Costs O(n_singletons) extra rows per
    batch (linear, not K-multiplied), so cheap given singletons are rare.
    """

    def __init__(
        self,
        train_dsets: list[Dataset],
        p: int,
        k: int,
        batches_per_epoch: int,
        seed: int,
        hard_neg_frac: float = 0.0,
        hard_neg_pool_k: int = 20,
        use_singletons: bool = False,
        peg_dim: int = 1280,
    ):
        self.train_dsets = train_dsets
        self.p = p
        self.k = k
        self.batches_per_epoch = batches_per_epoch
        self.rng = random.Random(seed)
        self.hard_neg_frac = hard_neg_frac
        self.use_singletons = use_singletons

        # Precompute nearest-neighbor groups (per dataset) for hard-neg sampling.
        # Use PE-G centroid (already L2-normalized) for the similarity — cheap,
        # static, and matches the baseline's notion of "similar."
        self.neighbors: dict[str, dict[int, list[int]]] = {}
        if hard_neg_frac > 0:
            for ds in train_dsets:
                gids = list(ds.train_group_to_idxs.keys())
                if len(gids) < 2:
                    self.neighbors[ds.name] = {g: [] for g in gids}
                    continue
                # Compute per-group centroid in PE-G space (first peg_dim dims)
                # Re-L2-normalize after mean (centroid is no longer unit-norm).
                cents = np.zeros((len(gids), peg_dim), dtype=np.float32)
                for i, g in enumerate(gids):
                    rows = ds.train_group_to_idxs[g]
                    cents[i] = ds.features[rows, :peg_dim].mean(dim=0).numpy()
                cents /= np.linalg.norm(cents, axis=1, keepdims=True).clip(min=1e-8)
                # NumPy 2.x's float32 matmul SIMD kernel raises spurious
                # divide-by-zero / overflow / invalid-value RuntimeWarnings here
                # even though cents and the result are fully finite (it trips the
                # FP-exception flags internally). The neighbor pools are correct
                # regardless; silence the false alarm rather than perturb the math.
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    sims = cents @ cents.T  # (G, G)
                np.fill_diagonal(sims, -1.0)
                k_pool = min(hard_neg_pool_k, len(gids) - 1)
                # For each group, top-k_pool nearest groups (descending sim)
                self.neighbors[ds.name] = {
                    gids[i]: [gids[j] for j in np.argsort(-sims[i])[:k_pool]]
                    for i in range(len(gids))
                }
            print(f"  built hard-neg pools (frac={hard_neg_frac}, pool_k={hard_neg_pool_k})", file=sys.stderr)

    def __iter__(self):
        ds_cycle = [self.train_dsets[i % len(self.train_dsets)] for i in range(self.batches_per_epoch)]
        self.rng.shuffle(ds_cycle)
        for ds in ds_cycle:
            gids = list(ds.train_group_to_idxs.keys())
            if len(gids) < self.p:
                chosen = [self.rng.choice(gids) for _ in range(self.p)]
            elif self.hard_neg_frac > 0 and ds.name in self.neighbors:
                # Hard-neg sampling: pick 1 random anchor, then mix of hard + random
                anchor = self.rng.choice(gids)
                n_hard = int(round((self.p - 1) * self.hard_neg_frac))
                n_rand = self.p - 1 - n_hard
                hard_pool = [g for g in self.neighbors[ds.name][anchor] if g != anchor]
                hard_picks = self.rng.sample(hard_pool, min(n_hard, len(hard_pool)))
                remaining = [g for g in gids if g != anchor and g not in hard_picks]
                rand_picks = self.rng.sample(remaining, min(n_rand, len(remaining)))
                chosen = [anchor] + hard_picks + rand_picks
                # Pad if hard pool was too small
                while len(chosen) < self.p:
                    extra = self.rng.choice([g for g in gids if g not in chosen])
                    chosen.append(extra)
            else:
                chosen = self.rng.sample(gids, self.p)

            idxs: list[int] = []
            labels: list[int] = []
            for gid in chosen:
                pool = ds.train_group_to_idxs[gid]
                if len(pool) >= self.k:
                    picks = self.rng.sample(pool, self.k)
                else:
                    picks = [self.rng.choice(pool) for _ in range(self.k)]
                idxs.extend(picks)
                labels.extend([gid] * self.k)

            # Background negatives: append every singleton-group image. Sentinel
            # label = -(idx+1) — negative so it can't collide with real (>=0)
            # group ids, and unique per image so no two singletons match.
            if self.use_singletons:
                for pick in ds.singleton_idxs:
                    idxs.append(pick)
                    labels.append(-(pick + 1))

            yield ds, idxs, labels

    def __len__(self):
        return self.batches_per_epoch


# ── Evaluation ───────────────────────────────────────────────────────────────


def project_all(head: ProjectionHead, ds: Dataset, device, batch_size: int = 512) -> torch.Tensor:
    head.eval()
    out = []
    with torch.no_grad():
        for i in range(0, ds.features.shape[0], batch_size):
            batch = ds.features[i:i + batch_size].to(device)
            out.append(head(batch).cpu())
    head.train()
    return torch.cat(out, dim=0)  # (N, out_dim)


def pairwise_acc_holdout(proj: torch.Tensor, ds: Dataset, threshold: float | None = None) -> dict:
    """
    Pair-classification AUC on held-out groups: among same-model pairs where
    BOTH images are in holdout_group_to_idxs, distinguish same-group from
    different-group by cosine similarity. Returns AUC and best-threshold acc.
    """
    if not ds.holdout_group_to_idxs:
        return {"n_holdout_imgs": 0}
    holdout_idxs = []
    holdout_labels = []
    for gid, ix in ds.holdout_group_to_idxs.items():
        holdout_idxs.extend(ix)
        holdout_labels.extend([gid] * len(ix))
    z = proj[holdout_idxs]                       # (Nh, D)
    sims = (z @ z.t()).numpy()
    labs = np.array(holdout_labels)
    nh = len(holdout_idxs)
    iu = np.triu_indices(nh, k=1)
    sim_pairs = sims[iu]
    same = (labs[iu[0]] == labs[iu[1]]).astype(np.int8)

    # AUC via rank statistic
    order = np.argsort(-sim_pairs)              # most-similar first
    ranks_same = np.where(same[order] == 1)[0]
    n_pos = int(same.sum())
    n_neg = len(same) - n_pos
    if n_pos == 0 or n_neg == 0:
        auc = float("nan")
    else:
        # AUC = (sum of (n_neg - rank_among_negs) for positives) / (n_pos * n_neg)
        # Simpler: use Mann-Whitney U
        from scipy.stats import mannwhitneyu  # noqa: F401  (optional)
        # Avoid scipy dependency — compute by sorting
        idx_sort = np.argsort(sim_pairs)
        ranks = np.empty_like(idx_sort, dtype=np.float64)
        ranks[idx_sort] = np.arange(1, len(sim_pairs) + 1, dtype=np.float64)
        sum_ranks_pos = ranks[same == 1].sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    # Best-threshold pair accuracy (sweep)
    # For tractability, sample thresholds across distinct sims
    qs = np.quantile(sim_pairs, np.linspace(0.0, 1.0, 201))
    best_acc = 0.0
    best_thr = 0.0
    for thr in qs:
        pred = (sim_pairs >= thr).astype(np.int8)
        acc = float((pred == same).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return {
        "n_holdout_imgs": nh,
        "n_holdout_pairs": int(len(sim_pairs)),
        "n_same_pairs": n_pos,
        "auc": float(auc),
        "best_pair_acc": best_acc,
        "best_threshold": best_thr,
    }


# ── Output writers ───────────────────────────────────────────────────────────


def write_dist_matrix(proj: torch.Tensor, path: str):
    """
    Write condensed cosine-distance matrix in cluster-tool's expected format:
      u64 LE n_images, then n*(n-1)/2 f64 LE distances (i<j, row-major).
    Distance = 1 - cos_sim, clamped to [0, 2].
    """
    n = proj.shape[0]
    z = F.normalize(proj.float(), dim=-1)  # ensure unit norm
    # Compute upper triangle, vectorized by row
    out = np.empty(n * (n - 1) // 2, dtype=np.float64)
    off = 0
    for i in range(n - 1):
        sims = (z[i:i + 1] @ z[i + 1:].t()).numpy().reshape(-1)
        d = np.clip(1.0 - sims, 0.0, 2.0).astype(np.float64)
        out[off:off + d.shape[0]] = d
        off += d.shape[0]
    assert off == out.shape[0]
    with open(path, "wb") as f:
        f.write(np.uint64(n).tobytes())
        f.write(out.tobytes())


def filenames_from_groups(ds: Dataset, group_to_idxs: dict[int, list[int]]) -> list[str]:
    out = []
    for ix_list in group_to_idxs.values():
        for i in ix_list:
            out.append(ds.filenames[i])
    return sorted(out)


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_dataset_arg(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"--dataset wants name:dir, got {s!r}")
    name, d = s.split(":", 1)
    return name, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, type=parse_dataset_arg,
                    help="name:dir — can repeat")
    ap.add_argument("--train", type=lambda s: s.split(","), required=True,
                    help="comma-separated dataset names to train on")
    ap.add_argument("--eval", type=lambda s: s.split(","), default=None,
                    help="comma-separated dataset names to write projections for (default = all)")
    ap.add_argument("--within-holdout-frac", type=float, default=0.0,
                    help="fraction of GROUPS in each training set to hold out")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--out-dim", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use-dinov3-patches", action="store_true",
                    help="concatenate flattened 7x7 DINOv3 patches (49*768=37632d) to input")
    ap.add_argument("--use-augmented-views", action="store_true",
                    help="load pre-extracted augmented views and sample randomly per image during training")
    ap.add_argument("--mixup-alpha", type=float, default=0.0,
                    help="same-group MixUp Beta(α,α) parameter; 0 disables. Typical: 0.2-0.4")
    ap.add_argument("--drop-color-prob", type=float, default=0.0,
                    help="per-image probability of zeroing color features (background-masking analog)")
    ap.add_argument("--drop-peg-prob", type=float, default=0.0,
                    help="per-image probability of zeroing PE-G features (control)")
    ap.add_argument("--feature-dropout", type=float, default=0.0,
                    help="per-dim Bernoulli dropout on the raw input vector")
    ap.add_argument("--feature-noise", type=float, default=0.0,
                    help="Gaussian additive noise σ on the raw input vector. Typical: 0.01-0.05")
    ap.add_argument("--cross-mixup-prob", type=float, default=0.0,
                    help="per-image probability of cross-group MixUp (creates soft-label boundary samples)")
    ap.add_argument("--cross-mixup-alpha", type=float, default=0.4,
                    help="Beta(α,α) parameter for cross-group MixUp; only used when --cross-mixup-prob > 0")
    ap.add_argument("--hard-neg-frac", type=float, default=0.0,
                    help="fraction of in-batch negative groups drawn from each anchor group's nearest neighbors (PE-G centroid). 0 = uniform random sampling.")
    ap.add_argument("--hard-neg-pool-k", type=int, default=20,
                    help="how many nearest-neighbor groups to consider for hard-negative sampling")
    ap.add_argument("--use-singleton-negatives", action="store_true",
                    help="append every explicit singleton-group image to each batch "
                         "as a background negative (unique sentinel label so it never "
                         "appears as a positive). Cheap: ~0-4 extra rows per batch.")
    ap.add_argument("--dedup-self-pairs", action="store_true",
                    help="Mask ALL same-source-image pairs (not only the i==i "
                         "diagonal) from the SupCon numerator AND denominator. "
                         "Fixes tiny groups (<K images) that the PKSampler draws "
                         "with replacement, which otherwise inject cos=1 "
                         "self-as-positive pairs that dominate the softmax.")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--blend-aware", action="store_true",
                    help="Train the contrastive loss on the blended (learned + zero-shot) "
                         "similarity the clustering pipeline scores on, so the head learns "
                         "the residual rather than re-learning zero-shot structure.")
    ap.add_argument("--blend-weight", type=float, default=0.6,
                    help="Learned-head fraction used by --blend-aware; match the inference blend.")
    ap.add_argument("--arcface-weight", type=float, default=1.0,
                    help="0 disables ArcFace; total_loss = supcon + arcface_weight * arcface")
    ap.add_argument("--arcface-margin", type=float, default=0.3)
    ap.add_argument("--arcface-scale", type=float, default=30.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batches-per-epoch", type=int, default=200)
    ap.add_argument("--p-groups", type=int, default=16, help="groups per batch")
    ap.add_argument("--k-images", type=int, default=8, help="images per group per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-clip", type=float, default=5.0,
                    help="max grad norm for clip_grad_norm_ (0 disables). Guards against "
                         "mid-training divergence in aggressive configs.")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected)")
    args = ap.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    print(f"device: {device}", file=sys.stderr)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    datasets: dict[str, Dataset] = {}
    for name, d in args.dataset:
        print(f"loading dataset {name} from {d}", file=sys.stderr)
        datasets[name] = load_dataset(name, d, use_dinov3_patches=args.use_dinov3_patches, use_augmented_views=args.use_augmented_views)
    in_dim = next(iter(datasets.values())).features.shape[1]
    for ds in datasets.values():
        assert ds.features.shape[1] == in_dim

    # Split training sets
    train_names = args.train
    eval_names = args.eval or list(datasets.keys())
    for name in train_names:
        if name not in datasets:
            sys.exit(f"--train {name} not in --dataset list")
        split_groups(datasets[name], args.within_holdout_frac, args.seed)
    train_dsets = [datasets[n] for n in train_names]
    if not any(ds.train_group_to_idxs for ds in train_dsets):
        sys.exit("no usable training groups (need ≥2 images per group)")

    # Model + per-dataset ArcFace heads + optimizer
    head = ProjectionHead(in_dim, hidden=args.hidden, out_dim=args.out_dim, dropout=args.dropout).to(device)
    arc_heads: dict[str, ArcFaceHead] = {}
    if args.arcface_weight > 0:
        for ds in train_dsets:
            n_cls = len(ds.train_group_to_idxs)
            if n_cls > 0:
                arc_heads[ds.name] = ArcFaceHead(
                    in_dim=args.out_dim, n_classes=n_cls,
                    margin=args.arcface_margin, scale=args.arcface_scale,
                ).to(device)
                print(f"  ArcFace[{ds.name}]: {n_cls} classes, m={args.arcface_margin}, s={args.arcface_scale}", file=sys.stderr)
    params = list(head.parameters())
    for arc in arc_heads.values():
        params += list(arc.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sampler = PKSampler(
        train_dsets, p=args.p_groups, k=args.k_images,
        batches_per_epoch=args.batches_per_epoch, seed=args.seed,
        hard_neg_frac=args.hard_neg_frac, hard_neg_pool_k=args.hard_neg_pool_k,
        use_singletons=args.use_singleton_negatives,
    )

    # Move features/views to device after the sampler builds its CPU-side
    # neighbor pools — saves a host→device copy of ~3MB of features per batch.
    for ds in datasets.values():
        ds.features = ds.features.to(device)
        if ds.views_features is not None:
            ds.views_features = ds.views_features.to(device)
            ds.views_valid_mask = ds.views_valid_mask.to(device)

    summary = {
        "args": {k: (v if not isinstance(v, list) else list(v)) for k, v in vars(args).items()},
        "datasets": {n: {"n_images": d.features.shape[0]} for n, d in datasets.items()},
        "epochs": [],
    }

    print(f"\ntraining: {train_names} → {sum(len(ds.train_group_to_idxs) for ds in train_dsets)} train groups", file=sys.stderr)
    print(f"batches/epoch={args.batches_per_epoch}  P×K={args.p_groups}×{args.k_images}={args.p_groups*args.k_images}", file=sys.stderr)

    for epoch in range(args.epochs):
        t0 = time.time()
        head.train()
        for arc in arc_heads.values():
            arc.train()
        # Accumulate on-device — pulling .item() per batch forces a CPU sync
        # that stalls the MPS queue between every step.
        ep_loss_t = torch.zeros((), device=device)
        ep_supcon_t = torch.zeros((), device=device)
        ep_arc_t = torch.zeros((), device=device)
        n_batches = 0
        # Drain the sampler upfront and bulk-transfer all idx+label tensors in
        # two host→device copies instead of one tiny copy per batch.
        epoch_batches = list(sampler)
        n_batches_ep = len(epoch_batches)
        all_idxs = np.concatenate([np.asarray(b[1], dtype=np.int64) for b in epoch_batches])
        all_labels = np.concatenate([np.asarray(b[2], dtype=np.int64) for b in epoch_batches])
        offsets = np.concatenate([[0], np.cumsum([len(b[1]) for b in epoch_batches])])
        all_idxs_t = torch.from_numpy(all_idxs).to(device, non_blocking=True)
        all_labels_t = torch.from_numpy(all_labels).to(device, non_blocking=True)
        # Bulk pre-sample per-batch randoms for the whole epoch — one Beta/rand
        # draw + transfer per augmentation instead of one per batch.
        n_group_ep = args.p_groups * args.k_images
        max_b_ep = int(max(len(b[1]) for b in epoch_batches))
        if args.mixup_alpha > 0:
            mixup_lams_bulk = (torch.distributions.Beta(args.mixup_alpha, args.mixup_alpha)
                               .sample((n_batches_ep, n_group_ep, 1))
                               .to(device, non_blocking=True))
        else:
            mixup_lams_bulk = None
        if args.cross_mixup_prob > 0:
            cross_lams_bulk = (torch.distributions.Beta(args.cross_mixup_alpha, args.cross_mixup_alpha)
                               .sample((n_batches_ep, max_b_ep))
                               .to(device, non_blocking=True))
        else:
            cross_lams_bulk = None
        if args.drop_color_prob > 0:
            drop_color_bulk = (torch.rand(n_batches_ep, max_b_ep, 1, device=device)
                               < args.drop_color_prob).float()
        else:
            drop_color_bulk = None
        # Precompute cross-mixup partner indices and do_mix masks for the whole
        # epoch. Partner selection only operates on the group prefix (first P*K
        # rows); singletons aren't passed through cross_mixup.
        if args.cross_mixup_prob > 0:
            group_labels_np = np.stack([np.asarray(b[2][:n_group_ep], dtype=np.int64)
                                        for b in epoch_batches])
            group_labels_t = torch.from_numpy(group_labels_np).to(device, non_blocking=True)
            diff_bulk = (group_labels_t.unsqueeze(1) != group_labels_t.unsqueeze(2)).float()
            has_diff_partner_bulk = diff_bulk.sum(dim=-1) > 0
            rand_partner_bulk = torch.rand(n_batches_ep, n_group_ep, n_group_ep, device=device)
            partners_bulk = torch.where(
                diff_bulk > 0, rand_partner_bulk, rand_partner_bulk.new_full((), -1.0)
            ).argmax(dim=-1)
            # Free the (n_b, n_group, n_group) intermediates immediately — only
            # partners and the has-partner mask are needed downstream.
            del diff_bulk, rand_partner_bulk
            # Fold the diff-group guard into do_mix: a row with no different-group
            # partner (degenerate batch with < P distinct groups) must not mix.
            do_mix_bulk = ((torch.rand(n_batches_ep, n_group_ep, device=device) < args.cross_mixup_prob)
                           & has_diff_partner_bulk)
            # Bulk-densify labels: per-batch group ids → [0, n_classes_b) dense
            # column indices, plus the n_classes value per batch. Done once on
            # the host then transferred as one (n_b, n_group) tensor.
            label_cols_np = np.empty((n_batches_ep, n_group_ep), dtype=np.int64)
            n_classes_per_batch = np.empty(n_batches_ep, dtype=np.int64)
            for bi, b in enumerate(epoch_batches):
                grp = b[2][:n_group_ep]
                unique = sorted(set(grp))
                gid_to_col = {g: c for c, g in enumerate(unique)}
                label_cols_np[bi] = [gid_to_col[g] for g in grp]
                n_classes_per_batch[bi] = len(unique)
            label_cols_bulk = torch.from_numpy(label_cols_np).to(device, non_blocking=True)
        else:
            partners_bulk = None
            do_mix_bulk = None
            label_cols_bulk = None
            n_classes_per_batch = None
        # Bulk-transfer ArcFace class indices for the whole epoch (one copy vs a
        # per-batch torch.tensor+to(device)). Rows for datasets without an
        # ArcFace head are never read, so they can stay uninitialized.
        if arc_heads:
            arc_labels_np = np.empty((n_batches_ep, n_group_ep), dtype=np.int64)
            for bi, (ds_b, _, labels_b) in enumerate(epoch_batches):
                if ds_b.name in arc_heads:
                    arc_labels_np[bi] = [ds_b.gid_to_cls[g] for g in labels_b[:n_group_ep]]
            arc_labels_bulk = torch.from_numpy(arc_labels_np).to(device, non_blocking=True)
        else:
            arc_labels_bulk = None
        for batch_i, (ds, idxs, labels) in enumerate(epoch_batches):
            # Singletons are appended after the P*K group rows; mixup/ArcFace
            # operate only on the group prefix, they ride along only as
            # sup_con_loss negatives.
            n_group = args.p_groups * args.k_images
            has_singletons = len(idxs) > n_group
            idx_tensor = all_idxs_t[offsets[batch_i]:offsets[batch_i+1]]
            if ds.views_features is not None:
                # Pick view in [0, K] inclusive (0 = original). Rows without valid
                # augmented data fall back to the original feature.
                k_views = ds.views_features.shape[1]
                view_choices = torch.randint(0, k_views + 1, (len(idxs),), device=device)
                use_orig = (view_choices == 0) | (~ds.views_valid_mask[idx_tensor])
                orig_feats = ds.features[idx_tensor]
                aug_view_idx = (view_choices - 1).clamp(min=0)
                aug_feats = ds.views_features[idx_tensor, aug_view_idx]
                batch_feats = torch.where(use_orig.unsqueeze(1), orig_feats, aug_feats)
            else:
                batch_feats = ds.features[idx_tensor]
            if (args.mixup_alpha > 0 or args.drop_color_prob > 0 or args.drop_peg_prob > 0
                    or args.feature_dropout > 0 or args.feature_noise > 0):
                b_now = len(idxs)
                batch_feats = apply_augmentations(
                    batch_feats, k=args.k_images,
                    mixup_alpha=args.mixup_alpha,
                    drop_color_prob=args.drop_color_prob,
                    drop_peg_prob=args.drop_peg_prob,
                    feature_dropout=args.feature_dropout,
                    feature_noise=args.feature_noise,
                    peg_dim=1280, color_dim=693,
                    mixup_end=n_group if has_singletons else None,
                    mixup_lam=mixup_lams_bulk[batch_i] if mixup_lams_bulk is not None else None,
                    drop_color_mask=drop_color_bulk[batch_i, :b_now] if drop_color_bulk is not None else None,
                )
            supcon_labels = all_labels_t[offsets[batch_i]:offsets[batch_i+1]]
            if args.cross_mixup_prob > 0:
                cross_lam_b = cross_lams_bulk[batch_i] if cross_lams_bulk is not None else None
                partner_b = partners_bulk[batch_i] if partners_bulk is not None else None
                do_mix_b = do_mix_bulk[batch_i] if do_mix_bulk is not None else None
                label_col_b = label_cols_bulk[batch_i] if label_cols_bulk is not None else None
                n_classes_b = int(n_classes_per_batch[batch_i]) if n_classes_per_batch is not None else None
                if has_singletons:
                    mixed_group, group_pos_weights = apply_cross_mixup(
                        batch_feats[:n_group], labels[:n_group],
                        alpha=args.cross_mixup_alpha,
                        prob=args.cross_mixup_prob,
                        device=device,
                        cross_lam=cross_lam_b[:n_group] if cross_lam_b is not None else None,
                        partner=partner_b,
                        do_mix=do_mix_b,
                        label_col=label_col_b,
                        n_classes=n_classes_b,
                    )
                    batch_feats = torch.cat([mixed_group, batch_feats[n_group:]], dim=0)
                    # Singletons have unique sentinel labels → zero pos overlap
                    # with every other row, so the bottom-right / off-diagonal
                    # blocks are all zeros.
                    pos_weights = torch.zeros(len(idxs), len(idxs), device=device)
                    pos_weights[:n_group, :n_group] = group_pos_weights
                else:
                    batch_feats, pos_weights = apply_cross_mixup(
                        batch_feats, labels,
                        alpha=args.cross_mixup_alpha,
                        prob=args.cross_mixup_prob,
                        device=device,
                        cross_lam=cross_lam_b[:len(idxs)] if cross_lam_b is not None else None,
                        partner=partner_b,
                        do_mix=do_mix_b,
                        label_col=label_col_b,
                        n_classes=n_classes_b,
                    )
            else:
                pos_weights = hard_label_weights(supcon_labels)
            z = head(batch_feats)
            base_sim = zeroshot_base_sim(batch_feats) if args.blend_aware else None
            l_supcon = sup_con_loss(
                z, pos_weights, temperature=args.temperature,
                base_sim=base_sim, blend_w=args.blend_weight,
                self_idx=idx_tensor if args.dedup_self_pairs else None,
            )
            if ds.name in arc_heads:
                # Singletons are not in gid_to_cls — restrict ArcFace to the
                # group prefix.
                l_arc = arc_heads[ds.name](z[:n_group], arc_labels_bulk[batch_i])
            else:
                l_arc = torch.zeros((), device=device)
            loss = l_supcon + args.arcface_weight * l_arc
            opt.zero_grad()
            loss.backward()
            # Gradient clipping guards against mid-training divergence: aggressive
            # configs (small K, high LR) can explode the grad norm and run away to
            # NaN with no containment. Only activates above the threshold, so stable
            # configs are unaffected. (Distinct from the MPS LayerNorm-backward NaN
            # workaround in ManualLayerNorm, which is a structural fix.)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip)
            opt.step()
            with torch.no_grad():
                ep_loss_t += loss.detach()
                ep_supcon_t += l_supcon.detach()
                if isinstance(l_arc, torch.Tensor):
                    ep_arc_t += l_arc.detach()
            n_batches += 1
        denom = max(1, n_batches)
        mean_loss = (ep_loss_t / denom).item()
        mean_supcon = (ep_supcon_t / denom).item()
        mean_arc = (ep_arc_t / denom).item()
        dt = time.time() - t0

        # Eval: holdout-group pair AUC for each training dataset
        ep_metrics: dict = {
            "epoch": epoch, "loss": mean_loss, "supcon": mean_supcon, "arc": mean_arc, "dt_sec": round(dt, 2),
        }
        for ds in train_dsets:
            if not ds.holdout_group_to_idxs:
                continue
            proj = project_all(head, ds, device)
            m = pairwise_acc_holdout(proj, ds)
            ep_metrics[f"holdout/{ds.name}"] = m
        summary["epochs"].append(ep_metrics)
        msg = f"epoch {epoch+1:3d}/{args.epochs}  loss={mean_loss:.4f}(sc={mean_supcon:.3f},arc={mean_arc:.3f})  ({dt:.1f}s)"
        for ds in train_dsets:
            k = f"holdout/{ds.name}"
            if k in ep_metrics:
                m = ep_metrics[k]
                msg += f"  | {ds.name} holdout: AUC={m['auc']:.4f} pairAcc={m['best_pair_acc']:.4f}"
        print(msg, file=sys.stderr)

    # Save head
    torch.save(head.state_dict(), out_dir / "proj_head.pt")

    # Project + write artifacts for every eval dataset
    print("\nwriting per-dataset projection artifacts...", file=sys.stderr)
    for name in eval_names:
        ds = datasets[name]
        proj = project_all(head, ds, device)
        np.save(out_dir / f"{name}_proj.npy", proj.numpy())
        write_dist_matrix(proj, str(out_dir / f"{name}_dist_matrix.bin"))
        with open(out_dir / f"{name}_filenames.json", "w") as f:
            json.dump(ds.filenames, f)
        if ds.train_group_to_idxs or ds.holdout_group_to_idxs:
            with open(out_dir / f"{name}_train_filenames.json", "w") as f:
                json.dump(filenames_from_groups(ds, ds.train_group_to_idxs), f)
            with open(out_dir / f"{name}_holdout_filenames.json", "w") as f:
                json.dump(filenames_from_groups(ds, ds.holdout_group_to_idxs), f)
        print(f"  [{name}] wrote proj.npy + dist_matrix.bin (N={proj.shape[0]} D={proj.shape[1]})", file=sys.stderr)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\ndone. artifacts in {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
