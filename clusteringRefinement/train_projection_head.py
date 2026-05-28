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
    # Augmented views: (N, K, D) tensor. None if not loaded. K=0 if no augmented
    # views exist for this dataset. views_complete_through tracks the prefix that
    # has actual augmented data; indices >= it fall back to the original.
    views_features: torch.Tensor | None = None
    views_complete_through: int = 0


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
    views_complete = 0
    if use_augmented_views:
        peg_v_path = os.path.join(cache, "pecore_g_views.npy")
        col_v_path = os.path.join(cache, "color_views.npy")
        meta_path = os.path.join(cache, "views_meta.json")
        if all(os.path.exists(p) for p in [peg_v_path, col_v_path, meta_path]):
            with open(meta_path) as f:
                vm = json.load(f)
            views_complete = int(vm.get("completed_through", 0))
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
                print(f"  [{name}] loaded {peg_v.shape[1]} augmented views (complete through {views_complete}/{n})", file=sys.stderr)
        else:
            print(f"  [{name}] augmented views not found in {cache}", file=sys.stderr)

    return Dataset(
        name=name,
        target_dir=target_dir,
        filenames=filenames,
        features=torch.from_numpy(features),
        group_id=group_id,
        views_features=views_features,
        views_complete_through=views_complete,
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


class ProjectionHead(nn.Module):
    """LayerNorm → MLP → L2-normalize. Backbone-free; expects concat features."""

    def __init__(self, in_dim: int, hidden: int = 1024, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
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


def sup_con_loss(
    z: torch.Tensor,
    pos_weights: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Soft-label supervised contrastive loss.
    z: (B, D) L2-normalized features.
    pos_weights: (B, B) where pos_weights[i, j] ∈ [0, 1] is how much sample j
                 counts as a positive for anchor i. For hard labels this is just
                 (label[i] == label[j]).float(). For mixed samples (cross-group
                 MixUp) it's the soft-label overlap. Self-pairs are masked.
    """
    device = z.device
    b = z.shape[0]
    sim = z @ z.t() / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    mask_self = torch.eye(b, dtype=torch.bool, device=device)
    mask_valid = (~mask_self).float()

    exp_sim = torch.exp(sim) * mask_valid
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    W = pos_weights * mask_valid
    pos_weight_sum = W.sum(dim=1)
    has_pos = pos_weight_sum > 1e-6
    if not has_pos.any():
        return torch.zeros((), device=device)
    mean_log_prob_pos = (W * log_prob).sum(dim=1) / pos_weight_sum.clamp(min=1e-12)
    return -mean_log_prob_pos[has_pos].mean()


def hard_label_weights(labels: torch.Tensor) -> torch.Tensor:
    """Convert (B,) int labels to (B, B) hard pos-weight matrix."""
    return (labels.unsqueeze(0) == labels.unsqueeze(1)).float()


def apply_cross_mixup(
    feats: torch.Tensor,
    labels: list[int],
    alpha: float,
    prob: float,
    device,
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
    unique_gids = sorted(set(labels))
    gid_to_col = {g: c for c, g in enumerate(unique_gids)}
    label_col = torch.tensor([gid_to_col[g] for g in labels], dtype=torch.long, device=device)

    # Hard one-hot label table.
    L = torch.zeros(b, len(unique_gids), device=device)
    L.scatter_(1, label_col.unsqueeze(1), 1.0)

    if prob <= 0 or alpha <= 0:
        return feats.clone(), L @ L.t()

    # Which rows get mixed up?
    do_mix = torch.rand(b, device=device) < prob               # (B,)
    # Diff-group candidate matrix: (B, B), 1 if labels differ, 0 otherwise.
    diff_mat = (label_col.unsqueeze(0) != label_col.unsqueeze(1)).float()
    # Sample a partner per row from diff-group candidates. Rows where do_mix is
    # False or no diff-group partner exists get a self-pair (later masked out).
    row_sums = diff_mat.sum(dim=1, keepdim=True)
    has_partner = (row_sums.squeeze(1) > 0) & do_mix
    # multinomial requires non-zero row sums; substitute uniform for rows with no
    # partners (those rows won't be used since has_partner=False).
    safe_diff = torch.where(row_sums > 0, diff_mat, torch.ones_like(diff_mat))
    partner = torch.multinomial(safe_diff, 1).squeeze(1)       # (B,)

    # λ ~ Beta(α, α). Pin λ=1 for rows we don't mix → out_feats[i] == feats[i].
    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.sample((b,)).to(device)
    lam = torch.where(has_partner, lam, torch.ones_like(lam))
    lam_v = lam.unsqueeze(1)

    out_feats = lam_v * feats + (1 - lam_v) * feats[partner]

    # Soft label = λ * one_hot(self) + (1-λ) * one_hot(partner).
    L = lam_v * L + (1 - lam_v) * L[partner]

    return out_feats, L @ L.t()


# ── Sampler ──────────────────────────────────────────────────────────────────


# ── Augmentations (feature-space) ────────────────────────────────────────────


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
        partner = torch.arange(b, device=feats.device)
        for g_start in range(0, mix_b, k):
            block = list(range(g_start, g_start + k))
            shifted = block[1:] + block[:1]
            for i, p in zip(block, shifted):
                partner[i] = p
        beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
        lam = beta.sample((mix_b, 1)).to(feats.device)
        out[:mix_b] = lam * out[:mix_b] + (1 - lam) * feats[partner[:mix_b]]

    # Drop color: with per-image prob, zero the color slice. Closest feature-
    # space analog to background-masking — color histograms heavily encode
    # backdrop appearance, so this attacks the most plausible shortcut.
    if drop_color_prob > 0 and color_dim > 0:
        mask = (torch.rand(b, device=feats.device) < drop_color_prob).float().unsqueeze(1)
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
    ap.add_argument("--temperature", type=float, default=0.1)
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
        ep_loss = 0.0
        ep_supcon = 0.0
        ep_arc = 0.0
        n_batches = 0
        for ds, idxs, labels in sampler:
            # Singletons are appended after the P*K group rows; mixup/ArcFace
            # operate only on the group prefix, they ride along only as
            # sup_con_loss negatives.
            n_group = args.p_groups * args.k_images
            has_singletons = len(idxs) > n_group
            if ds.views_features is not None:
                # For each sample, pick view in [0, K] inclusive. 0 = original.
                # Indices past views_complete_through fall back to original.
                k_views = ds.views_features.shape[1]
                idx_tensor = torch.tensor(idxs, dtype=torch.long)
                view_choices = torch.randint(0, k_views + 1, (len(idxs),))
                use_orig = (view_choices == 0) | (idx_tensor >= ds.views_complete_through)
                orig_feats = ds.features[idx_tensor]
                aug_view_idx = (view_choices - 1).clamp(min=0)
                aug_feats = ds.views_features[idx_tensor, aug_view_idx]
                batch_feats = torch.where(use_orig.unsqueeze(1), orig_feats, aug_feats).to(device)
            else:
                batch_feats = ds.features[idxs].to(device)
            if (args.mixup_alpha > 0 or args.drop_color_prob > 0 or args.drop_peg_prob > 0
                    or args.feature_dropout > 0 or args.feature_noise > 0):
                batch_feats = apply_augmentations(
                    batch_feats, k=args.k_images,
                    mixup_alpha=args.mixup_alpha,
                    drop_color_prob=args.drop_color_prob,
                    drop_peg_prob=args.drop_peg_prob,
                    feature_dropout=args.feature_dropout,
                    feature_noise=args.feature_noise,
                    peg_dim=1280, color_dim=693,
                    mixup_end=n_group if has_singletons else None,
                )
            supcon_labels = torch.tensor(labels, dtype=torch.long, device=device)
            if args.cross_mixup_prob > 0:
                if has_singletons:
                    mixed_group, group_pos_weights = apply_cross_mixup(
                        batch_feats[:n_group], labels[:n_group],
                        alpha=args.cross_mixup_alpha,
                        prob=args.cross_mixup_prob,
                        device=device,
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
                    )
            else:
                pos_weights = hard_label_weights(supcon_labels)
            z = head(batch_feats)
            l_supcon = sup_con_loss(z, pos_weights, temperature=args.temperature)
            if ds.name in arc_heads:
                # Singletons are not in gid_to_cls — restrict ArcFace to the
                # group prefix.
                arc_labels = torch.tensor(
                    [ds.gid_to_cls[g] for g in labels[:n_group]],
                    dtype=torch.long, device=device,
                )
                l_arc = arc_heads[ds.name](z[:n_group], arc_labels)
            else:
                l_arc = torch.zeros((), device=device)
            loss = l_supcon + args.arcface_weight * l_arc
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            ep_supcon += l_supcon.item()
            ep_arc += l_arc.item() if isinstance(l_arc, torch.Tensor) else 0.0
            n_batches += 1
        mean_loss = ep_loss / max(1, n_batches)
        mean_supcon = ep_supcon / max(1, n_batches)
        mean_arc = ep_arc / max(1, n_batches)
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
