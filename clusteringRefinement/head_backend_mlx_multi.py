"""Batched multi-seed MLX trainer: all ensemble seeds in ONE compiled graph.

Trains S projection heads simultaneously by stacking their parameters into
(S, ...) tensors and running every step as batched matmuls. The per-step FLOPs
are identical to S sequential runs, but the effective GEMM batch triples and
all host/dispatch overhead is paid once — measured faster than S concurrent
single-seed processes (which contend for the GPU anyway).

Exactness/parity with S independent single-seed runs:
  - Each lane's PKSampler and host Beta(α,α) draws use that lane's seed, so a
    lane sees EXACTLY the batches + mixup λs its standalone run would see.
  - Lane init replays mx.random.seed(seed) → MLXProjectionHead(...), so initial
    weights are bit-identical to the standalone run's.
  - Lanes never mix: parameters are blockwise, the loss is a sum of per-lane
    losses, grad-clip is per-lane, and AdamW is elementwise (bias-correction
    step count is global and identical for every lane).
  - Batches are padded to a fixed B_max so one compiled signature serves every
    step; pad columns are excluded from both the softmax denominator (-1e9) and
    the positive weights, and pad rows have no positives so they're never
    anchors — the padded loss equals the unpadded loss exactly.
  - What DOES differ from standalone runs: the in-graph augmentation/dropout
    RNG is one shared stream, and batched-kernel float reduction order — both
    statistically neutral (same class as the torch↔mlx difference, validated
    in LEARNED_HEAD.md).

Supports the deployed/LOMO feature set: --input-mods, augmented views, mixup,
drop-color/peg, feature dropout/noise, cross-mixup, hard-neg sampling,
singleton negatives, dedup-self-pairs, grad clip, cosine LR. Errors out on
ArcFace / shoot-context / blend-aware / within-holdout (none used by current
configs); single-seed paths keep full feature coverage.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import mlx.core as mx
import mlx.nn as mnn
import mlx.optimizers as moptim
from mlx.utils import tree_map

from head_backend_mlx import MLXProjectionHead

PAD_LABEL_BASE = -1_000_000_000  # below any singleton sentinel -(idx+1)


class MultiProjectionHead(mnn.Module):
    """S stacked ProjectionHeads: every parameter carries a leading (S,) dim;
    forward is batched GEMMs over (S, B, ·)."""

    def __init__(self, ln_w, ln_b, w1, b1, w2, b2, w3, b3, dropout: float):
        super().__init__()
        self.ln_w, self.ln_b = ln_w, ln_b   # (S, D)
        self.w1, self.b1 = w1, b1           # (S, hidden, D), (S, hidden)
        self.w2, self.b2 = w2, b2
        self.w3, self.b3 = w3, b3
        self.drop = mnn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:   # (S, B, D) → (S, B, out)
        x = (mx.fast.layer_norm(x, None, None, 1e-5)
             * self.ln_w[:, None, :] + self.ln_b[:, None, :])
        x = self.drop(mnn.gelu(mx.matmul(x, self.w1.transpose(0, 2, 1)) + self.b1[:, None, :]))
        x = self.drop(mnn.gelu(mx.matmul(x, self.w2.transpose(0, 2, 1)) + self.b2[:, None, :]))
        z = mx.matmul(x, self.w3.transpose(0, 2, 1)) + self.b3[:, None, :]
        return z / mx.maximum(mx.linalg.norm(z, axis=-1, keepdims=True), 1e-12)


def _stacked_init(seeds, in_dim, hidden, out_dim, dropout) -> MultiProjectionHead:
    """Replay each seed's standalone init (mx.random.seed(s) → head ctor) and
    stack — lane s starts bit-identical to a single-seed run."""
    lanes = []
    for s in seeds:
        mx.random.seed(s)
        h = MLXProjectionHead(in_dim, hidden=hidden, out_dim=out_dim, dropout=dropout)
        mx.eval(h.parameters())
        lanes.append(h.parameters())
    def stk(*path):
        out = []
        for p in lanes:
            v = p
            for k in path:
                v = v[k]
            out.append(v)
        return mx.stack(out)
    return MultiProjectionHead(
        stk("in_norm", "weight"), stk("in_norm", "bias"),
        stk("lin1", "weight"), stk("lin1", "bias"),
        stk("lin2", "weight"), stk("lin2", "bias"),
        stk("lin3", "weight"), stk("lin3", "bias"),
        dropout,
    )


def _clip_per_lane(grads, max_norm: float, S: int):
    """torch clip_grad_norm_ semantics, applied independently per lane:
    scale_s = min(1, max_norm / (||g_s|| + 1e-6))."""
    total_sq = None
    def acc(g):
        nonlocal total_sq
        sq = g.reshape(S, -1).square().sum(axis=1)
        total_sq = sq if total_sq is None else total_sq + sq
        return g
    tree_map(acc, grads)
    scale = mx.minimum(1.0, max_norm / (mx.sqrt(total_sq) + 1e-6))   # (S,)
    return tree_map(lambda g: g * scale.reshape((S,) + (1,) * (g.ndim - 1)), grads)


def run_mlx_training_multi(args, datasets, train_dsets, samplers, summary, in_dim: int):
    """Train len(args.seeds) heads at once. Returns [(seed, state_dict_np,
    project_fn), ...] in seed order."""
    seeds = args.seeds
    S = len(seeds)
    unsupported = [flag for flag, on in [
        ("--arcface-weight > 0", args.arcface_weight > 0),
        ("--shoot-context", args.shoot_context),
        ("--blend-aware", args.blend_aware),
        ("--within-holdout-frac > 0", args.within_holdout_frac > 0),
    ] if on]
    if unsupported:
        sys.exit(f"--seeds (batched multi-seed) does not support: {unsupported} — "
                 f"run per-seed single trainings instead")

    n_group = args.p_groups * args.k_images
    max_singles = max((len(ds.singleton_idxs) for ds in train_dsets), default=0)
    B = n_group + (max_singles if args.use_singleton_negatives else 0)
    peg_dim = 1280 if "peg" in args.input_mods else 0
    color_dim = ((693 + (77 if args.global_color else 0))
                 if "color" in args.input_mods else 0)
    C = args.p_groups

    # ── global feature bank: originals first, then each dataset's views flat ──
    offsets: dict[str, int] = {}
    feat_parts, cur = [], 0
    for ds in train_dsets:
        offsets[ds.name] = cur
        feat_parts.append(ds.features.numpy())
        cur += ds.features.shape[0]
    n_tot = cur
    view_base_np = np.zeros(n_tot, dtype=np.int64)
    vmask_np = np.zeros(n_tot, dtype=bool)
    k_views_by_ds: dict[str, int] = {}
    view_parts, vcur = [], n_tot
    for ds in train_dsets:
        if ds.views_features is None:
            k_views_by_ds[ds.name] = 0
            continue
        n, k = ds.views_features.shape[0], ds.views_features.shape[1]
        k_views_by_ds[ds.name] = k
        off = offsets[ds.name]
        view_parts.append(ds.views_features.numpy().reshape(n * k, -1))
        view_base_np[off:off + n] = vcur + np.arange(n, dtype=np.int64) * k
        vmask_np[off:off + n] = ds.views_valid_mask.numpy()
        vcur += n * k
    bank = mx.array(np.concatenate(feat_parts + view_parts, axis=0))
    view_base = mx.array(view_base_np)
    vmask_g = mx.array(vmask_np)
    has_views = bool(view_parts) and args.use_augmented_views
    del feat_parts, view_parts
    mx.eval(bank)

    model = _stacked_init(seeds, in_dim, args.hidden, args.out_dim, args.dropout)
    mx.random.seed(seeds[0])  # post-init stream for in-graph aug/dropout RNG
    if args.lr_schedule == "cosine":
        lr: object = moptim.cosine_decay(args.lr, args.epochs * args.batches_per_epoch)
    else:
        lr = args.lr
    opt = moptim.AdamW(learning_rate=lr, weight_decay=args.weight_decay,
                       bias_correction=True)
    opt.init(model.trainable_parameters())
    state = [model.state, opt.state, mx.random.state]
    beta_rngs = [np.random.default_rng(s) for s in seeds]

    pos = np.arange(n_group)
    rot_partner = mx.array((pos // args.k_images * args.k_images
                            + (pos % args.k_images + 1) % args.k_images).astype(np.int32))
    eye = mx.eye(B)
    lane_idx = mx.arange(S)[:, None]

    def loss_fn(m, bf, W, neg_inf, mask_valid):
        z = m(bf)
        sim = mx.matmul(z, z.transpose(0, 2, 1)) / args.temperature
        logits = sim + neg_inf
        log_prob = logits - mx.logsumexp(logits, axis=2, keepdims=True)
        Wm = W * mask_valid
        pos_sum = Wm.sum(axis=2)
        has_pos = (pos_sum > 1e-6).astype(mx.float32)
        mlp = (Wm * log_prob).sum(axis=2) / mx.maximum(pos_sum, 1e-12)
        per_lane = -(mlp * has_pos).sum(axis=1) / mx.maximum(has_pos.sum(axis=1), 1.0)
        return per_lane.sum(), per_lane

    vg = mnn.value_and_grad(model, loss_fn)

    def step(idx, labels, label_col, k_lane, valid, mix_lam, cross_lam):
        # idx (S,B) global rows; labels (S,B); label_col (S,n_group);
        # k_lane (S,) per-lane view count; valid (S,B) real-row mask.
        if has_views:
            u = mx.random.uniform(shape=(S, B))
            vc = mx.minimum(mx.floor(u * (k_lane[:, None] + 1)).astype(mx.int32),
                            k_lane[:, None])
            use_orig = (vc == 0) | (~vmask_g[idx])
            flat = mx.where(use_orig, idx, view_base[idx] + (vc - 1))
            bf = bank[flat]
        else:
            bf = bank[idx]
        if args.mixup_alpha > 0 and args.k_images >= 2:
            pref = bf[:, :n_group]
            mixed = mix_lam * pref + (1 - mix_lam) * pref[:, rot_partner]
            bf = mx.concatenate([mixed, bf[:, n_group:]], axis=1)
        if args.drop_color_prob > 0 and color_dim > 0:
            m = (mx.random.uniform(shape=(S, B, 1)) < args.drop_color_prob).astype(mx.float32)
            bf = mx.concatenate([bf[:, :, :peg_dim],
                                 bf[:, :, peg_dim:peg_dim + color_dim] * (1 - m),
                                 bf[:, :, peg_dim + color_dim:]], axis=2)
        if args.drop_peg_prob > 0 and peg_dim > 0:
            m = (mx.random.uniform(shape=(S, B, 1)) < args.drop_peg_prob).astype(mx.float32)
            bf = mx.concatenate([bf[:, :, :peg_dim] * (1 - m), bf[:, :, peg_dim:]], axis=2)
        if args.feature_dropout > 0:
            keep = 1 - args.feature_dropout
            m = (mx.random.uniform(shape=bf.shape) < keep).astype(mx.float32) / keep
            bf = bf * m
        if args.feature_noise > 0:
            bf = bf + mx.random.normal(bf.shape) * args.feature_noise

        if args.cross_mixup_prob > 0 and args.cross_mixup_alpha > 0:
            L = (label_col[:, :, None] == mx.arange(C)[None, None, :]).astype(mx.float32)
            diff = label_col[:, :, None] != label_col[:, None, :]
            rand = mx.random.uniform(shape=(S, n_group, n_group))
            partner = mx.argmax(mx.where(diff, rand, -1.0), axis=2)        # (S, n_group)
            do_mix = ((mx.random.uniform(shape=(S, n_group)) < args.cross_mixup_prob)
                      & mx.any(diff, axis=2))
            lam = mx.where(do_mix, cross_lam, mx.ones_like(cross_lam))[:, :, None]
            pref = bf[:, :n_group]
            mixed = lam * pref + (1 - lam) * pref[lane_idx, partner]
            bf = mx.concatenate([mixed, bf[:, n_group:]], axis=1)
            L = lam * L + (1 - lam) * L[lane_idx, partner]
            Wg = mx.matmul(L, L.transpose(0, 2, 1))
            W = mx.pad(Wg, ((0, 0), (0, B - n_group), (0, B - n_group)))
        else:
            W = (labels[:, :, None] == labels[:, None, :]).astype(mx.float32)

        if args.dedup_self_pairs:
            same = idx[:, :, None] == idx[:, None, :]
            base_neg = same.astype(mx.float32) * -1e9
            base_valid = (~same).astype(mx.float32)
        else:
            base_neg = eye * -1e9
            base_valid = 1.0 - eye
        vcol = valid.astype(mx.float32)[:, None, :]
        neg_inf = base_neg + (1.0 - vcol) * -1e9      # pad cols leave the softmax
        mask_valid = base_valid * vcol                # …and the positive weights

        (loss, per_lane), grads = vg(model, bf, W, neg_inf, mask_valid)
        if args.grad_clip > 0:
            grads = _clip_per_lane(grads, args.grad_clip, S)
        opt.update(model, grads)
        return loss, per_lane

    step_c = mx.compile(step, inputs=state, outputs=state)

    n_train_groups = sum(len(ds.train_group_to_idxs) for ds in train_dsets)
    print(f"\ntraining (mlx, batched seeds {seeds}): {len(train_dsets)} datasets → "
          f"{n_train_groups} train groups", file=sys.stderr)
    print(f"batches/epoch={args.batches_per_epoch}  P×K={args.p_groups}×{args.k_images}"
          f"={n_group}  B_max={B}", file=sys.stderr)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        per_seed_batches = [list(s) for s in samplers]
        nb = args.batches_per_epoch
        idx_np = np.zeros((nb, S, B), dtype=np.int32)
        lab_np = np.empty((nb, S, B), dtype=np.int32)
        lab_np[:] = (PAD_LABEL_BASE - np.arange(B, dtype=np.int32))[None, None, :]
        valid_np = np.zeros((nb, S, B), dtype=bool)
        col_np = np.zeros((nb, S, n_group), dtype=np.int32)
        k_np = np.zeros((nb, S), dtype=np.int32)
        for s, batches in enumerate(per_seed_batches):
            for i, (ds, idxs, labels) in enumerate(batches):
                b = len(idxs)
                idx_np[i, s, :b] = offsets[ds.name] + np.asarray(idxs, dtype=np.int32)
                lab_np[i, s, :b] = labels
                valid_np[i, s, :b] = True
                k_np[i, s] = k_views_by_ds[ds.name] if args.use_augmented_views else 0
                if args.cross_mixup_prob > 0:
                    grp = labels[:n_group]
                    gid_to_col = {g: c for c, g in enumerate(sorted(set(grp)))}
                    col_np[i, s] = [gid_to_col[g] for g in grp]
        # Per-lane Beta draws from that lane's own rng — identical to the λs a
        # standalone --seed s run would draw this epoch.
        if args.mixup_alpha > 0:
            mix_l = np.stack([r.beta(args.mixup_alpha, args.mixup_alpha,
                                     (nb, n_group, 1)).astype(np.float32)
                              for r in beta_rngs], axis=1)        # (nb, S, n_group, 1)
        else:
            mix_l = np.zeros((nb, S, n_group, 1), dtype=np.float32)
        if args.cross_mixup_prob > 0:
            cross_l = np.stack([r.beta(args.cross_mixup_alpha, args.cross_mixup_alpha,
                                       (nb, n_group)).astype(np.float32)
                                for r in beta_rngs], axis=1)      # (nb, S, n_group)
        else:
            cross_l = np.zeros((nb, S, n_group), dtype=np.float32)

        ep_loss = mx.zeros(())
        ep_lane = mx.zeros((S,))
        for i in range(nb):
            loss, per_lane = step_c(
                mx.array(idx_np[i]), mx.array(lab_np[i]), mx.array(col_np[i]),
                mx.array(k_np[i]), mx.array(valid_np[i]),
                mx.array(mix_l[i]), mx.array(cross_l[i]),
            )
            ep_loss = ep_loss + loss
            ep_lane = ep_lane + per_lane
            mx.async_eval(loss, state)
        mx.eval(ep_loss, ep_lane, state)
        lane_means = (ep_lane / max(1, nb)).tolist()
        dt = time.time() - t0
        summary["epochs"].append({
            "epoch": epoch,
            "loss_per_seed": {str(s): round(v, 4) for s, v in zip(seeds, lane_means)},
            "dt_sec": round(dt, 2),
        })
        lane_str = "  ".join(f"s{s}={v:.4f}" for s, v in zip(seeds, lane_means))
        print(f"epoch {epoch+1:3d}/{args.epochs}  {lane_str}  ({dt:.1f}s)", file=sys.stderr)

    # ── per-seed exports ──────────────────────────────────────────────────────
    p = model.parameters()
    feats_cache: dict[str, mx.array] = {}

    def make_lane(s_i: int):
        lane_sd = {
            "in_norm.weight": np.array(p["ln_w"][s_i]),
            "in_norm.bias": np.array(p["ln_b"][s_i]),
            "net.0.weight": np.array(p["w1"][s_i]),
            "net.0.bias": np.array(p["b1"][s_i]),
            "net.3.weight": np.array(p["w2"][s_i]),
            "net.3.bias": np.array(p["b2"][s_i]),
            "net.6.weight": np.array(p["w3"][s_i]),
            "net.6.bias": np.array(p["b3"][s_i]),
        }
        head = MLXProjectionHead(in_dim, hidden=args.hidden, out_dim=args.out_dim,
                                 dropout=args.dropout)
        head.load_weights([("in_norm.weight", p["ln_w"][s_i]),
                           ("in_norm.bias", p["ln_b"][s_i]),
                           ("lin1.weight", p["w1"][s_i]), ("lin1.bias", p["b1"][s_i]),
                           ("lin2.weight", p["w2"][s_i]), ("lin2.bias", p["b2"][s_i]),
                           ("lin3.weight", p["w3"][s_i]), ("lin3.bias", p["b3"][s_i])])
        head.eval()

        def project(name: str, batch_size: int = 2048) -> np.ndarray:
            f = feats_cache.get(name)
            if f is None:
                f = feats_cache[name] = mx.array(datasets[name].features.numpy())
            outs = []
            for i in range(0, f.shape[0], batch_size):
                z = head(f[i:i + batch_size])
                mx.eval(z)
                outs.append(np.array(z))
            return np.concatenate(outs, axis=0)

        return lane_sd, project

    return [(seed, *make_lane(i)) for i, seed in enumerate(seeds)]
