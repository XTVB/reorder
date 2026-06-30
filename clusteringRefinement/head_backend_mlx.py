"""MLX training backend for train_projection_head.py.

Same training semantics as the torch/MPS path, executed with MLX. The training
step is dispatch-bound in torch (the M-series GPU drains the queue faster than
Python can enqueue ~100 kernels per step); MLX's lazy graphs + mx.compile cut
the per-step cost to near the GPU floor.

Parity notes (vs the torch path):
  - The PKSampler, group splits, and all host-side RNG (python `random`) are
    shared with the torch path, so a given --seed draws the SAME batches.
  - Init distributions match torch exactly in shape: Linear weights/biases are
    U(±1/√fan_in) in both frameworks; LayerNorm starts at identity; ArcFace
    prototypes are N(0, 0.01²).
  - Beta(α,α) mixup draws happen on host via numpy (MLX has no Beta sampler);
    all other augmentation randomness is drawn in-graph from MLX's seeded RNG.
  - Device RNG streams differ from torch, so a given --seed is *statistically*
    equivalent to the torch backend, not bit-identical.
  - proj_head.pt is still written as a torch state_dict (the rest of the
    pipeline — extract_features.py — loads it with torch); conversion happens
    in train_projection_head.py after training.

Everything here mirrors the torch implementation op-for-op; see
train_projection_head.py for the canonical documentation of each feature.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

import mlx.core as mx
import mlx.nn as mnn
import mlx.optimizers as moptim


# ── Model ────────────────────────────────────────────────────────────────────


class MLXProjectionHead(mnn.Module):
    """LayerNorm → MLP → L2-normalize. Mirrors ProjectionHead (torch)."""

    def __init__(self, in_dim: int, hidden: int = 1024, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_norm = mnn.LayerNorm(in_dim)
        self.lin1 = mnn.Linear(in_dim, hidden)
        self.lin2 = mnn.Linear(hidden, hidden // 2)
        self.lin3 = mnn.Linear(hidden // 2, out_dim)
        self.drop = mnn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.drop(mnn.gelu(self.lin1(self.in_norm(x))))
        x = self.drop(mnn.gelu(self.lin2(x)))
        z = self.lin3(x)
        return z / mx.maximum(mx.linalg.norm(z, axis=-1, keepdims=True), 1e-12)


class MLXArcFace(mnn.Module):
    """Mirrors ArcFaceHead (torch): margin-augmented cosine logits + CE."""

    def __init__(self, in_dim: int, n_classes: int, margin: float = 0.3, scale: float = 30.0):
        super().__init__()
        self.weight = mx.random.normal((n_classes, in_dim)) * 0.01
        self.n_classes = n_classes
        self.scale = scale
        self._cos_m = float(np.cos(margin))
        self._sin_m = float(np.sin(margin))
        self._th = float(np.cos(np.pi - margin))
        self._mm = float(np.sin(np.pi - margin) * margin)

    def __call__(self, z: mx.array, labels: mx.array) -> mx.array:
        w = self.weight / mx.maximum(mx.linalg.norm(self.weight, axis=1, keepdims=True), 1e-12)
        cos = mx.clip(z @ w.T, -1.0 + 1e-7, 1.0 - 1e-7)
        sin = mx.sqrt(1.0 - cos * cos)
        cos_phi = cos * self._cos_m - sin * self._sin_m
        cos_phi = mx.where(cos > self._th, cos_phi, cos - self._mm)
        one_hot = (labels[:, None] == mx.arange(self.n_classes)[None, :]).astype(mx.float32)
        logits = (one_hot * cos_phi + (1.0 - one_hot) * cos) * self.scale
        return mnn.losses.cross_entropy(logits, labels, reduction="mean")


class _TrainState(mnn.Module):
    """Container so one value_and_grad covers the head + every ArcFace head."""

    def __init__(self, head: MLXProjectionHead, arc_heads: dict):
        super().__init__()
        self.head = head
        self.arc = arc_heads  # {} when ArcFace is off


# ── Per-dataset device arrays ────────────────────────────────────────────────


@dataclass
class _DeviceDS:
    feats: mx.array                 # (N, D)
    views_all: mx.array | None      # (N, K+1, D) — view 0 is the ORIGINAL row
    vmask: mx.array | None          # (N,) bool
    context: mx.array | None        # (D,)
    n: int


def _to_device(ds) -> _DeviceDS:
    feats_np = ds.features.numpy() if hasattr(ds.features, "numpy") else np.asarray(ds.features)
    views_all = vmask = None
    if ds.views_features is not None:
        # Prepend the original features as view 0 so the train step picks its
        # row with ONE gather (views_all[idx, view_sel]) instead of gathering
        # originals + augmented separately and where-selecting — same math,
        # half the gather traffic.
        views_all = mx.array(np.concatenate(
            [feats_np[:, None, :], ds.views_features.numpy()], axis=1))
        vmask = mx.array(ds.views_valid_mask.numpy())
    ctx = mx.array(ds.context.numpy()) if ds.context is not None else None
    return _DeviceDS(feats=mx.array(feats_np), views_all=views_all, vmask=vmask,
                     context=ctx, n=feats_np.shape[0])


def make_device_cache(datasets) -> dict[str, _DeviceDS]:
    """Upload every dataset's arrays once; share across folds (--lomo)."""
    return {n: _to_device(d) for n, d in datasets.items()}


# ── Training ─────────────────────────────────────────────────────────────────


def run_mlx_training(args, datasets, train_dsets, sampler, summary,
                     pairwise_acc_holdout, in_dim: int,
                     dev_cache: dict[str, _DeviceDS] | None = None):
    """Train the head with MLX. Returns (torch_style_state_dict_of_numpy, project_fn)
    where project_fn(name) -> (N, out_dim) float32 numpy array.
    dev_cache: pre-uploaded device arrays (make_device_cache) shared across
    folds in --lomo mode; None uploads fresh."""
    mx.random.seed(args.seed)
    beta_rng = np.random.default_rng(args.seed)

    dev: dict[str, _DeviceDS] = (dev_cache if dev_cache is not None
                                 else {n: _to_device(d) for n, d in datasets.items()})

    head = MLXProjectionHead(in_dim, hidden=args.hidden, out_dim=args.out_dim,
                             dropout=args.dropout)
    arc_heads: dict[str, MLXArcFace] = {}
    if args.arcface_weight > 0:
        for ds in train_dsets:
            n_cls = len(ds.train_group_to_idxs)
            if n_cls > 0:
                arc_heads[ds.name] = MLXArcFace(args.out_dim, n_cls,
                                                margin=args.arcface_margin,
                                                scale=args.arcface_scale)
                print(f"  ArcFace[{ds.name}]: {n_cls} classes, "
                      f"m={args.arcface_margin}, s={args.arcface_scale}", file=sys.stderr)
    container = _TrainState(head, arc_heads)
    mx.eval(container.parameters())

    if args.lr_schedule == "cosine":
        lr: object = moptim.cosine_decay(args.lr, args.epochs * args.batches_per_epoch)
    else:
        lr = args.lr
    opt_head = moptim.AdamW(learning_rate=lr, weight_decay=args.weight_decay,
                            bias_correction=True)
    opt_head.init(head.trainable_parameters())
    arc_opts = {}
    for name, arc in arc_heads.items():
        # One optimizer per ArcFace head, stepped only when its dataset is the
        # batch's dataset — matches torch, where inactive heads have grad=None
        # and AdamW skips them (no moment/step updates). With --lr-schedule the
        # arc decay clock ticks on its OWN step count (slower than the head's);
        # torch shares one clock — moot while every config runs ArcFace off.
        o = moptim.AdamW(learning_rate=lr, weight_decay=args.weight_decay,
                         bias_correction=True)
        o.init(arc.trainable_parameters())
        arc_opts[name] = o

    n_group = args.p_groups * args.k_images
    peg_dim = 1280 if "peg" in args.input_mods else 0
    color_dim = ((693 + (77 if args.global_color else 0))
                 if "color" in args.input_mods else 0)
    use_ctx = bool(args.shoot_context)

    state = [container.state, opt_head.state, *[o.state for o in arc_opts.values()],
             mx.random.state]

    # Same-group mixup partner: rotate within each K-block (constant per run).
    pos = np.arange(n_group)
    rot_partner = mx.array((pos // args.k_images * args.k_images
                            + (pos - pos // args.k_images * args.k_images + 1) % args.k_images
                            ).astype(np.int32))

    def loss_fn(model: _TrainState, ds_name, bf, W, neg_inf, mask_valid, arc_cls):
        z = model.head(bf)
        if args.blend_aware:
            # Mirrors zeroshot_base_sim: slices the FINAL head input at 1280.
            peg = bf[:, :1280] * 1.0
            col = bf[:, 1280:]
            col = col / mx.maximum(mx.linalg.norm(col, axis=1, keepdims=True), 1e-12) * 0.8
            base = mx.concatenate([peg, col], axis=1)
            base = base / mx.maximum(mx.linalg.norm(base, axis=1, keepdims=True), 1e-12)
            base_sim = mx.stop_gradient(base @ base.T)
            sim = (args.blend_weight * (z @ z.T)
                   + (1.0 - args.blend_weight) * base_sim) / args.temperature
        else:
            sim = z @ z.T / args.temperature
        logits = sim + neg_inf
        log_prob = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        Wm = W * mask_valid
        pos_sum = Wm.sum(axis=1)
        has_pos = (pos_sum > 1e-6).astype(mx.float32)
        mean_log_prob_pos = (Wm * log_prob).sum(axis=1) / mx.maximum(pos_sum, 1e-12)
        l_supcon = -(mean_log_prob_pos * has_pos).sum() / mx.maximum(has_pos.sum(), 1.0)
        if ds_name in model.arc:
            l_arc = model.arc[ds_name](z[:n_group], arc_cls)
        else:
            l_arc = mx.zeros(())
        return l_supcon + args.arcface_weight * l_arc, (l_supcon, l_arc)

    vg = mnn.value_and_grad(container, loss_fn)

    def build_step(ds_name: str, b: int):
        """Compile one training step for (dataset, batch-size). The dataset's
        feature/view arrays and all python scalars are baked in as constants."""
        d = dev[ds_name]
        has_singletons = b > n_group
        has_arc = ds_name in arc_heads

        def step(idx, labels, label_col, arc_cls, mix_lam, cross_lam, ctx_vec):
            # ── batch features (views pick) ─────────────────────────────────
            if d.views_all is not None:
                k_views = d.views_all.shape[1] - 1
                vc = mx.random.randint(0, k_views + 1, (b,))
                use_orig = (vc == 0) | (~d.vmask[idx])
                # view 0 of views_all IS the original row → one fused gather.
                view_sel = mx.where(use_orig, mx.zeros_like(vc), vc)
                bf = d.views_all[idx, view_sel]
            else:
                bf = d.feats[idx]

            # ── feature-space augmentations (same order as torch) ───────────
            if args.mixup_alpha > 0 and args.k_images >= 2:
                mixed = mix_lam * bf[:n_group] + (1 - mix_lam) * bf[rot_partner]
                bf = mx.concatenate([mixed, bf[n_group:]], axis=0) if has_singletons else mixed
            if args.drop_color_prob > 0 and color_dim > 0:
                m = (mx.random.uniform(shape=(b, 1)) < args.drop_color_prob).astype(mx.float32)
                bf = mx.concatenate([bf[:, :peg_dim],
                                     bf[:, peg_dim:peg_dim + color_dim] * (1 - m),
                                     bf[:, peg_dim + color_dim:]], axis=1)
            if args.drop_peg_prob > 0 and peg_dim > 0:
                m = (mx.random.uniform(shape=(b, 1)) < args.drop_peg_prob).astype(mx.float32)
                bf = mx.concatenate([bf[:, :peg_dim] * (1 - m), bf[:, peg_dim:]], axis=1)
            if args.feature_dropout > 0:
                keep = 1 - args.feature_dropout
                m = (mx.random.uniform(shape=bf.shape) < keep).astype(mx.float32) / keep
                bf = bf * m
            if args.feature_noise > 0:
                bf = bf + mx.random.normal(bf.shape) * args.feature_noise

            # ── cross-group mixup + pos-weight matrix ───────────────────────
            if args.cross_mixup_prob > 0 and args.cross_mixup_alpha > 0:
                C = args.p_groups  # padded one-hot width; zero cols are inert in L@Lᵀ
                L = (label_col[:, None] == mx.arange(C)[None, :]).astype(mx.float32)
                diff = label_col[:, None] != label_col[None, :]
                rand = mx.random.uniform(shape=(n_group, n_group))
                partner = mx.argmax(mx.where(diff, rand, -1.0), axis=1)
                do_mix = ((mx.random.uniform(shape=(n_group,)) < args.cross_mixup_prob)
                          & mx.any(diff, axis=1))
                lam = mx.where(do_mix, cross_lam, mx.ones_like(cross_lam))[:, None]
                mixed = lam * bf[:n_group] + (1 - lam) * bf[partner]
                bf = mx.concatenate([mixed, bf[n_group:]], axis=0) if has_singletons else mixed
                L = lam * L + (1 - lam) * L[partner]
                Wg = L @ L.T
                # Singletons have unique sentinel labels → zero pos overlap with
                # everything; zero-pad the group block out to (b, b).
                W = (mx.pad(Wg, ((0, b - n_group), (0, b - n_group)))
                     if has_singletons else Wg)
            else:
                W = (labels[:, None] == labels[None, :]).astype(mx.float32)

            # ── shoot context ────────────────────────────────────────────────
            if use_ctx:
                ctx = mx.broadcast_to(ctx_vec[None, :], (b, ctx_vec.shape[0]))
                if args.ctx_dropout > 0:
                    keep = (mx.random.uniform(shape=(b, 1)) >= args.ctx_dropout
                            ).astype(ctx.dtype)
                    ctx = ctx * keep
                bf = mx.concatenate([bf, ctx], axis=1)

            # ── self-pair masks ──────────────────────────────────────────────
            if args.dedup_self_pairs:
                same = idx[:, None] == idx[None, :]
                neg_inf = same.astype(mx.float32) * -1e9
                mask_valid = (~same).astype(mx.float32)
            else:
                eye = mx.eye(b)
                neg_inf = eye * -1e9
                mask_valid = 1.0 - eye

            (loss, (l_supcon, l_arc)), grads = vg(container, ds_name, bf, W,
                                                  neg_inf, mask_valid, arc_cls)
            head_grads, _ = ((grads["head"], None) if args.grad_clip <= 0
                             else moptim.clip_grad_norm(grads["head"], args.grad_clip))
            opt_head.update(container.head, head_grads)
            if has_arc:
                arc_opts[ds_name].update(container.arc[ds_name], grads["arc"][ds_name])
            return loss, l_supcon, l_arc

        return mx.compile(step, inputs=state, outputs=state)

    step_cache: dict[tuple[str, int], object] = {}

    n_train_groups = sum(len(ds.train_group_to_idxs) for ds in train_dsets)
    print(f"\ntraining (mlx): {[d.name for d in train_dsets]} → {n_train_groups} train groups",
          file=sys.stderr)
    print(f"batches/epoch={args.batches_per_epoch}  "
          f"P×K={args.p_groups}×{args.k_images}={n_group}", file=sys.stderr)

    zero_arc = mx.zeros((1,), dtype=mx.int32)

    def project_all(name: str, batch_size: int = 2048) -> np.ndarray:
        head.eval()
        d = dev[name]
        outs = []
        for i in range(0, d.n, batch_size):
            x = d.feats[i:i + batch_size]
            if use_ctx:
                x = mx.concatenate(
                    [x, mx.broadcast_to(d.context[None, :], (x.shape[0], d.context.shape[0]))],
                    axis=1)
            z = head(x)
            mx.eval(z)
            outs.append(np.array(z))
        head.train()
        return np.concatenate(outs, axis=0)

    for epoch in range(args.epochs):
        t0 = time.time()
        head.train()
        epoch_batches = list(sampler)
        nb = len(epoch_batches)
        # Host Beta draws for the whole epoch (numpy — MLX has no Beta sampler).
        mix_lams = (beta_rng.beta(args.mixup_alpha, args.mixup_alpha,
                                  (nb, n_group, 1)).astype(np.float32)
                    if args.mixup_alpha > 0 else None)
        cross_lams = (beta_rng.beta(args.cross_mixup_alpha, args.cross_mixup_alpha,
                                    (nb, n_group)).astype(np.float32)
                      if args.cross_mixup_prob > 0 else None)
        dummy_lam_mix = mx.zeros((n_group, 1))
        dummy_lam_cross = mx.zeros((n_group,))
        dummy_ctx = mx.zeros((1,))

        ep_loss = mx.zeros(())
        ep_supcon = mx.zeros(())
        ep_arc = mx.zeros(())
        for bi, (ds, idxs, labels) in enumerate(epoch_batches):
            b = len(idxs)
            key = (ds.name, b)
            fn = step_cache.get(key)
            if fn is None:
                fn = step_cache[key] = build_step(ds.name, b)
            idx_mx = mx.array(np.asarray(idxs, dtype=np.int32))
            labels_mx = mx.array(np.asarray(labels, dtype=np.int32))
            if args.cross_mixup_prob > 0:
                grp = labels[:n_group]
                gid_to_col = {g: c for c, g in enumerate(sorted(set(grp)))}
                label_col = mx.array(np.fromiter((gid_to_col[g] for g in grp),
                                                 dtype=np.int32, count=n_group))
            else:
                label_col = zero_arc
            if ds.name in arc_heads:
                arc_cls = mx.array(np.fromiter((ds.gid_to_cls[g] for g in labels[:n_group]),
                                               dtype=np.int32, count=n_group))
            else:
                arc_cls = zero_arc
            loss, l_supcon, l_arc = fn(
                idx_mx, labels_mx, label_col, arc_cls,
                mx.array(mix_lams[bi]) if mix_lams is not None else dummy_lam_mix,
                mx.array(cross_lams[bi]) if cross_lams is not None else dummy_lam_cross,
                dev[ds.name].context if use_ctx else dummy_ctx,
            )
            ep_loss = ep_loss + loss
            ep_supcon = ep_supcon + l_supcon
            ep_arc = ep_arc + l_arc
            mx.async_eval(loss, state)
        mx.eval(ep_loss, ep_supcon, ep_arc, state)
        denom = max(1, nb)
        mean_loss = ep_loss.item() / denom
        mean_supcon = ep_supcon.item() / denom
        mean_arc = ep_arc.item() / denom
        dt = time.time() - t0

        ep_metrics: dict = {"epoch": epoch, "loss": mean_loss, "supcon": mean_supcon,
                            "arc": mean_arc, "dt_sec": round(dt, 2)}
        for ds in train_dsets:
            if not ds.holdout_group_to_idxs:
                continue
            m = pairwise_acc_holdout(project_all(ds.name), ds)
            ep_metrics[f"holdout/{ds.name}"] = m
        summary["epochs"].append(ep_metrics)
        msg = (f"epoch {epoch+1:3d}/{args.epochs}  loss={mean_loss:.4f}"
               f"(sc={mean_supcon:.3f},arc={mean_arc:.3f})  ({dt:.1f}s)")
        for ds in train_dsets:
            k = f"holdout/{ds.name}"
            if k in ep_metrics:
                m = ep_metrics[k]
                msg += f"  | {ds.name} holdout: AUC={m['auc']:.4f} pairAcc={m['best_pair_acc']:.4f}"
        print(msg, file=sys.stderr)

    # torch-compatible state_dict (numpy values; caller wraps in torch tensors)
    p = head.parameters()
    sd = {
        "in_norm.weight": np.array(p["in_norm"]["weight"]),
        "in_norm.bias": np.array(p["in_norm"]["bias"]),
        "net.0.weight": np.array(p["lin1"]["weight"]),
        "net.0.bias": np.array(p["lin1"]["bias"]),
        "net.3.weight": np.array(p["lin2"]["weight"]),
        "net.3.bias": np.array(p["lin2"]["bias"]),
        "net.6.weight": np.array(p["lin3"]["weight"]),
        "net.6.bias": np.array(p["lin3"]["bias"]),
    }
    return sd, project_all
