#!/usr/bin/env python3
"""Extract intermediate PE-Core-bigG layer features for the learned-head
intermediate-layer experiment.

For each candidate transformer block, pool its token output three ways
(mean / gem3 / attnpool) into vectors and store them row-aligned to
embeddings_hash_cache.npz['hashes'] (so they line up with the cached
pecore_g / color for the head to consume directly). Capturing several poolings in
the one forward pass is nearly free, so the post-extraction search can cover
pooling x layer x blend rather than pre-committing to one pooling.

Only images that belong to a group in .reorder-groups.json are encoded — the
head trains exclusively on grouped images (ungrouped ones are dropped by
train_projection_head.py), so ungrouped rows stay zero-filled in the output
arrays. Consumers already warn/skip on zero rows.

Backends:
  mlx (default)  — uses the native MLX PE-Core port (PECoreBigG.forward_capture),
                   ~2 img/s, fp32. Same numerical regime as the deployed pecore_g
                   extraction, and the path we'd ship if this lands.
  pytorch        — open_clip + forward hooks, ~1.7 img/s, fp16. Fallback/debug.
                   (mean agrees with mlx to ~0.99 cosine; attnpool is fp16-sensitive.)

Resumable + checkpointed like pixel-aug: writes every --checkpoint-every images
and skips already-done rows on re-run (tracked in pe_layers_meta.json).

Output (in <cache>):
  pe_layers_L<NN>_<pool>.npy   one (N, D) array per (layer, pooling), npz-hash order
  pe_layers_meta.json          {version, n_images, layers, poolings, backend, completed_through}
"""
import argparse, json, os, sys, time
import numpy as np

VERSION = "pe-layers-v2"
CHECKPOINT_EVERY = 200
POOLINGS = ["mean", "gem3", "attnpool"]
HW = 448
MLX_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "scripts", "mlx_pe_core"))
# CLIP/PE preprocessing constants (shared by both backends).
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


def _torchvision_preprocess():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(HW, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(HW),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def build_encoder_mlx(layers):
    """encode(pil_list) -> {(L, pool): (B, D) np.float32}, depth. MLX fp32."""
    import mlx.core as mx
    import torch
    if MLX_DIR not in sys.path:
        sys.path.insert(0, MLX_DIR)
    from model import PECoreBigG
    weights = os.path.expanduser("~/.cache/mlx-pe-core-bigg.safetensors")
    if not os.path.exists(weights):
        sys.exit(f"MLX weights not found at {weights}; run extract_features.py once "
                 f"(or scripts/mlx_pe_core/convert.py) to create them, or use --backend pytorch")
    mdl = PECoreBigG()
    mdl.load_weights(weights, strict=False)
    mdl.set_dtype(mx.float32)
    mdl.eval()
    mx.eval(mdl.parameters())
    depth = len(mdl.blocks)
    preprocess = _torchvision_preprocess()

    def encode(pils):
        batch = torch.stack([preprocess(im) for im in pils]).numpy().transpose(0, 2, 3, 1)
        out = mdl.forward_capture(mx.array(batch).astype(mx.float32), layers, tuple(POOLINGS))
        mx.eval(list(out.values()))
        return {k: np.array(v).astype(np.float32) for k, v in out.items()}

    return encode, depth


def build_encoder_pytorch(layers):
    """encode(pil_list) -> {(L, pool): (B, D) np.float32}, depth. open_clip fp16 + hooks."""
    import copy, torch, open_clip
    dev = torch.device("mps")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "PE-Core-bigG-14-448", pretrained="meta", device=dev)
    model.eval(); model = model.half()
    trunk = getattr(model.visual, "trunk", None)
    ap = getattr(trunk, "attn_pool", None)
    fn = getattr(trunk, "norm", None)
    ap_f = copy.deepcopy(ap).float().to(dev).eval() if ap is not None else None
    fn_f = copy.deepcopy(fn).float().to(dev).eval() if fn is not None else None
    for mod in (ap_f, fn_f):
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad_(False)
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 12 and (best is None or len(mod) > len(best[1])):
            best = (name, mod)
    blocks = best[1]; depth = len(blocks)
    cap = {}
    def mk(i):
        def hook(m, inp, out):
            cap[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        return hook
    for L in layers:
        blocks[L].register_forward_hook(mk(L))

    def to_BND(t, B):
        t = t.float()
        if t.dim() != 3:
            return None
        if t.shape[0] == B and t.shape[1] != B:
            return t
        if t.shape[1] == B:
            return t.transpose(0, 1)
        return t

    def encode(pils):
        tens = torch.stack([preprocess(im) for im in pils]).to(dev).half()
        out = {}
        with torch.no_grad():
            model.encode_image(tens)
            B = tens.shape[0]
            for L in layers:
                x = to_BND(cap[L], B)
                out[(L, "mean")] = x.mean(1).float().cpu().numpy()
                out[(L, "gem3")] = (x.clamp(min=1e-6) ** 3).mean(1).pow(1.0 / 3.0).float().cpu().numpy()
                if ap_f is not None:
                    a = ap_f(fn_f(x) if fn_f is not None else x)
                    out[(L, "attnpool")] = (a.mean(1) if a.dim() == 3 else a).float().cpu().numpy()
        return out

    return encode, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target_dir")
    ap.add_argument("--layers", default="42,44,46,47", help="comma-separated block indices")
    ap.add_argument("--backend", choices=["mlx", "pytorch"], default="mlx")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--all-images", action="store_true",
                    help="extract EVERY image, not just grouped ones. Required for "
                         "production/inference use (a fresh folder has no groups, and "
                         "the head only uses the layer when it covers all images); the "
                         "default grouped-only mode is for training-set extraction.")
    args = ap.parse_args()
    from PIL import Image

    cache = args.cache_dir or os.path.join(args.target_dir, ".reorder-cache")
    npz_path = os.path.join(cache, "embeddings_hash_cache.npz")
    ch_path = os.path.join(cache, "content_hashes.json")
    if not os.path.exists(npz_path):
        sys.exit(f"no embeddings_hash_cache.npz at {npz_path}; run extract_features.py first")
    if not os.path.exists(ch_path):
        sys.exit(f"no content_hashes.json at {ch_path}")
    hashes = [str(h) for h in np.load(npz_path, allow_pickle=False)["hashes"]]
    n = len(hashes)
    layers = [int(x) for x in args.layers.split(",")]
    ch = json.load(open(ch_path))                       # {filename: hash}
    hash2fn = {}
    for fn, h in ch.items():
        hash2fn.setdefault(str(h), fn)
    paths = [os.path.join(args.target_dir, hash2fn[h]) if h in hash2fn else None for h in hashes]
    missing = sum(p is None for p in paths)

    # Default: only grouped images feed the head (training drops ungrouped
    # entirely), so only their rows get encoded; the rest stay zero. --all-images
    # encodes everything — needed for inference, where the head uses the layer
    # only when every image is covered (see extract_features._load_full_pe_layer).
    if args.all_images:
        wanted = [p is not None for p in paths]
    else:
        groups_path = os.path.join(args.target_dir, ".reorder-groups.json")
        if not os.path.exists(groups_path):
            print(f"  {os.path.basename(args.target_dir)}: no .reorder-groups.json — "
                  f"nothing to extract (grouped-only mode; pass --all-images for inference)",
                  file=sys.stderr)
            return
        graw = json.load(open(groups_path))
        glist = graw if isinstance(graw, list) else graw.get("groups", [])
        grouped_hashes = {str(ch[fn]) for g in glist for fn in g["images"] if fn in ch}
        wanted = [h in grouped_hashes for h in hashes]

    meta_path = os.path.join(cache, "pe_layers_meta.json")
    keys = [(L, p) for L in layers for p in POOLINGS]
    arr_paths = {(L, p): os.path.join(cache, f"pe_layers_L{L:02d}_{p}.npy") for (L, p) in keys}
    meta = None
    if os.path.exists(meta_path):
        m = json.load(open(meta_path))
        # A prior grouped-only run must NOT satisfy an --all-images request (its
        # ungrouped rows are still zero) — treat the coverage change as a mismatch.
        prior_all = m.get("all_images", not m.get("grouped_only", True))
        if (m.get("version") == VERSION and m.get("n_images") == n
                and m.get("layers") == layers and m.get("poolings") == POOLINGS
                and prior_all == bool(args.all_images)):
            meta = m
        else:
            print("  pe_layers_meta.json mismatch — starting fresh", file=sys.stderr)
    if meta is None:
        meta = {"version": VERSION, "n_images": n, "layers": layers,
                "poolings": POOLINGS, "backend": args.backend,
                "grouped_only": not args.all_images, "all_images": bool(args.all_images),
                "completed_through": 0}

    start = meta["completed_through"]
    arrays = {}
    for k in keys:
        if start > 0 and os.path.exists(arr_paths[k]):
            a = np.load(arr_paths[k]); arrays[k] = a if a.shape[0] == n else None
        else:
            arrays[k] = None
    if any(arrays[k] is None for k in keys):
        start = 0; meta["completed_through"] = 0
        arrays = {k: None for k in keys}
    if start >= n:
        print(f"  {os.path.basename(args.target_dir)}: already complete ({n} imgs)", file=sys.stderr)
        return

    name = os.path.basename(args.target_dir)
    # completed_through is a row-index watermark over the full npz order: every
    # wanted (grouped) row below it is encoded. Processing `todo` in ascending
    # index order preserves that invariant across pause/resume.
    todo = [j for j in range(start, n) if paths[j] is not None and wanted[j]]
    n_grouped = sum(wanted)
    if not todo:
        if all(arrays[k] is None for k in keys):
            print(f"  {name}: no grouped images to extract — skipping", file=sys.stderr)
            return
        meta["completed_through"] = n
        json.dump(meta, open(meta_path, "w"))
        print(f"  {name}: done (no grouped rows left past resume point)", file=sys.stderr, flush=True)
        return
    print(f"  {name}: {len(todo)} of {n_grouped} grouped imgs to encode "
          f"({n} rows total, {missing} missing), layers {layers} x {POOLINGS}, "
          f"backend={args.backend}, resume@{start}", file=sys.stderr, flush=True)
    encode, depth = (build_encoder_mlx(layers) if args.backend == "mlx"
                     else build_encoder_pytorch(layers))
    if any(L >= depth for L in layers):
        sys.exit(f"layer index >= depth {depth}")

    bs = args.batch_size
    done_imgs = 0; since = 0; t0 = time.time()
    for s in range(0, len(todo), bs):
        idxs = todo[s:s + bs]
        imgs, valid = [], []
        for j in idxs:
            try:
                imgs.append(Image.open(paths[j]).convert("RGB")); valid.append(j)
            except Exception as e:
                print(f"    skip {paths[j]}: {e}", file=sys.stderr)
        if imgs:
            reps = encode(imgs)
            for (L, p), arr in reps.items():
                if arrays[(L, p)] is None:
                    arrays[(L, p)] = np.zeros((n, arr.shape[1]), dtype=np.float32)
                for kk, j in enumerate(valid):
                    arrays[(L, p)][j] = arr[kk]
            done_imgs += len(valid); since += len(valid)
        last = s + bs >= len(todo)
        if since >= args.checkpoint_every or last:
            for k in keys:
                if arrays[k] is not None:
                    np.save(arr_paths[k], arrays[k])
            meta["completed_through"] = n if last else idxs[-1] + 1
            json.dump(meta, open(meta_path, "w"))
            since = 0
            print(f"    {name} {done_imgs}/{len(todo)} grouped imgs (row {idxs[-1] + 1}/{n}, "
                  f"{done_imgs/max(time.time()-t0,1e-9):.1f} img/s)", file=sys.stderr, flush=True)
    print(f"  {name}: done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
