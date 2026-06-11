#!/usr/bin/env python3
"""Benchmark harness for PE-Core-G MLX extraction.

Phases measured independently:
  1. JPEG decode + torchvision preprocess (single / multi thread)
  2. Pure GPU inference: sweep dtype x batch x {eager, compile, async-pipelined}
  3. Correctness: cosine sim of every config vs the fp32 batch-8 reference

Usage:
    python bench_peg.py <image_dir> [--quick]
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "mlx_pe_core"))

WEIGHTS = os.path.expanduser("~/.cache/mlx-pe-core-bigg.safetensors")
HW = 448


def load_and_preprocess(image_dir, n_threads=1):
    """Replicates extract_open_clip's MLX preprocess exactly. Returns NHWC fp32."""
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from PIL import Image
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(HW, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(HW),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    def _one(f):
        img = Image.open(os.path.join(image_dir, f)).convert("RGB")
        return preprocess(img)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        tensors = list(pool.map(_one, files))
    dt = time.time() - t0
    batch = torch.stack(tensors).numpy().transpose(0, 2, 3, 1).copy()
    return files, batch, dt


def make_model(dtype_name):
    import mlx.core as mx
    from model import PECoreBigG

    dtype = getattr(mx, dtype_name)
    m = PECoreBigG()
    m.load_weights(WEIGHTS, strict=False)
    m.set_dtype(dtype)
    m.eval()
    mx.eval(m.parameters())
    return m, dtype


def run_inference(model, dtype, batch_np, batch_size, mode="eager", compiled_fn=None):
    """One full pass over batch_np. Returns (embs fp32 normalized, wall_seconds)."""
    import mlx.core as mx

    n = batch_np.shape[0]
    fn = compiled_fn if compiled_fn is not None else model
    outs = []
    t0 = time.time()
    if mode == "async":
        pending = []
        for s in range(0, n, batch_size):
            xb = mx.array(batch_np[s:s + batch_size]).astype(dtype)
            out = fn(xb)
            mx.async_eval(out)
            pending.append(out)
        mx.eval(pending)
        outs = pending
    else:
        for s in range(0, n, batch_size):
            xb = mx.array(batch_np[s:s + batch_size]).astype(dtype)
            out = fn(xb)
            mx.eval(out)
            outs.append(out)
    dt = time.time() - t0
    embs = np.concatenate([np.array(o, copy=False).astype(np.float32) for o in outs])
    embs /= np.maximum(np.linalg.norm(embs, axis=-1, keepdims=True), 1e-12)
    return embs, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    import mlx.core as mx

    results = {}

    # ── Phase 1: preprocess timing ────────────────────────────────────────
    print("== Phase 1: decode + preprocess ==", flush=True)
    files, batch_np, _ = load_and_preprocess(args.image_dir, n_threads=1)  # warm FS cache
    for nt in (1, 4, 8):
        _, b, dt = load_and_preprocess(args.image_dir, n_threads=nt)
        print(f"  threads={nt}: {dt:.2f}s ({len(files)/dt:.1f} img/s)", flush=True)
        results[f"preprocess_t{nt}"] = dt
    n = batch_np.shape[0]
    print(f"  {n} images, batch tensor {batch_np.shape}", flush=True)

    # ── Phase 2+3: inference sweep ────────────────────────────────────────
    ref = None
    dtypes = ["float32", "bfloat16", "float16"]
    batches = [8, 16] if args.quick else [8, 16, 32]

    for dtype_name in dtypes:
        print(f"== dtype {dtype_name} ==", flush=True)
        t0 = time.time()
        model, dtype = make_model(dtype_name)
        print(f"  model load+cast: {time.time()-t0:.1f}s", flush=True)

        # Warmup (Metal kernel compile)
        run_inference(model, dtype, batch_np[:8], 8)

        for bs in batches:
            for mode in ("eager", "async"):
                embs, dt = run_inference(model, dtype, batch_np, bs, mode=mode)
                key = f"{dtype_name}_b{bs}_{mode}"
                if ref is None:
                    ref = embs
                cos = float(np.mean(np.sum(embs * ref, axis=1)))
                cos_min = float(np.min(np.sum(embs * ref, axis=1)))
                results[key] = {"sec": dt, "img_s": n / dt, "cos_mean": cos, "cos_min": cos_min}
                print(f"  b={bs:2d} {mode:5s}: {dt:6.2f}s  {n/dt:5.2f} img/s  "
                      f"cos mean={cos:.6f} min={cos_min:.6f}", flush=True)

        # mx.compile at the best batch size (eager-equivalent semantics)
        if not args.quick or dtype_name != "bfloat16":
            cfn = mx.compile(model.__call__)
            run_inference(model, dtype, batch_np[:8], 8, compiled_fn=cfn)  # warmup/trace
            embs, dt = run_inference(model, dtype, batch_np, 8, compiled_fn=cfn)
            cos = float(np.mean(np.sum(embs * ref, axis=1)))
            results[f"{dtype_name}_b8_compile"] = {"sec": dt, "img_s": n / dt, "cos_mean": cos}
            print(f"  b= 8 compile: {dt:6.2f}s  {n/dt:5.2f} img/s  cos mean={cos:.6f}", flush=True)

        del model
        gc.collect()
        mx.clear_cache()

    # ── 8-bit quantization (fp16 base) ────────────────────────────────────
    print("== quantized (8-bit, fp16) ==", flush=True)
    import mlx.nn as nn
    model, dtype = make_model("float16")
    nn.quantize(model, group_size=64, bits=8,
                class_predicate=lambda p, m: isinstance(m, nn.Linear))
    mx.eval(model.parameters())
    run_inference(model, dtype, batch_np[:8], 8)
    for bs in batches:
        embs, dt = run_inference(model, dtype, batch_np, bs, mode="async")
        cos = float(np.mean(np.sum(embs * ref, axis=1)))
        cos_min = float(np.min(np.sum(embs * ref, axis=1)))
        results[f"q8_b{bs}_async"] = {"sec": dt, "img_s": n / dt, "cos_mean": cos, "cos_min": cos_min}
        print(f"  b={bs:2d} async: {dt:6.2f}s  {n/dt:5.2f} img/s  "
              f"cos mean={cos:.6f} min={cos_min:.6f}", flush=True)
    del model
    gc.collect()
    mx.clear_cache()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
