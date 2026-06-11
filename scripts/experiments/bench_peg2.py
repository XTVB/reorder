#!/usr/bin/env python3
"""Focused follow-up: does mx.compile win once batch shapes are uniform?

Run 1 found compile slower at b8 on 30 images — but 30 % 8 = 6 forces a
mid-run retrace. Here we use exactly 24 images (3 uniform batches of 8) so
compile is traced once, plus a b4 check and a shapeless-tail padding variant.
"""
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from bench_peg import load_and_preprocess, make_model, run_inference  # noqa: E402


def main():
    import mlx.core as mx

    image_dir = sys.argv[1]
    files, batch_np, _ = load_and_preprocess(image_dir, n_threads=8)
    batch24 = batch_np[:24]  # 3 uniform batches of 8
    n = batch24.shape[0]

    ref = None
    for dtype_name in ("float32", "float16"):
        print(f"== dtype {dtype_name} (24 imgs, uniform batches) ==", flush=True)
        model, dtype = make_model(dtype_name)
        run_inference(model, dtype, batch24[:8], 8)  # warmup

        # eager baselines
        for bs in (4, 8):
            embs, dt = run_inference(model, dtype, batch24, bs)
            if ref is None:
                ref = embs
            cos = float(np.min(np.sum(embs * ref, axis=1)))
            print(f"  b={bs} eager  : {dt:6.2f}s  {n/dt:5.2f} img/s  cos_min={cos:.6f}", flush=True)

        # compiled, uniform shape only
        cfn = mx.compile(model.__call__)
        run_inference(model, dtype, batch24[:8], 8, compiled_fn=cfn)  # trace once
        for trial in range(2):
            embs, dt = run_inference(model, dtype, batch24, 8, compiled_fn=cfn)
            cos = float(np.min(np.sum(embs * ref, axis=1)))
            print(f"  b=8 compile{trial}: {dt:6.2f}s  {n/dt:5.2f} img/s  cos_min={cos:.6f}", flush=True)

        # compiled + async pipelining
        embs, dt = run_inference(model, dtype, batch24, 8, mode="async", compiled_fn=cfn)
        cos = float(np.min(np.sum(embs * ref, axis=1)))
        print(f"  b=8 comp+async: {dt:6.2f}s  {n/dt:5.2f} img/s  cos_min={cos:.6f}", flush=True)

        del model, cfn
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
