"""Convert PE-Core-bigG weights from open_clip (timm-format) state_dict to MLX safetensors.

Steps:
  1. Load PE-Core-bigG-14-448 via open_clip (with pretrained='meta'). open_clip applies
     timm's _convert_pe() to remap Meta's checkpoint keys to timm's conventions.
  2. Strip the 'visual.trunk.' prefix added by open_clip's TimmModel wrapper.
  3. Transpose patch_embed Conv2d weight: PyTorch (O,I,kH,kW) -> MLX (O,kH,kW,I).
  4. Save as a single safetensors file.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


def convert(out_path: str, dtype: str = "float32"):
    import open_clip

    log = lambda msg: print(msg, file=sys.stderr, flush=True)

    log("Loading open_clip PE-Core-bigG (downloads/uses meta weights)...")
    model, _, _ = open_clip.create_model_and_transforms(
        "PE-Core-bigG-14-448", pretrained="meta"
    )
    sd = model.visual.trunk.state_dict()
    log(f"  trunk state_dict: {len(sd)} tensors")

    np_dtype = getattr(np, dtype)
    out: dict[str, np.ndarray] = {}

    for k, v in sd.items():
        # Skip non-persistent buffers (rope.pos_embed) — we rebuild from constants.
        if k.startswith("rope."):
            continue

        arr = v.detach().cpu().numpy()

        if k == "patch_embed.proj.weight":
            # PyTorch (O, I, kH, kW) -> MLX (O, kH, kW, I)
            arr = arr.transpose(0, 2, 3, 1)
            log(f"  transposed patch_embed.proj.weight -> {arr.shape}")

        out[k] = arr.astype(np_dtype)

    log(f"Writing {len(out)} tensors to {out_path}")
    # safetensors prefers contiguous arrays
    out = {k: np.ascontiguousarray(v) for k, v in out.items()}
    from safetensors.numpy import save_file
    save_file(out, out_path)
    log(f"Saved {sum(v.nbytes for v in out.values()) / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/mlx-pe-core-bigg.safetensors"))
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = ap.parse_args()
    convert(args.out, args.dtype)


if __name__ == "__main__":
    main()
