#!/usr/bin/env python3
"""
Global (un-gridded) color histograms — the pre-3x3 extraction, resurrected.

Reproduces the original extract_features.py color features exactly as they were
before commit 1527cf4 switched to the 3x3 spatial grid: one 77-d vector
(36+16+16 HSV histogram + 9 RGB moments) over a 128x128 thumbnail of the whole
image. The 3x3 switch was motivated by spatial information but never benchmarked
against this — global_color_eval.py does that comparison zero-shot.

Writes a per-dataset sidecar  .reorder-cache/global_color_cache.npz
(hashes, color_global, version) — does NOT touch the production embeddings NPZ.
Idempotent: skips a dataset whose sidecar already covers its hashes.

  python clusteringRefinement/extract_global_color.py            # EVAL_SET (excl M7/M14/M15)
  python clusteringRefinement/extract_global_color.py --datasets M20 M11
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lomo_common import BASE, NAMES, EVAL_SET  # single-source dataset registry  # noqa: E402

THUMB = 128                  # the original (pre-3x3) thumbnail size
DIM = 36 + 16 + 16 + 9       # 77
VERSION = "hsv-rgb-77d-global-v1"


def global_color_features(img_rgb):
    """The original pre-1527cf4 extract_color_features, verbatim: 77-d over the whole thumb."""
    thumb = img_rgb.resize((THUMB, THUMB))
    arr = np.array(thumb, dtype=np.float32)
    hsv = np.array(thumb.convert("HSV"), dtype=np.float32)

    feats = []
    for ch, bins in [(0, 36), (1, 16), (2, 16)]:
        h, _ = np.histogram(hsv[:, :, ch], bins=bins, range=(0, 256))
        h = h.astype(np.float32) / (h.sum() + 1e-10)
        feats.extend(h)
    for ch in range(3):
        d = arr[:, :, ch]
        mu, sigma = d.mean(), d.std()
        feats.extend([
            mu / 256.0,
            sigma / 128.0,
            float(np.mean(((d - mu) / max(sigma, 1.0)) ** 3)) / 5.0,
        ])
    return np.array(feats, dtype=np.float32)


def _worker(args):
    image_dir, fname = args
    try:
        from PIL import Image
        return (global_color_features(Image.open(os.path.join(image_dir, fname)).convert("RGB")), None)
    except Exception as e:
        return (None, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    targets = args.datasets or EVAL_SET
    n_workers = os.cpu_count() or 4

    for tgt in targets:
        d = f"{BASE}/ClusteringBenchmark{NAMES[tgt]}"
        cache = f"{d}/.reorder-cache"
        sidecar = f"{cache}/global_color_cache.npz"
        ch = json.load(open(f"{cache}/content_hashes.json"))
        fns = sorted(ch.keys())
        want = {ch[f] for f in fns}
        if os.path.exists(sidecar) and not args.force:
            prev = np.load(sidecar, allow_pickle=False)
            if str(prev["version"]) == VERSION and want <= set(prev["hashes"].tolist()):
                print(f"{tgt}: cached ({len(fns)} imgs), skip", flush=True)
                continue

        t0 = time.time()
        feats = np.zeros((len(fns), DIM), np.float32)
        errors = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_worker, (d, f)) for f in fns]
            for i, fut in enumerate(futures):
                feat, err = fut.result()
                if err is not None:
                    print(f"  WARNING {tgt}/{fns[i]}: {err}", file=sys.stderr)
                    errors += 1
                else:
                    feats[i] = feat
        np.savez_compressed(sidecar, hashes=np.array([ch[f] for f in fns]),
                            color_global=feats, version=np.array(VERSION))
        print(f"{tgt}: done {len(fns)} imgs  errors={errors}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
