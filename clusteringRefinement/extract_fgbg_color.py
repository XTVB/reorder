#!/usr/bin/env python3
"""
Foreground/background-split color histograms (spatial color attribution probe).

Person mask via torchvision DeepLabV3 (no new dependency), then compute the SAME
693-d HSV+RGB-moment color features as scripts/extract_features.py — but separately
over person (fg) and non-person (bg) pixels. The hypothesis (LEARNED_HEAD.md
discussion): a single histogram discards which-region-is-which-color, so groups that
share a palette but distribute it differently between subject and scene are
indistinguishable to color; splitting recovers that.

Writes a per-dataset sidecar  .reorder-cache/fgbg_color_cache.npz
(hashes, color_fg, color_bg, bg_coverage, version) — does NOT touch the production
embeddings NPZ. Idempotent: skips a dataset whose sidecar already covers its hashes.

  python clusteringRefinement/extract_fgbg_color.py            # 17-set (excl M7/M14/M15)
  python clusteringRefinement/extract_fgbg_color.py --datasets M20 M11
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
from PIL import Image
import torch
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights)

sys.path.insert(0, os.path.dirname(__file__))
from lomo_common import BASE, NAMES, EVAL_SET  # single-source dataset registry  # noqa: E402

GRID, THUMB = 3, 144
CELL = THUMB // GRID                 # 48
SEG_RES = 256
PERSON = 15                          # VOC 'person' class index
MIN_PX = 8                           # a cell needs ≥ this many class pixels to emit features
VERSION = "fgbg-dlv3mnv3-hsvrgb3x3-693d-v1"


def cell_feats(arr_cell, hsv_cell, m):
    """The extract_features.py 77-d cell features, but only over pixels where m is True."""
    if int(m.sum()) < MIN_PX:
        return [0.0] * 77
    feats = []
    for ch, bins in [(0, 36), (1, 16), (2, 16)]:
        h, _ = np.histogram(hsv_cell[:, :, ch][m], bins=bins, range=(0, 256))
        h = h.astype(np.float32) / (h.sum() + 1e-10)
        feats.extend(h)
    for ch in range(3):
        d = arr_cell[:, :, ch][m]
        mu, sigma = d.mean(), d.std()
        feats.extend([mu / 256.0, sigma / 128.0,
                      float(np.mean(((d - mu) / max(sigma, 1.0)) ** 3)) / 5.0])
    return feats


def masked_color(arr, hsv, fgmask):
    """Return (fg_693, bg_693) computed over person / non-person pixels of a 144px thumb."""
    fg, bg = [], []
    for gy in range(GRID):
        for gx in range(GRID):
            y0, x0 = gy * CELL, gx * CELL
            ac, hc = arr[y0:y0+CELL, x0:x0+CELL], hsv[y0:y0+CELL, x0:x0+CELL]
            mc = fgmask[y0:y0+CELL, x0:x0+CELL]
            fg.extend(cell_feats(ac, hc, mc))
            bg.extend(cell_feats(ac, hc, ~mc))
    return np.array(fg, np.float32), np.array(bg, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    targets = args.datasets or EVAL_SET

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    assert weights.meta["categories"][PERSON] == "person"
    model = deeplabv3_mobilenet_v3_large(weights=weights).eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    print(f"device={dev}  datasets={targets}", flush=True)

    for tgt in targets:
        d = f"{BASE}/ClusteringBenchmark{NAMES[tgt]}"
        cache = f"{d}/.reorder-cache"
        sidecar = f"{cache}/fgbg_color_cache.npz"
        ch = json.load(open(f"{cache}/content_hashes.json"))
        fns = sorted(ch.keys())
        want = {ch[f] for f in fns}
        if os.path.exists(sidecar) and not args.force:
            prev = np.load(sidecar, allow_pickle=False)
            if str(prev["version"]) == VERSION and want <= set(prev["hashes"].tolist()):
                print(f"{tgt}: cached ({len(fns)} imgs), skip", flush=True)
                continue

        t0 = time.time()
        n = len(fns)
        FG = np.zeros((n, 693), np.float32)
        BG = np.zeros((n, 693), np.float32)
        COV = np.zeros(n, np.float32)
        errors = 0
        for i in range(0, n, args.batch):
            chunk = fns[i:i + args.batch]
            thumbs, hsvs, segs, ok = [], [], [], []
            for f in chunk:
                try:
                    im = Image.open(os.path.join(d, f)).convert("RGB")
                except Exception:
                    thumbs.append(None); hsvs.append(None); segs.append(np.zeros((SEG_RES, SEG_RES, 3), np.float32)); ok.append(False)
                    continue
                t144 = im.resize((THUMB, THUMB))
                thumbs.append(np.asarray(t144, np.float32))
                hsvs.append(np.asarray(t144.convert("HSV"), np.float32))
                segs.append(np.asarray(im.resize((SEG_RES, SEG_RES)), np.float32) / 255.0)
                ok.append(True)
            bt = torch.from_numpy(np.stack(segs)).permute(0, 3, 1, 2).to(dev)
            bt = (bt - mean) / std
            with torch.no_grad():
                logits = model(bt)["out"]
            person = (logits.argmax(1) == PERSON).float().unsqueeze(1)
            person = torch.nn.functional.interpolate(person, size=(THUMB, THUMB), mode="nearest")[:, 0]
            person = person.bool().cpu().numpy()
            for j, f in enumerate(chunk):
                if not ok[j]:
                    errors += 1
                    continue
                fgm = person[j]
                FG[i + j], BG[i + j] = masked_color(thumbs[j], hsvs[j], fgm)
                COV[i + j] = 1.0 - float(fgm.mean())
            if (i // args.batch) % 25 == 0:
                print(f"  {tgt}: {min(i+args.batch,n)}/{n}  ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(sidecar, hashes=np.array([ch[f] for f in fns]),
                            color_fg=FG, color_bg=BG, bg_coverage=COV,
                            version=np.array(VERSION))
        print(f"{tgt}: done {n} imgs  mean bg_cov={COV.mean():.2f}  "
              f"(<0.05 bg: {(COV<0.05).mean()*100:.0f}%)  errors={errors}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
