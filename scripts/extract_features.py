#!/usr/bin/env python3
"""Extract CLIP + color features from images, cached by content hash.

Only does feature extraction — no clustering. Outputs a manifest JSON to stdout.
Progress is reported on stderr.

Usage:
    python3 extract_features.py <image_dir> [--cache-dir <dir>] [--batch-size N]

Cache is stored in <image_dir>/.reorder-cache/ by default.
Features are keyed by content hash (blake2b of first 16KB + file size),
so renaming files does not invalidate the cache.

Output files:
    <cache_dir>/clip_embeddings.npz        - clip (N,512) + color (N,77) float32
    <cache_dir>/clip_embeddings.filenames.json - ordered list of filenames matching rows
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def content_hash(filepath: str) -> str:
    """Fast content-based hash: blake2b(first 16KB + file size)."""
    size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        head = f.read(16384)
    return hashlib.blake2b(head, digest_size=16, key=size.to_bytes(8, "big")).hexdigest()


def extract_color_features(img_rgb, thumb_size=128):
    """Extract HSV histogram + RGB color moments (77 dimensions)."""
    thumb = img_rgb.resize((thumb_size, thumb_size))
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


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP + color features")
    parser.add_argument("image_dir", help="Directory containing images")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache directory (default: <image_dir>/.reorder-cache)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    image_dir = os.path.abspath(args.image_dir)
    cache_dir = args.cache_dir or os.path.join(image_dir, ".reorder-cache")
    os.makedirs(cache_dir, exist_ok=True)

    hash_cache_path = os.path.join(cache_dir, "clip_hash_cache.npz")  # keyed by content hash
    npz_path = os.path.join(cache_dir, "clip_embeddings.npz")  # ordered by filename (for Rust)
    filenames_path = os.path.join(cache_dir, "clip_embeddings.filenames.json")

    # Find image files
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts
    )
    print(f"Found {len(image_files)} images in {image_dir}", file=sys.stderr)

    # Compute content hashes for all current files
    print("Hashing files...", file=sys.stderr)
    t0 = time.time()
    current_hashes = {}
    for f in image_files:
        current_hashes[f] = content_hash(os.path.join(image_dir, f))
    print(f"  Hashed {len(current_hashes)} files in {time.time()-t0:.1f}s", file=sys.stderr)

    # Load existing cache (keyed by content hash)
    cached_hashes = {}
    cached_clip = None
    cached_color = None
    if os.path.exists(hash_cache_path):
        try:
            data = np.load(hash_cache_path, allow_pickle=True)
            cached_hash_list = list(data["hashes"])
            cached_clip = data["clip"]
            cached_color = data["color"]
            cached_hashes = {h: i for i, h in enumerate(cached_hash_list)}
            print(f"  Loaded hash cache: {len(cached_hashes)} entries", file=sys.stderr)
        except Exception as e:
            print(f"  Hash cache corrupt, rebuilding: {e}", file=sys.stderr)

    # Find which hashes need extraction
    needed = []
    for f in image_files:
        h = current_hashes[f]
        if h not in cached_hashes:
            needed.append((f, h))

    if not needed:
        print("All features cached, nothing to extract.", file=sys.stderr)
        # Still need to write filename-ordered Rust files (filenames may have changed)
        hash_to_idx_local = cached_hashes
        ordered_clip = np.zeros((len(image_files), cached_clip.shape[1]), dtype=np.float32)
        ordered_color = np.zeros((len(image_files), cached_color.shape[1]), dtype=np.float32)
        for i, f in enumerate(image_files):
            h = current_hashes[f]
            idx = hash_to_idx_local[h]
            ordered_clip[i] = cached_clip[idx]
            ordered_color[i] = cached_color[idx]
        np.savez_compressed(npz_path, clip=ordered_clip, color=ordered_color)
        with open(filenames_path, "w") as fp:
            json.dump(image_files, fp)
        json.dump({
            "total": len(image_files),
            "cached": len(image_files),
            "extracted": 0,
            "cache_path": npz_path,
        }, sys.stdout)
        return

    print(f"Need to extract {len(needed)} new images ({len(image_files) - len(needed)} cached)",
          file=sys.stderr)

    # Setup CLIP model
    import torch
    import open_clip
    from PIL import Image

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Using device: {device}", file=sys.stderr)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()

    # Extract features for new images
    new_hashes = []
    new_clip = []
    new_color = []
    t0 = time.time()

    for batch_start in range(0, len(needed), args.batch_size):
        batch = needed[batch_start:batch_start + args.batch_size]
        batch_tensors = []
        batch_colors = []

        for fname, h in batch:
            path = os.path.join(image_dir, fname)
            try:
                img = Image.open(path).convert("RGB")
                batch_tensors.append(preprocess(img))
                batch_colors.append(extract_color_features(img))
            except Exception as e:
                print(f"  WARNING: skipping {fname}: {e}", file=sys.stderr)
                batch_tensors.append(torch.zeros(3, 224, 224))
                batch_colors.append(np.zeros(77, dtype=np.float32))

        batch_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            clip_embs = model.encode_image(batch_tensor)
            clip_embs = clip_embs / clip_embs.norm(dim=-1, keepdim=True)
        clip_np = clip_embs.cpu().numpy().astype(np.float32)

        new_clip.append(clip_np)
        new_color.append(np.array(batch_colors, dtype=np.float32))
        for _, h in batch:
            new_hashes.append(h)

        done = batch_start + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(needed) - done) / rate if rate > 0 else 0
        print(f"  {done}/{len(needed)} ({done/len(needed)*100:.0f}%) "
              f"- {rate:.1f} img/s - ETA {eta:.0f}s", file=sys.stderr)

    new_clip = np.vstack(new_clip)
    new_color = np.vstack(new_color)

    # Merge with existing cache
    if cached_clip is not None and len(cached_hashes) > 0:
        all_hash_list = list(cached_hashes.keys()) + new_hashes
        all_clip = np.vstack([cached_clip, new_clip])
        all_color = np.vstack([cached_color, new_color])
    else:
        all_hash_list = new_hashes
        all_clip = new_clip
        all_color = new_color

    # Rebuild hash→index map
    hash_to_idx = {h: i for i, h in enumerate(all_hash_list)}

    # Save hash-keyed cache (survives renames)
    np.savez_compressed(
        hash_cache_path,
        hashes=np.array(all_hash_list),
        clip=all_clip,
        color=all_color,
    )
    print(f"  Saved hash cache: {hash_cache_path}", file=sys.stderr)

    # Write filename-ordered arrays for the Rust tool
    ordered_clip = np.zeros((len(image_files), all_clip.shape[1]), dtype=np.float32)
    ordered_color = np.zeros((len(image_files), all_color.shape[1]), dtype=np.float32)
    for i, f in enumerate(image_files):
        h = current_hashes[f]
        idx = hash_to_idx[h]
        ordered_clip[i] = all_clip[idx]
        ordered_color[i] = all_color[idx]

    np.savez_compressed(npz_path, clip=ordered_clip, color=ordered_color)
    with open(filenames_path, "w") as fp:
        json.dump(image_files, fp)

    print(f"  Saved Rust embeddings: {npz_path}", file=sys.stderr)
    print(f"  Saved filenames: {filenames_path}", file=sys.stderr)

    json.dump({
        "total": len(image_files),
        "cached": len(image_files) - len(needed),
        "extracted": len(needed),
        "cache_path": npz_path,
    }, sys.stdout)


if __name__ == "__main__":
    main()
