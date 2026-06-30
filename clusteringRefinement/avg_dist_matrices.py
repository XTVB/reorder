#!/usr/bin/env python3
"""Average N condensed distance matrices (cluster-tool .bin format) into one.

Linear in the deployed ensemble's math: the runtime head concatenates per-seed
L2-normed projections scaled by 1/√n, so the ensemble cosine — and therefore
the ensemble condensed distance — is exactly the per-seed mean. Averaging the
per-seed *_dist_matrix.bin files reproduces it for LOMO scoring.

Usage: avg_dist_matrices.py OUT.bin IN1.bin IN2.bin [IN3.bin ...]
"""
import struct
import sys

import numpy as np


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    out_path, in_paths = sys.argv[1], sys.argv[2:]
    acc = None
    n0 = None
    for p in in_paths:
        with open(p, "rb") as f:
            raw = f.read()
        n = struct.unpack("<Q", raw[:8])[0]
        d = np.frombuffer(raw[8:], dtype=np.float64)
        if n0 is None:
            n0, acc = n, d.copy()
        else:
            if n != n0 or d.shape != acc.shape:
                sys.exit(f"{p}: shape mismatch (n={n} vs {n0})")
            acc += d
    acc /= len(in_paths)
    with open(out_path, "wb") as f:
        f.write(np.uint64(n0).tobytes())
        f.write(acc.tobytes())
    print(f"averaged {len(in_paths)} matrices (n={n0}) → {out_path}")


if __name__ == "__main__":
    main()
