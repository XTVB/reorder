"""2D-axial RoPE buffer construction matching timm's RotaryEmbeddingCat.

Buffer layout for PE-Core-bigG (head_dim=96, grid 32x32):
    shape (1024, 192) = (num_tokens, 2 * head_dim)
    last-dim split: [sin_all (96) || cos_all (96)]
    sin_all = [sin_x (48) || sin_y (48)]
    each axial half: 24 frequencies, each pair-repeated to 48 (interleaved rotation).

grid_indexing='xy' + grid_offset=0.0 + temperature=10000 (PE-G specifics).
Tokens are row-major: token i*W + j has y=i, x=j.
"""
import numpy as np


def build_rope_buf(
    grid_h: int = 32,
    grid_w: int = 32,
    head_dim: int = 96,
    theta: float = 10000.0,
    grid_offset: float = 0.0,
):
    half = head_dim // 2  # per-axis dim (48)
    num_freqs = half // 2  # frequencies per axis (24)

    inv_freq = 1.0 / (theta ** (np.arange(0, half, 2, dtype=np.float32) / half))

    x = np.arange(grid_w, dtype=np.float32) + grid_offset
    y = np.arange(grid_h, dtype=np.float32) + grid_offset
    X, Y = np.meshgrid(x, y, indexing="xy")
    X_flat = X.flatten()  # token k = row*W + col → X = col
    Y_flat = Y.flatten()  # token k = row*W + col → Y = row

    angles_x = np.outer(X_flat, inv_freq)  # (N, 24)
    angles_y = np.outer(Y_flat, inv_freq)  # (N, 24)

    # Pair-repeat each freq for interleaved pair rotation: (N, 24) -> (N, 48)
    angles_x = np.repeat(angles_x, 2, axis=-1)
    angles_y = np.repeat(angles_y, 2, axis=-1)

    sin_x = np.sin(angles_x)
    sin_y = np.sin(angles_y)
    cos_x = np.cos(angles_x)
    cos_y = np.cos(angles_y)

    sin_all = np.concatenate([sin_x, sin_y], axis=-1)  # (N, 96)
    cos_all = np.concatenate([cos_x, cos_y], axis=-1)  # (N, 96)
    return np.concatenate([sin_all, cos_all], axis=-1).astype(np.float32)


if __name__ == "__main__":
    timm_buf = np.load("/tmp/timm_rope_buf.npy")
    my_buf = build_rope_buf()
    diff = np.abs(timm_buf - my_buf).max()
    print(f"timm shape: {timm_buf.shape}, my shape: {my_buf.shape}")
    print(f"max abs diff: {diff:.3e}")
    if diff < 1e-5:
        print("RoPE buffer match")
    else:
        print("MISMATCH — investigating")
        # Identify which slice differs
        for name, sl in [("sin_x", slice(0, 48)), ("sin_y", slice(48, 96)),
                          ("cos_x", slice(96, 144)), ("cos_y", slice(144, 192))]:
            d = np.abs(timm_buf[:, sl] - my_buf[:, sl]).max()
            print(f"  {name}: max diff {d:.3e}")
