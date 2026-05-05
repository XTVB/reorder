use std::collections::HashMap;
use std::path::Path;

/// In-memory patch tokens reindexed from cache (hash) order to filename order.
///
/// Layout: row-major [n_images, n_patches, patch_dim], stride_image = n_patches * patch_dim.
/// Tokens are L2-normalized per patch (done upstream in Python extraction).
pub(crate) struct Patches {
    pub data: Vec<f32>,
    pub n_images: usize,
    pub n_patches: usize,
    pub patch_dim: usize,
    pub stride_image: usize,
    pub fname_to_idx: HashMap<String, usize>,
}

/// Load DINOv3 patch tokens from a .npy file, reindexed from cache row order
/// (the order the file was extracted in) to sorted-filename order. The cache
/// layout uses content-hash keys, so renames don't invalidate; we map from
/// hash → cache row → output row via `content_hashes.json` and
/// `patches_hashes.json`.
pub(crate) fn load_patches(
    patches_cache: &str,
    content_hashes_path: &str,
    patches_hashes_path: &str,
) -> Patches {
    // ── Load content_hashes.json → sorted filenames + hash lookup ──────
    let (content_hashes, filenames, fname_to_idx) =
        reorder_common::load_content_hashes_sorted(Path::new(content_hashes_path));
    let n_images = filenames.len();

    // ── Load patches hash order → cache row mapping ─────────────────────
    let patches_hash_order: Vec<String> = {
        let content =
            std::fs::read_to_string(patches_hashes_path).expect("read patches_hashes.json");
        serde_json::from_str(&content).expect("parse patches_hashes.json")
    };
    let patch_hash_to_row: HashMap<&str, usize> = patches_hash_order
        .iter()
        .enumerate()
        .map(|(i, h)| (h.as_str(), i))
        .collect();

    // Build filename → patches cache row mapping
    let fname_to_patch_row: Vec<usize> = filenames
        .iter()
        .map(|f| {
            let hash = content_hashes
                .get(f)
                .unwrap_or_else(|| panic!("No hash for {}", f));
            *patch_hash_to_row.get(hash.as_str()).unwrap_or_else(|| {
                panic!(
                    "Hash {} (file {}) not in patches cache — re-run extraction with --required dinov3",
                    hash, f
                )
            })
        })
        .collect();

    // ── Load DINOv3 patch tokens ─────────────────────────────────────────
    // Shape: [N_cache, N_patches, patch_dim], dtype: float32, L2-normalized per patch
    eprintln!("Loading DINOv3 patches from {}...", patches_cache);
    let npy_bytes = std::fs::read(patches_cache).expect("read patches cache file");

    // Parse .npy header
    assert!(npy_bytes.len() > 10, "npy file too small");
    assert_eq!(&npy_bytes[..6], b"\x93NUMPY", "Invalid npy magic");
    let header_len = u16::from_le_bytes([npy_bytes[8], npy_bytes[9]]) as usize;
    let data_start = 10 + header_len;
    let header_str = std::str::from_utf8(&npy_bytes[10..data_start])
        .expect("header not utf8")
        .trim();

    // Parse shape from header (e.g. "'shape': (5195, 49, 768)")
    let shape_start = header_str.find("'shape': (").expect("no shape in header") + 10;
    let shape_end = header_str[shape_start..].find(')').expect("no shape close") + shape_start;
    let shape_str = &header_str[shape_start..shape_end];
    let shape_dims: Vec<usize> = shape_str
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("bad shape dim"))
        .collect();
    assert_eq!(shape_dims.len(), 3, "Expected 3D array");
    let n_cache_entries = shape_dims[0];
    let n_patches = shape_dims[1];
    let patch_dim = shape_dims[2];
    assert!(
        n_cache_entries >= patches_hash_order.len(),
        "Patches cache has {} entries but hash order has {}",
        n_cache_entries,
        patches_hash_order.len()
    );

    // Reinterpret data bytes as f32 slice, reindex to filename order, then drop the original
    let stride_image = n_patches * patch_dim;
    let mut patches_reindexed: Vec<f32> = vec![0.0; n_images * stride_image];
    {
        let data_bytes = &npy_bytes[data_start..];
        let n_floats = data_bytes.len() / 4;
        let cache_flat: &[f32] =
            unsafe { std::slice::from_raw_parts(data_bytes.as_ptr() as *const f32, n_floats) };
        for (i, &cache_row) in fname_to_patch_row.iter().enumerate() {
            let src = &cache_flat[cache_row * stride_image..(cache_row + 1) * stride_image];
            patches_reindexed[i * stride_image..(i + 1) * stride_image].copy_from_slice(src);
        }
    }
    drop(npy_bytes); // free ~1GB original buffer now that reindexing is done

    eprintln!(
        "Loaded {} images × {} patches × {}d = {:.1} GB (reindexed from {} cache entries)",
        n_images,
        n_patches,
        patch_dim,
        (n_images * stride_image) as f64 * 4.0 / 1e9,
        n_cache_entries,
    );

    Patches {
        data: patches_reindexed,
        n_images,
        n_patches,
        patch_dim,
        stride_image,
        fname_to_idx,
    }
}
