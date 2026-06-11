//! Shared NPZ embedding loading for the two Rust binaries.
//!
//! Both `cluster-tool` and `group-similarity` consume the same
//! `embeddings_hash_cache.npz` + `hash_cache_order.json` pair written by the
//! Python extraction stage, reindex rows from content-hash/cache order to
//! sorted-filename order, and blend the active models via weighted
//! concatenation. This module is the single copy of that pipeline.

use ndarray::{Array2, Ix2, OwnedRepr};
use ndarray_npy::NpzReader;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Per-model weight spec: (npz array name, weight, needs per-row L2 norm).
/// dinov3/pecore_g/learned_proj are already L2-normalized by the Python
/// extraction stage; color histograms are not.
pub fn emb_specs(
    color: f32,
    dinov3: f32,
    pecore_g: f32,
    learned_proj: f32,
) -> Vec<(&'static str, f32, bool)> {
    vec![
        ("dinov3", dinov3, false),
        ("pecore_g", pecore_g, false),
        ("color", color, true),
        ("learned_proj", learned_proj, false),
    ]
}

/// Read `hash_cache_order.json` (NPZ row order, one content hash per row) and
/// map each filename to its NPZ row via `content_hashes`. Panics on a missing
/// file or a hash absent from the cache — both mean extraction must re-run.
pub fn load_fname_to_cache_row(
    hash_order_path: &str,
    content_hashes: &HashMap<String, String>,
    filenames: &[String],
) -> Vec<usize> {
    let content = std::fs::read_to_string(hash_order_path)
        .unwrap_or_else(|_| panic!("Missing hash_cache_order.json: {}", hash_order_path));
    let hash_order: Vec<String> = serde_json::from_str(&content)
        .unwrap_or_else(|_| panic!("Invalid hash_cache_order.json: {}", hash_order_path));
    let hash_to_cache_row: HashMap<&str, usize> = hash_order
        .iter()
        .enumerate()
        .map(|(i, h)| (h.as_str(), i))
        .collect();
    filenames
        .iter()
        .map(|f| {
            let hash = content_hashes
                .get(f)
                .unwrap_or_else(|| panic!("No hash for {}", f));
            *hash_to_cache_row.get(hash.as_str()).unwrap_or_else(|| {
                panic!(
                    "Hash {} (file {}) not in embeddings cache — re-run extraction",
                    hash, f
                )
            })
        })
        .collect()
}

/// Load every positively-weighted model array from the NPZ hash cache,
/// reindexed from cache order to filename order. Models missing from the
/// cache are skipped with a warning; panics if none of the requested
/// models are present.
pub fn load_model_arrays(
    hash_cache_path: &Path,
    fname_to_cache_row: &[usize],
    specs: &[(&'static str, f32, bool)],
) -> Vec<(Array2<f32>, f32, bool)> {
    let n_images = fname_to_cache_row.len();
    eprintln!("Loading embeddings from {}...", hash_cache_path.display());
    let file = File::open(hash_cache_path).expect("open hash cache npz");
    let mut npz = NpzReader::new(file).expect("read npz");

    let mut loaded: Vec<(Array2<f32>, f32, bool)> = Vec::new();
    let mut active_desc: Vec<String> = Vec::new();
    for &(name, w, norm) in specs.iter().filter(|(_, w, _)| *w > 0.0) {
        match npz.by_name::<OwnedRepr<f32>, Ix2>(name) {
            Ok(hash_ordered) => {
                let dim = hash_ordered.ncols();
                let mut arr = Array2::<f32>::zeros((n_images, dim));
                for (i, &cache_row) in fname_to_cache_row.iter().enumerate() {
                    arr.row_mut(i).assign(&hash_ordered.row(cache_row));
                }
                active_desc.push(format!("{}={}d×{}", name, dim, w));
                loaded.push((arr, w, norm));
            }
            Err(_) => {
                eprintln!(
                    "WARNING: '{}' array not found in hash cache (weight={:.2}), skipping",
                    name, w
                );
            }
        }
    }
    assert!(
        !loaded.is_empty(),
        "no requested model arrays found in {}",
        hash_cache_path.display()
    );
    eprintln!(
        "Loaded {} images, active: {}",
        n_images,
        active_desc.join(", ")
    );
    loaded
}

/// Weighted concatenation of the per-model rows into one flat row-major
/// `[n, dim]` buffer. Pass `l2_normalize_rows = true` when downstream code
/// computes cosine similarity as a plain dot product between rows.
pub fn build_combined_features_flat(
    arrays: &[(Array2<f32>, f32, bool)],
    n: usize,
    l2_normalize_rows: bool,
) -> (Vec<f32>, usize) {
    let slices: Vec<(&[f32], usize, f32, bool)> = arrays
        .iter()
        .map(|(a, w, norm)| {
            let data = a.as_slice().expect("array must be contiguous");
            (data, a.ncols(), *w, *norm)
        })
        .collect();
    let combined_dim: usize = slices.iter().map(|(_, d, _, _)| *d).sum();
    let mut features = vec![0.0f32; n * combined_dim];

    for i in 0..n {
        let out = &mut features[i * combined_dim..][..combined_dim];
        let mut offset = 0;

        for &(data, dim, weight, needs_norm) in &slices {
            let row = &data[i * dim..][..dim];
            if needs_norm {
                let norm_sq: f32 = row.iter().map(|&x| x * x).sum();
                let norm = norm_sq.sqrt().max(1e-10);
                for (o, &v) in out[offset..offset + dim].iter_mut().zip(row) {
                    *o = (v / norm) * weight;
                }
            } else {
                for (o, &v) in out[offset..offset + dim].iter_mut().zip(row) {
                    *o = v * weight;
                }
            }
            offset += dim;
        }

        if l2_normalize_rows {
            let norm_sq: f32 = out.iter().map(|&x| x * x).sum();
            let norm = norm_sq.sqrt().max(1e-10);
            for o in out.iter_mut() {
                *o /= norm;
            }
        }
    }

    (features, combined_dim)
}
