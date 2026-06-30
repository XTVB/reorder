use std::collections::HashMap;
use std::path::Path;

use reorder_common::embeddings::{
    build_combined_features_flat, emb_specs, load_fname_to_cache_row, load_model_arrays,
};

/// Combined per-image feature vectors, reindexed from NPZ (hash) order to
/// sorted-filename order. Each row is the weighted concatenation of the active
/// per-model embeddings, then L2-normalized — so a dot product between two rows
/// equals their cosine similarity (matching cluster-tool's concat-then-cosine).
///
/// Layout: row-major [n_images, dim], stride_image = dim.
pub(crate) struct Embeddings {
    pub data: Vec<f32>,
    pub dim: usize,
    pub fname_to_idx: HashMap<String, usize>,
}

/// Load the per-model NPZ embeddings, blend the models with positive weight into
/// one combined vector per image (weighted concatenation), and L2-normalize each
/// row. Reindexes from content-hash/cache order to sorted-filename order via
/// `content_hashes.json` + `hash_cache_order.json` so renames don't invalidate.
pub(crate) fn load_combined_embeddings(
    hash_cache_path: &str,
    content_hashes_path: &str,
    hash_order_path: &str,
    color: f32,
    dinov3: f32,
    pecore_g: f32,
    learned_proj: f32,
    learned_proj_peg: f32,
    learned_proj_color: f32,
) -> Embeddings {
    assert!(
        !hash_cache_path.is_empty(),
        "--hash-cache is required for embeddings mode"
    );
    assert!(
        !hash_order_path.is_empty(),
        "--hash-order is required for embeddings mode"
    );

    let (content_hashes, filenames, fname_to_idx) =
        reorder_common::load_content_hashes_sorted(Path::new(content_hashes_path));
    let fname_to_cache_row =
        load_fname_to_cache_row(hash_order_path, &content_hashes, &filenames);

    let specs = emb_specs(
        color,
        dinov3,
        pecore_g,
        learned_proj,
        learned_proj_peg,
        learned_proj_color,
    );
    assert!(
        specs.iter().any(|(_, w, _)| *w > 0.0),
        "embeddings mode requires at least one positive model weight"
    );
    let loaded = load_model_arrays(Path::new(hash_cache_path), &fname_to_cache_row, &specs);
    let (data, dim) = build_combined_features_flat(&loaded, filenames.len(), true);
    eprintln!("Combined feature dim: {} (L2-normalized rows)", dim);

    Embeddings {
        data,
        dim,
        fname_to_idx,
    }
}
