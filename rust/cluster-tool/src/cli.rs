use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Hierarchical agglomerative clustering with pre-seeded groups")]
pub(crate) struct Cli {
    /// Path to hash-keyed cache .npz (embeddings_hash_cache.npz)
    #[arg(long)]
    pub hash_cache: PathBuf,

    /// Linkage method: ward | average | complete. Default: average.
    /// Average linkage works best with the re-ranking distance matrix; ward is
    /// the historical default for raw cosine.
    #[arg(long, default_value = "average")]
    pub linkage: String,

    /// Path to content_hashes.json (filename → content hash)
    #[arg(long)]
    pub content_hashes: String,

    /// Path to hash_cache_order.json (hash list in NPZ row order)
    #[arg(long)]
    pub hash_order: String,

    /// Path to .reorder-groups.json
    #[arg(long, default_value = "")]
    pub groups: String,

    /// Number of clusters to produce
    #[arg(long, default_value_t = 200)]
    pub n_clusters: usize,

    /// Output path for linkage tree binary
    #[arg(long, default_value = "")]
    pub output_tree: String,

    /// Color feature weight
    #[arg(long, default_value_t = 0.0)]
    pub color_weight: f32,

    /// DINOv3 CLS token weight
    #[arg(long, default_value_t = 0.0)]
    pub dinov3_weight: f32,

    /// PE-Core-bigG feature weight
    #[arg(long, default_value_t = 0.0)]
    pub pecore_g_weight: f32,

    /// Learned-projection-head feature weight. The features are pre-computed at
    /// extraction time by pushing PE-G + color through the trained head and
    /// stored as `learned_proj` in the hash cache NPZ. Already L2-normalized.
    #[arg(long, default_value_t = 0.0)]
    pub learned_proj_weight: f32,

    /// Path to precomputed condensed distance matrix binary
    #[arg(long, default_value = "")]
    pub dist_matrix: String,

    /// Weight for the precomputed distance matrix when blending with embedding distances.
    /// 1.0 = patches only, 0.0 = embeddings only, 0.5 = equal blend.
    #[arg(long, default_value_t = 1.0)]
    pub dist_matrix_weight: f32,

    /// Optional path to a JSON array of filenames. When provided, clustering is
    /// restricted to this subset (applied before group loading and embedding load).
    /// Incompatible with --dist-matrix for now (matrix is indexed on the full set).
    #[arg(long, default_value = "")]
    pub filenames: String,

    /// Path to JSON `[{ "image_filename": "...", "group_id": "..." }, ...]`
    /// of image↔group cannot-link constraints. Each pair becomes a 1e18
    /// distance sentinel between the image and the group's representative,
    /// preventing the image from ever joining that group during NNC.
    #[arg(long, default_value = "")]
    pub cannot_link: String,

    /// Path to JSON `[{ "group_id": "..." }, ...]` of locked groups.
    /// For each locked group G, sets dist(i, G_rep) = 1e18 for every active
    /// index i ≠ G_rep, fully isolating G from further merges. Locked groups'
    /// clusters reproduce exactly at any tree cut.
    #[arg(long, default_value = "")]
    pub locked_groups: String,
}
