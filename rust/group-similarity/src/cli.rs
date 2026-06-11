use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum Mode {
    /// Pairwise group scoring for the Merge Suggestions UI (DINOv3 patches).
    #[value(name = "merge-suggestions")]
    MergeSuggestions,
    /// Full pairwise condensed distance matrix for patch-weighted clustering.
    #[value(name = "dist-matrix")]
    DistMatrix,
    /// Pairwise group scoring for the Merge Suggestions UI using the weighted
    /// blend of CLS embeddings (pe-g + color + learned-head + dinov3).
    #[value(name = "embeddings")]
    Embeddings,
}

impl Mode {
    /// Patch-based modes read the DINOv3 patch token cache; the embeddings mode
    /// reads the per-model NPZ instead.
    pub(crate) fn uses_patches(self) -> bool {
        matches!(self, Mode::MergeSuggestions | Mode::DistMatrix)
    }
}

#[derive(Parser)]
#[command(about = "Compute pairwise group similarity (DINOv3 patches or embedding blend)")]
pub(crate) struct Cli {
    /// Path to hash-keyed patches cache .npy (dinov3_patches_hash_cache.npy).
    /// Required for patch-based modes; unused by the embeddings mode.
    #[arg(long, default_value = "")]
    pub patches_cache: String,

    /// Path to content_hashes.json (filename → content hash)
    #[arg(long)]
    pub content_hashes: String,

    /// Path to dinov3_patches_hashes.json (hash list in patches cache row order).
    /// Required for patch-based modes; unused by the embeddings mode.
    #[arg(long, default_value = "")]
    pub patches_hashes: String,

    /// Path to .reorder-groups.json
    #[arg(long)]
    pub groups: String,

    /// Minimum patch_median similarity to include in output (0.0-1.0)
    #[arg(long, default_value_t = 0.0)]
    pub min_score: f32,

    /// Skip pairs whose combined image count exceeds this value (0 = no limit)
    #[arg(long, default_value_t = 0)]
    pub max_combined_size: usize,

    /// Optional JSON file of group-id pairs to skip entirely (merge-suggestions
    /// mode only). Shape: [{ "groupA": "...", "groupB": "..." }, ...]. Pairs
    /// are unordered.
    #[arg(long, default_value = "")]
    pub rejected_pairs: String,

    /// Mode: "merge-suggestions" (default), "dist-matrix", or "embeddings"
    #[arg(long, value_enum, default_value_t = Mode::MergeSuggestions)]
    pub mode: Mode,

    /// Output path for condensed distance matrix binary (dist-matrix mode only)
    #[arg(long, default_value = "")]
    pub output: String,

    // ── Embeddings mode inputs (mirrors cluster-tool) ─────────────────────
    /// Path to embeddings_hash_cache.npz (per-model CLS embeddings)
    #[arg(long, default_value = "")]
    pub hash_cache: String,

    /// Path to hash_cache_order.json (hash list in NPZ row order)
    #[arg(long, default_value = "")]
    pub hash_order: String,

    /// Per-model blend weights — same semantics as cluster-tool. A model is
    /// loaded only when its weight is > 0; color is L2-normalized per row first.
    #[arg(long, default_value_t = 0.0)]
    pub color_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    pub dinov3_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    pub pecore_g_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    pub learned_proj_weight: f32,
}
