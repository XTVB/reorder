use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum Mode {
    /// Pairwise group scoring for the Merge Suggestions UI.
    #[value(name = "merge-suggestions")]
    MergeSuggestions,
    /// Full pairwise condensed distance matrix for patch-weighted clustering.
    #[value(name = "dist-matrix")]
    DistMatrix,
}

#[derive(Parser)]
#[command(about = "Compute pairwise group similarity using DINOv3 patch matching")]
pub(crate) struct Cli {
    /// Path to hash-keyed patches cache .npy (dinov3_patches_hash_cache.npy)
    #[arg(long)]
    pub patches_cache: String,

    /// Path to content_hashes.json (filename → content hash)
    #[arg(long)]
    pub content_hashes: String,

    /// Path to dinov3_patches_hashes.json (hash list in patches cache row order)
    #[arg(long)]
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

    /// Mode: "merge-suggestions" (default) or "dist-matrix"
    #[arg(long, value_enum, default_value_t = Mode::MergeSuggestions)]
    pub mode: Mode,

    /// Output path for condensed distance matrix binary (dist-matrix mode only)
    #[arg(long, default_value = "")]
    pub output: String,
}
