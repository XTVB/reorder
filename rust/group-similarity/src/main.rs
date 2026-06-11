use clap::Parser;

mod cli;
mod embeddings;
mod modes;
mod patches;
mod score;

use crate::cli::{Cli, Mode};
use crate::embeddings::load_combined_embeddings;
use crate::patches::load_patches;

fn main() {
    let cli = Cli::parse();

    if cli.mode.uses_patches() {
        let patches = load_patches(&cli.patches_cache, &cli.content_hashes, &cli.patches_hashes);
        match cli.mode {
            Mode::DistMatrix => modes::dist_matrix::run(&patches, &cli.output),
            Mode::MergeSuggestions => modes::merge_suggestions::run(
                &patches,
                &cli.groups,
                cli.min_score,
                cli.max_combined_size,
                &cli.rejected_pairs,
            ),
            Mode::Embeddings => unreachable!("embeddings mode does not use patches"),
        }
    } else {
        let emb = load_combined_embeddings(
            &cli.hash_cache,
            &cli.content_hashes,
            &cli.hash_order,
            cli.color_weight,
            cli.dinov3_weight,
            cli.pecore_g_weight,
            cli.learned_proj_weight,
        );
        modes::embeddings::run(
            &emb,
            &cli.groups,
            cli.min_score,
            cli.max_combined_size,
            &cli.rejected_pairs,
        );
    }
}
