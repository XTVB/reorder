use clap::Parser;

mod cli;
mod modes;
mod patches;
mod score;

use crate::cli::{Cli, Mode};
use crate::patches::load_patches;

fn main() {
    let cli = Cli::parse();

    let patches = load_patches(&cli.patches_cache, &cli.content_hashes, &cli.patches_hashes);

    match cli.mode {
        Mode::DistMatrix => modes::dist_matrix::run(&patches, &cli.output),
        Mode::MergeSuggestions => modes::merge_suggestions::run(
            &patches,
            &cli.groups,
            cli.min_score,
            cli.max_combined_size,
        ),
    }
}
