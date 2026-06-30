use clap::Parser;
use std::collections::{HashMap, HashSet};
use std::path::Path;

mod cli;
mod distances;
mod io;
mod linkage;
mod tree;

use crate::cli::Cli;
use crate::io::{load_cannot_link_pairs, load_groups, load_locked_group_ids};
use crate::linkage::{linkage_cosine, Linkage};
use crate::tree::{build_output_from_image_labels, cut_tree, save_linkage_tree};

fn main() {
    let cli = Cli::parse();

    let use_dist_matrix = !cli.dist_matrix.is_empty();
    let use_filename_subset = !cli.filenames.is_empty();

    if use_dist_matrix && use_filename_subset {
        eprintln!("ERROR: --filenames and --dist-matrix are incompatible (matrix is indexed on the full image set).");
        std::process::exit(2);
    }

    // Load content_hashes.json → sorted filenames + hash lookup
    let (content_hashes, mut filenames, _) =
        reorder_common::load_content_hashes_sorted(Path::new(&cli.content_hashes));

    // Narrow to the subset-of-filenames if requested. Applied before group
    // loading and embedding reindex so the rest of the pipeline is unchanged.
    if use_filename_subset {
        let subset_raw = std::fs::read_to_string(&cli.filenames)
            .unwrap_or_else(|_| panic!("Missing --filenames path: {}", cli.filenames));
        let subset_list: Vec<String> = serde_json::from_str(&subset_raw)
            .unwrap_or_else(|_| panic!("Invalid --filenames JSON: {}", cli.filenames));
        let subset_set: HashSet<String> = subset_list.into_iter().collect();
        let before = filenames.len();
        filenames.retain(|f| subset_set.contains(f));
        eprintln!(
            "Subset mode: narrowed {} → {} filenames (from {})",
            before,
            filenames.len(),
            cli.filenames
        );
        if filenames.len() < 2 {
            eprintln!("ERROR: subset has fewer than 2 filenames after intersection with content_hashes.json");
            std::process::exit(2);
        }
    }

    let n_images = filenames.len();

    let fname_to_idx: HashMap<&str, usize> = filenames
        .iter()
        .enumerate()
        .map(|(i, f)| (f.as_str(), i))
        .collect();

    let fname_to_cache_row =
        reorder_common::embeddings::load_fname_to_cache_row(&cli.hash_order, &content_hashes, &filenames);

    // Load groups
    let groups = load_groups(&cli.groups, &fname_to_idx);
    eprintln!("Loaded {} confirmed groups", groups.len());

    // Load embeddings if ANY model weight is positive. The previous behavior
    // skipped embedding loading whenever --dist-matrix was provided, which made
    // blending the precomputed matrix with embedding distances impossible. Now
    // the two coexist: with both, linkage.rs blends them via dist_matrix_weight.
    let emb_specs = reorder_common::embeddings::emb_specs(
        cli.color_weight,
        cli.dinov3_weight,
        cli.pecore_g_weight,
        cli.learned_proj_weight,
        cli.learned_proj_peg_weight,
        cli.learned_proj_color_weight,
    );
    let any_active = emb_specs.iter().any(|(_, w, _)| *w > 0.0);
    let features_flat: Vec<f32>;
    let feat_dim: usize;

    if !any_active {
        eprintln!("No embedding weights set — using precomputed distance matrix only");
        features_flat = vec![];
        feat_dim = 0;
    } else {
        let loaded = reorder_common::embeddings::load_model_arrays(
            &cli.hash_cache,
            &fname_to_cache_row,
            &emb_specs,
        );
        let (ff, fd) =
            reorder_common::embeddings::build_combined_features_flat(&loaded, n_images, false);
        features_flat = ff;
        feat_dim = fd;
        eprintln!("Combined feature dim: {}", feat_dim);
    }

    // Find which images are in any group
    let mut grouped_images: HashSet<usize> = HashSet::new();
    for g in &groups {
        for &idx in &g.member_indices {
            grouped_images.insert(idx);
        }
    }

    let n_groups = groups.len();
    let mut ungrouped_img_indices: Vec<usize> = Vec::new();
    for i in 0..n_images {
        if !grouped_images.contains(&i) {
            ungrouped_img_indices.push(i);
        }
    }
    eprintln!(
        "Initial clusters: {} ({} groups + {} ungrouped)",
        n_groups + ungrouped_img_indices.len(),
        n_groups,
        ungrouped_img_indices.len()
    );

    // Load precomputed distance matrix if provided
    let precomputed_dist: Option<(Vec<f64>, f32)> = if use_dist_matrix {
        eprintln!("Loading precomputed distance matrix from {}...", cli.dist_matrix);
        let bytes = std::fs::read(&cli.dist_matrix).expect("read dist matrix");
        let stored_n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(
            stored_n, n_images,
            "Distance matrix has {} images but embeddings has {}",
            stored_n, n_images
        );
        let n_pairs = n_images * (n_images - 1) / 2;
        let data_bytes = &bytes[8..];
        assert_eq!(
            data_bytes.len(),
            n_pairs * 8,
            "Distance matrix data size mismatch"
        );
        let dist: Vec<f64> = unsafe {
            std::slice::from_raw_parts(data_bytes.as_ptr() as *const f64, n_pairs)
        }
        .to_vec();
        let w = cli.dist_matrix_weight;
        eprintln!("  Loaded {} distances (weight={})", n_pairs, w);
        Some((dist, w))
    } else {
        None
    };

    let cannot_link_pairs = load_cannot_link_pairs(&cli.cannot_link, &fname_to_idx);
    let locked_group_ids = load_locked_group_ids(&cli.locked_groups);
    if !cannot_link_pairs.is_empty() {
        eprintln!(
            "Loaded {} image↔group cannot-link constraints",
            cannot_link_pairs.len()
        );
    }
    if !locked_group_ids.is_empty() {
        eprintln!("Loaded {} group-lock constraints", locked_group_ids.len());
    }

    // Run hierarchical agglomerative linkage
    let linkage = Linkage::parse(&cli.linkage);
    eprintln!("Running {:?} linkage...", linkage);
    let merge_steps = linkage_cosine(
        &features_flat,
        feat_dim,
        n_images,
        &groups,
        &ungrouped_img_indices,
        precomputed_dist,
        linkage,
        &cannot_link_pairs,
        &locked_group_ids,
    );
    eprintln!("Linkage complete: {} merge steps", merge_steps.len());

    // Sort main steps by distance for the tree file (so Bun can cut correctly)
    let n_pre_merges = groups
        .iter()
        .map(|g| g.member_indices.len().saturating_sub(1))
        .sum::<usize>();
    let mut sorted_steps = merge_steps.clone();
    sorted_steps[n_pre_merges..].sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Save linkage tree
    if !cli.output_tree.is_empty() {
        save_linkage_tree(&sorted_steps, n_images, n_pre_merges, n_groups, &cli.output_tree);
        eprintln!("Saved linkage tree to {}", cli.output_tree);
    }

    // Cut tree at requested N using sorted steps (already sorted, no re-sort needed)
    let n_initial_after_premerge = n_images - n_pre_merges;
    let labels = cut_tree(
        &sorted_steps,
        n_images,
        n_initial_after_premerge,
        cli.n_clusters,
        n_pre_merges,
        n_groups,
    );

    // Build output — labels are per original image, need to group them
    let output = build_output_from_image_labels(
        &labels,
        &filenames,
        &groups,
        cli.n_clusters,
        &cli.output_tree,
    );

    serde_json::to_writer(std::io::stdout().lock(), &output).expect("Failed to write JSON");
}
