use byteorder::{LittleEndian, WriteBytesExt};
use clap::Parser;
use ndarray::Array2;
use ndarray_npy::NpzReader;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Ward's linkage clustering with pre-seeded groups")]
struct Cli {
    /// Path to hash-keyed cache .npz (clip_hash_cache.npz)
    #[arg(long)]
    hash_cache: PathBuf,

    /// Path to content_hashes.json (filename → content hash)
    #[arg(long)]
    content_hashes: String,

    /// Path to hash_cache_order.json (hash list in NPZ row order)
    #[arg(long)]
    hash_order: String,

    /// Path to .reorder-groups.json
    #[arg(long, default_value = "")]
    groups: String,

    /// Number of clusters to produce
    #[arg(long, default_value_t = 200)]
    n_clusters: usize,

    /// Output path for linkage tree binary
    #[arg(long, default_value = "")]
    output_tree: String,

    /// CLIP feature weight
    #[arg(long, default_value_t = 1.0)]
    clip_weight: f32,

    /// Color feature weight
    #[arg(long, default_value_t = 0.5)]
    color_weight: f32,

    /// DINOv2 feature weight
    #[arg(long, default_value_t = 0.0)]
    dino_weight: f32,

    /// DINOv3 CLS token weight
    #[arg(long, default_value_t = 0.0)]
    dinov3_weight: f32,

    /// PE-Core-L feature weight
    #[arg(long, default_value_t = 0.0)]
    pecore_l_weight: f32,

    /// PE-Core-bigG feature weight
    #[arg(long, default_value_t = 0.0)]
    pecore_g_weight: f32,

    /// Path to precomputed condensed distance matrix binary
    #[arg(long, default_value = "")]
    dist_matrix: String,

    /// Weight for the precomputed distance matrix when blending with embedding distances.
    /// 1.0 = patches only, 0.0 = embeddings only, 0.5 = equal blend.
    #[arg(long, default_value_t = 1.0)]
    dist_matrix_weight: f32,
}

// ── Data types ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ReorderGroup {
    id: String,
    name: String,
    images: Vec<String>,
}

#[derive(Serialize)]
struct OutputCluster {
    id: String,
    images: Vec<String>,
    confirmed_group: Option<ConfirmedGroupInfo>,
}

#[derive(Serialize)]
struct ConfirmedGroupInfo {
    id: String,
    name: String,
    images: Vec<String>,
}

#[derive(Serialize)]
struct Output {
    clusters: Vec<OutputCluster>,
    n_clusters: usize,
    tree_path: String,
}

/// A merge step in the linkage tree.
#[derive(Clone, Copy)]
struct MergeStep {
    cluster_a: u32,
    cluster_b: u32,
    distance: f32,
    new_size: u32,
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    let use_dist_matrix = !cli.dist_matrix.is_empty();

    // Load content_hashes.json → sorted filenames + hash lookup
    let content_hashes: HashMap<String, String> = {
        let content = std::fs::read_to_string(&cli.content_hashes)
            .unwrap_or_else(|_| panic!("Missing content_hashes.json: {}", cli.content_hashes));
        serde_json::from_str(&content)
            .unwrap_or_else(|_| panic!("Invalid content_hashes.json: {}", cli.content_hashes))
    };
    let mut filenames: Vec<String> = content_hashes.keys().cloned().collect();
    filenames.sort();
    let n_images = filenames.len();

    let fname_to_idx: HashMap<&str, usize> = filenames
        .iter()
        .enumerate()
        .map(|(i, f)| (f.as_str(), i))
        .collect();

    // Load hash_cache_order.json → NPZ row mapping
    let hash_order: Vec<String> = {
        let content = std::fs::read_to_string(&cli.hash_order)
            .unwrap_or_else(|_| panic!("Missing hash_cache_order.json: {}", cli.hash_order));
        serde_json::from_str(&content)
            .unwrap_or_else(|_| panic!("Invalid hash_cache_order.json: {}", cli.hash_order))
    };
    let hash_to_cache_row: HashMap<&str, usize> = hash_order
        .iter()
        .enumerate()
        .map(|(i, h)| (h.as_str(), i))
        .collect();

    // Build filename → NPZ cache row mapping
    let fname_to_cache_row: Vec<usize> = filenames
        .iter()
        .map(|f| {
            let hash = content_hashes.get(f).unwrap_or_else(|| panic!("No hash for {}", f));
            *hash_to_cache_row.get(hash.as_str())
                .unwrap_or_else(|| panic!("Hash {} (file {}) not found in cache — re-run extraction", hash, f))
        })
        .collect();

    // Load groups
    let groups = load_groups(&cli.groups, &fname_to_idx);
    eprintln!("Loaded {} confirmed groups", groups.len());

    // Load embeddings (skip if using precomputed distance matrix)
    let features_flat: Vec<f32>;
    let feat_dim: usize;

    if use_dist_matrix {
        eprintln!("Using precomputed distance matrix — skipping embedding loading");
        features_flat = vec![];
        feat_dim = 0;
    } else {
        eprintln!("Loading embeddings from {:?}...", cli.hash_cache);
        let file = File::open(&cli.hash_cache).expect("Failed to open hash cache file");
        let mut npz = NpzReader::new(file).expect("Failed to read npz");

        let emb_specs: Vec<(&str, f32, bool)> = vec![
            ("clip", cli.clip_weight, false),
            ("dino", cli.dino_weight, false),
            ("dinov3", cli.dinov3_weight, false),
            ("pecore_l", cli.pecore_l_weight, false),
            ("pecore_g", cli.pecore_g_weight, false),
            ("color", cli.color_weight, true),
        ];
        let loaded: Vec<(Array2<f32>, f32, bool)> = emb_specs
            .iter()
            .filter(|(_, w, _)| *w > 0.0)
            .filter_map(|(name, w, norm)| {
                match npz.by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>(name) {
                    Ok(hash_ordered_arr) => {
                        // Reindex from hash/cache order to filename order
                        let dim = hash_ordered_arr.ncols();
                        let mut arr = Array2::<f32>::zeros((n_images, dim));
                        for (i, &cache_row) in fname_to_cache_row.iter().enumerate() {
                            arr.row_mut(i).assign(&hash_ordered_arr.row(cache_row));
                        }
                        Some((arr, *w, *norm))
                    }
                    Err(_) => {
                        eprintln!("WARNING: '{}' array not found in hash cache (weight={:.1}), skipping", name, w);
                        None
                    }
                }
            })
            .collect();

        let active_desc: Vec<String> = emb_specs
            .iter()
            .filter(|(_, w, _)| *w > 0.0)
            .zip(loaded.iter())
            .map(|((name, w, _), (arr, _, _))| format!("{}={}d×{}", name, arr.ncols(), w))
            .collect();
        eprintln!("Loaded {} images, active: {}", n_images, active_desc.join(", "));

        let emb_arrays: Vec<(&Array2<f32>, f32, bool)> = loaded.iter().map(|(a, w, n)| (a, *w, *n)).collect();
        let (ff, fd) = build_combined_features_flat(&emb_arrays, n_images);
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
        assert_eq!(stored_n, n_images,
            "Distance matrix has {} images but embeddings has {}", stored_n, n_images);
        let n_pairs = n_images * (n_images - 1) / 2;
        let data_bytes = &bytes[8..];
        assert_eq!(data_bytes.len(), n_pairs * 8, "Distance matrix data size mismatch");
        let dist: Vec<f64> = unsafe {
            std::slice::from_raw_parts(data_bytes.as_ptr() as *const f64, n_pairs)
        }.to_vec();
        let w = cli.dist_matrix_weight;
        eprintln!("  Loaded {} distances (weight={})", n_pairs, w);
        Some((dist, w))
    } else {
        None
    };

    // Run Ward's linkage
    eprintln!("Running Ward's linkage...");
    let merge_steps = wards_linkage_cosine(
        &features_flat, feat_dim, n_images, &groups, &ungrouped_img_indices,
        precomputed_dist,
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

// ── Feature combination ──────────────────────────────────────────────────────

/// Returns a flat row-major Vec<f32> and the feature dimension.
/// Using a flat Vec instead of ndarray removes ndarray indexing overhead in
/// the hot distance-computation loop.
/// Build combined feature vector from multiple embedding arrays.
/// Each entry is (array, weight, needs_l2_norm). Arrays already L2-normalized
/// from Python have needs_l2_norm=false; color features need per-row normalization.
fn build_combined_features_flat(
    arrays: &[(&Array2<f32>, f32, bool)],
    n: usize,
) -> (Vec<f32>, usize) {
    let combined_dim: usize = arrays.iter().map(|(a, _, _)| a.ncols()).sum();
    let mut features = vec![0.0f32; n * combined_dim];

    // Pre-extract contiguous slices and dims
    let slices: Vec<(&[f32], usize, f32, bool)> = arrays
        .iter()
        .map(|(a, w, norm)| {
            let data = a.as_slice().expect("array must be contiguous");
            (data, a.ncols(), *w, *norm)
        })
        .collect();

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
    }

    (features, combined_dim)
}

// ── Group loading ────────────────────────────────────────────────────────────

struct LoadedGroup {
    id: String,
    name: String,
    member_indices: Vec<usize>,
    member_filenames: Vec<String>,
}

fn load_groups(groups_path: &str, fname_to_idx: &HashMap<&str, usize>) -> Vec<LoadedGroup> {
    if groups_path.is_empty() {
        return vec![];
    }
    let Ok(content) = std::fs::read_to_string(groups_path) else {
        return vec![];
    };
    let Ok(raw_groups): Result<Vec<ReorderGroup>, _> = serde_json::from_str(&content) else {
        return vec![];
    };

    raw_groups
        .into_iter()
        .filter_map(|g| {
            let mut indices = Vec::new();
            let mut fnames = Vec::new();
            for f in &g.images {
                if let Some(&idx) = fname_to_idx.get(f.as_str()) {
                    indices.push(idx);
                    fnames.push(f.clone());
                }
            }
            if indices.len() < 2 {
                return None;
            }
            Some(LoadedGroup {
                id: g.id,
                name: g.name,
                member_indices: indices,
                member_filenames: fnames,
            })
        })
        .collect()
}

// ── Flat condensed distance matrix ───────────────────────────────────────────
//
// We store the upper-triangle of the n×n distance matrix as a flat Vec<f64>
// using the standard condensed index formula. This replaces both the old
// `img_dist` array and the `dist_row: Vec<Vec<f64>>` — eliminating the O(n²)
// copy and halving peak memory.
//
// condensed_idx(i, j, n) gives the offset for i < j.

#[inline(always)]
fn condensed_idx(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i < j, "condensed_idx requires i < j, got i={} j={}", i, j);
    // Row i has (n - i - 1) entries, starting at offset: i*n - i*(i+1)/2
    i * n - i * (i + 1) / 2 + j - i - 1
}

#[inline(always)]
fn get_dist(dist: &[f64], i: usize, j: usize, n: usize) -> f64 {
    if i < j {
        dist[condensed_idx(i, j, n)]
    } else {
        dist[condensed_idx(j, i, n)]
    }
}

#[inline(always)]
fn set_dist(dist: &mut [f64], i: usize, j: usize, n: usize, val: f64) {
    if i < j {
        dist[condensed_idx(i, j, n)] = val;
    } else {
        dist[condensed_idx(j, i, n)] = val;
    }
}

// ── Ward's linkage with cosine distances and pre-seeded groups ───────────────
//
// Matches scipy's `linkage(pdist(X, metric='cosine'), method='ward')`:
// 1. Compute pairwise cosine distances between ALL individual images
// 2. For pre-seeded groups: simulate the merges using Lance-Williams to get
//    correct distances from each group to everything else
// 3. Run NNC + Lance-Williams on the resulting distance matrix
//
// This ensures the clustering results match the validated scipy approach.

fn wards_linkage_cosine(
    features: &[f32],       // flat row-major, shape [n_images × feat_dim]
    feat_dim: usize,
    n_images: usize,
    groups: &[LoadedGroup],
    ungrouped: &[usize],
    precomputed_dist: Option<(Vec<f64>, f32)>, // (distances, weight)
) -> Vec<MergeStep> {
    let n_groups = groups.len();
    let n_ungrouped = ungrouped.len();

    let n_pairs = n_images * (n_images - 1) / 2;

    // ── Step 1: Get pairwise distances ───────────────────────────────────
    let has_features = feat_dim > 0;

    let skip_cosine = match &precomputed_dist {
        Some((_, w)) => !has_features || *w >= 1.0,
        None => !has_features,
    };

    let mut dist: Vec<f64> = if skip_cosine {
        let (precomp, _) = precomputed_dist.expect("skip_cosine implies precomputed");
        eprintln!("  Using precomputed distance matrix only ({} pairs)", precomp.len());
        precomp
    } else {
        eprintln!("  Computing cosine distances for {} images...", n_images);

        let norms: Vec<f64> = (0..n_images)
            .map(|i| {
                features[i * feat_dim..][..feat_dim]
                    .iter()
                    .map(|&x| (x as f64) * (x as f64))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        let mut dist: Vec<f64> = vec![0.0f64; n_pairs];
        let mut row_slices: Vec<(usize, &mut [f64])> = Vec::with_capacity(n_images - 1);
        {
            let mut remaining = dist.as_mut_slice();
            for i in 0..n_images - 1 {
                let count = n_images - i - 1;
                let (chunk, rest) = remaining.split_at_mut(count);
                row_slices.push((i, chunk));
                remaining = rest;
            }
        }

        row_slices.par_iter_mut().for_each(|(i, slice)| {
            let i = *i;
            let row_i = &features[i * feat_dim..][..feat_dim];
            let ni = norms[i];

            for (k, slot) in slice.iter_mut().enumerate() {
                let j = i + 1 + k;
                let row_j = &features[j * feat_dim..][..feat_dim];
                let nj = norms[j];

                let dot: f64 = row_i
                    .iter()
                    .zip(row_j.iter())
                    .map(|(&a, &b)| (a as f64) * (b as f64))
                    .sum();

                let denom = ni * nj;
                let cos_sim = if denom > 1e-20 { dot / denom } else { 0.0 };
                *slot = (1.0 - cos_sim).max(0.0);
            }
        });

        // Blend in precomputed distances if provided: dist = w*precomp + (1-w)*cos
        if let Some((precomp, weight)) = precomputed_dist {
            eprintln!("  Blending precomputed (weight={}) with cosine distances...", weight);
            let w = weight as f64;
            dist.par_iter_mut().zip(precomp.par_iter()).for_each(|(d, &p)| {
                *d = w * p + (1.0 - w) * *d;
            });
        }

        eprintln!("  Distances computed.");
        dist
    };

    // ── Step 2: Pre-merge groups using Lance-Williams updates ─────────────
    //
    // We work directly in `dist` (the condensed flat array). The Ward update
    // formula is unchanged — only the access pattern differs: instead of
    // dist_row[i][j-i-1] we use dist[condensed_idx(...)].
    //
    // The active_indices list avoids scanning deactivated slots in both the
    // Ward update and the NNC inner loop.

    let _n_initial = n_groups + n_ungrouped;

    let mut size = vec![1.0f64; n_images];

    // Maintain a sorted list of active indices for fast NN scanning.
    // Using a Vec rather than a BTreeSet keeps iteration cache-friendly.
    let mut active_indices: Vec<usize> = (0..n_images).collect();

    // Pre-compute sorted members and representative (highest index) for each group
    let group_sorted: Vec<Vec<usize>> = groups
        .iter()
        .filter(|g| g.member_indices.len() >= 2)
        .map(|g| {
            let mut m = g.member_indices.clone();
            m.sort();
            m
        })
        .collect();

    eprintln!("  Pre-merging {} groups...", n_groups);
    let mut pre_merge_steps: Vec<MergeStep> = Vec::new();

    for members in &group_sorted {
        let target = *members.last().unwrap();

        for &member in &members[..members.len() - 1] {
            let x = member; // x < target always (sorted)
            let y = target;
            let merge_dist = get_dist(&dist, x, y, n_images);

            let nx = size[x];
            let ny = size[y];
            let new_size = nx + ny;

            // Ward update into slot y for all other active clusters
            for &i in &active_indices {
                if i == x || i == y {
                    continue;
                }
                let ni = size[i];
                let d_ix = get_dist(&dist, i, x, n_images);
                let d_iy = get_dist(&dist, i, y, n_images);
                let t = 1.0 / (ni + new_size);
                let d_new = ((ni + nx) * t * d_ix * d_ix
                    + (ni + ny) * t * d_iy * d_iy
                    - ni * t * merge_dist * merge_dist)
                    .max(0.0)
                    .sqrt();
                set_dist(&mut dist, i, y, n_images, d_new);
            }

            size[x] = 0.0;
            size[y] = new_size;

            // Remove x from active_indices (it's sorted, binary search is O(log n))
            if let Ok(pos) = active_indices.binary_search(&x) {
                active_indices.remove(pos);
            }

            pre_merge_steps.push(MergeStep {
                cluster_a: x as u32,
                cluster_b: y as u32,
                distance: merge_dist as f32,
                new_size: new_size as u32,
            });
        }
    }
    eprintln!(
        "  Pre-merged {} steps, {} active clusters remain",
        pre_merge_steps.len(),
        active_indices.len()
    );

    // ── Prevent confirmed groups from ever being merged with each other ───
    // We use 1e18 rather than f64::MAX because the Ward update formula squares
    // distances — MAX² overflows to infinity and the subtraction produces NaN.
    const GROUP_BARRIER: f64 = 1e18;
    let group_reps: Vec<usize> = group_sorted.iter().map(|m| *m.last().unwrap()).collect();
    for i in 0..group_reps.len() {
        for j in (i + 1)..group_reps.len() {
            set_dist(&mut dist, group_reps[i], group_reps[j], n_images, GROUP_BARRIER);
        }
    }

    // ── Step 3: NNC (nearest-neighbor chain) Ward's linkage ──────────────
    //
    // Same algorithm as before; key improvements:
    //   - NN scan iterates active_indices (shrinking list) instead of 0..n_images
    //   - Ward update also iterates active_indices
    //   - active_indices.remove() is O(n) but called only n times total
    //
    // The O(n) scan per NN search is unavoidable without a different data
    // structure (e.g. ball-tree), which would require fundamental changes to
    // the Ward update semantics.

    let n_remaining = active_indices.len();
    eprintln!("  Running NNC on {} clusters...", n_remaining);

    let mut merge_steps: Vec<MergeStep> = Vec::with_capacity(n_remaining - 1);
    let mut chain: Vec<usize> = Vec::with_capacity(n_remaining);

    for step in 0..(n_remaining - 1) {
        if step % 2000 == 0 && step > 0 {
            eprintln!("  merge step {}/{}", step, n_remaining - 1);
        }

        // If chain is empty, seed with first active cluster
        if chain.is_empty() {
            chain.push(active_indices[0]);
        }

        loop {
            let x = *chain.last().unwrap();

            // Scipy tie-breaking: prefer previous chain element as the
            // initial candidate (only replaced on strictly-less-than).
            let mut y;
            let mut current_min;
            if chain.len() >= 2 {
                y = chain[chain.len() - 2];
                current_min = get_dist(&dist, x, y, n_images);
            } else {
                y = usize::MAX; // sentinel — will be overwritten on first valid candidate
                current_min = f64::MAX;
            }

            // Scan active clusters for the nearest neighbor of x.
            // active_indices shrinks monotonically, so this loop gets faster
            // as the algorithm progresses (average size n/2 over all merges).
            for &i in &active_indices {
                if i == x {
                    continue;
                }
                let d = get_dist(&dist, x, i, n_images);
                if d < current_min {
                    current_min = d;
                    y = i;
                }
            }

            // Check if y is the previous chain element (reciprocal NN pair)
            if chain.len() >= 2 && y == chain[chain.len() - 2] {
                chain.pop();
                chain.pop();

                // Convention: x = min, y = max. Deactivate x, reuse y.
                let (x, y) = if x < y { (x, y) } else { (y, x) };

                let nx = size[x];
                let ny = size[y];
                let new_size = nx + ny;

                merge_steps.push(MergeStep {
                    cluster_a: x as u32,
                    cluster_b: y as u32,
                    distance: current_min as f32,
                    new_size: new_size as u32,
                });

                // Ward update into slot y for all remaining active clusters
                for &i in &active_indices {
                    if i == x || i == y {
                        continue;
                    }
                    let ni = size[i];
                    let d_ix = get_dist(&dist, i, x, n_images);
                    let d_iy = get_dist(&dist, i, y, n_images);
                    let t = 1.0 / (ni + new_size);
                    let d_new = ((ni + nx) * t * d_ix * d_ix
                        + (ni + ny) * t * d_iy * d_iy
                        - ni * t * current_min * current_min)
                        .max(0.0)
                        .sqrt();
                    set_dist(&mut dist, i, y, n_images, d_new);
                }

                size[x] = 0.0;
                size[y] = new_size;

                if let Ok(pos) = active_indices.binary_search(&x) {
                    active_indices.remove(pos);
                }

                break;
            } else {
                chain.push(y);
            }
        }
    }

    // Combine pre-merge steps + main merge steps
    let mut all_steps = pre_merge_steps;
    all_steps.extend(merge_steps);
    all_steps
}

// ── Tree cutting ─────────────────────────────────────────────────────────────

fn cut_tree(
    merge_steps: &[MergeStep], // already sorted from main()
    n_images: usize,
    n_after_premerge: usize,
    n_clusters: usize,
    n_pre_merges: usize,
    n_groups: usize,
) -> Vec<u32> {
    // Union-find over original image indices.
    // The input steps are already sorted by distance (done in main() before
    // saving the tree), so we don't re-sort here.
    let mut parent = vec![0u32; n_images];
    for i in 0..n_images {
        parent[i] = i as u32;
    }

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let p = parent[x as usize];
            parent[x as usize] = parent[p as usize];
            x = p;
        }
        x
    }

    // Apply all pre-merge steps (first n_pre_merges entries — forced)
    for step in merge_steps.iter().take(n_pre_merges) {
        let ra = find(&mut parent, step.cluster_a);
        let rb = find(&mut parent, step.cluster_b);
        if ra != rb {
            parent[ra as usize] = rb;
        }
    }

    // Main steps are already sorted by distance in the input slice.
    // Apply sorted main steps until we reach n_clusters.
    // Never go below n_groups clusters — confirmed groups must stay separate.
    let min_clusters = n_clusters.max(n_groups);
    let main_merges_needed = if min_clusters >= n_after_premerge {
        0
    } else {
        n_after_premerge - min_clusters
    };

    for step in merge_steps[n_pre_merges..].iter().take(main_merges_needed) {
        let ra = find(&mut parent, step.cluster_a);
        let rb = find(&mut parent, step.cluster_b);
        if ra != rb {
            parent[ra as usize] = rb;
        }
    }

    // Get cluster label for each image
    let roots: Vec<u32> = (0..n_images).map(|i| find(&mut parent, i as u32)).collect();

    // Renumber contiguously
    let mut seen: HashMap<u32, u32> = HashMap::new();
    let mut next_label = 0u32;
    roots
        .iter()
        .map(|&r| {
            *seen.entry(r).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            })
        })
        .collect()
}

// ── Linkage tree I/O ─────────────────────────────────────────────────────────

fn save_linkage_tree(steps: &[MergeStep], n_images: usize, n_pre_merges: usize, n_groups: usize, path: &str) {
    let file = File::create(path).expect("Failed to create tree file");
    let mut w = BufWriter::new(file);

    // Header: n_images, n_pre_merges, n_groups, n_total_steps
    w.write_u32::<LittleEndian>(n_images as u32).unwrap();
    w.write_u32::<LittleEndian>(n_pre_merges as u32).unwrap();
    w.write_u32::<LittleEndian>(n_groups as u32).unwrap();
    w.write_u32::<LittleEndian>(steps.len() as u32).unwrap();

    for step in steps {
        w.write_u32::<LittleEndian>(step.cluster_a).unwrap();
        w.write_u32::<LittleEndian>(step.cluster_b).unwrap();
        w.write_f32::<LittleEndian>(step.distance).unwrap();
        w.write_u32::<LittleEndian>(step.new_size).unwrap();
    }
    w.flush().unwrap();
}

// ── Output construction ──────────────────────────────────────────────────────

fn build_output_from_image_labels(
    labels: &[u32], // one label per original image
    filenames: &[String],
    groups: &[LoadedGroup],
    n_clusters: usize,
    tree_path: &str,
) -> Output {
    // Group images by cluster label
    let mut cluster_images: HashMap<u32, Vec<usize>> = HashMap::new();
    for (img_idx, &label) in labels.iter().enumerate() {
        cluster_images.entry(label).or_default().push(img_idx);
    }

    // Build index of which images belong to which confirmed group
    let mut img_to_group: HashMap<usize, usize> = HashMap::new();
    for (gi, group) in groups.iter().enumerate() {
        for &idx in &group.member_indices {
            img_to_group.insert(idx, gi);
        }
    }

    let mut output_clusters = Vec::new();
    let mut sorted_labels: Vec<u32> = cluster_images.keys().copied().collect();
    sorted_labels.sort();

    for (ci, &label) in sorted_labels.iter().enumerate() {
        let members = &cluster_images[&label];

        let image_filenames: Vec<String> = members.iter().map(|&i| filenames[i].clone()).collect();

        // Check if this cluster contains any confirmed group
        let confirmed = members
            .iter()
            .find_map(|&idx| img_to_group.get(&idx))
            .map(|&gi| &groups[gi]);

        let mut sorted_filenames = image_filenames;
        sorted_filenames.sort();

        output_clusters.push(OutputCluster {
            id: format!("cluster_{}", ci),
            images: sorted_filenames,
            confirmed_group: confirmed.map(|g| ConfirmedGroupInfo {
                id: g.id.clone(),
                name: g.name.clone(),
                images: g.member_filenames.clone(),
            }),
        });
    }

    output_clusters.sort_by(|a, b| b.images.len().cmp(&a.images.len()));

    Output {
        clusters: output_clusters,
        n_clusters,
        tree_path: tree_path.to_string(),
    }
}
