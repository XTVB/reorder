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
    /// Path to .npz file with 'clip' and 'color' arrays + 'filenames'
    #[arg(long)]
    embeddings: PathBuf,

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

    // Load embeddings
    eprintln!("Loading embeddings from {:?}...", cli.embeddings);
    let file = File::open(&cli.embeddings).expect("Failed to open embeddings file");
    let mut npz = NpzReader::new(file).expect("Failed to read npz");

    let clip: Array2<f32> = npz.by_name("clip").expect("Missing 'clip' array");
    let color: Array2<f32> = npz.by_name("color").expect("Missing 'color' array");

    // Filenames are stored in a separate JSON file (numpy can't round-trip strings reliably)
    let filenames_path = cli.embeddings.with_extension("filenames.json");
    let filenames: Vec<String> = {
        let content = std::fs::read_to_string(&filenames_path)
            .unwrap_or_else(|_| panic!("Missing filenames file: {:?}", filenames_path));
        serde_json::from_str(&content)
            .unwrap_or_else(|_| panic!("Invalid filenames JSON: {:?}", filenames_path))
    };

    let n_images = clip.nrows();
    eprintln!(
        "Loaded {} images, CLIP dim={}, color dim={}",
        n_images,
        clip.ncols(),
        color.ncols()
    );

    // Build combined feature vectors as a flat row-major Vec<f32> for cache-efficient
    // access in the parallel distance computation.
    let (features_flat, feat_dim) =
        build_combined_features_flat(&clip, &color, cli.clip_weight, cli.color_weight);
    eprintln!("Combined feature dim: {}", feat_dim);

    // Build filename→index map
    let fname_to_idx: HashMap<&str, usize> = filenames
        .iter()
        .enumerate()
        .map(|(i, f)| (f.as_str(), i))
        .collect();

    // Load groups
    let groups = load_groups(&cli.groups, &fname_to_idx);
    eprintln!("Loaded {} confirmed groups", groups.len());

    // Find which images are in any group
    let mut grouped_images: HashSet<usize> = HashSet::new();
    for g in &groups {
        for &idx in &g.member_indices {
            grouped_images.insert(idx);
        }
    }

    // Build initial cluster set: one per group + one per ungrouped image
    let n_groups = groups.len();
    let mut ungrouped_img_indices: Vec<usize> = Vec::new();
    for i in 0..n_images {
        if !grouped_images.contains(&i) {
            ungrouped_img_indices.push(i);
        }
    }
    let n_initial = n_groups + ungrouped_img_indices.len();
    eprintln!(
        "Initial clusters: {} ({} groups + {} ungrouped)",
        n_initial,
        n_groups,
        ungrouped_img_indices.len()
    );

    // Run Ward's linkage with cosine distances and pre-seeded groups
    eprintln!("Running Ward's linkage (cosine + pre-seeded groups)...");
    let merge_steps =
        wards_linkage_cosine(&features_flat, feat_dim, n_images, &groups, &ungrouped_img_indices);
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
        save_linkage_tree(&sorted_steps, n_images, n_pre_merges, &cli.output_tree);
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
fn build_combined_features_flat(
    clip: &Array2<f32>,
    color: &Array2<f32>,
    clip_weight: f32,
    color_weight: f32,
) -> (Vec<f32>, usize) {
    let n = clip.nrows();
    let clip_dim = clip.ncols();
    let color_dim = color.ncols();
    let combined_dim = clip_dim + color_dim;
    let mut features = vec![0.0f32; n * combined_dim];

    // ndarray rows are contiguous in standard C (row-major) layout
    let clip_data = clip.as_slice().expect("clip array must be contiguous");
    let color_data = color.as_slice().expect("color array must be contiguous");

    for i in 0..n {
        let out = &mut features[i * combined_dim..][..combined_dim];

        // CLIP features (already L2-normalized from Python)
        let clip_row = &clip_data[i * clip_dim..][..clip_dim];
        for (o, &c) in out[..clip_dim].iter_mut().zip(clip_row) {
            *o = c * clip_weight;
        }

        // L2-normalize color features per row, then weight
        let color_row = &color_data[i * color_dim..][..color_dim];
        let color_norm_sq: f32 = color_row.iter().map(|&x| x * x).sum();
        let color_norm = color_norm_sq.sqrt().max(1e-10);
        for (o, &c) in out[clip_dim..].iter_mut().zip(color_row) {
            *o = (c / color_norm) * color_weight;
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
) -> Vec<MergeStep> {
    let n_groups = groups.len();
    let n_ungrouped = ungrouped.len();

    // ── Step 1: Compute pairwise cosine distances ─────────────────────────
    //
    // Key optimizations vs the original:
    //
    //   a) Single flat Vec<f64> — no second allocation, no Vec<Vec> copy.
    //      Replaces both `img_dist` and `dist_row` from the original code.
    //
    //   b) Parallel outer loop via rayon across 14 cores.
    //      Row i writes to dist[offset_i .. offset_i + (n-i-1)].
    //      These slices are provably non-overlapping; we use a SendPtr wrapper
    //      (a standard Rust pattern) to send the raw pointer across threads.
    //
    //   c) The dot-product inner loop operates on contiguous f32 slices.
    //      LLVM auto-vectorizes this to NEON FMLA on M4 (8-wide f32 SIMD).
    //
    //   d) Norms are precomputed once per image, not per pair.

    eprintln!("  Computing cosine distances for {} images...", n_images);

    // Precompute L2 norms in f64
    let norms: Vec<f64> = (0..n_images)
        .map(|i| {
            let row = &features[i * feat_dim..][..feat_dim];
            row.iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    let n_pairs = n_images * (n_images - 1) / 2;
    // Single flat condensed distance matrix — used throughout the entire algorithm.
    let mut dist: Vec<f64> = vec![0.0f64; n_pairs];

    // Parallel write: split `dist` into variable-length row chunks (row i has
    // n_images - i - 1 entries) and process each chunk on its own rayon worker.
    //
    // We use a recursive split-and-conquer via `par_bridge` + an iterator that
    // yields (&mut [f64], row_index) pairs. A simpler equivalent is to collect
    // split indices and use `split_at_mut` recursively — but the cleanest safe
    // Rust idiom for variable-length chunks is to build the mutable sub-slices
    // via a sequential split loop and then call `par_bridge` on the result.
    //
    // However, rayon requires the iterator items to be `Send`. `&mut [f64]` is
    // `Send`, so we build a Vec of (row_index, &mut [f64]) pairs by splitting
    // `dist` sequentially, then process them in parallel.

    // Build (row_index, &mut slice) pairs by sequential split
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

            // Contiguous f32 zip → LLVM NEON FMLA auto-vectorization
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

    eprintln!("  Cosine distances computed.");

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

    eprintln!("  Pre-merging {} groups...", n_groups);
    let mut pre_merge_steps: Vec<MergeStep> = Vec::new();

    for group in groups {
        if group.member_indices.len() < 2 {
            continue;
        }
        let mut members = group.member_indices.clone();
        members.sort();
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
    let main_merges_needed = if n_clusters >= n_after_premerge {
        0
    } else {
        n_after_premerge - n_clusters
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

fn save_linkage_tree(steps: &[MergeStep], n_images: usize, n_pre_merges: usize, path: &str) {
    let file = File::create(path).expect("Failed to create tree file");
    let mut w = BufWriter::new(file);

    // Header: n_images, n_pre_merges, n_total_steps
    w.write_u32::<LittleEndian>(n_images as u32).unwrap();
    w.write_u32::<LittleEndian>(n_pre_merges as u32).unwrap();
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
