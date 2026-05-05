use rayon::prelude::*;
use std::collections::HashMap;

use crate::distances::{get_dist, set_dist};
use crate::io::LoadedGroup;

/// A merge step in the linkage tree.
#[derive(Clone, Copy)]
pub(crate) struct MergeStep {
    pub cluster_a: u32,
    pub cluster_b: u32,
    pub distance: f32,
    pub new_size: u32,
}

/// Hierarchical linkage method (Lance-Williams family).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Linkage {
    Ward,
    Average,
    Complete,
}

impl Linkage {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "ward" => Self::Ward,
            "average" | "upgma" => Self::Average,
            "complete" => Self::Complete,
            other => panic!(
                "unknown linkage method: '{}' (expected ward|average|complete)",
                other
            ),
        }
    }
}

/// Lance-Williams update: distance from cluster i to merged (x ∪ y), given
/// existing distances and cluster sizes. Returns the new d(i, xy).
#[inline(always)]
pub(crate) fn lance_williams(
    linkage: Linkage,
    d_ix: f64,
    d_iy: f64,
    d_xy: f64,
    ni: f64,
    nx: f64,
    ny: f64,
) -> f64 {
    match linkage {
        Linkage::Ward => {
            let t = 1.0 / (ni + nx + ny);
            ((ni + nx) * t * d_ix * d_ix
                + (ni + ny) * t * d_iy * d_iy
                - ni * t * d_xy * d_xy)
                .max(0.0)
                .sqrt()
        }
        Linkage::Average => {
            // UPGMA: weighted by cluster size
            (nx * d_ix + ny * d_iy) / (nx + ny)
        }
        Linkage::Complete => d_ix.max(d_iy),
    }
}

// ── Sentinel-barrier value for hard cannot-link / locked-group constraints ───
//
// We use 1e18 rather than f64::MAX because Ward's update squares distances —
// f64::MAX² overflows. 1e18 is safe across all linkage methods. The recut
// filter in src/cluster/linkage.ts uses `distance < 1e10` to detect these
// barriers, so the magic number must stay aligned across both sides.
pub(crate) const GROUP_BARRIER: f64 = 1e18;

// ── Hierarchical agglomerative linkage with cosine distances and pre-seeded groups ───
//
// Matches scipy's `linkage(pdist(X, metric='cosine'), method=<linkage>)`:
// 1. Compute pairwise cosine distances between ALL individual images
//    (or use a precomputed distance matrix, optionally blended with cosine).
// 2. For pre-seeded groups: simulate the merges using Lance-Williams to get
//    correct distances from each group to everything else.
// 3. Run NNC + Lance-Williams on the resulting distance matrix.
//
// `linkage` selects the Lance-Williams variant (Ward / Average / Complete).
// Average linkage works best with the re-ranking distance matrix; Ward is the
// historical default for raw cosine.

#[allow(clippy::too_many_arguments)]
pub(crate) fn linkage_cosine(
    features: &[f32], // flat row-major, shape [n_images × feat_dim]
    feat_dim: usize,
    n_images: usize,
    groups: &[LoadedGroup],
    ungrouped: &[usize],
    precomputed_dist: Option<(Vec<f64>, f32)>, // (distances, weight)
    linkage: Linkage,
    cannot_link: &[(usize, String)], // (image_idx, group_id) — barrier between image and group's rep
    locked_groups: &[String],        // group_ids that are sealed: nothing else may merge in
) -> Vec<MergeStep> {
    let n_groups = groups.len();
    let n_ungrouped = ungrouped.len();
    let _n_initial = n_groups + n_ungrouped;

    // ── Block 1: build the pairwise distance matrix ──────────────────────
    let mut dist = compute_distances(features, feat_dim, n_images, precomputed_dist);

    // Per-cluster size; updated as merges happen.
    let mut size = vec![1.0f64; n_images];

    // Sorted list of active indices for fast NN scanning.
    // Using a Vec rather than a BTreeSet keeps iteration cache-friendly.
    let mut active_indices: Vec<usize> = (0..n_images).collect();

    // ── Block 2: pre-merge confirmed groups ──────────────────────────────
    let pre_merge_steps = premerge_groups(
        &mut dist,
        n_images,
        groups,
        linkage,
        &mut size,
        &mut active_indices,
        cannot_link,
        locked_groups,
    );

    // ── Block 3: NNC main loop ───────────────────────────────────────────
    let merge_steps = nnc_loop(
        &mut dist,
        n_images,
        linkage,
        &mut size,
        &mut active_indices,
    );

    // Combine pre-merge steps + main merge steps
    let mut all_steps = pre_merge_steps;
    all_steps.extend(merge_steps);
    all_steps
}

// ── Block 1: pairwise distance matrix ────────────────────────────────────────
//
// Builds the condensed-upper-triangle distance matrix from per-model features
// and/or a precomputed distance matrix, optionally blending the two.
fn compute_distances(
    features: &[f32],
    feat_dim: usize,
    n_images: usize,
    precomputed_dist: Option<(Vec<f64>, f32)>,
) -> Vec<f64> {
    let n_pairs = n_images * (n_images - 1) / 2;
    let has_features = feat_dim > 0;

    let skip_cosine = match &precomputed_dist {
        Some((_, w)) => !has_features || *w >= 1.0,
        None => !has_features,
    };

    if skip_cosine {
        let (precomp, _) = precomputed_dist.expect("skip_cosine implies precomputed");
        eprintln!(
            "  Using precomputed distance matrix only ({} pairs)",
            precomp.len()
        );
        return precomp;
    }

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
        eprintln!(
            "  Blending precomputed (weight={}) with cosine distances...",
            weight
        );
        let w = weight as f64;
        dist.par_iter_mut()
            .zip(precomp.par_iter())
            .for_each(|(d, &p)| {
                *d = w * p + (1.0 - w) * *d;
            });
    }

    eprintln!("  Distances computed.");
    dist
}

// ── Block 2: pre-merge confirmed groups ──────────────────────────────────────
//
// Simulates merges of confirmed groups via Lance-Williams so the rest of the
// pipeline sees them as already-clustered. After merging, we install sentinel
// barriers (GROUP_BARRIER) between confirmed groups so they can't be merged
// with each other, and apply user cannot-link / group-lock constraints.
#[allow(clippy::too_many_arguments)]
fn premerge_groups(
    dist: &mut [f64],
    n_images: usize,
    groups: &[LoadedGroup],
    linkage: Linkage,
    size: &mut [f64],
    active_indices: &mut Vec<usize>,
    cannot_link: &[(usize, String)],
    locked_groups: &[String],
) -> Vec<MergeStep> {
    let n_groups = groups.len();

    // Pre-compute sorted members for each group.
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
            let merge_dist = get_dist(dist, x, y, n_images);

            let nx = size[x];
            let ny = size[y];
            let new_size = nx + ny;

            for &i in active_indices.iter() {
                if i == x || i == y {
                    continue;
                }
                let ni = size[i];
                let d_ix = get_dist(dist, i, x, n_images);
                let d_iy = get_dist(dist, i, y, n_images);
                let d_new = lance_williams(linkage, d_ix, d_iy, merge_dist, ni, nx, ny);
                set_dist(dist, i, y, n_images, d_new);
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
    let group_reps: Vec<usize> = group_sorted.iter().map(|m| *m.last().unwrap()).collect();
    for i in 0..group_reps.len() {
        for j in (i + 1)..group_reps.len() {
            set_dist(dist, group_reps[i], group_reps[j], n_images, GROUP_BARRIER);
        }
    }

    // ── Apply user cannot-link / group-lock constraints ───────────────────
    if !cannot_link.is_empty() || !locked_groups.is_empty() {
        // group_id → rep_idx map: group_sorted parallels groups.iter().filter(...);
        // zip the filtered iterator with group_reps to recover the IDs.
        let group_id_to_rep: HashMap<&str, usize> = groups
            .iter()
            .filter(|g| g.member_indices.len() >= 2)
            .zip(group_reps.iter())
            .map(|(g, &rep)| (g.id.as_str(), rep))
            .collect();

        let mut applied_cl = 0usize;
        for (img_idx, group_id) in cannot_link {
            let Some(&rep_idx) = group_id_to_rep.get(group_id.as_str()) else {
                continue; // group no longer exists
            };
            if *img_idx == rep_idx {
                continue; // image is itself the rep — nothing sensible to do
            }
            set_dist(dist, *img_idx, rep_idx, n_images, GROUP_BARRIER);
            applied_cl += 1;
        }
        if applied_cl > 0 {
            eprintln!("  Applied {} cannot-link barriers", applied_cl);
        }

        let mut applied_lock = 0usize;
        for group_id in locked_groups {
            let Some(&rep_idx) = group_id_to_rep.get(group_id.as_str()) else {
                continue;
            };
            for &i in active_indices.iter() {
                if i == rep_idx {
                    continue;
                }
                set_dist(dist, i, rep_idx, n_images, GROUP_BARRIER);
            }
            applied_lock += 1;
        }
        if applied_lock > 0 {
            eprintln!("  Applied {} group-lock barriers", applied_lock);
        }
    }

    pre_merge_steps
}

// ── Block 3: NNC (nearest-neighbor chain) main loop ──────────────────────────
//
// Runs Murtagh's nearest-neighbor-chain algorithm with Lance-Williams updates.
// active_indices shrinks monotonically; the NN scan iterates only that list.
fn nnc_loop(
    dist: &mut [f64],
    n_images: usize,
    linkage: Linkage,
    size: &mut [f64],
    active_indices: &mut Vec<usize>,
) -> Vec<MergeStep> {
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
                current_min = get_dist(dist, x, y, n_images);
            } else {
                y = usize::MAX; // sentinel — overwritten on first valid candidate
                current_min = f64::MAX;
            }

            // Scan active clusters for the nearest neighbor of x.
            for &i in active_indices.iter() {
                if i == x {
                    continue;
                }
                let d = get_dist(dist, x, i, n_images);
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

                for &i in active_indices.iter() {
                    if i == x || i == y {
                        continue;
                    }
                    let ni = size[i];
                    let d_ix = get_dist(dist, i, x, n_images);
                    let d_iy = get_dist(dist, i, y, n_images);
                    let d_new = lance_williams(linkage, d_ix, d_iy, current_min, ni, nx, ny);
                    set_dist(dist, i, y, n_images, d_new);
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

    merge_steps
}
