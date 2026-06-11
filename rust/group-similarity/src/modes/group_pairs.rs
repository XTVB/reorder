use rayon::prelude::*;
use reorder_common::LoadedGroup;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Shared group-pair scoring used by both merge-suggestions modes (DINOv3
/// patches and embedding blend). The only difference between the modes is how a
/// single image pair is scored, so that is injected as a closure; everything
/// else — pair enumeration, rejected/oversize filtering, median/p75/best
/// reduction, ranking, and the JSON wire shape — is identical.

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RejectedPairInput {
    group_a: String,
    group_b: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GroupPairResult {
    group_a: String,
    group_b: String,
    size_a: usize,
    size_b: usize,
    /// Median of per-image-pair similarity scores (primary metric)
    patch_median: f32,
    /// 75th percentile (lower end — worst matches)
    patch_p75: f32,
    /// Best image pair's similarity score
    patch_best: f32,
    closest_pair: (String, String),
}

/// Load the rejected-pairs JSON file and return an unordered-pair lookup set.
/// Returns an empty set when the path is empty or the file can't be read —
/// rejection is a hint, not a correctness requirement.
fn load_rejected_pairs(path: &str) -> HashSet<(String, String)> {
    if path.is_empty() {
        return HashSet::new();
    }
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Could not read --rejected-pairs file {}: {}", path, e);
            return HashSet::new();
        }
    };
    let entries: Vec<RejectedPairInput> = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Could not parse --rejected-pairs file {}: {}", path, e);
            return HashSet::new();
        }
    };
    let mut set = HashSet::with_capacity(entries.len());
    for e in entries {
        let (a, b) = if e.group_a <= e.group_b {
            (e.group_a, e.group_b)
        } else {
            (e.group_b, e.group_a)
        };
        set.insert((a, b));
    }
    set
}

/// Compute per-group-pair similarity scores and return them ranked descending by
/// median score. `score_pair(img_a, img_b)` returns the similarity for one image
/// pair (global image indices); `metric_label` is used only for progress logs.
pub(crate) fn compute_group_pairs<F>(
    groups: &[LoadedGroup],
    min_score: f32,
    max_combined_size: usize,
    rejected_pairs_path: &str,
    metric_label: &str,
    score_pair: F,
) -> Vec<GroupPairResult>
where
    F: Fn(usize, usize) -> f32 + Sync,
{
    let rejected = load_rejected_pairs(rejected_pairs_path);
    if !rejected.is_empty() {
        eprintln!("Skipping {} rejected group pair(s)", rejected.len());
    }

    let n_groups = groups.len();
    eprintln!("Loaded {} confirmed groups", n_groups);
    if n_groups < 2 {
        return vec![];
    }

    let n_full_pairs = n_groups * (n_groups - 1) / 2;
    // Pre-resolve rejected id-pairs to index-pairs (i < j) so the per-pair
    // filter is a hash lookup with no allocation.
    let rejected_idx: HashSet<(usize, usize)> = if rejected.is_empty() {
        HashSet::new()
    } else {
        let id_to_idx: HashMap<&str, usize> = groups
            .iter()
            .enumerate()
            .map(|(i, g)| (g.id.as_str(), i))
            .collect();
        let mut set = HashSet::with_capacity(rejected.len());
        for (a, b) in &rejected {
            if let (Some(&i), Some(&j)) = (id_to_idx.get(a.as_str()), id_to_idx.get(b.as_str())) {
                let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
                set.insert((lo, hi));
            }
        }
        set
    };
    let pair_indices: Vec<(usize, usize)> = (0..n_groups)
        .flat_map(|i| ((i + 1)..n_groups).map(move |j| (i, j)))
        .filter(|&(i, j)| {
            if rejected_idx.contains(&(i, j)) {
                return false;
            }
            max_combined_size == 0
                || groups[i].member_indices.len() + groups[j].member_indices.len()
                    <= max_combined_size
        })
        .collect();
    let n_total_pairs = pair_indices.len();
    if max_combined_size > 0 {
        eprintln!(
            "Computing {} for {}/{} group pairs (max combined size = {})...",
            metric_label, n_total_pairs, n_full_pairs, max_combined_size,
        );
    } else {
        eprintln!("Computing {} for {} group pairs...", metric_label, n_total_pairs);
    }

    let done_counter = AtomicUsize::new(0);
    let t0 = std::time::Instant::now();
    let progress_interval = (n_total_pairs / 50).max(1);

    let results: Vec<GroupPairResult> = pair_indices
        .par_iter()
        .filter_map(|&(gi, gj)| {
            let a_indices = &groups[gi].member_indices;
            let b_indices = &groups[gj].member_indices;
            let size_a = a_indices.len();
            let size_b = b_indices.len();

            let n_image_pairs = size_a * size_b;
            let mut scores = Vec::with_capacity(n_image_pairs);
            let mut best_score = f32::MIN;
            let mut best_a = 0usize;
            let mut best_b = 0usize;

            for (ai, &ia) in a_indices.iter().enumerate() {
                for (bi, &ib) in b_indices.iter().enumerate() {
                    let s = score_pair(ia, ib);
                    scores.push(s);
                    if s > best_score {
                        best_score = s;
                        best_a = ai;
                        best_b = bi;
                    }
                }
            }

            let prev = done_counter.fetch_add(1, Ordering::Relaxed);
            let done = prev + 1;
            if done % progress_interval == 0 || done == n_total_pairs {
                let pct = done as f64 / n_total_pairs as f64 * 100.0;
                let elapsed = t0.elapsed().as_secs_f64();
                let eta = if done > 0 {
                    elapsed / done as f64 * (n_total_pairs - done) as f64
                } else {
                    0.0
                };
                eprintln!(
                    "progress: {}/{} ({:.0}%) - ETA {:.0}s",
                    done, n_total_pairs, pct, eta
                );
            }

            scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let n = scores.len();
            let patch_median = scores[n / 2];
            let patch_p75 = scores[n * 3 / 4];

            if patch_median < min_score {
                return None;
            }

            Some(GroupPairResult {
                group_a: groups[gi].id.clone(),
                group_b: groups[gj].id.clone(),
                size_a,
                size_b,
                patch_median,
                patch_p75,
                patch_best: best_score,
                closest_pair: (
                    groups[gi].member_filenames[best_a].clone(),
                    groups[gj].member_filenames[best_b].clone(),
                ),
            })
        })
        .collect();

    let mut sorted = results;
    sorted.sort_by(|a, b| {
        b.patch_median
            .partial_cmp(&a.patch_median)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("Output {} group pairs", sorted.len());
    sorted
}
