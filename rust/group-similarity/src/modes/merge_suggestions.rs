use rayon::prelude::*;
use reorder_common::LoadedGroup;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::patches::Patches;
use crate::score::patch_match_score;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RejectedPairInput {
    group_a: String,
    group_b: String,
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

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GroupPairResult {
    group_a: String,
    group_b: String,
    size_a: usize,
    size_b: usize,
    /// Median of per-image-pair patch match scores (primary metric)
    patch_median: f32,
    /// 75th percentile (lower end — worst matches)
    patch_p75: f32,
    /// Best image pair's patch match score
    patch_best: f32,
    closest_pair: (String, String),
}

/// Compute per-group-pair patch-match scores and emit ranked JSON to stdout.
/// Used by the Merge Suggestions UI to surface candidate groups likely to
/// belong together.
pub(crate) fn run(
    patches: &Patches,
    groups_path: &str,
    min_score: f32,
    max_combined_size: usize,
    rejected_pairs_path: &str,
) {
    let rejected = load_rejected_pairs(rejected_pairs_path);
    if !rejected.is_empty() {
        eprintln!("Skipping {} rejected group pair(s)", rejected.len());
    }
    let stride_image = patches.stride_image;
    let n_patches = patches.n_patches;
    let patch_dim = patches.patch_dim;
    let patches_flat: &[f32] = &patches.data;

    // ── Load groups ──────────────────────────────────────────────────────
    let groups: Vec<LoadedGroup> = reorder_common::load_groups(groups_path, |f| {
        patches.fname_to_idx.get(f).copied()
    });
    let n_groups = groups.len();
    eprintln!("Loaded {} confirmed groups", n_groups);

    if n_groups < 2 {
        let empty: Vec<GroupPairResult> = vec![];
        serde_json::to_writer(std::io::stdout().lock(), &empty).unwrap();
        return;
    }

    // ── Compute patch match scores for all group pairs (parallel) ────────
    let n_full_pairs = n_groups * (n_groups - 1) / 2;
    let is_rejected = |i: usize, j: usize| -> bool {
        if rejected.is_empty() {
            return false;
        }
        let (a, b) = (&groups[i].id, &groups[j].id);
        let key = if a <= b {
            (a.clone(), b.clone())
        } else {
            (b.clone(), a.clone())
        };
        rejected.contains(&key)
    };
    let pair_indices: Vec<(usize, usize)> = (0..n_groups)
        .flat_map(|i| ((i + 1)..n_groups).map(move |j| (i, j)))
        .filter(|&(i, j)| {
            if is_rejected(i, j) {
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
            "Computing patch match scores for {}/{} group pairs (max combined size = {})...",
            n_total_pairs, n_full_pairs, max_combined_size,
        );
    } else {
        eprintln!(
            "Computing patch match scores for {} group pairs...",
            n_total_pairs
        );
    }

    // Thread-local similarity buffer to avoid per-call allocation
    thread_local! {
        static SIM_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
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

            // For each image pair (a, b), compute bidirectional patch match score
            let n_image_pairs = size_a * size_b;
            let mut scores = Vec::with_capacity(n_image_pairs);
            let mut best_score = f32::MIN;
            let mut best_a = 0usize;
            let mut best_b = 0usize;

            SIM_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                for (ai, &ia) in a_indices.iter().enumerate() {
                    for (bi, &ib) in b_indices.iter().enumerate() {
                        let s = patch_match_score(
                            patches_flat,
                            stride_image,
                            n_patches,
                            patch_dim,
                            ia,
                            ib,
                            &mut buf,
                        );
                        scores.push(s);
                        if s > best_score {
                            best_score = s;
                            best_a = ai;
                            best_b = bi;
                        }
                    }
                }
            });

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
    serde_json::to_writer(std::io::stdout().lock(), &sorted).expect("write JSON");
}
