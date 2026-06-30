//! order-tool — batch similarity ordering ("Sort Similar") for the reorder page.
//!
//! Loads the cached per-model embeddings once, then for each ordering *job* (a
//! list of image filenames — the ungrouped set, or one per selected group)
//! builds the blended weighted-cosine distance matrix and reduces it to a 1-D
//! sequence via one of six modes (see `ordering.rs`). Jobs are independent and
//! run in parallel (rayon); the heavy matrix build parallelizes per-row.
//!
//! This is the Rust replacement for `src/cluster/image-ordering.ts` — same
//! per-model linear-weighted blend, same modes, same tie-breaks — moved off the
//! JS event loop so 4–8k-image ungrouped sorts run fast.
//!
//! Input: embedding paths + per-model `--*-weight` flags + a `--jobs` JSON file
//!   { "mode": "...", "minimalLocality"?, "stableClusters"?, "gatherMinGain"?,
//!     "jobs": [ { "id": "...", "filenames": [...] }, ... ] }
//! Output (stdout): [ { "id", "orderedIds": [...], "skipped", "clusters"?,
//!   "moved"? }, ... ]
//! Progress: `progress:`-prefixed lines on stderr (forwarded by the TS spawner).

mod ordering;

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrd};

use clap::Parser;
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use ordering::{order_by_distance_matrix, OrderParams};
use reorder_common::embeddings::{emb_specs, load_fname_to_cache_row, load_model_arrays};
use reorder_common::load_content_hashes_sorted;

#[derive(Parser, Debug)]
#[command(about = "Batch similarity ordering over cached embeddings")]
struct Args {
    #[arg(long)]
    hash_cache: String,
    #[arg(long)]
    content_hashes: String,
    #[arg(long)]
    hash_order: String,
    /// Path to the jobs JSON file.
    #[arg(long)]
    jobs: String,

    #[arg(long, default_value_t = 0.0)]
    color_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    dinov3_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    pecore_g_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    learned_proj_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    learned_proj_peg_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    learned_proj_color_weight: f32,

    /// Hard cap on images per job (non-tree modes).
    #[arg(long, default_value_t = 20000)]
    max_images: usize,
    /// Hard cap on images per job in tree mode (O(n³)).
    #[arg(long, default_value_t = 2500)]
    max_images_tree: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct JobsInput {
    mode: String,
    #[serde(default)]
    minimal_locality: Option<usize>,
    #[serde(default)]
    stable_clusters: Option<usize>,
    #[serde(default)]
    gather_min_gain: Option<f64>,
    jobs: Vec<JobSpec>,
}

#[derive(Deserialize)]
struct JobSpec {
    id: String,
    filenames: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JobResult {
    id: String,
    ordered_ids: Vec<String>,
    skipped: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    clusters: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moved: Option<usize>,
}

/// A model's subset-gathered, L2-normalized rows (f64) plus its blend weight.
struct ModelRows {
    rows: Vec<f64>,
    dim: usize,
    weight: f64,
}

fn main() {
    let args = Args::parse();

    let input_text = std::fs::read_to_string(&args.jobs)
        .unwrap_or_else(|e| panic!("read jobs file {}: {}", args.jobs, e));
    let input: JobsInput = serde_json::from_str(&input_text).expect("parse jobs JSON");

    // Cap check up front so we fail fast with a clear message (the TS caller
    // also gates, but this is the safety net for a runaway tree-mode request).
    let is_tree = input.mode == ordering::MODE_TREE;
    for j in &input.jobs {
        let n = j.filenames.len();
        if is_tree && n > args.max_images_tree {
            eprintln!(
                "Tree mode is limited to {} images ({} requested) — use Chain, Spectral or Minimal.",
                args.max_images_tree, n
            );
            std::process::exit(2);
        }
        if n > args.max_images {
            eprintln!(
                "Too many images to sort ({}, limit {}) — select a subset or group some first.",
                n, args.max_images
            );
            std::process::exit(2);
        }
    }

    // Load every positively-weighted model array once, in filename order.
    let (content_hashes, filenames, fname_to_idx) =
        load_content_hashes_sorted(Path::new(&args.content_hashes));
    let fname_to_cache_row =
        load_fname_to_cache_row(&args.hash_order, &content_hashes, &filenames);
    let specs = emb_specs(
        args.color_weight,
        args.dinov3_weight,
        args.pecore_g_weight,
        args.learned_proj_weight,
        args.learned_proj_peg_weight,
        args.learned_proj_color_weight,
    );
    assert!(
        specs.iter().any(|(_, w, _)| *w > 0.0),
        "at least one positive model weight is required"
    );
    let loaded: Vec<(Array2<f32>, f32, bool)> =
        load_model_arrays(Path::new(&args.hash_cache), &fname_to_cache_row, &specs);
    // Flat (&data, dim, weight) views; the matrix build normalizes per row, so
    // the per-model `needs_norm` flag is irrelevant here (matches image-ordering.ts).
    let models: Vec<(&[f32], usize, f64)> = loaded
        .iter()
        .map(|(a, w, _)| {
            (
                a.as_slice().expect("contiguous embedding array"),
                a.ncols(),
                *w as f64,
            )
        })
        .collect();

    let params = OrderParams {
        mode: input.mode.clone(),
        minimal_locality: input.minimal_locality,
        stable_clusters: input.stable_clusters,
        gather_min_gain: input.gather_min_gain,
    };

    let total = input.jobs.len();
    eprintln!("progress: Ordering {} job(s) ({})...", total, input.mode);

    let results: Vec<JobResult> = if total == 1 {
        // Single (usually large, e.g. ungrouped) job: stream finer progress.
        vec![process_job(
            &input.jobs[0],
            &models,
            &fname_to_idx,
            &params,
            true,
        )]
    } else {
        let done = AtomicUsize::new(0);
        let mut out: Vec<JobResult> = input
            .jobs
            .par_iter()
            .map(|job| {
                let r = process_job(job, &models, &fname_to_idx, &params, false);
                let c = done.fetch_add(1, AtomicOrd::Relaxed) + 1;
                eprintln!("progress: Ordered {}/{} groups...", c, total);
                r
            })
            .collect();
        // par_iter preserves order, but be explicit about the output contract.
        out.sort_by_key(|r| {
            input
                .jobs
                .iter()
                .position(|j| j.id == r.id)
                .unwrap_or(usize::MAX)
        });
        out
    };

    serde_json::to_writer(std::io::stdout().lock(), &results).expect("write JSON");
}

fn process_job(
    job: &JobSpec,
    models: &[(&[f32], usize, f64)],
    fname_to_idx: &std::collections::HashMap<String, usize>,
    params: &OrderParams,
    progress: bool,
) -> JobResult {
    // De-duplicate while preserving incoming order; drop filenames with no
    // embedding row (they keep their slot client-side, counted as skipped).
    let mut seen: HashSet<&str> = HashSet::new();
    let mut known_fn: Vec<String> = Vec::new();
    let mut known_idx: Vec<usize> = Vec::new();
    for f in &job.filenames {
        if !seen.insert(f.as_str()) {
            continue;
        }
        if let Some(&idx) = fname_to_idx.get(f) {
            known_fn.push(f.clone());
            known_idx.push(idx);
        }
    }
    let n = known_idx.len();
    let skipped = seen.len() - n;

    if n <= 2 {
        return JobResult {
            id: job.id.clone(),
            ordered_ids: known_fn,
            skipped,
            clusters: None,
            moved: None,
        };
    }

    if progress {
        eprintln!("progress: Building distance matrix for {} images...", n);
    }

    // Per-model: gather the subset rows and L2-normalize each (always — rows
    // sanitized at extraction may not be unit-norm), into f64 for parity.
    let total_weight: f64 = models.iter().map(|(_, _, w)| *w).sum();
    let model_rows: Vec<ModelRows> = models
        .iter()
        .map(|&(data, dim, weight)| {
            let mut rows = vec![0f64; n * dim];
            for i in 0..n {
                let src = &data[known_idx[i] * dim..][..dim];
                let mut s = 0f64;
                for &x in src {
                    s += x as f64 * x as f64;
                }
                let mut nrm = s.sqrt();
                if nrm == 0.0 {
                    nrm = 1e-10;
                }
                let out = &mut rows[i * dim..][..dim];
                for (o, &x) in out.iter_mut().zip(src) {
                    *o = x as f64 / nrm;
                }
            }
            ModelRows { rows, dim, weight }
        })
        .collect();

    // Blended cosine distance matrix, parallel per row (each thread writes its
    // own upper-triangle row segment). dist(i,j) = Σ w·max(0,1−cos)/Σw.
    let mut dist = vec![0f64; n * n];
    dist.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for (j, slot) in row.iter_mut().enumerate().take(n).skip(i + 1) {
            let mut s = 0f64;
            for m in &model_rows {
                let a = &m.rows[i * m.dim..][..m.dim];
                let b = &m.rows[j * m.dim..][..m.dim];
                let mut dot = 0f64;
                for d in 0..m.dim {
                    dot += a[d] * b[d];
                }
                s += m.weight * (1.0 - dot).max(0.0);
            }
            let raw = s / total_weight;
            *slot = if raw.is_finite() { raw } else { 2.0 };
        }
    });
    // Mirror upper triangle into the lower; diagonal stays 0.
    for i in 0..n {
        for j in (i + 1)..n {
            dist[j * n + i] = dist[i * n + j];
        }
    }

    if progress {
        eprintln!("progress: Ordering {} images ({})...", n, params.mode);
    }
    let (order, info) = order_by_distance_matrix(&dist, n, params);
    let ordered_ids = order.into_iter().map(|i| known_fn[i].clone()).collect();

    JobResult {
        id: job.id.clone(),
        ordered_ids,
        skipped,
        clusters: info.clusters,
        moved: info.moved,
    }
}
