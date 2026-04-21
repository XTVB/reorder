use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Compute pairwise group similarity using DINOv3 patch matching")]
struct Cli {
    /// Path to hash-keyed patches cache .npy (dinov3_patches_hash_cache.npy)
    #[arg(long)]
    patches_cache: String,

    /// Path to content_hashes.json (filename → content hash)
    #[arg(long)]
    content_hashes: String,

    /// Path to dinov3_patches_hashes.json (hash list in patches cache row order)
    #[arg(long)]
    patches_hashes: String,

    /// Path to .reorder-groups.json
    #[arg(long)]
    groups: String,

    /// Minimum patch_median similarity to include in output (0.0-1.0)
    #[arg(long, default_value_t = 0.0)]
    min_score: f32,

    /// Skip pairs whose combined image count exceeds this value (0 = no limit)
    #[arg(long, default_value_t = 0)]
    max_combined_size: usize,

    /// Mode: "merge-suggestions" (default) or "dist-matrix"
    #[arg(long, default_value = "merge-suggestions")]
    mode: String,

    /// Output path for condensed distance matrix binary (dist-matrix mode only)
    #[arg(long, default_value = "")]
    output: String,
}

// ── Data types ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ReorderGroup {
    id: String,
    #[allow(dead_code)]
    name: String,
    images: Vec<String>,
}

struct Group {
    id: String,
    indices: Vec<usize>,
    filenames: Vec<String>,
}

#[derive(Serialize)]
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

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    // ── Load content_hashes.json → sorted filenames + hash lookup ──────
    let content_hashes: HashMap<String, String> = {
        let content = std::fs::read_to_string(&cli.content_hashes).expect("read content_hashes.json");
        serde_json::from_str(&content).expect("parse content_hashes.json")
    };
    let mut filenames: Vec<String> = content_hashes.keys().cloned().collect();
    filenames.sort();
    let n_images = filenames.len();
    let fname_to_idx: HashMap<&str, usize> =
        filenames.iter().enumerate().map(|(i, f)| (f.as_str(), i)).collect();

    // ── Load patches hash order → cache row mapping ─────────────────────
    let patches_hash_order: Vec<String> = {
        let content = std::fs::read_to_string(&cli.patches_hashes).expect("read patches_hashes.json");
        serde_json::from_str(&content).expect("parse patches_hashes.json")
    };
    let patch_hash_to_row: HashMap<&str, usize> = patches_hash_order
        .iter()
        .enumerate()
        .map(|(i, h)| (h.as_str(), i))
        .collect();

    // Build filename → patches cache row mapping
    let fname_to_patch_row: Vec<usize> = filenames
        .iter()
        .map(|f| {
            let hash = content_hashes.get(f).unwrap_or_else(|| panic!("No hash for {}", f));
            *patch_hash_to_row.get(hash.as_str())
                .unwrap_or_else(|| panic!("Hash {} (file {}) not in patches cache — re-run extraction with --required dinov3", hash, f))
        })
        .collect();

    // ── Load DINOv3 patch tokens ─────────────────────────────────────────
    // Shape: [N_cache, N_patches, patch_dim], dtype: float32, L2-normalized per patch
    eprintln!("Loading DINOv3 patches from {}...", cli.patches_cache);
    let npy_bytes = std::fs::read(&cli.patches_cache).expect("read patches cache file");

    // Parse .npy header
    assert!(npy_bytes.len() > 10, "npy file too small");
    assert_eq!(&npy_bytes[..6], b"\x93NUMPY", "Invalid npy magic");
    let header_len = u16::from_le_bytes([npy_bytes[8], npy_bytes[9]]) as usize;
    let data_start = 10 + header_len;
    let header_str = std::str::from_utf8(&npy_bytes[10..data_start])
        .expect("header not utf8")
        .trim();

    // Parse shape from header (e.g. "'shape': (5195, 49, 768)")
    let shape_start = header_str.find("'shape': (").expect("no shape in header") + 10;
    let shape_end = header_str[shape_start..].find(')').expect("no shape close") + shape_start;
    let shape_str = &header_str[shape_start..shape_end];
    let shape_dims: Vec<usize> = shape_str
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("bad shape dim"))
        .collect();
    assert_eq!(shape_dims.len(), 3, "Expected 3D array");
    let n_cache_entries = shape_dims[0];
    let n_patches = shape_dims[1];
    let patch_dim = shape_dims[2];
    assert!(n_cache_entries >= patches_hash_order.len(),
        "Patches cache has {} entries but hash order has {}",
        n_cache_entries, patches_hash_order.len());

    // Reinterpret data bytes as f32 slice, reindex to filename order, then drop the original
    let stride_image = n_patches * patch_dim;
    let mut patches_reindexed: Vec<f32> = vec![0.0; n_images * stride_image];
    {
        let data_bytes = &npy_bytes[data_start..];
        let n_floats = data_bytes.len() / 4;
        let cache_flat: &[f32] = unsafe {
            std::slice::from_raw_parts(data_bytes.as_ptr() as *const f32, n_floats)
        };
        for (i, &cache_row) in fname_to_patch_row.iter().enumerate() {
            let src = &cache_flat[cache_row * stride_image..(cache_row + 1) * stride_image];
            patches_reindexed[i * stride_image..(i + 1) * stride_image].copy_from_slice(src);
        }
    }
    drop(npy_bytes); // free ~1GB original buffer now that reindexing is done
    let patches_flat: &[f32] = &patches_reindexed;
    eprintln!(
        "Loaded {} images × {} patches × {}d = {:.1} GB (reindexed from {} cache entries)",
        n_images, n_patches, patch_dim,
        (n_images * stride_image) as f64 * 4.0 / 1e9,
        n_cache_entries,
    );

    // ── dist-matrix mode: compute full pairwise distance matrix ────────
    if cli.mode == "dist-matrix" {
        assert!(!cli.output.is_empty(), "--output is required for dist-matrix mode");
        eprintln!("Computing full pairwise patch distance matrix for {} images...", n_images);

        let n_pairs = n_images * (n_images - 1) / 2;
        eprintln!("  {} pairs, output size: {:.0} MB", n_pairs, n_pairs as f64 * 8.0 / 1e6);

        // Allocate condensed distance matrix (upper triangle, f64)
        let mut dist: Vec<f64> = vec![0.0f64; n_pairs];

        // Build (row_index, &mut slice) pairs for parallel write
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

        // Thread-local sim buffer
        use std::cell::RefCell;
        thread_local! {
            static SIM_BUF2: RefCell<Vec<f32>> = RefCell::new(Vec::new());
        }

        let t0 = std::time::Instant::now();
        row_slices.par_iter_mut().for_each(|(i, slice)| {
            let i = *i;
            SIM_BUF2.with(|buf| {
                let mut buf = buf.borrow_mut();
                for (k, slot) in slice.iter_mut().enumerate() {
                    let j = i + 1 + k;
                    let sim = patch_match_score(
                        patches_flat, stride_image, n_patches, patch_dim, i, j, &mut buf,
                    );
                    // Convert similarity [0,1] to distance [0,1] (1 - sim)
                    *slot = (1.0 - sim as f64).max(0.0);
                }
            });
            if i % 500 == 0 && i > 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                let done_pairs: usize = (0..i).map(|r| n_images - r - 1).sum();
                let rate = done_pairs as f64 / elapsed;
                let eta = (n_pairs - done_pairs) as f64 / rate;
                eprintln!("  row {}/{} ({:.0}%) - {:.0} pairs/s - ETA {:.0}s",
                    i, n_images, done_pairs as f64 / n_pairs as f64 * 100.0, rate, eta);
            }
        });
        eprintln!("  Distance matrix computed in {:.1}s", t0.elapsed().as_secs_f64());

        // Write as binary: header (u64 LE n_images) + flat f64 array
        use std::io::Write;
        let file = std::fs::File::create(&cli.output).expect("create output file");
        let mut w = std::io::BufWriter::new(file);
        w.write_all(&(n_images as u64).to_le_bytes()).expect("write header");
        // Write f64 values as raw bytes
        let byte_slice = unsafe {
            std::slice::from_raw_parts(dist.as_ptr() as *const u8, dist.len() * 8)
        };
        w.write_all(byte_slice).expect("write dist data");
        w.flush().expect("flush");

        eprintln!("  Saved to {}", cli.output);
        return;
    }

    // ── Load groups ──────────────────────────────────────────────────────
    let raw_groups: Vec<ReorderGroup> =
        serde_json::from_str(&std::fs::read_to_string(&cli.groups).expect("read groups"))
            .expect("parse groups");

    let groups: Vec<Group> = raw_groups
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
            Some(Group { id: g.id, indices, filenames: fnames })
        })
        .collect();
    let n_groups = groups.len();
    eprintln!("Loaded {} groups (2+ images)", n_groups);

    if n_groups < 2 {
        let empty: Vec<GroupPairResult> = vec![];
        serde_json::to_writer(std::io::stdout().lock(), &empty).unwrap();
        return;
    }

    // ── Compute patch match scores for all group pairs (parallel) ────────
    let n_full_pairs = n_groups * (n_groups - 1) / 2;
    let max_combined = cli.max_combined_size;
    let pair_indices: Vec<(usize, usize)> = (0..n_groups)
        .flat_map(|i| ((i + 1)..n_groups).map(move |j| (i, j)))
        .filter(|&(i, j)| {
            max_combined == 0
                || groups[i].indices.len() + groups[j].indices.len() <= max_combined
        })
        .collect();
    let n_total_pairs = pair_indices.len();
    if max_combined > 0 {
        eprintln!(
            "Computing patch match scores for {}/{} group pairs (max combined size = {})...",
            n_total_pairs, n_full_pairs, max_combined,
        );
    } else {
        eprintln!("Computing patch match scores for {} group pairs...", n_total_pairs);
    }

    // Thread-local similarity buffer to avoid per-call allocation
    use std::cell::RefCell;
    thread_local! {
        static SIM_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    }

    let done_counter = AtomicUsize::new(0);
    let t0 = std::time::Instant::now();
    let progress_interval = (n_total_pairs / 50).max(1);

    let results: Vec<GroupPairResult> = pair_indices
        .par_iter()
        .filter_map(|&(gi, gj)| {
            let a_indices = &groups[gi].indices;
            let b_indices = &groups[gj].indices;
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
                            patches_flat, stride_image, n_patches, patch_dim, ia, ib,
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
                let eta = if done > 0 { elapsed / done as f64 * (n_total_pairs - done) as f64 } else { 0.0 };
                eprintln!("progress: {}/{} ({:.0}%) - ETA {:.0}s", done, n_total_pairs, pct, eta);
            }

            scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let n = scores.len();
            let patch_median = scores[n / 2];
            let patch_p75 = scores[n * 3 / 4];

            if patch_median < cli.min_score {
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
                    groups[gi].filenames[best_a].clone(),
                    groups[gj].filenames[best_b].clone(),
                ),
            })
        })
        .collect();

    let mut sorted = results;
    sorted.sort_by(|a, b| b.patch_median.partial_cmp(&a.patch_median).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("Output {} group pairs", sorted.len());
    serde_json::to_writer(std::io::stdout().lock(), &sorted).expect("write JSON");
}

// ── Apple Accelerate BLAS FFI ─────────────────────────────────────────────────

#[allow(non_camel_case_types)]
type CBLAS_ORDER = i32;
#[allow(non_camel_case_types)]
type CBLAS_TRANSPOSE = i32;
const CBLAS_ROW_MAJOR: CBLAS_ORDER = 101;
const CBLAS_NO_TRANS: CBLAS_TRANSPOSE = 111;
const CBLAS_TRANS: CBLAS_TRANSPOSE = 112;

unsafe extern "C" {
    fn cblas_sgemm(
        order: CBLAS_ORDER,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: i32, n: i32, k: i32,
        alpha: f32,
        a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32,
        c: *mut f32, ldc: i32,
    );
}

// ── Patch match score ────────────────────────────────────────────────────────
//
// For two images A and B, each with N_PATCHES patch vectors (L2-normalized):
//   1. Compute similarity matrix S = A × B^T using BLAS sgemm (NEON-accelerated)
//   2. For each row (patch in A), take max → best match in B
//   3. For each col (patch in B), take max → best match in A
//   4. Return the mean of all best-match similarities
//
// Uses a thread-local buffer for the similarity matrix to avoid allocation.

fn patch_match_score(
    patches: &[f32],
    stride_image: usize,
    n_patches: usize,
    dim: usize,
    img_a: usize,
    img_b: usize,
    sim_buf: &mut Vec<f32>,
) -> f32 {
    let m = n_patches as i32;
    let k = dim as i32;

    let a_ptr = &patches[img_a * stride_image] as *const f32;
    let b_ptr = &patches[img_b * stride_image] as *const f32;

    // S = A × B^T, shape [m, m]
    sim_buf.resize((n_patches * n_patches) as usize, 0.0);
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS, CBLAS_TRANS,
            m, m, k,
            1.0,
            a_ptr, k,
            b_ptr, k,
            0.0,
            sim_buf.as_mut_ptr(), m,
        );
    }

    // Single pass: accumulate row maxes (A→B) and track col maxes (B→A) in parallel.
    use std::cell::RefCell;
    thread_local! {
        static COL_MAX: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    }
    COL_MAX.with(|cm| {
        let mut col_max = cm.borrow_mut();
        col_max.clear();
        col_max.resize(n_patches, f32::MIN);
        let mut row_sum = 0.0f32;
        for row in 0..n_patches {
            let mut row_best = f32::MIN;
            let base = row * n_patches;
            for col in 0..n_patches {
                let v = sim_buf[base + col];
                if v > row_best { row_best = v; }
                if v > col_max[col] { col_max[col] = v; }
            }
            row_sum += row_best;
        }
        let col_sum: f32 = col_max.iter().sum();
        (row_sum + col_sum) / (2.0 * n_patches as f32)
    })
}
