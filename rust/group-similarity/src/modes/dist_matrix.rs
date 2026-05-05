use rayon::prelude::*;
use std::cell::RefCell;

use crate::patches::Patches;
use crate::score::patch_match_score;

/// Compute the full pairwise patch-distance matrix and write it to disk as a
/// condensed upper-triangle binary: `u64 LE n_images` header followed by
/// `n_pairs * f64` little-endian distances. The output is consumed by
/// cluster-tool's `--dist-matrix` flag.
pub(crate) fn run(patches: &Patches, output: &str) {
    assert!(
        !output.is_empty(),
        "--output is required for dist-matrix mode"
    );

    let n_images = patches.n_images;
    let stride_image = patches.stride_image;
    let n_patches = patches.n_patches;
    let patch_dim = patches.patch_dim;
    let patches_flat: &[f32] = &patches.data;

    eprintln!(
        "Computing full pairwise patch distance matrix for {} images...",
        n_images
    );

    let n_pairs = n_images * (n_images - 1) / 2;
    eprintln!(
        "  {} pairs, output size: {:.0} MB",
        n_pairs,
        n_pairs as f64 * 8.0 / 1e6
    );

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
    thread_local! {
        static SIM_BUF2: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    }

    let t0 = std::time::Instant::now();
    row_slices.par_iter_mut().for_each(|(i, slice)| {
        let i = *i;
        SIM_BUF2.with(|buf| {
            let mut buf = buf.borrow_mut();
            for (k, slot) in slice.iter_mut().enumerate() {
                let j = i + 1 + k;
                let sim = patch_match_score(
                    patches_flat,
                    stride_image,
                    n_patches,
                    patch_dim,
                    i,
                    j,
                    &mut buf,
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
            eprintln!(
                "  row {}/{} ({:.0}%) - {:.0} pairs/s - ETA {:.0}s",
                i,
                n_images,
                done_pairs as f64 / n_pairs as f64 * 100.0,
                rate,
                eta
            );
        }
    });
    eprintln!(
        "  Distance matrix computed in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    // Write as binary: header (u64 LE n_images) + flat f64 array
    use std::io::Write;
    let file = std::fs::File::create(output).expect("create output file");
    let mut w = std::io::BufWriter::new(file);
    w.write_all(&(n_images as u64).to_le_bytes())
        .expect("write header");
    // Write f64 values as raw bytes
    let byte_slice =
        unsafe { std::slice::from_raw_parts(dist.as_ptr() as *const u8, dist.len() * 8) };
    w.write_all(byte_slice).expect("write dist data");
    w.flush().expect("flush");

    eprintln!("  Saved to {}", output);
}
