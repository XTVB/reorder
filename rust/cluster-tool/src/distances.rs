use ndarray::Array2;

// ── Flat condensed distance matrix ───────────────────────────────────────────
//
// We store the upper-triangle of the n×n distance matrix as a flat Vec<f64>
// using the standard condensed index formula. This replaces both the old
// `img_dist` array and the `dist_row: Vec<Vec<f64>>` — eliminating the O(n²)
// copy and halving peak memory.
//
// condensed_idx(i, j, n) gives the offset for i < j.

#[inline(always)]
pub(crate) fn condensed_idx(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i < j, "condensed_idx requires i < j, got i={} j={}", i, j);
    // Row i has (n - i - 1) entries, starting at offset: i*n - i*(i+1)/2
    i * n - i * (i + 1) / 2 + j - i - 1
}

#[inline(always)]
pub(crate) fn get_dist(dist: &[f64], i: usize, j: usize, n: usize) -> f64 {
    if i < j {
        dist[condensed_idx(i, j, n)]
    } else {
        dist[condensed_idx(j, i, n)]
    }
}

#[inline(always)]
pub(crate) fn set_dist(dist: &mut [f64], i: usize, j: usize, n: usize, val: f64) {
    if i < j {
        dist[condensed_idx(i, j, n)] = val;
    } else {
        dist[condensed_idx(j, i, n)] = val;
    }
}

// ── Feature combination ──────────────────────────────────────────────────────

/// Returns a flat row-major Vec<f32> and the feature dimension.
/// Using a flat Vec instead of ndarray removes ndarray indexing overhead in
/// the hot distance-computation loop.
/// Build combined feature vector from multiple embedding arrays.
/// Each entry is (array, weight, needs_l2_norm). Arrays already L2-normalized
/// from Python have needs_l2_norm=false; color features need per-row normalization.
pub(crate) fn build_combined_features_flat(
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
