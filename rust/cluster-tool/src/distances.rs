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
