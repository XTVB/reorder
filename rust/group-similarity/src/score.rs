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
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
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

pub(crate) fn patch_match_score(
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
    sim_buf.resize(n_patches * n_patches, 0.0);
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m,
            m,
            k,
            1.0,
            a_ptr,
            k,
            b_ptr,
            k,
            0.0,
            sim_buf.as_mut_ptr(),
            m,
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
                if v > row_best {
                    row_best = v;
                }
                if v > col_max[col] {
                    col_max[col] = v;
                }
            }
            row_sum += row_best;
        }
        let col_sum: f32 = col_max.iter().sum();
        (row_sum + col_sum) / (2.0 * n_patches as f32)
    })
}
