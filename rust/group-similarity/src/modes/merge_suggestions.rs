use reorder_common::LoadedGroup;
use std::cell::RefCell;

use crate::modes::group_pairs::compute_group_pairs;
use crate::patches::Patches;
use crate::score::patch_match_score;

/// Compute per-group-pair DINOv3 patch-match scores and emit ranked JSON to
/// stdout. Used by the Merge Suggestions UI to surface candidate groups likely
/// to belong together.
pub(crate) fn run(
    patches: &Patches,
    groups_path: &str,
    min_score: f32,
    max_combined_size: usize,
    rejected_pairs_path: &str,
) {
    let stride_image = patches.stride_image;
    let n_patches = patches.n_patches;
    let patch_dim = patches.patch_dim;
    let patches_flat: &[f32] = &patches.data;

    let groups: Vec<LoadedGroup> =
        reorder_common::load_groups(groups_path, |f| patches.fname_to_idx.get(f).copied());

    // Thread-local similarity buffer to avoid per-call allocation.
    thread_local! {
        static SIM_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    }

    let results = compute_group_pairs(
        &groups,
        min_score,
        max_combined_size,
        rejected_pairs_path,
        "patch match scores",
        |ia, ib| {
            SIM_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                patch_match_score(patches_flat, stride_image, n_patches, patch_dim, ia, ib, &mut buf)
            })
        },
    );

    serde_json::to_writer(std::io::stdout().lock(), &results).expect("write JSON");
}
