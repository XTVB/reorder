use reorder_common::LoadedGroup;

use crate::embeddings::Embeddings;
use crate::modes::group_pairs::compute_group_pairs;

/// Compute per-group-pair similarity scores from the weighted blend of CLS
/// embeddings and emit ranked JSON to stdout. Rows are L2-normalized at load
/// time, so the per-image-pair score is just a dot product (== cosine).
pub(crate) fn run(
    emb: &Embeddings,
    groups_path: &str,
    min_score: f32,
    max_combined_size: usize,
    rejected_pairs_path: &str,
) {
    let dim = emb.dim;
    let data: &[f32] = &emb.data;

    let groups: Vec<LoadedGroup> =
        reorder_common::load_groups(groups_path, |f| emb.fname_to_idx.get(f).copied());

    let results = compute_group_pairs(
        &groups,
        min_score,
        max_combined_size,
        rejected_pairs_path,
        "embedding cosine scores",
        |ia, ib| {
            let a = &data[ia * dim..][..dim];
            let b = &data[ib * dim..][..dim];
            a.iter().zip(b).map(|(&x, &y)| x * y).sum()
        },
    );

    serde_json::to_writer(std::io::stdout().lock(), &results).expect("write JSON");
}
