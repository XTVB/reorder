use serde::Deserialize;
use std::collections::HashMap;

pub(crate) use reorder_common::LoadedGroup;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CannotLinkInput {
    pub image_filename: String,
    pub group_id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LockedGroupInput {
    pub group_id: String,
}

pub(crate) fn load_groups(
    groups_path: &str,
    fname_to_idx: &HashMap<&str, usize>,
) -> Vec<LoadedGroup> {
    reorder_common::load_groups(groups_path, |f| fname_to_idx.get(f).copied())
}

/// Parse a JSON array file as `Vec<T>`, returning an empty vec on missing
/// path, unreadable file, or parse failure (all treated as "no constraints").
fn load_json_array<T: serde::de::DeserializeOwned>(path: &str) -> Vec<T> {
    if path.is_empty() {
        return vec![];
    }
    let Ok(content) = std::fs::read_to_string(path) else {
        return vec![];
    };
    serde_json::from_str(&content).unwrap_or_default()
}

/// Resolve cannot-link pairs to indices, dropping entries with unknown filenames.
pub(crate) fn load_cannot_link_pairs(
    path: &str,
    fname_to_idx: &HashMap<&str, usize>,
) -> Vec<(usize, String)> {
    load_json_array::<CannotLinkInput>(path)
        .into_iter()
        .filter_map(|c| {
            fname_to_idx
                .get(c.image_filename.as_str())
                .map(|&idx| (idx, c.group_id))
        })
        .collect()
}

pub(crate) fn load_locked_group_ids(path: &str) -> Vec<String> {
    load_json_array::<LockedGroupInput>(path)
        .into_iter()
        .map(|l| l.group_id)
        .collect()
}
