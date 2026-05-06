//! Shared types and parsing helpers for the reorder Rust binaries.
//!
//! Both `cluster-tool` and `group-similarity` parse the same on-disk
//! `groups.json` and `content_hashes.json` formats; this crate centralises
//! those types so the two binaries can't drift.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// On-disk shape of a confirmed reorder group as written by the TS server
/// to `.reorder-groups.json`.
#[derive(Debug, Deserialize)]
pub struct ReorderGroup {
    pub id: String,
    pub name: String,
    pub images: Vec<String>,
}

/// A confirmed group resolved against a known image set: each filename has
/// been mapped to its index. Groups with no surviving members are dropped;
/// singletons are kept so they're treated as confirmed clusters of size 1
/// (sealed against merging with other confirmed groups).
#[derive(Debug)]
pub struct LoadedGroup {
    pub id: String,
    pub name: String,
    pub member_indices: Vec<usize>,
    pub member_filenames: Vec<String>,
}

/// Load `groups.json` and resolve each group's filenames against the caller's
/// filename → index mapping. Groups with no surviving members are dropped.
///
/// `fname_to_idx` is a closure rather than a borrow of a specific HashMap so
/// callers can use either `HashMap<&str, usize>` or `HashMap<String, usize>`
/// without forcing a particular ownership shape.
///
/// Missing path, unreadable file, or malformed JSON all return an empty Vec
/// (matching the prior cluster-tool behaviour: groups are an optional input).
pub fn load_groups<F>(path: &str, fname_to_idx: F) -> Vec<LoadedGroup>
where
    F: Fn(&str) -> Option<usize>,
{
    if path.is_empty() {
        return vec![];
    }
    let Ok(content) = std::fs::read_to_string(path) else {
        return vec![];
    };
    let Ok(raw_groups): Result<Vec<ReorderGroup>, _> = serde_json::from_str(&content) else {
        return vec![];
    };

    raw_groups
        .into_iter()
        .filter_map(|g| {
            let mut indices = Vec::new();
            let mut fnames = Vec::new();
            for f in &g.images {
                if let Some(idx) = fname_to_idx(f.as_str()) {
                    indices.push(idx);
                    fnames.push(f.clone());
                }
            }
            if indices.is_empty() {
                return None;
            }
            Some(LoadedGroup {
                id: g.id,
                name: g.name,
                member_indices: indices,
                member_filenames: fnames,
            })
        })
        .collect()
}

/// Read `content_hashes.json` (a `{filename: hash}` map) and return the
/// sorted filenames vector together with a filename → index lookup map.
///
/// The two binaries both build this exact structure as the canonical
/// "image set" for downstream indexing. Panics if the file is missing or
/// malformed — both binaries treat this as an unrecoverable input error.
pub fn load_content_hashes_sorted(
    path: &Path,
) -> (HashMap<String, String>, Vec<String>, HashMap<String, usize>) {
    let path_str = path.display();
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Missing content_hashes.json: {}", path_str));
    let content_hashes: HashMap<String, String> = serde_json::from_str(&content)
        .unwrap_or_else(|_| panic!("Invalid content_hashes.json: {}", path_str));
    let mut filenames: Vec<String> = content_hashes.keys().cloned().collect();
    filenames.sort();
    let fname_to_idx: HashMap<String, usize> = filenames
        .iter()
        .enumerate()
        .map(|(i, f)| (f.clone(), i))
        .collect();
    (content_hashes, filenames, fname_to_idx)
}
