use byteorder::{LittleEndian, WriteBytesExt};
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::io::LoadedGroup;
use crate::linkage::MergeStep;

// ── Output JSON schema ───────────────────────────────────────────────────────

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OutputCluster {
    pub id: String,
    pub images: Vec<String>,
    pub confirmed_group: Option<ConfirmedGroupInfo>,
}

#[derive(Serialize)]
pub(crate) struct ConfirmedGroupInfo {
    pub id: String,
    pub name: String,
    pub images: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Output {
    pub clusters: Vec<OutputCluster>,
    pub n_clusters: usize,
    pub tree_path: String,
}

// ── Tree cutting ─────────────────────────────────────────────────────────────

pub(crate) fn cut_tree(
    merge_steps: &[MergeStep], // already sorted from main()
    n_images: usize,
    n_after_premerge: usize,
    n_clusters: usize,
    n_pre_merges: usize,
    n_groups: usize,
) -> Vec<u32> {
    // Union-find over original image indices.
    // The input steps are already sorted by distance (done in main() before
    // saving the tree), so we don't re-sort here.
    let mut parent = vec![0u32; n_images];
    for i in 0..n_images {
        parent[i] = i as u32;
    }

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let p = parent[x as usize];
            parent[x as usize] = parent[p as usize];
            x = p;
        }
        x
    }

    // Apply all pre-merge steps (first n_pre_merges entries — forced)
    for step in merge_steps.iter().take(n_pre_merges) {
        let ra = find(&mut parent, step.cluster_a);
        let rb = find(&mut parent, step.cluster_b);
        if ra != rb {
            parent[ra as usize] = rb;
        }
    }

    // Main steps are already sorted by distance in the input slice.
    // Apply sorted main steps until we reach n_clusters.
    // Never go below n_groups clusters — confirmed groups must stay separate.
    let min_clusters = n_clusters.max(n_groups);
    let main_merges_needed = if min_clusters >= n_after_premerge {
        0
    } else {
        n_after_premerge - min_clusters
    };

    for step in merge_steps[n_pre_merges..]
        .iter()
        .take(main_merges_needed)
    {
        let ra = find(&mut parent, step.cluster_a);
        let rb = find(&mut parent, step.cluster_b);
        if ra != rb {
            parent[ra as usize] = rb;
        }
    }

    // Get cluster label for each image
    let roots: Vec<u32> = (0..n_images).map(|i| find(&mut parent, i as u32)).collect();

    // Renumber contiguously
    let mut seen: HashMap<u32, u32> = HashMap::new();
    let mut next_label = 0u32;
    roots
        .iter()
        .map(|&r| {
            *seen.entry(r).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            })
        })
        .collect()
}

// ── Linkage tree I/O ─────────────────────────────────────────────────────────

pub(crate) fn save_linkage_tree(
    steps: &[MergeStep],
    n_images: usize,
    n_pre_merges: usize,
    n_groups: usize,
    path: &str,
) {
    let file = File::create(path).expect("Failed to create tree file");
    let mut w = BufWriter::new(file);

    // Header: n_images, n_pre_merges, n_groups, n_total_steps
    w.write_u32::<LittleEndian>(n_images as u32).unwrap();
    w.write_u32::<LittleEndian>(n_pre_merges as u32).unwrap();
    w.write_u32::<LittleEndian>(n_groups as u32).unwrap();
    w.write_u32::<LittleEndian>(steps.len() as u32).unwrap();

    for step in steps {
        w.write_u32::<LittleEndian>(step.cluster_a).unwrap();
        w.write_u32::<LittleEndian>(step.cluster_b).unwrap();
        w.write_f32::<LittleEndian>(step.distance).unwrap();
        w.write_u32::<LittleEndian>(step.new_size).unwrap();
    }
    w.flush().unwrap();
}

// ── Output construction ──────────────────────────────────────────────────────

pub(crate) fn build_output_from_image_labels(
    labels: &[u32], // one label per original image
    filenames: &[String],
    groups: &[LoadedGroup],
    n_clusters: usize,
    tree_path: &str,
) -> Output {
    // Group images by cluster label
    let mut cluster_images: HashMap<u32, Vec<usize>> = HashMap::new();
    for (img_idx, &label) in labels.iter().enumerate() {
        cluster_images.entry(label).or_default().push(img_idx);
    }

    // Build index of which images belong to which confirmed group
    let mut img_to_group: HashMap<usize, usize> = HashMap::new();
    for (gi, group) in groups.iter().enumerate() {
        for &idx in &group.member_indices {
            img_to_group.insert(idx, gi);
        }
    }

    let mut output_clusters = Vec::new();
    let mut sorted_labels: Vec<u32> = cluster_images.keys().copied().collect();
    sorted_labels.sort();

    for (ci, &label) in sorted_labels.iter().enumerate() {
        let members = &cluster_images[&label];

        let image_filenames: Vec<String> =
            members.iter().map(|&i| filenames[i].clone()).collect();

        // Check if this cluster contains any confirmed group
        let confirmed = members
            .iter()
            .find_map(|&idx| img_to_group.get(&idx))
            .map(|&gi| &groups[gi]);

        let mut sorted_filenames = image_filenames;
        sorted_filenames.sort();

        output_clusters.push(OutputCluster {
            id: format!("cluster_{}", ci),
            images: sorted_filenames,
            confirmed_group: confirmed.map(|g| ConfirmedGroupInfo {
                id: g.id.clone(),
                name: g.name.clone(),
                images: g.member_filenames.clone(),
            }),
        });
    }

    output_clusters.sort_by(|a, b| b.images.len().cmp(&a.images.len()));

    Output {
        clusters: output_clusters,
        n_clusters,
        tree_path: tree_path.to_string(),
    }
}
