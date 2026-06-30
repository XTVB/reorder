//! 1-D similarity ordering ("seriation") over a precomputed distance matrix —
//! a faithful port of `src/cluster/group-ordering.ts`'s `orderByDistanceMatrix`
//! and its six modes. Distances are a flat row-major `n×n` `f64` buffer with a
//! zero diagonal (matrix build lives in `main.rs`). All arithmetic is `f64` to
//! match the TypeScript reference bit-for-bit; tie-breaks and iteration caps are
//! preserved so the Rust output matches the JS output exactly.

/// Cosine distance tops out at 2; this matches the TS reference for non-finite
/// blended values (it never appears in the image path, but kept for parity).
pub const MODE_CHAIN: &str = "chain";
pub const MODE_TREE: &str = "tree";
pub const MODE_SPECTRAL: &str = "spectral";
pub const MODE_MINIMAL: &str = "minimal";
pub const MODE_STABLE: &str = "stable";
pub const MODE_GATHER: &str = "gather";

const MINIMAL_LOCALITY: usize = 5;
const GATHER_MIN_GAIN: f64 = 0.02;

pub struct OrderParams {
    pub mode: String,
    pub minimal_locality: Option<usize>,
    pub stable_clusters: Option<usize>,
    pub gather_min_gain: Option<f64>,
}

#[derive(Default)]
pub struct ModeInfo {
    pub clusters: Option<usize>,
    pub moved: Option<usize>,
}

#[inline(always)]
fn at(dist: &[f64], n: usize, i: usize, j: usize) -> f64 {
    dist[i * n + j]
}

/// Mode dispatch. `dist` is a symmetric flat `n×n` matrix with zero diagonal.
/// The anchor is always position 0 (the image path passes `known[0]`), matching
/// `image-ordering.ts`. Returns the ordered indices into `0..n` plus mode info.
pub fn order_by_distance_matrix(dist: &[f64], n: usize, p: &OrderParams) -> (Vec<usize>, ModeInfo) {
    let anchor = 0usize;
    if n <= 2 {
        // anchor == 0 ⇒ identity (TS reverses only when anchor == 1).
        return ((0..n).collect(), ModeInfo::default());
    }

    let mut info = ModeInfo::default();
    let order = match p.mode.as_str() {
        MODE_TREE => {
            let tree = build_average_linkage_tree(dist, n);
            orient_to_anchor(optimal_leaf_order(&tree, dist, n), anchor)
        }
        MODE_SPECTRAL => orient_to_anchor(spectral_order(dist, n), anchor),
        MODE_MINIMAL => {
            let base: Vec<usize> = (0..n).collect();
            let locality = p.minimal_locality.unwrap_or(MINIMAL_LOCALITY).max(1);
            minimal_improve(&base, dist, n, locality)
        }
        MODE_STABLE => {
            let (order, clusters) = stable_cluster_order(dist, n, p.stable_clusters);
            info.clusters = Some(clusters);
            order
        }
        MODE_GATHER => {
            let (order, moved) = gather_order(dist, n, p.gather_min_gain.unwrap_or(GATHER_MIN_GAIN));
            info.moved = Some(moved);
            order
        }
        MODE_CHAIN => two_opt(greedy_chain(dist, n, anchor), dist, n),
        // Any unrecognized mode falls back to chain (matches the TS default).
        _ => two_opt(greedy_chain(dist, n, anchor), dist, n),
    };
    (order, info)
}

/// Reverse the sequence when that moves the anchor closer to the front.
fn orient_to_anchor(mut order: Vec<usize>, anchor: usize) -> Vec<usize> {
    let idx = order.iter().position(|&x| x == anchor).unwrap_or(0);
    if idx > order.len() - 1 - idx {
        order.reverse();
    }
    order
}

// ---------------------------------------------------------------------------
// chain: greedy nearest-neighbor + 2-opt

fn greedy_chain(dist: &[f64], n: usize, start: usize) -> Vec<usize> {
    let mut visited = vec![false; n];
    visited[start] = true;
    let mut order = Vec::with_capacity(n);
    order.push(start);
    while order.len() < n {
        let last = *order.last().unwrap();
        let mut best_idx = usize::MAX;
        let mut best_d = f64::INFINITY;
        for i in 0..n {
            if visited[i] {
                continue;
            }
            let d = at(dist, n, last, i);
            if d < best_d {
                best_d = d;
                best_idx = i;
            }
        }
        visited[best_idx] = true;
        order.push(best_idx);
    }
    order
}

/// 2-opt for open paths: reverse `order[i..=j]` whenever that lowers the summed
/// adjacent distance. Position 0 (the anchor) never moves; capped at 200 passes.
fn two_opt(mut order: Vec<usize>, dist: &[f64], n: usize) -> Vec<usize> {
    let mut improved = true;
    let mut passes = 0;
    while improved && passes < 200 {
        passes += 1;
        improved = false;
        for i in 1..n.saturating_sub(1) {
            for j in (i + 1)..n {
                let before = at(dist, n, order[i - 1], order[i])
                    + if j + 1 < n {
                        at(dist, n, order[j], order[j + 1])
                    } else {
                        0.0
                    };
                let after = at(dist, n, order[i - 1], order[j])
                    + if j + 1 < n {
                        at(dist, n, order[i], order[j + 1])
                    } else {
                        0.0
                    };
                if after < before - 1e-12 {
                    order[i..=j].reverse();
                    improved = true;
                }
            }
        }
    }
    order
}

// ---------------------------------------------------------------------------
// minimal: hill-climb from the incoming order with strictly local moves

fn minimal_improve(base: &[usize], dist: &[f64], n: usize, locality: usize) -> Vec<usize> {
    let mut orig_idx = vec![0usize; n];
    for (i, &g) in base.iter().enumerate() {
        orig_idx[g] = i;
    }

    let path_cost = |o: &[usize]| -> f64 {
        let mut c = 0.0;
        for k in 0..n.saturating_sub(1) {
            c += at(dist, n, o[k], o[k + 1]);
        }
        c
    };
    let within_drift = |o: &[usize]| -> bool {
        o.iter()
            .enumerate()
            .all(|(i, &g)| (i as i64 - orig_idx[g] as i64).abs() as usize <= locality)
    };

    let mut order = base.to_vec();
    let mut cost = path_cost(&order);
    let mut improved = true;
    let mut passes = 0;
    while improved && passes < 50 {
        passes += 1;
        improved = false;

        // Short segment reversals (span ≤ locality).
        for i in 1..n.saturating_sub(1) {
            for j in (i + 1)..(i + locality).min(n) {
                let mut cand = order.clone();
                cand[i..=j].reverse();
                let c = path_cost(&cand);
                if c < cost - 1e-12 && within_drift(&cand) {
                    order = cand;
                    cost = c;
                    improved = true;
                }
            }
        }

        // Single-group relocations (≤ locality positions away).
        for pp in 1..n {
            let lo = if pp > locality { pp - locality } else { 1 };
            let hi = (n - 1).min(pp + locality);
            for q in lo..=hi {
                if q == pp {
                    continue;
                }
                let mut cand = order.clone();
                let x = cand.remove(pp);
                cand.insert(q, x);
                let c = path_cost(&cand);
                if c < cost - 1e-12 && within_drift(&cand) {
                    order = cand;
                    cost = c;
                    improved = true;
                }
            }
        }
    }
    order
}

// ---------------------------------------------------------------------------
// stable: similarity decides membership, the incoming order decides sequence

fn stable_cluster_order(dist: &[f64], n: usize, forced: Option<usize>) -> (Vec<usize>, usize) {
    let mut merges = average_linkage_merges(dist, n);
    merges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let applied: usize = match forced {
        Some(fc) if fc >= 1 => n - n.min(fc),
        _ => {
            let mut best_gap = -1.0f64;
            let mut best_idx = merges.len().saturating_sub(1);
            for i in 0..merges.len().saturating_sub(1) {
                let gap = merges[i + 1].2 - merges[i].2;
                if gap > best_gap {
                    best_gap = gap;
                    best_idx = i;
                }
            }
            best_idx + 1
        }
    };

    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for m in merges.iter().take(applied) {
        let ra = find(&mut parent, m.0);
        let rb = find(&mut parent, m.1);
        parent[ra] = rb;
    }

    // First-appearance bucket order; members ascending == incoming order.
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut bucket_start: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut buckets: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        match bucket_start.get(&root) {
            Some(&bi) => buckets[bi].push(i),
            None => {
                bucket_start.insert(root, buckets.len());
                buckets.push(vec![i]);
            }
        }
    }
    let clusters = buckets.len();
    for b in buckets {
        order.extend(b);
    }
    (order, clusters)
}

// ---------------------------------------------------------------------------
// gather: per-item stray reinsertion, the incoming order is the default

fn gather_order(dist: &[f64], n: usize, min_gain: f64) -> (Vec<usize>, usize) {
    let tol = min_gain.max(0.0);
    let mut sib_cut = vec![0f64; n];
    for i in 0..n {
        let mut nearest = f64::INFINITY;
        let mut sum = 0.0;
        for j in 0..n {
            if j == i {
                continue;
            }
            let v = at(dist, n, i, j);
            sum += v;
            if v < nearest {
                nearest = v;
            }
        }
        sib_cut[i] = (nearest + sum / (n - 1) as f64) / 2.0;
    }

    let mut placed: Vec<usize> = vec![0];
    let mut run_id: Vec<usize> = vec![0];
    let mut next_run = 1usize;
    let mut moved = 0usize;
    for x in 1..n {
        let m = placed.len();
        let append_cost = at(dist, n, x, placed[m - 1]);
        let mut k = m; // append by default
        let mut run: i64 = -1; // -1 → start a new singleton run

        let mut end: i64 = m as i64 - 1;
        let mut found = false;
        while end >= 0 && !found {
            let mut start = end;
            while start - 1 >= 0 && run_id[(start - 1) as usize] == run_id[end as usize] {
                start -= 1;
            }
            let mut i = end;
            while i >= start && !found {
                let p = placed[i as usize];
                found = at(dist, n, x, p) <= sib_cut[x].min(sib_cut[p]);
                i -= 1;
            }
            if found {
                let e = end as usize;
                if e == m - 1 {
                    run = run_id[e] as i64;
                } else {
                    let cost = at(dist, n, x, placed[e]) + at(dist, n, x, placed[e + 1])
                        - at(dist, n, placed[e], placed[e + 1]);
                    if append_cost - cost > tol {
                        k = e + 1;
                        run = run_id[e] as i64;
                        moved += 1;
                    }
                }
            }
            end = start - 1;
        }
        let rid = if run == -1 {
            let r = next_run;
            next_run += 1;
            r
        } else {
            run as usize
        };
        placed.insert(k, x);
        run_id.insert(k, rid);
    }
    (placed, moved)
}

// ---------------------------------------------------------------------------
// average-linkage merge list (NN-chain, O(n²)) — used by stable mode

/// Returns merges as `(a, b, height)` with `a < b`, the lower slot surviving.
fn average_linkage_merges(dist: &[f64], n: usize) -> Vec<(usize, usize, f64)> {
    let mut d = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = at(dist, n, i, j);
        }
    }
    let mut size = vec![1usize; n];
    let mut active = vec![true; n];
    let mut merges: Vec<(usize, usize, f64)> = Vec::with_capacity(n.saturating_sub(1));
    let mut chain: Vec<usize> = Vec::new();
    let mut start = 0usize;
    while merges.len() < n - 1 {
        if chain.is_empty() {
            while !active[start] {
                start += 1;
            }
            chain.push(start);
        }
        loop {
            let x = *chain.last().unwrap();
            let prev: i64 = if chain.len() > 1 {
                chain[chain.len() - 2] as i64
            } else {
                -1
            };
            let mut best = if prev >= 0 {
                d[x * n + prev as usize]
            } else {
                f64::INFINITY
            };
            let mut y: i64 = prev;
            for i in 0..n {
                if !active[i] || i == x || i as i64 == prev {
                    continue;
                }
                if d[x * n + i] < best {
                    best = d[x * n + i];
                    y = i as i64;
                }
            }
            if y != prev {
                chain.push(y as usize);
                continue;
            }
            // Reciprocal nearest neighbors: merge x and prev.
            chain.pop();
            chain.pop();
            let yy = y as usize;
            let a = x.min(yy);
            let b = x.max(yy);
            merges.push((a, b, best));
            let total = size[a] + size[b];
            for k in 0..n {
                if !active[k] || k == a || k == b {
                    continue;
                }
                let dk = (size[a] as f64 * d[a * n + k] + size[b] as f64 * d[b * n + k])
                    / total as f64;
                d[a * n + k] = dk;
                d[k * n + a] = dk;
            }
            active[b] = false;
            size[a] = total;
            break;
        }
    }
    merges
}

// ---------------------------------------------------------------------------
// tree: average-linkage clustering + optimal leaf ordering (Bar-Joseph)

struct TreeNode {
    leaf: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
    leaves: Vec<usize>,
}

struct Tree {
    nodes: Vec<TreeNode>,
    root: usize,
}

fn build_average_linkage_tree(dist: &[f64], n: usize) -> Tree {
    let mut nodes: Vec<TreeNode> = Vec::with_capacity(2 * n);
    // items: (node_idx, size); cd: working pairwise distances between clusters.
    let mut items: Vec<(usize, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        nodes.push(TreeNode {
            leaf: Some(i),
            left: None,
            right: None,
            leaves: vec![i],
        });
        items.push((i, 1));
    }
    let mut cd: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| at(dist, n, i, j)).collect())
        .collect();

    while items.len() > 1 {
        let mut bi = 0usize;
        let mut bj = 1usize;
        let mut best = f64::INFINITY;
        for i in 0..items.len() {
            for j in (i + 1)..items.len() {
                if cd[i][j] < best {
                    best = cd[i][j];
                    bi = i;
                    bj = j;
                }
            }
        }
        let (a_idx, a_size) = items[bi];
        let (b_idx, b_size) = items[bj];
        let merged_size = a_size + b_size;
        let mut leaves = nodes[a_idx].leaves.clone();
        leaves.extend_from_slice(&nodes[b_idx].leaves);
        let merged_idx = nodes.len();
        nodes.push(TreeNode {
            leaf: None,
            left: Some(a_idx),
            right: Some(b_idx),
            leaves,
        });
        // Lance-Williams average-linkage update into slot bi.
        for k in 0..items.len() {
            if k == bi || k == bj {
                continue;
            }
            let dk = (a_size as f64 * cd[k][bi] + b_size as f64 * cd[k][bj]) / merged_size as f64;
            cd[k][bi] = dk;
            cd[bi][k] = dk;
        }
        items[bi] = (merged_idx, merged_size);
        items.remove(bj);
        cd.remove(bj);
        for row in cd.iter_mut() {
            row.remove(bj);
        }
    }
    Tree {
        root: items[0].0,
        nodes,
    }
}

struct NodeMemo {
    cost: std::collections::HashMap<(usize, usize), f64>,
    choice: std::collections::HashMap<(usize, usize), (usize, usize)>,
}

fn optimal_leaf_order(tree: &Tree, dist: &[f64], n: usize) -> Vec<usize> {
    let mut memo: Vec<Option<NodeMemo>> = (0..tree.nodes.len()).map(|_| None).collect();

    fn get_cost(
        cost: &std::collections::HashMap<(usize, usize), f64>,
        a: usize,
        b: usize,
    ) -> Option<f64> {
        cost.get(&(a, b)).or_else(|| cost.get(&(b, a))).copied()
    }

    // Post-order solve: children before parents (arena indices are not ordered,
    // so recurse explicitly).
    fn solve(idx: usize, tree: &Tree, dist: &[f64], n: usize, memo: &mut Vec<Option<NodeMemo>>) {
        if memo[idx].is_some() {
            return;
        }
        let mut cost = std::collections::HashMap::new();
        let mut choice = std::collections::HashMap::new();
        if let Some(leaf) = tree.nodes[idx].leaf {
            cost.insert((leaf, leaf), 0.0);
        } else {
            let li = tree.nodes[idx].left.unwrap();
            let ri = tree.nodes[idx].right.unwrap();
            solve(li, tree, dist, n, memo);
            solve(ri, tree, dist, n, memo);
            let left_leaves = &tree.nodes[li].leaves;
            let right_leaves = &tree.nodes[ri].leaves;
            let lc = &memo[li].as_ref().unwrap().cost;
            let rc = &memo[ri].as_ref().unwrap().cost;
            for &l in left_leaves {
                // partial[k] = min over m: M(left, l, m) + dist(m, k)
                let mut partial: std::collections::HashMap<usize, (f64, usize)> =
                    std::collections::HashMap::new();
                for &k in right_leaves {
                    partial.insert(k, (f64::INFINITY, usize::MAX));
                }
                for &m in left_leaves {
                    let cl = match get_cost(lc, l, m) {
                        Some(v) => v,
                        None => continue,
                    };
                    for &k in right_leaves {
                        let c = cl + at(dist, n, m, k);
                        let p = partial.get_mut(&k).unwrap();
                        if c < p.0 {
                            p.0 = c;
                            p.1 = m;
                        }
                    }
                }
                for &r in right_leaves {
                    let mut best = f64::INFINITY;
                    let mut bm = usize::MAX;
                    let mut bk = usize::MAX;
                    for &k in right_leaves {
                        let cr = match get_cost(rc, k, r) {
                            Some(v) => v,
                            None => continue,
                        };
                        let p = partial.get(&k).unwrap();
                        if p.0 + cr < best {
                            best = p.0 + cr;
                            bm = p.1;
                            bk = k;
                        }
                    }
                    cost.insert((l, r), best);
                    choice.insert((l, r), (bm, bk));
                }
            }
        }
        memo[idx] = Some(NodeMemo { cost, choice });
    }

    solve(tree.root, tree, dist, n, &mut memo);

    fn build(
        idx: usize,
        l: usize,
        r: usize,
        tree: &Tree,
        memo: &[Option<NodeMemo>],
    ) -> Vec<usize> {
        if let Some(leaf) = tree.nodes[idx].leaf {
            return vec![leaf];
        }
        let choice = &memo[idx].as_ref().unwrap().choice;
        match choice.get(&(l, r)) {
            None => {
                // Stored in the opposite orientation.
                let mut v = build(idx, r, l, tree, memo);
                v.reverse();
                v
            }
            Some(&(m, k)) => {
                let li = tree.nodes[idx].left.unwrap();
                let ri = tree.nodes[idx].right.unwrap();
                let mut out = build(li, l, m, tree, memo);
                out.extend(build(ri, k, r, tree, memo));
                out
            }
        }
    }

    let root_cost = &memo[tree.root].as_ref().unwrap().cost;
    let mut bl = usize::MAX;
    let mut br = usize::MAX;
    let mut best = f64::INFINITY;
    for (&(l, r), &c) in root_cost.iter() {
        if c < best {
            best = c;
            bl = l;
            br = r;
        }
    }
    build(tree.root, bl, br, tree, &memo)
}

// ---------------------------------------------------------------------------
// spectral: Fiedler-vector seriation

fn spectral_order(dist: &[f64], n: usize) -> Vec<usize> {
    // sim(i,j) = 0 on the diagonal, else max(0, 1 - dist) — matching the TS
    // reference where the similarity matrix has a zero diagonal.
    let sim = |i: usize, j: usize| -> f64 {
        if i == j {
            0.0
        } else {
            (1.0 - at(dist, n, i, j)).max(0.0)
        }
    };
    let mut deg = vec![0f64; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += sim(i, j);
        }
        deg[i] = s;
    }
    let c = deg.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 2.0 + 1.0;

    let mut v: Vec<f64> = (0..n).map(|i| i as f64 - (n as f64 - 1.0) / 2.0).collect();
    normalize(&mut v);
    let mut next = vec![0f64; n];
    for _ in 0..1000 {
        for i in 0..n {
            let mut s = (c - deg[i]) * v[i];
            for j in 0..n {
                s += sim(i, j) * v[j];
            }
            next[i] = s;
        }
        let mean: f64 = next.iter().sum::<f64>() / n as f64;
        for x in next.iter_mut() {
            *x -= mean;
        }
        normalize(&mut next);
        let mut diff = 0.0f64;
        for i in 0..n {
            diff = diff.max((next[i] - v[i]).abs());
            v[i] = next[i];
        }
        if diff < 1e-9 {
            break;
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        v[a]
            .partial_cmp(&v[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

fn normalize(v: &mut [f64]) {
    let mut s = 0.0;
    for &x in v.iter() {
        s += x * x;
    }
    let nrm = s.sqrt();
    let nrm = if nrm == 0.0 { 1.0 } else { nrm };
    for x in v.iter_mut() {
        *x /= nrm;
    }
}
