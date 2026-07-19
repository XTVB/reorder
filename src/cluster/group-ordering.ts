// Group-ordering algorithms for the reorder page's "Sort Similar" button:
// reduce pairwise group similarity (GroupPairResult.patchMedian) to a 1D
// sequence where similar groups sit next to each other.
//
//   - "chain":    greedy nearest-neighbor chain from the anchor, improved by
//                 2-opt segment reversal until no adjacent-distance gain.
//   - "tree":     average-linkage clustering over the groups + optimal leaf
//                 ordering — families of related groups stay contiguous as
//                 blocks, with similar blocks adjacent.
//   - "spectral": seriation along the Fiedler vector of the similarity graph —
//                 the single dominant gradient across the whole collection.
//   - "minimal":  the incoming order is the answer, modulo small local moves
//                 (short reversals, single-group relocations) that improve
//                 adjacent similarity; nothing drifts more than a few
//                 positions from where it started.
//   - "stable":   similarity decides membership, the incoming order decides
//                 sequence — cluster into sets, emit sets by first appearance
//                 with each set's members in their incoming order. For
//                 collections that are already roughly ordered but lightly
//                 scrambled.
//   - "gather":   like stable but with per-item decisions instead of a global
//                 cluster cut: walk the incoming order, each item either keeps
//                 its slot or jumps back beside an earlier item it clearly
//                 matches — moves need a minimum adjacency gain, and near-tied
//                 placements resolve to the least-displacing one, so siblings
//                 never reorder.
//
// The anchor group starts the sequence in chain mode and is pinned in minimal
// mode. Tree/spectral orderings have an intrinsic axis, so there the anchor
// only picks the direction: the sequence is reversed when that moves the
// anchor closer to the front.

import type { GroupOrderMode } from "../shared/types.ts";
import { mergePairKey } from "./constraints.ts";
import type { GroupPairResult } from "./merge-suggestions.ts";

// Cosine distance tops out at 2; pairs absent from the similarity results
// must rank below any scored pair. (Merge-page rejections are deliberately
// kept in the results here — see includeRejected in computeMergeSuggestions —
// so a rejected-but-similar pair still sorts adjacent.)
const MISSING_DIST = 3;

/** Mode-specific diagnostics, surfaced in the client's completion toast. */
export interface OrderModeInfo {
  /** Stable mode: how many sets the cut produced. */
  clusters?: number;
  /** Gather mode: how many items left their incoming slot. */
  moved?: number;
}

export interface OrderByMatrixOpts {
  mode?: GroupOrderMode;
  anchorId?: string;
  minimalLocality?: number;
  /** Stable mode: force this many sets instead of the automatic gap cut. */
  stableClusters?: number;
  /** Gather mode: minimum cost improvement (blended distance) to move at all. */
  gatherMinGain?: number;
  onModeInfo?: (info: OrderModeInfo) => void;
}

export function orderGroupsBySimilarity(
  groupIds: string[],
  pairs: GroupPairResult[],
  opts?: OrderByMatrixOpts,
): string[] {
  const n = groupIds.length;
  const idToIdx = new Map(groupIds.map((id, i) => [id, i]));

  // Dedupe pairs by canonical key (defensive; the Rust binary emits each once).
  const seen = new Set<string>();
  const dist: number[][] = Array.from({ length: n }, () => new Array(n).fill(MISSING_DIST));
  const sim: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) dist[i]![i] = 0;
  for (const p of pairs) {
    const a = idToIdx.get(p.groupA);
    const b = idToIdx.get(p.groupB);
    if (a === undefined || b === undefined || a === b) continue;
    const key = mergePairKey(p.groupA, p.groupB);
    if (seen.has(key)) continue;
    seen.add(key);
    dist[a]![b] = dist[b]![a] = 1 - p.patchMedian;
    sim[a]![b] = sim[b]![a] = Math.max(0, p.patchMedian);
  }

  return orderByDistanceMatrix(groupIds, dist, sim, opts);
}

/**
 * Mode dispatch over prebuilt matrices for group ordering (matrices from Rust
 * pair results). The image-ordering equivalent of this lives in the Rust
 * `order-tool` binary (see src/cluster/order-batch.ts). `dist[i][j]` and
 * `sim[i][j]` are indexed by position in `ids`.
 */
function orderByDistanceMatrix(
  ids: string[],
  dist: number[][],
  sim: number[][],
  opts?: OrderByMatrixOpts,
): string[] {
  const n = ids.length;
  const anchor = opts?.anchorId !== undefined ? Math.max(0, ids.indexOf(opts.anchorId)) : 0;
  if (n <= 2) {
    return anchor === 1 ? [...ids].reverse() : [...ids];
  }

  let order: number[];
  switch (opts?.mode ?? "chain") {
    case "tree":
      order = orientToAnchor(optimalLeafOrder(buildAverageLinkageTree(dist), dist), anchor);
      break;
    case "spectral":
      order = orientToAnchor(spectralOrder(sim), anchor);
      break;
    case "minimal":
      order = minimalImprove(
        Array.from({ length: n }, (_, i) => i),
        dist,
        opts?.minimalLocality,
      );
      break;
    case "stable":
      order = stableClusterOrder(dist, opts?.stableClusters, opts?.onModeInfo);
      break;
    case "gather":
      order = gatherOrder(dist, opts?.gatherMinGain, opts?.onModeInfo);
      break;
    default:
      order = twoOpt(greedyChain(dist, anchor), dist);
      break;
  }
  return order.map((i) => ids[i]!);
}

/** Reverse the sequence when that moves the anchor closer to the front. */
function orientToAnchor(order: number[], anchor: number): number[] {
  const idx = order.indexOf(anchor);
  return idx > order.length - 1 - idx ? order.reverse() : order;
}

// ---------------------------------------------------------------------------
// chain: greedy nearest-neighbor + 2-opt

function greedyChain(dist: number[][], start: number): number[] {
  const n = dist.length;
  const visited: boolean[] = new Array(n).fill(false);
  visited[start] = true;
  const order = [start];
  while (order.length < n) {
    const last = order[order.length - 1]!;
    let bestIdx = -1;
    let bestD = Infinity;
    for (let i = 0; i < n; i++) {
      if (visited[i]) continue;
      if (dist[last]![i]! < bestD) {
        bestD = dist[last]![i]!;
        bestIdx = i;
      }
    }
    visited[bestIdx] = true;
    order.push(bestIdx);
  }
  return order;
}

/**
 * 2-opt for open paths: reverse order[i..j] whenever that lowers the summed
 * adjacent distance, until a full pass finds no improvement. Position 0 (the
 * anchor) never moves.
 */
function twoOpt(order: number[], dist: number[][]): number[] {
  const n = order.length;
  let improved = true;
  let passes = 0;
  while (improved && passes++ < 200) {
    improved = false;
    for (let i = 1; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        // Reversing swaps edges (i-1,i) and (j,j+1) for (i-1,j) and (i,j+1);
        // when j is the last element only the left edge changes.
        const before =
          dist[order[i - 1]!]![order[i]!]! + (j + 1 < n ? dist[order[j]!]![order[j + 1]!]! : 0);
        const after =
          dist[order[i - 1]!]![order[j]!]! + (j + 1 < n ? dist[order[i]!]![order[j + 1]!]! : 0);
        if (after < before - 1e-12) {
          for (let a = i, b = j; a < b; a++, b--) {
            const t = order[a]!;
            order[a] = order[b]!;
            order[b] = t;
          }
          improved = true;
        }
      }
    }
  }
  return order;
}

// ---------------------------------------------------------------------------
// minimal: hill-climb from the incoming order with strictly local moves

// How far a group may end up from its original position, and the max span of
// a single move. Small on purpose — this mode is "tidy up", not "rearrange".
const MINIMAL_LOCALITY = 5;

/**
 * Improve the incoming order with local moves only: reversals of short
 * segments and relocations of single groups, both spanning at most
 * MINIMAL_LOCALITY positions. A move is rejected if it would leave any group
 * more than MINIMAL_LOCALITY positions from its ORIGINAL slot, so repeated
 * passes cannot drift. Position 0 is pinned.
 */
function minimalImprove(base: number[], dist: number[][], locality = MINIMAL_LOCALITY): number[] {
  const n = base.length;
  const origIdx = new Array<number>(n);
  base.forEach((g, i) => {
    origIdx[g] = i;
  });

  const pathCost = (o: number[]): number => {
    let c = 0;
    for (let k = 0; k + 1 < n; k++) c += dist[o[k]!]![o[k + 1]!]!;
    return c;
  };
  const withinDrift = (o: number[]): boolean =>
    o.every((g, i) => Math.abs(i - origIdx[g]!) <= locality);

  let order = [...base];
  let cost = pathCost(order);
  let improved = true;
  let passes = 0;
  while (improved && passes++ < 50) {
    improved = false;

    // Short segment reversals (span ≤ MINIMAL_LOCALITY).
    for (let i = 1; i < n - 1; i++) {
      for (let j = i + 1; j < Math.min(i + locality, n); j++) {
        const cand = [...order];
        for (let a = i, b = j; a < b; a++, b--) {
          const t = cand[a]!;
          cand[a] = cand[b]!;
          cand[b] = t;
        }
        const c = pathCost(cand);
        if (c < cost - 1e-12 && withinDrift(cand)) {
          order = cand;
          cost = c;
          improved = true;
        }
      }
    }

    // Single-group relocations (≤ MINIMAL_LOCALITY positions away).
    for (let p = 1; p < n; p++) {
      const lo = Math.max(1, p - locality);
      const hi = Math.min(n - 1, p + locality);
      for (let q = lo; q <= hi; q++) {
        if (q === p) continue;
        const cand = [...order];
        const [x] = cand.splice(p, 1);
        cand.splice(q, 0, x!);
        const c = pathCost(cand);
        if (c < cost - 1e-12 && withinDrift(cand)) {
          order = cand;
          cost = c;
          improved = true;
        }
      }
    }
  }
  return order;
}

// ---------------------------------------------------------------------------
// stable: similarity decides membership, the incoming order decides sequence

/**
 * Cluster items by average linkage, then emit the clusters in order of first
 * appearance with each cluster's members in their incoming order. Two items
 * never swap relative order unless an intervening set is pulled out from
 * between them, so an already-correct sequence survives untouched — only the
 * strays travel, to wherever the rest of their set first appears.
 *
 * The cut: `forcedClusters` when given, otherwise the largest gap in the
 * sorted merge-height sequence (within-set merges are dense and low,
 * between-set merges jump).
 */
function stableClusterOrder(
  dist: number[][],
  forcedClusters?: number,
  onInfo?: (info: OrderModeInfo) => void,
): number[] {
  const n = dist.length;
  const merges = averageLinkageMerges(dist).sort((a, b) => a.h - b.h);

  let applied: number; // merges to apply = n − cluster count
  if (forcedClusters !== undefined && forcedClusters >= 1) {
    applied = n - Math.min(n, Math.round(forcedClusters));
  } else {
    let bestGap = -1;
    let bestIdx = merges.length - 1; // no interior gap → everything is one set
    for (let i = 0; i + 1 < merges.length; i++) {
      const gap = merges[i + 1]!.h - merges[i]!.h;
      if (gap > bestGap) {
        bestGap = gap;
        bestIdx = i;
      }
    }
    applied = bestIdx + 1;
  }

  // Average linkage is monotone, so applying the lowest `applied` merges via
  // union-find reproduces the flat clusters of the dendrogram cut.
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]!]!;
      x = parent[x]!;
    }
    return x;
  };
  for (let i = 0; i < applied; i++) {
    const m = merges[i]!;
    parent[find(m.a)] = find(m.b);
  }

  // Scanning indices in incoming order makes bucket insertion order = first
  // appearance and bucket contents ascending — exactly the output contract.
  const buckets = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const root = find(i);
    const bucket = buckets.get(root);
    if (bucket) bucket.push(i);
    else buckets.set(root, [i]);
  }
  onInfo?.({ clusters: buckets.size });
  return [...buckets.values()].flat();
}

// ---------------------------------------------------------------------------
// gather: per-item stray reinsertion, the incoming order is the default

// Default for `minGain`: how much (in blended-cosine units) a move must
// improve adjacency over staying put. Low on purpose: false candidates are
// already screened by the sibling cutoff, and in compressed regimes (every
// image the same person) genuine improvements can be as small as the
// intra/inter margin itself — a higher floor mostly suppresses correct moves.
const GATHER_MIN_GAIN = 0.02;

/**
 * Walk the items in their incoming order, building the output left to right
 * as a sequence of runs (emergent sets). Each item either keeps its slot
 * (append, the default — an already-correct order is reproduced exactly) or
 * moves to the end of the rightmost run containing a sibling of it, when that
 * improves adjacency by more than `minGain`.
 *
 * Three structural choices do the heavy lifting, with no global cluster count:
 * - Sibling recognition uses a per-item cutoff — the midpoint between the
 *   item's nearest-neighbor distance (its intra floor) and its mean distance
 *   to everything (its foreign scale) — so it adapts to each item's own
 *   regime, whether sets are tight and far apart or everything is one person
 *   and the margins are thin. A bond must clear BOTH items' cutoffs (min):
 *   a singleton's floor is already foreign-range, and its cutoff must not
 *   vouch for a bond on its own.
 * - Placement is only ever at a run's end, never interior — an item can
 *   rejoin its set from any distance, but can never land between siblings.
 * - Only the RIGHTMOST sibling-holding run is considered: any slot further
 *   left would put the item before that sibling. Together with run-end
 *   placement this makes within-set incoming order survive by construction.
 */
function gatherOrder(
  dist: number[][],
  minGain = GATHER_MIN_GAIN,
  onInfo?: (info: OrderModeInfo) => void,
): number[] {
  const n = dist.length;
  const tol = Math.max(0, minGain);
  const sibCut = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const row = dist[i]!;
    let nearest = Infinity;
    let sum = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      sum += row[j]!;
      if (row[j]! < nearest) nearest = row[j]!;
    }
    sibCut[i] = (nearest + sum / (n - 1)) / 2;
  }
  // Runs stay contiguous in `placed` (insertions only happen at run ends), so
  // a parallel run-id array spliced alongside is enough to delimit them.
  const placed: number[] = [0];
  const runId: number[] = [0];
  let nextRun = 1;
  let moved = 0;
  for (let x = 1; x < n; x++) {
    const m = placed.length;
    const row = dist[x]!;
    const appendCost = row[placed[m - 1]!]!;
    let k = m; // append by default
    let run = -1; // -1 → start a new singleton run
    // Find the rightmost run containing a sibling; it alone decides.
    let end = m - 1;
    let found = false;
    while (end >= 0 && !found) {
      let start = end;
      while (start - 1 >= 0 && runId[start - 1] === runId[end]) start--;
      for (let i = end; i >= start && !found; i--) {
        const p = placed[i]!;
        found = row[p]! <= Math.min(sibCut[x]!, sibCut[p]!);
      }
      if (found) {
        if (end === m - 1) {
          // The tail run's end slot IS the append slot — adopt, stay put.
          run = runId[end]!;
        } else {
          // Moving swaps the run-boundary edge for two new ones; appending
          // adds one edge after the current last item.
          const cost =
            row[placed[end]!]! + row[placed[end + 1]!]! - dist[placed[end]!]![placed[end + 1]!]!;
          if (appendCost - cost > tol) {
            k = end + 1;
            run = runId[end]!;
            moved++;
          }
        }
      }
      end = start - 1;
    }
    placed.splice(k, 0, x);
    runId.splice(k, 0, run === -1 ? nextRun++ : run);
  }
  onInfo?.({ moved });
  return placed;
}

/**
 * Average-linkage agglomeration via the nearest-neighbor chain algorithm —
 * O(n²), unlike buildAverageLinkageTree's O(n³) (tree mode needs the explicit
 * tree for leaf ordering; here only the merge list matters). Each merge keeps
 * the lower slot as the surviving cluster, so union-find over slots
 * reconstructs the clusters.
 */
function averageLinkageMerges(dist: number[][]): { a: number; b: number; h: number }[] {
  const n = dist.length;
  const d = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) d[i * n + j] = dist[i]![j]!;
  const size = new Array<number>(n).fill(1);
  const active = new Array<boolean>(n).fill(true);
  const merges: { a: number; b: number; h: number }[] = [];
  const chain: number[] = [];
  let start = 0;
  while (merges.length < n - 1) {
    if (chain.length === 0) {
      while (!active[start]) start++;
      chain.push(start);
    }
    for (;;) {
      const x = chain[chain.length - 1]!;
      const prev = chain.length > 1 ? chain[chain.length - 2]! : -1;
      // Nearest active neighbor; ties prefer the chain predecessor so
      // reciprocal pairs terminate the walk.
      let best = prev >= 0 ? d[x * n + prev]! : Infinity;
      let y = prev;
      for (let i = 0; i < n; i++) {
        if (!active[i] || i === x || i === prev) continue;
        if (d[x * n + i]! < best) {
          best = d[x * n + i]!;
          y = i;
        }
      }
      if (y !== prev) {
        chain.push(y);
        continue;
      }
      // x and prev are reciprocal nearest neighbors: merge them.
      chain.pop();
      chain.pop();
      const a = Math.min(x, y);
      const b = Math.max(x, y);
      merges.push({ a, b, h: best });
      // Lance-Williams update for average linkage into the surviving slot.
      const total = size[a]! + size[b]!;
      for (let k = 0; k < n; k++) {
        if (!active[k] || k === a || k === b) continue;
        const dk = (size[a]! * d[a * n + k]! + size[b]! * d[b * n + k]!) / total;
        d[a * n + k] = dk;
        d[k * n + a] = dk;
      }
      active[b] = false;
      size[a] = total;
      break;
    }
  }
  return merges;
}

// ---------------------------------------------------------------------------
// tree: average-linkage clustering + optimal leaf ordering

interface TreeNode {
  leaf: number | null;
  left: TreeNode | null;
  right: TreeNode | null;
  leaves: number[];
}

function buildAverageLinkageTree(dist: number[][]): TreeNode {
  const items: { node: TreeNode; size: number }[] = dist.map((_, i) => ({
    node: { leaf: i, left: null, right: null, leaves: [i] },
    size: 1,
  }));
  // Working copy of pairwise distances between current clusters.
  const cd = dist.map((row) => [...row]);
  while (items.length > 1) {
    let bi = 0;
    let bj = 1;
    let best = Infinity;
    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) {
        if (cd[i]![j]! < best) {
          best = cd[i]![j]!;
          bi = i;
          bj = j;
        }
      }
    }
    const a = items[bi]!;
    const b = items[bj]!;
    const merged = {
      node: {
        leaf: null,
        left: a.node,
        right: b.node,
        leaves: [...a.node.leaves, ...b.node.leaves],
      },
      size: a.size + b.size,
    };
    // Lance-Williams update for average linkage: d(k, a∪b) = (nₐ·d(k,a) + n_b·d(k,b)) / (nₐ+n_b)
    for (let k = 0; k < items.length; k++) {
      if (k === bi || k === bj) continue;
      const dk = (a.size * cd[k]![bi]! + b.size * cd[k]![bj]!) / merged.size;
      cd[k]![bi] = dk;
      cd[bi]![k] = dk;
    }
    items[bi] = merged;
    items.splice(bj, 1);
    cd.splice(bj, 1);
    for (const row of cd) row.splice(bj, 1);
  }
  return items[0]!.node;
}

/**
 * Optimal leaf ordering (Bar-Joseph): among all orderings reachable by
 * flipping subtrees, find one minimizing the summed distance between adjacent
 * leaves. M(v, l, r) = best cost for v's leaves with endpoints l and r; at an
 * internal node the junction (m, k) between the children is chosen by DP.
 */
function optimalLeafOrder(root: TreeNode, dist: number[][]): number[] {
  const n = dist.length;
  // Per node: cost/choice keyed by l*n+r with l in the left child and r in the
  // right child. The reversed orientation has equal cost and is looked up via
  // the symmetric key.
  const memo = new Map<TreeNode, { cost: Map<number, number>; choice: Map<number, number> }>();

  const getCost = (cost: Map<number, number>, a: number, b: number): number | undefined =>
    cost.get(a * n + b) ?? cost.get(b * n + a);

  function solve(node: TreeNode): Map<number, number> {
    const existing = memo.get(node);
    if (existing) return existing.cost;
    const cost = new Map<number, number>();
    const choice = new Map<number, number>();
    if (node.leaf !== null) {
      cost.set(node.leaf * n + node.leaf, 0);
    } else {
      const L = solve(node.left!);
      const R = solve(node.right!);
      for (const l of node.left!.leaves) {
        // partial[k] = min over m: M(left, l, m) + dist(m, k)
        const partial = new Map<number, { cost: number; m: number }>();
        for (const k of node.right!.leaves) partial.set(k, { cost: Infinity, m: -1 });
        for (const m of node.left!.leaves) {
          const cl = getCost(L, l, m);
          if (cl === undefined) continue;
          for (const k of node.right!.leaves) {
            const c = cl + dist[m]![k]!;
            const p = partial.get(k)!;
            if (c < p.cost) {
              p.cost = c;
              p.m = m;
            }
          }
        }
        for (const r of node.right!.leaves) {
          let best = Infinity;
          let bm = -1;
          let bk = -1;
          for (const k of node.right!.leaves) {
            const cr = getCost(R, k, r);
            if (cr === undefined) continue;
            const p = partial.get(k)!;
            if (p.cost + cr < best) {
              best = p.cost + cr;
              bm = p.m;
              bk = k;
            }
          }
          cost.set(l * n + r, best);
          choice.set(l * n + r, bm * n + bk);
        }
      }
    }
    memo.set(node, { cost, choice });
    return cost;
  }

  function build(node: TreeNode, l: number, r: number): number[] {
    if (node.leaf !== null) return [node.leaf];
    const { choice } = memo.get(node)!;
    const c = choice.get(l * n + r);
    if (c === undefined) return build(node, r, l).reverse(); // stored in the opposite orientation
    const m = Math.floor(c / n);
    const k = c % n;
    return [...build(node.left!, l, m), ...build(node.right!, k, r)];
  }

  const rootCost = solve(root);
  let bl = -1;
  let br = -1;
  let best = Infinity;
  for (const [key, c] of rootCost) {
    if (c < best) {
      best = c;
      bl = Math.floor(key / n);
      br = key % n;
    }
  }
  return build(root, bl, br);
}

// ---------------------------------------------------------------------------
// spectral: Fiedler-vector seriation

function normalize(v: number[]): void {
  let s = 0;
  for (const x of v) s += x * x;
  const nrm = Math.sqrt(s) || 1;
  for (let i = 0; i < v.length; i++) v[i] = v[i]! / nrm;
}

/**
 * Order by the Fiedler vector (second-smallest eigenvector of the graph
 * Laplacian L = D − W). Power iteration on B = cI − L flips the spectrum so
 * the smallest eigenvectors of L dominate; deflating the all-ones kernel each
 * step leaves the Fiedler vector.
 */
function spectralOrder(sim: number[][]): number[] {
  const n = sim.length;
  const deg = sim.map((row) => row.reduce((acc, x) => acc + x, 0));
  const c = Math.max(...deg) * 2 + 1; // > λmax(L) ⇒ B is positive definite, no sign flapping
  const v: number[] = Array.from({ length: n }, (_, i) => i - (n - 1) / 2); // deterministic, ⊥ ones
  normalize(v);
  const next: number[] = new Array(n).fill(0);
  for (let iter = 0; iter < 1000; iter++) {
    for (let i = 0; i < n; i++) {
      let s = (c - deg[i]!) * v[i]!;
      for (let j = 0; j < n; j++) s += sim[i]![j]! * v[j]!;
      next[i] = s;
    }
    const mean = next.reduce((acc, x) => acc + x, 0) / n;
    for (let i = 0; i < n; i++) next[i] = next[i]! - mean;
    normalize(next);
    let diff = 0;
    for (let i = 0; i < n; i++) {
      diff = Math.max(diff, Math.abs(next[i]! - v[i]!));
      v[i] = next[i]!;
    }
    if (diff < 1e-9) break;
  }
  return Array.from({ length: n }, (_, i) => i).sort((a, b) => v[a]! - v[b]! || a - b);
}
