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
//
// The anchor group starts the sequence in chain mode and is pinned in minimal
// mode. Tree/spectral orderings have an intrinsic axis, so there the anchor
// only picks the direction: the sequence is reversed when that moves the
// anchor closer to the front.

import type { GroupOrderMode } from "../shared/types.ts";
import { mergePairKey } from "./constraints.ts";
import type { GroupPairResult } from "./merge-suggestions.ts";

// Cosine distance tops out at 2; unscored pairs (rejected, or absent from the
// similarity results) must rank below any scored pair.
const MISSING_DIST = 3;

export function orderGroupsBySimilarity(
  groupIds: string[],
  pairs: GroupPairResult[],
  opts?: { mode?: GroupOrderMode; anchorId?: string; minimalLocality?: number },
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
 * Mode dispatch over prebuilt matrices — shared by group ordering (matrices
 * from Rust pair results) and ungrouped-image ordering (matrices from a
 * direct embedding blend). `dist[i][j]` and `sim[i][j]` are indexed by
 * position in `ids`.
 */
export function orderByDistanceMatrix(
  ids: string[],
  dist: number[][],
  sim: number[][],
  opts?: { mode?: GroupOrderMode; anchorId?: string; minimalLocality?: number },
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
