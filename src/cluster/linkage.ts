// Linkage tree: parsing the binary format produced by the Rust cluster-tool
// and re-cutting it on the Bun side (instant, no Rust round-trip required).
//
// Three cut modes:
//   - recutTree(N)        — cut to a fixed N clusters
//   - recutTreeByThreshold — cut at a distance threshold
//   - recutTreeAdaptive   — HDBSCAN-style stability extraction (preserved
//                           per the refactor plan)

import { existsSync, readFileSync, statSync } from "node:fs";
import { unlink } from "node:fs/promises";
import { linkageTreePath } from "../fs/paths.ts";
import type { DistanceProfile } from "../shared/types.ts";

export interface LinkageTree {
  nImages: number;
  nPreMerges: number;
  nGroups: number;
  steps: { clusterA: number; clusterB: number; distance: number; newSize: number }[];
}

let _treeCache: { targetDir: string; mtime: number; tree: LinkageTree } | null = null;

export function parseLinkageTree(path: string): LinkageTree {
  const buf = readFileSync(path);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);

  let offset = 0;
  const nImages = view.getUint32(offset, true);
  offset += 4;
  const nPreMerges = view.getUint32(offset, true);
  offset += 4;
  const nGroups = view.getUint32(offset, true);
  offset += 4;
  const nSteps = view.getUint32(offset, true);
  offset += 4;

  const steps = [];
  for (let i = 0; i < nSteps; i++) {
    const clusterA = view.getUint32(offset, true);
    offset += 4;
    const clusterB = view.getUint32(offset, true);
    offset += 4;
    const distance = view.getFloat32(offset, true);
    offset += 4;
    const newSize = view.getUint32(offset, true);
    offset += 4;
    steps.push({ clusterA, clusterB, distance, newSize });
  }

  return { nImages, nPreMerges, nGroups, steps };
}

export function loadTree(targetDir: string): LinkageTree {
  const path = linkageTreePath(targetDir);
  if (!existsSync(path)) {
    throw new Error("No linkage tree found. Run full clustering first.");
  }
  const mtime = statSync(path).mtimeMs;
  if (_treeCache && _treeCache.targetDir === targetDir && _treeCache.mtime === mtime) {
    return _treeCache.tree;
  }
  const tree = parseLinkageTree(path);
  _treeCache = { targetDir, mtime, tree };
  return tree;
}

export function clearTreeCache(): void {
  _treeCache = null;
}

/** Delete the on-disk linkage tree (no-op if absent). */
export async function removeLinkageTree(targetDir: string): Promise<void> {
  try {
    await unlink(linkageTreePath(targetDir));
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "ENOENT") throw err;
  }
}

/** Apply pre-merges, then main merges up to `maxMainMerges`, return labels. */
export function cutTree(tree: LinkageTree, maxMainMerges: number): number[] {
  const { nImages, nPreMerges, steps } = tree;

  const parent = new Int32Array(nImages);
  for (let i = 0; i < nImages; i++) parent[i] = i;

  // Path-halving union-find: each step skips to grandparent, flattening the tree
  function find(x: number): number {
    let p = parent[x]!;
    while (p !== x) {
      const gp = parent[p]!;
      parent[x] = gp;
      x = p;
      p = gp;
    }
    return x;
  }

  for (let i = 0; i < nPreMerges; i++) {
    const s = steps[i]!;
    const ra = find(s.clusterA);
    const rb = find(s.clusterB);
    if (ra !== rb) parent[ra] = rb;
  }

  for (let i = 0; i < maxMainMerges; i++) {
    const s = steps[nPreMerges + i]!;
    const ra = find(s.clusterA);
    const rb = find(s.clusterB);
    if (ra !== rb) parent[ra] = rb;
  }

  const remap = new Map<number, number>();
  let nextLabel = 0;
  const labels = new Array<number>(nImages);
  for (let i = 0; i < nImages; i++) {
    const r = find(i);
    if (!remap.has(r)) remap.set(r, nextLabel++);
    labels[i] = remap.get(r)!;
  }

  return labels;
}

export function distanceProfileFromTree(tree: LinkageTree): DistanceProfile {
  const distances = tree.steps
    .slice(tree.nPreMerges)
    .map((s) => s.distance)
    .filter((d) => d < 1e10);
  return {
    distances,
    nAfterPremerge: tree.nImages - tree.nPreMerges,
    nGroups: tree.nGroups,
  };
}

export function recutTree(
  targetDir: string,
  nClusters: number,
): { labels: number[]; nClusters: number; distanceProfile: DistanceProfile } {
  const tree = loadTree(targetDir);
  const nAfterPremerge = tree.nImages - tree.nPreMerges;
  // Never go below nGroups clusters — confirmed groups must stay separate
  const minClusters = Math.max(nClusters, tree.nGroups);
  const mainMergesNeeded = Math.max(0, nAfterPremerge - minClusters);
  const labels = cutTree(tree, mainMergesNeeded);
  return {
    labels,
    nClusters: nAfterPremerge - mainMergesNeeded,
    distanceProfile: distanceProfileFromTree(tree),
  };
}

export function recutTreeByThreshold(
  targetDir: string,
  threshold: number,
): { labels: number[]; nClusters: number; distanceProfile: DistanceProfile } {
  const tree = loadTree(targetDir);
  const { nPreMerges, steps } = tree;
  const nAfterPremerge = tree.nImages - nPreMerges;

  // Count main merges below threshold (steps are sorted by distance;
  // group-to-group sentinel distances ~1e18 are naturally excluded)
  let mainMerges = 0;
  for (let i = nPreMerges; i < steps.length; i++) {
    if (steps[i]!.distance >= threshold) break;
    mainMerges++;
  }

  const labels = cutTree(tree, mainMerges);
  return {
    labels,
    nClusters: nAfterPremerge - mainMerges,
    distanceProfile: distanceProfileFromTree(tree),
  };
}

/**
 * HDBSCAN-style stability-based cluster extraction with condensed tree.
 *
 * Instead of a global threshold, scores each potential cluster by how long it
 * persists in the hierarchy. Clusters that exist over a wide range of distances
 * (high stability) are "real" groups — they naturally handle variable sizes.
 *
 * Algorithm:
 * 1. Build the full binary tree from the linkage steps (each merge creates a node)
 * 2. Condense: remove nodes smaller than minClusterSize, propagating their
 *    members up to the parent (these become "noise" that falls out of small clusters)
 * 3. Compute stability for each condensed node: Σ (1/λ_birth - 1/λ_death) per point,
 *    where λ = distance (using distance directly since Ward distances are monotonic)
 * 4. Bottom-up selection: for each node, if its own stability > sum of children's
 *    selected stability, select it and deselect children. Otherwise propagate.
 *
 * minClusterSize controls granularity: higher = fewer, larger clusters.
 * Typical range: 3–15.
 */
export function recutTreeAdaptive(
  targetDir: string,
  minClusterSize: number,
): { labels: number[]; nClusters: number; distanceProfile: DistanceProfile } {
  const tree = loadTree(targetDir);
  const { nImages, steps } = tree;

  const ufParent = new Int32Array(nImages);
  const ufSize = new Int32Array(nImages).fill(1);
  for (let i = 0; i < nImages; i++) ufParent[i] = i;
  function find(x: number): number {
    while (ufParent[x]! !== x) {
      ufParent[x] = ufParent[ufParent[x]!]!;
      x = ufParent[x]!;
    }
    return x;
  }

  // ── Build condensed tree ──────────────────────────────────────────────
  interface CNode {
    id: number;
    birthDist: number;
    deathDist: number;
    size: number;
    children: number[];
    parentId: number;
  }
  const cnodes: CNode[] = [];
  let nextCid = 0;
  const repCnode = new Map<number, number>();
  const imgCnode = new Int32Array(nImages).fill(-1);

  for (const s of steps) {
    if (s.distance >= 1e10) break;
    const ra = find(s.clusterA),
      rb = find(s.clusterB);
    if (ra === rb) continue;
    const sA = ufSize[ra]!,
      sB = ufSize[rb]!;
    const cA = repCnode.get(ra),
      cB = repCnode.get(rb);

    if (sA >= minClusterSize && sB >= minClusterSize) {
      if (cA !== undefined) cnodes[cA]!.deathDist = s.distance;
      if (cB !== undefined) cnodes[cB]!.deathDist = s.distance;
      const pid = nextCid++;
      cnodes.push({
        id: pid,
        birthDist: s.distance,
        deathDist: Infinity,
        size: sA + sB,
        children: [],
        parentId: -1,
      });
      if (cA !== undefined) {
        cnodes[pid]!.children.push(cA);
        cnodes[cA]!.parentId = pid;
      }
      if (cB !== undefined) {
        cnodes[pid]!.children.push(cB);
        cnodes[cB]!.parentId = pid;
      }
      ufParent[ra] = rb;
      ufSize[rb] = sA + sB;
      repCnode.delete(ra);
      repCnode.set(rb, pid);
    } else {
      ufParent[ra] = rb;
      ufSize[rb] = sA + sB;
      const ms = ufSize[rb]!;
      if (!repCnode.has(rb) && ms >= minClusterSize) {
        const id = nextCid++;
        cnodes.push({
          id,
          birthDist: s.distance,
          deathDist: Infinity,
          size: ms,
          children: [],
          parentId: -1,
        });
        repCnode.set(rb, id);
        for (let j = 0; j < nImages; j++) {
          if (find(j) === rb && imgCnode[j] === -1) imgCnode[j] = id;
        }
      }
      if (cA !== undefined && repCnode.has(rb)) {
        const pc = repCnode.get(rb)!;
        cnodes[cA]!.deathDist = s.distance;
        cnodes[pc]!.children.push(cA);
        cnodes[cA]!.parentId = pc;
      }
      repCnode.delete(ra);
    }
  }

  // Remaining unmapped images get their root's condensed node
  for (let i = 0; i < nImages; i++) {
    if (imgCnode[i] === -1) {
      const root = find(i);
      const cid = repCnode.get(root);
      if (cid !== undefined) imgCnode[i] = cid;
    }
  }

  if (cnodes.length === 0) {
    return recutTree(targetDir, Math.max(50, Math.floor(nImages / 30)));
  }

  // ── Stability ─────────────────────────────────────────────────────────
  const cstab = new Float64Array(cnodes.length);
  for (let i = 0; i < cnodes.length; i++) {
    const c = cnodes[i]!;
    const lb = c.birthDist > 0 ? 1 / c.birthDist : 0;
    const ld = c.deathDist < Infinity && c.deathDist > 0 ? 1 / c.deathDist : 0;
    cstab[i] = c.size * Math.max(0, lb - ld);
  }

  // ── Bottom-up selection (topological order: children before parents) ──
  const topoOrder: number[] = [];
  const vis = new Uint8Array(cnodes.length);
  function visit(id: number) {
    if (vis[id]) return;
    vis[id] = 1;
    for (const ch of cnodes[id]!.children) visit(ch);
    topoOrder.push(id);
  }
  for (let i = 0; i < cnodes.length; i++) visit(i);

  const csel = new Uint8Array(cnodes.length);
  const cbest = new Float64Array(cnodes.length);
  for (const i of topoOrder) {
    const c = cnodes[i]!;
    let childSum = 0;
    for (const ch of c.children) childSum += cbest[ch]!;
    if (cstab[i]! > childSum || c.children.length === 0) {
      csel[i] = 1;
      cbest[i] = cstab[i]!;
      const stk = [...c.children];
      while (stk.length) {
        const d = stk.pop()!;
        csel[d] = 0;
        stk.push(...cnodes[d]!.children);
      }
    } else {
      cbest[i] = childSum;
    }
  }

  // ── Label assignment: walk up condensed tree from each image ──────────
  const labels = new Int32Array(nImages).fill(-1);
  const clusterMap = new Map<number, number>();
  let nextLabel = 0;

  for (let i = 0; i < nImages; i++) {
    let cid = imgCnode[i]!;
    while (cid >= 0) {
      if (csel[cid]) {
        if (!clusterMap.has(cid)) clusterMap.set(cid, nextLabel++);
        labels[i] = clusterMap.get(cid)!;
        break;
      }
      cid = cnodes[cid]!.parentId;
    }
  }

  // ── Orphan assignment via merge-step nearest-neighbor ─────────────────
  // For each orphan, replay merge steps with linked-list cluster tracking.
  // When an orphan's cluster merges with one containing labeled images,
  // assign the orphan that label (nearest neighbor in the merge tree).
  const orphanSet = new Set<number>();
  for (let i = 0; i < nImages; i++) if (labels[i]! < 0) orphanSet.add(i);

  if (orphanSet.size > 0) {
    for (let i = 0; i < nImages; i++) {
      ufParent[i] = i;
      ufSize[i] = 1;
    }
    const next = new Int32Array(nImages).fill(-1);
    const head = new Int32Array(nImages);
    const tail = new Int32Array(nImages);
    for (let i = 0; i < nImages; i++) {
      head[i] = i;
      tail[i] = i;
    }

    for (const s of steps) {
      if (s.distance >= 1e10 || orphanSet.size === 0) break;
      const ra = find(s.clusterA),
        rb = find(s.clusterB);
      if (ra === rb) continue;

      let labelA = -1,
        labelB = -1;
      for (let j = head[ra]!; j >= 0; j = next[j]!) {
        if (labels[j]! >= 0) {
          labelA = labels[j]!;
          break;
        }
      }
      for (let j = head[rb]!; j >= 0; j = next[j]!) {
        if (labels[j]! >= 0) {
          labelB = labels[j]!;
          break;
        }
      }

      if (labelA >= 0 && labelB < 0) {
        for (let j = head[rb]!; j >= 0; j = next[j]!) {
          if (labels[j]! < 0) {
            labels[j] = labelA;
            orphanSet.delete(j);
          }
        }
      } else if (labelB >= 0 && labelA < 0) {
        for (let j = head[ra]!; j >= 0; j = next[j]!) {
          if (labels[j]! < 0) {
            labels[j] = labelB;
            orphanSet.delete(j);
          }
        }
      }

      ufParent[ra] = rb;
      ufSize[rb] = ufSize[ra]! + ufSize[rb]!;
      next[tail[rb]!] = head[ra]!;
      tail[rb] = tail[ra]!;
    }

    // Any remaining orphans: walk cluster list to find a labeled neighbor
    for (const i of orphanSet) {
      const root = find(i);
      for (let j = head[root]!; j >= 0; j = next[j]!) {
        if (labels[j]! >= 0) {
          labels[i] = labels[j]!;
          break;
        }
      }
      if (labels[i]! < 0) labels[i] = nextLabel++;
    }
  }

  return {
    labels: Array.from(labels),
    nClusters: nextLabel,
    distanceProfile: distanceProfileFromTree(tree),
  };
}

export function getDistanceProfile(targetDir: string): DistanceProfile {
  return distanceProfileFromTree(loadTree(targetDir));
}
