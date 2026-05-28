/**
 * Server-side support for the tree-navigation cluster operations
 * (merge / split / expand) defined in CLUSTERING_UI_SPLIT_SPEC.md.
 *
 * All operations work on top of the cached linkage tree + cached embeddings.
 * They never mutate persistent state — the client is the source of truth for
 * the in-progress cluster view; the server only computes derived values
 * (metrics, candidate rankings, binary splits).
 */

import { loadGroups } from "../fs/groups.ts";
import type { ClusterResultData, WeightConfig } from "../shared/types.ts";
import {
  cachedHashMapping,
  loadModelEmbedding,
  MODEL_KEYS,
  type ModelEmbedding,
  type ModelKey,
} from "./embeddings.ts";
import { type LinkageTree, loadTree } from "./linkage.ts";

/**
 * Path-halving find for union-find structures backed by an Int32Array parent
 * map. Mutates `parent` to halve paths during traversal — amortized near-O(1).
 */
function pathHalvingFind(parent: Int32Array, x: number): number {
  let p = parent[x]!;
  while (p !== x) {
    const gp = parent[p]!;
    parent[x] = gp;
    x = p;
    p = gp;
  }
  return x;
}

// ── Embedding blend ──────────────────────────────────────────────────────────

interface ActiveModel {
  emb: ModelEmbedding;
  weight: number;
}

function loadActiveModels(targetDir: string, weights: WeightConfig | undefined): ActiveModel[] {
  const known = new Set<string>(MODEL_KEYS);
  const out: ActiveModel[] = [];
  if (weights) {
    for (const [key, val] of Object.entries(weights)) {
      const w = val ?? 0;
      if (w > 0 && known.has(key)) {
        try {
          out.push({ emb: loadModelEmbedding(targetDir, key as ModelKey), weight: w });
        } catch {
          // model missing — fall through, may end up empty
        }
      }
    }
  }
  if (out.length === 0) {
    out.push({ emb: loadModelEmbedding(targetDir, "pecore_g"), weight: 1.0 });
  }
  return out;
}

function l2Norm(row: Float32Array | Float64Array, start: number, dim: number): number {
  let s = 0;
  for (let i = 0; i < dim; i++) {
    const v = row[start + i]!;
    s += v * v;
  }
  return Math.sqrt(s);
}

/** Weighted-cosine distance from a single image's row to a centroid (per active model). */
function imageToCentroidDist(
  models: ActiveModel[],
  centroids: Float64Array[],
  imgIdx: number,
): number {
  let total = 0;
  let totalWeight = 0;
  for (let m = 0; m < models.length; m++) {
    const { emb, weight } = models[m]!;
    const { data, dim, normalized } = emb;
    const centroid = centroids[m]!;
    const start = imgIdx * dim;
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += centroid[d]! * data[start + d]!;
    if (!normalized) {
      const nrm = l2Norm(data, start, dim) || 1e-10;
      dot /= nrm;
    }
    total += weight * Math.max(0, 1 - dot);
    totalWeight += weight;
  }
  return totalWeight > 0 ? total / totalWeight : 0;
}

/** Compute and L2-normalize per-model centroid for an image-index set. */
function buildCentroids(models: ActiveModel[], imageIndices: number[]): Float64Array[] {
  const centroids: Float64Array[] = [];
  for (const { emb } of models) {
    const { data, dim, normalized } = emb;
    const c = new Float64Array(dim);
    for (const idx of imageIndices) {
      const start = idx * dim;
      if (normalized) {
        for (let d = 0; d < dim; d++) c[d] = c[d]! + data[start + d]!;
      } else {
        const nrm = l2Norm(data, start, dim) || 1e-10;
        for (let d = 0; d < dim; d++) c[d] = c[d]! + data[start + d]! / nrm;
      }
    }
    if (imageIndices.length > 0) {
      for (let d = 0; d < dim; d++) c[d] = c[d]! / imageIndices.length;
    }
    const cNorm = l2Norm(c, 0, dim) || 1e-10;
    for (let d = 0; d < dim; d++) c[d] = c[d]! / cNorm;
    centroids.push(c);
  }
  return centroids;
}

/** Pairwise blended-cosine distance between two image rows. */
function pairwiseDist(models: ActiveModel[], i: number, j: number): number {
  let total = 0;
  let totalWeight = 0;
  for (const { emb, weight } of models) {
    const { data, dim, normalized } = emb;
    const aStart = i * dim;
    const bStart = j * dim;
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += data[aStart + d]! * data[bStart + d]!;
    if (!normalized) {
      const an = l2Norm(data, aStart, dim) || 1e-10;
      const bn = l2Norm(data, bStart, dim) || 1e-10;
      dot /= an * bn;
    }
    total += weight * Math.max(0, 1 - dot);
    totalWeight += weight;
  }
  return totalWeight > 0 ? total / totalWeight : 0;
}

// ── Per-cluster metrics ──────────────────────────────────────────────────────

export interface ClusterMetrics {
  cohesion: number; // max intra-pair distance
  isolation: number; // death distance (Infinity if root or singleton-without-merge)
  stability: number; // (death - birth) / death; 0 when undefined
  birth: number;
  death: number;
}

interface MetricsInput {
  id: string;
  images: string[];
}

/** Find each cluster's birth/death distances by replaying the linkage tree. */
function computeBirthDeath(
  tree: LinkageTree,
  fnToIdx: Map<string, number>,
  clusters: MetricsInput[],
): Map<string, { birth: number; death: number }> {
  const result = new Map<string, { birth: number; death: number }>();
  const { nImages, steps } = tree;

  // Scratch buffers reused across clusters — reset at the top of each iteration.
  const inCluster = new Uint8Array(nImages);
  const parent = new Int32Array(nImages);
  const sizeIn = new Int32Array(nImages);
  const sizeOut = new Int32Array(nImages);
  const find = (x: number) => pathHalvingFind(parent, x);

  for (const c of clusters) {
    inCluster.fill(0);
    let target = 0;
    for (const f of c.images) {
      const idx = fnToIdx.get(f);
      if (idx !== undefined && !inCluster[idx]) {
        inCluster[idx] = 1;
        target++;
      }
    }

    if (target === 0) {
      result.set(c.id, { birth: 0, death: Infinity });
      continue;
    }

    for (let i = 0; i < nImages; i++) {
      parent[i] = i;
      sizeIn[i] = inCluster[i] ? 1 : 0;
      sizeOut[i] = inCluster[i] ? 0 : 1;
    }

    let birth = target === 1 ? 0 : -Infinity;
    let formed = target === 1; // singletons form trivially
    let death = Infinity;
    let formedRoot = -1;

    if (target === 1) {
      // A singleton's "death" is the first merge that pulls it into anything else.
      for (const s of steps) {
        if (s.distance >= 1e10) break;
        const ra = find(s.clusterA);
        const rb = find(s.clusterB);
        if (ra === rb) continue;
        if (sizeIn[ra]! + sizeIn[rb]! >= 1) {
          death = s.distance;
          break;
        }
      }
      result.set(c.id, { birth: 0, death });
      continue;
    }

    for (const s of steps) {
      if (s.distance >= 1e10) break;
      const ra = find(s.clusterA);
      const rb = find(s.clusterB);
      if (ra === rb) continue;

      const pureA = sizeOut[ra]! === 0 && sizeIn[ra]! > 0;
      const pureB = sizeOut[rb]! === 0 && sizeIn[rb]! > 0;
      const internal = pureA && pureB;
      if (internal) {
        // Walk up to the highest internal-merge distance encountered
        if (s.distance > birth) birth = s.distance;
      }

      // Union into rb
      parent[ra] = rb;
      sizeIn[rb] = sizeIn[rb]! + sizeIn[ra]!;
      sizeOut[rb] = sizeOut[rb]! + sizeOut[ra]!;

      if (!formed) {
        if (sizeIn[rb]! === target && sizeOut[rb]! === 0) {
          if (s.distance > birth) birth = s.distance;
          formed = true;
          formedRoot = rb;
        }
      } else {
        const r = find(formedRoot);
        if (sizeOut[r]! > 0) {
          death = s.distance;
          break;
        }
      }
    }

    if (!formed) {
      // Cluster never existed as an exact union-find component (rare, can happen
      // if the cluster spans across the linkage cut due to confirmed-group
      // pre-merges). Use the largest internal merge as birth, leave death=Inf.
      birth = birth === -Infinity ? 0 : birth;
    }
    if (birth === -Infinity) birth = 0;

    result.set(c.id, { birth, death });
  }
  return result;
}

/** Compute cohesion (max intra-pair distance) for one cluster, with capped sampling. */
function computeCohesion(
  models: ActiveModel[],
  fnToIdx: Map<string, number>,
  images: string[],
): number {
  const indices: number[] = [];
  for (const f of images) {
    const idx = fnToIdx.get(f);
    if (idx !== undefined) indices.push(idx);
  }
  if (indices.length < 2) return 0;

  // Cap pair count for huge clusters: random sample with a fixed seed-ish stride.
  const PAIR_CAP = 12000;
  const totalPairs = (indices.length * (indices.length - 1)) / 2;
  let maxDist = 0;
  if (totalPairs <= PAIR_CAP) {
    for (let i = 0; i < indices.length; i++) {
      for (let j = i + 1; j < indices.length; j++) {
        const d = pairwiseDist(models, indices[i]!, indices[j]!);
        if (d > maxDist) maxDist = d;
      }
    }
  } else {
    const stride = Math.max(1, Math.floor(totalPairs / PAIR_CAP));
    let pairIdx = 0;
    for (let i = 0; i < indices.length; i++) {
      for (let j = i + 1; j < indices.length; j++) {
        if (pairIdx % stride === 0) {
          const d = pairwiseDist(models, indices[i]!, indices[j]!);
          if (d > maxDist) maxDist = d;
        }
        pairIdx++;
      }
    }
  }
  return maxDist;
}

export function computeClusterMetrics(
  targetDir: string,
  clusters: MetricsInput[],
  weights: WeightConfig | undefined,
): Map<string, ClusterMetrics> {
  const tree = loadTree(targetDir);
  const mapping = cachedHashMapping(targetDir);
  const fnToIdx = mapping.fnToIdx;
  const models = loadActiveModels(targetDir, weights);

  const birthDeath = computeBirthDeath(tree, fnToIdx, clusters);
  const out = new Map<string, ClusterMetrics>();
  for (const c of clusters) {
    const bd = birthDeath.get(c.id) ?? { birth: 0, death: Infinity };
    const cohesion = computeCohesion(models, fnToIdx, c.images);
    const stability = bd.death === Infinity || bd.death <= 0 ? 0 : (bd.death - bd.birth) / bd.death;
    out.set(c.id, {
      cohesion,
      isolation: bd.death,
      stability,
      birth: bd.birth,
      death: bd.death,
    });
  }
  return out;
}

// ── Two-way binary split via linkage tree ────────────────────────────────────

export interface SplitResult {
  childA: ClusterResultData;
  childB: ClusterResultData;
}

/** Split `images` into two children using the linkage tree's last internal merge. */
export function splitClusterByTree(targetDir: string, images: string[]): SplitResult | null {
  if (images.length < 2) return null;

  const tree = loadTree(targetDir);
  const mapping = cachedHashMapping(targetDir);
  const fnToIdx = mapping.fnToIdx;
  const { nImages, steps } = tree;

  const inCluster = new Uint8Array(nImages);
  const idxToFn = new Map<number, string>();
  let target = 0;
  for (const f of images) {
    const idx = fnToIdx.get(f);
    if (idx !== undefined && !inCluster[idx]) {
      inCluster[idx] = 1;
      idxToFn.set(idx, f);
      target++;
    }
  }
  if (target < 2) return null;

  const parent = new Int32Array(nImages);
  const sizeIn = new Int32Array(nImages);
  const sizeOut = new Int32Array(nImages);
  for (let i = 0; i < nImages; i++) {
    parent[i] = i;
    sizeIn[i] = inCluster[i] ? 1 : 0;
    sizeOut[i] = inCluster[i] ? 0 : 1;
  }
  const find = (x: number) => pathHalvingFind(parent, x);

  let lastInternalMerge: { ra: number; rb: number; distance: number } | null = null;

  for (const s of steps) {
    if (s.distance >= 1e10) break;
    const ra = find(s.clusterA);
    const rb = find(s.clusterB);
    if (ra === rb) continue;

    const pureA = sizeOut[ra]! === 0 && sizeIn[ra]! > 0;
    const pureB = sizeOut[rb]! === 0 && sizeIn[rb]! > 0;
    const internal = pureA && pureB;
    if (internal) {
      lastInternalMerge = { ra, rb, distance: s.distance };
    }

    parent[ra] = rb;
    sizeIn[rb] = sizeIn[rb]! + sizeIn[ra]!;
    sizeOut[rb] = sizeOut[rb]! + sizeOut[ra]!;

    if (sizeIn[rb]! >= target && sizeOut[rb]! === 0) break;
    if (sizeOut[rb]! > 0 && sizeIn[rb]! >= target) break;
  }

  if (!lastInternalMerge) return null;

  for (let i = 0; i < nImages; i++) parent[i] = i;
  const find2 = (x: number) => pathHalvingFind(parent, x);

  const target1 = lastInternalMerge.ra;
  const target2 = lastInternalMerge.rb;
  for (const s of steps) {
    if (s.distance >= 1e10) break;
    if (s.distance >= lastInternalMerge.distance) break;
    const ra = find2(s.clusterA);
    const rb = find2(s.clusterB);
    if (ra === rb) continue;
    if ((ra === target1 && rb === target2) || (ra === target2 && rb === target1)) continue;
    parent[ra] = rb;
  }

  const rootToImages = new Map<number, string[]>();
  for (let i = 0; i < nImages; i++) {
    if (!inCluster[i]) continue;
    const r = find2(i);
    if (!rootToImages.has(r)) rootToImages.set(r, []);
    rootToImages.get(r)!.push(idxToFn.get(i)!);
  }

  const buckets = [...rootToImages.values()];
  if (buckets.length < 2) {
    const sorted = [...images].sort();
    const mid = Math.floor(sorted.length / 2);
    return buildSplitChildren(targetDir, sorted.slice(0, mid), sorted.slice(mid));
  }

  buckets.sort((a, b) => b.length - a.length);
  const a = buckets[0]!;
  const b = buckets[1]!;
  const aSet = new Set(a);
  const bSet = new Set(b);
  for (let i = 2; i < buckets.length; i++) {
    if (a.length <= b.length) {
      for (const f of buckets[i]!) {
        a.push(f);
        aSet.add(f);
      }
    } else {
      for (const f of buckets[i]!) {
        b.push(f);
        bSet.add(f);
      }
    }
  }

  return buildSplitChildren(targetDir, a, b);
}

function buildSplitChildren(targetDir: string, a: string[], b: string[]): SplitResult {
  const groups = loadGroups(targetDir);
  const fnToGroup = new Map<string, { id: string; name: string; images: string[] }>();
  for (const g of groups) {
    for (const f of g.images) fnToGroup.set(f, g);
  }
  const ts = Date.now();
  const rand = () => Math.random().toString(36).slice(2, 8);
  const buildChild = (images: string[], idx: number): ClusterResultData => {
    const confirmed = pickConfirmed(images, fnToGroup);
    return {
      id: `split_${ts}_${rand()}`,
      name: confirmed?.name ?? `Split ${idx + 1}`,
      images: images.slice().sort(),
      confirmedGroup: confirmed,
    };
  };
  return { childA: buildChild(a, 0), childB: buildChild(b, 1) };
}

function pickConfirmed(
  images: string[],
  fnToGroup: Map<string, { id: string; name: string; images: string[] }>,
) {
  for (const f of images) {
    const g = fnToGroup.get(f);
    if (g) return { id: g.id, name: g.name, images: g.images };
  }
  return null;
}

// ── Expand candidates ────────────────────────────────────────────────────────

export interface ExpandCandidate {
  filename: string;
  distance: number;
}

export interface ExpandResult {
  candidates: ExpandCandidate[];
  /** P90 of intra-pair cosine distance within the source cluster — the slider's 1× anchor. */
  p90Intra: number;
  /** Maximum candidate distance; useful for slider scaling. */
  maxDistance: number;
}

export function expandCandidates(
  targetDir: string,
  sourceImages: string[],
  weights: WeightConfig | undefined,
): ExpandResult {
  const mapping = cachedHashMapping(targetDir);
  const fnToIdx = mapping.fnToIdx;
  const models = loadActiveModels(targetDir, weights);

  const sourceSet = new Set(sourceImages);
  const sourceIdx = sourceImages
    .map((f) => fnToIdx.get(f))
    .filter((x): x is number => x !== undefined);
  if (sourceIdx.length === 0) {
    return { candidates: [], p90Intra: 0, maxDistance: 0 };
  }
  const centroids = buildCentroids(models, sourceIdx);

  const p90Intra = computeP90IntraPair(models, sourceIdx);

  const candidates: ExpandCandidate[] = [];
  let maxDistance = 0;
  for (let i = 0; i < mapping.nImages; i++) {
    const fn = mapping.filenames[i]!;
    if (sourceSet.has(fn)) continue;
    const d = imageToCentroidDist(models, centroids, i);
    candidates.push({ filename: fn, distance: d });
    if (d > maxDistance) maxDistance = d;
  }
  candidates.sort((a, b) => a.distance - b.distance);
  return { candidates, p90Intra, maxDistance };
}

function computeP90IntraPair(models: ActiveModel[], indices: number[]): number {
  if (indices.length < 2) return 0;
  const SAMPLE_CAP = 4000;
  const totalPairs = (indices.length * (indices.length - 1)) / 2;
  const dists: number[] = [];
  if (totalPairs <= SAMPLE_CAP) {
    for (let i = 0; i < indices.length; i++) {
      for (let j = i + 1; j < indices.length; j++) {
        dists.push(pairwiseDist(models, indices[i]!, indices[j]!));
      }
    }
  } else {
    const stride = Math.max(1, Math.floor(totalPairs / SAMPLE_CAP));
    let pairIdx = 0;
    for (let i = 0; i < indices.length; i++) {
      for (let j = i + 1; j < indices.length; j++) {
        if (pairIdx % stride === 0) dists.push(pairwiseDist(models, indices[i]!, indices[j]!));
        pairIdx++;
      }
    }
  }
  if (dists.length === 0) return 0;
  dists.sort((a, b) => a - b);
  const p90 = dists[Math.floor(dists.length * 0.9)] ?? dists[dists.length - 1]!;
  return p90;
}
