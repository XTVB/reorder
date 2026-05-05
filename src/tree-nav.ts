/**
 * Server-side support for the tree-navigation cluster operations
 * (merge / split / expand) defined in CLUSTERING_UI_SPLIT_SPEC.md.
 *
 * All operations work on top of the cached linkage tree + cached embeddings.
 * They never mutate persistent state — the client is the source of truth for
 * the in-progress cluster view; the server only computes derived values
 * (metrics, candidate rankings, binary splits).
 */

import type { ClusterResultData } from "./client/types.ts";
import {
  cachedHashMapping,
  computeAutoNames,
  type LinkageTree,
  loadGroups,
  loadModelEmbedding,
  loadTree,
  MODEL_KEYS,
  type ModelEmbedding,
  type ModelKey,
  type WeightConfig,
} from "./cluster.ts";

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
    // Fall back to CLIP at unit weight, matching nn-query semantics
    out.push({ emb: loadModelEmbedding(targetDir, "clip"), weight: 1.0 });
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
  const fnToIdx = new Map(mapping.filenames.map((f, i) => [f, i]));
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

// ── Merge candidate ranking ──────────────────────────────────────────────────

export interface MergeCandidateScore {
  id: string;
  distance: number;
}

interface CandidateInput {
  id: string;
  images: string[];
}

interface TreeNeighbours {
  /** Images on the other side of the first merge after source forms — its sibling. */
  sibling: Set<number>;
  /** Images on the other side of the SECOND merge above source — descendants of the parent's sibling. */
  cousins: Set<number>;
}

/**
 * Walk the linkage tree to find both the source's tree-sibling and tree-cousins.
 *
 * Tree sibling: the cluster the source would merge with first.
 * Tree cousins: the cluster on the other side of the next-next merge —
 *               i.e. descendants of the source's parent's sibling.
 * Returns null if the source never forms a clean component in the tree.
 */
function findTreeNeighbours(
  tree: LinkageTree,
  fnToIdx: Map<string, number>,
  sourceImages: string[],
): TreeNeighbours | null {
  const { nImages, steps } = tree;
  const inSource = new Uint8Array(nImages);
  let target = 0;
  for (const f of sourceImages) {
    const idx = fnToIdx.get(f);
    if (idx !== undefined && !inSource[idx]) {
      inSource[idx] = 1;
      target++;
    }
  }
  if (target === 0) return null;

  const parent = new Int32Array(nImages);
  const sizeIn = new Int32Array(nImages);
  const sizeOut = new Int32Array(nImages);
  // Linked-list of every image rooted at each component, so we can recover
  // membership cheaply when the merge that joins source with non-source fires.
  const next = new Int32Array(nImages).fill(-1);
  const head = new Int32Array(nImages);
  const tail = new Int32Array(nImages);
  for (let i = 0; i < nImages; i++) {
    parent[i] = i;
    sizeIn[i] = inSource[i] ? 1 : 0;
    sizeOut[i] = inSource[i] ? 0 : 1;
    head[i] = i;
    tail[i] = i;
  }
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

  let formed = false;
  let formedRoot = -1;
  // After the cluster forms, we need to find TWO upward merges:
  //   #1 unifies (source) with its sibling → sibling capture point
  //   #2 unifies (source ∪ sibling) with parent's sibling → cousins capture point
  let sibling: Set<number> | null = null;
  let cousins: Set<number> | null = null;
  let trackedRoot = -1; // the union-find root of (source ∪ sibling) after merge #1

  function captureSide(rootIdx: number): Set<number> {
    const out = new Set<number>();
    for (let j = head[rootIdx]!; j >= 0; j = next[j]!) out.add(j);
    return out;
  }

  for (const s of steps) {
    if (s.distance >= 1e10) break;
    const ra = find(s.clusterA);
    const rb = find(s.clusterB);
    if (ra === rb) continue;

    if (formed) {
      if (sibling === null) {
        const fr = find(formedRoot);
        if (fr === ra || fr === rb) {
          const otherRoot = fr === ra ? rb : ra;
          sibling = captureSide(otherRoot);
          // Mark the post-merge root for the next iteration's cousin detection.
          // Capture it BEFORE we union so we can re-find it on the next loop.
          trackedRoot = fr; // will be path-compressed; use find() on next iter
        }
      } else if (cousins === null) {
        const tr = find(trackedRoot);
        if (tr === ra || tr === rb) {
          const otherRoot = tr === ra ? rb : ra;
          cousins = captureSide(otherRoot);
        }
      }
    }

    // Union ra into rb (merge linked lists too)
    parent[ra] = rb;
    sizeIn[rb] = sizeIn[rb]! + sizeIn[ra]!;
    sizeOut[rb] = sizeOut[rb]! + sizeOut[ra]!;
    next[tail[rb]!] = head[ra]!;
    tail[rb] = tail[ra]!;

    if (!formed && sizeIn[rb]! === target && sizeOut[rb]! === 0) {
      formed = true;
      formedRoot = rb;
    }

    if (sibling !== null && cousins !== null) break;
  }
  if (sibling === null) return null;
  return { sibling, cousins: cousins ?? new Set() };
}

/**
 * Score how well a candidate's image-index set matches a target index set,
 * Jaccard-style. Candidates with no overlap return 0.
 */
function jaccardOverlap(candidateIdx: number[], target: Set<number>): number {
  if (target.size === 0 || candidateIdx.length === 0) return 0;
  let overlap = 0;
  for (const idx of candidateIdx) if (target.has(idx)) overlap++;
  if (overlap === 0) return 0;
  return overlap / (candidateIdx.length + target.size - overlap);
}

/**
 * Build the global "top-K nearest images" of the source by centroid distance,
 * returned as an index-set. Used as the kNN-overlap signal: candidates whose
 * images appear in this set share many nearest-neighbours with the source.
 */
function topKNearestIndices(
  models: ActiveModel[],
  centroids: Float64Array[],
  nImages: number,
  sourceSet: Set<number>,
  k: number,
): Set<number> {
  const heap: { idx: number; dist: number }[] = [];
  for (let i = 0; i < nImages; i++) {
    if (sourceSet.has(i)) continue;
    const d = imageToCentroidDist(models, centroids, i);
    if (heap.length < k) {
      heap.push({ idx: i, dist: d });
      if (heap.length === k) heap.sort((a, b) => b.dist - a.dist); // worst at front
    } else if (d < heap[0]!.dist) {
      heap[0] = { idx: i, dist: d };
      heap.sort((a, b) => b.dist - a.dist);
    }
  }
  return new Set(heap.map((e) => e.idx));
}

/**
 * Rank candidate clusters by a composite "likely-to-merge" signal, per the
 * spec ordering of factors:
 *   1. Tree sibling: the cluster source would merge with at the next
 *      linkage step (always rank 1 if present).
 *   2. Tree cousins: descendants of source's parent's sibling — bumped above
 *      the centroid baseline.
 *   3. Centroid distance under the active weighted-cosine metric.
 *   4. Shared-kNN overlap: clusters whose images appear among source's
 *      top-K nearest neighbours globally — small additional boost.
 * Output is a single ranked list; the user never sees which signal a
 * candidate came from.
 */
export function rankMergeCandidates(
  targetDir: string,
  sourceImages: string[],
  candidates: CandidateInput[],
  weights: WeightConfig | undefined,
): MergeCandidateScore[] {
  const mapping = cachedHashMapping(targetDir);
  const fnToIdx = new Map(mapping.filenames.map((f, i) => [f, i]));
  const models = loadActiveModels(targetDir, weights);

  const sourceIdx = sourceImages
    .map((f) => fnToIdx.get(f))
    .filter((x): x is number => x !== undefined);
  if (sourceIdx.length === 0) return [];
  const sourceSet = new Set(sourceIdx);
  const sourceCentroid = buildCentroids(models, sourceIdx);

  // Tree neighbours from the linkage tree.
  const neighbours = findTreeNeighbours(loadTree(targetDir), fnToIdx, sourceImages);
  const sibling = neighbours?.sibling ?? new Set<number>();
  const cousins = neighbours?.cousins ?? new Set<number>();

  // Pre-compute candidate index sets and centroid distances.
  type Scored = {
    id: string;
    distance: number;
    cIdx: number[];
    siblingScore: number;
    cousinScore: number;
  };
  const scored: Scored[] = [];
  for (const c of candidates) {
    const cIdx = c.images.map((f) => fnToIdx.get(f)).filter((x): x is number => x !== undefined);
    if (cIdx.length === 0) continue;
    const cCentroid = buildCentroids(models, cIdx);
    let total = 0;
    let totalWeight = 0;
    for (let m = 0; m < models.length; m++) {
      const { weight } = models[m]!;
      const sa = sourceCentroid[m]!;
      const ca = cCentroid[m]!;
      let dot = 0;
      for (let d = 0; d < sa.length; d++) dot += sa[d]! * ca[d]!;
      total += weight * Math.max(0, 1 - dot);
      totalWeight += weight;
    }
    const distance = totalWeight > 0 ? total / totalWeight : 0;
    scored.push({
      id: c.id,
      distance,
      cIdx,
      siblingScore: jaccardOverlap(cIdx, sibling),
      cousinScore: jaccardOverlap(cIdx, cousins),
    });
  }

  // Pick the single best tree-sibling candidate (highest Jaccard overlap).
  let bestSiblingId: string | null = null;
  let bestSiblingScore = 0;
  for (const s of scored) {
    if (s.siblingScore > bestSiblingScore) {
      bestSiblingScore = s.siblingScore;
      bestSiblingId = s.id;
    }
  }

  // kNN-overlap signal: top-K nearest images of source globally; candidates
  // whose images appear in that set get a small boost on the tail of the ranking.
  const KNN_K = Math.min(80, Math.max(20, sourceIdx.length * 4));
  const topK =
    candidates.length > 0
      ? topKNearestIndices(models, sourceCentroid, mapping.nImages, sourceSet, KNN_K)
      : new Set<number>();

  type Ranked = { id: string; distance: number; bucket: number; knnBoost: number };
  const ranked: Ranked[] = scored.map((s) => ({
    id: s.id,
    distance: s.distance,
    bucket: s.id === bestSiblingId ? 0 : s.cousinScore > 0 ? 1 : 2,
    knnBoost: jaccardOverlap(s.cIdx, topK),
  }));

  // Order: bucket 0 (sibling) → bucket 1 (cousins) → bucket 2 (other).
  // Within a bucket: ascending centroid distance, with kNN-overlap as tiebreak.
  ranked.sort((a, b) => {
    if (a.bucket !== b.bucket) return a.bucket - b.bucket;
    if (a.distance !== b.distance) return a.distance - b.distance;
    return b.knnBoost - a.knnBoost;
  });
  return ranked.map(({ id, distance }) => ({ id, distance }));
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
  const fnToIdx = new Map(mapping.filenames.map((f, i) => [f, i]));
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

  // Walk merges in order. The "last internal merge" — the one whose result is
  // the full cluster (or the largest internal-merge distance, if the cluster
  // never forms cleanly) — is what we undo to produce the binary split.
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

    // Detect cluster fully formed → stop walking
    parent[ra] = rb;
    sizeIn[rb] = sizeIn[rb]! + sizeIn[ra]!;
    sizeOut[rb] = sizeOut[rb]! + sizeOut[ra]!;

    if (sizeIn[rb]! >= target && sizeOut[rb]! === 0) break;
    if (sizeOut[rb]! > 0 && sizeIn[rb]! >= target) break;
  }

  if (!lastInternalMerge) return null;

  // Replay merges fresh, EXCEPT skip the chosen one. After replay, find the two
  // components inside the cluster; those are our two children.
  for (let i = 0; i < nImages; i++) parent[i] = i;
  function find2(x: number): number {
    let p = parent[x]!;
    while (p !== x) {
      const gp = parent[p]!;
      parent[x] = gp;
      x = p;
      p = gp;
    }
    return x;
  }

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

  // Each in-cluster image gets bucketed by its current root.
  const rootToImages = new Map<number, string[]>();
  for (let i = 0; i < nImages; i++) {
    if (!inCluster[i]) continue;
    const r = find2(i);
    if (!rootToImages.has(r)) rootToImages.set(r, []);
    rootToImages.get(r)!.push(idxToFn.get(i)!);
  }

  const buckets = [...rootToImages.values()];
  if (buckets.length < 2) {
    // Fallback: split arbitrarily down the middle so we always have two children.
    const sorted = [...images].sort();
    const mid = Math.floor(sorted.length / 2);
    return buildSplitChildren(targetDir, sorted.slice(0, mid), sorted.slice(mid));
  }

  // Take the two largest as the children, fold remainders into the closer one.
  buckets.sort((a, b) => b.length - a.length);
  const a = buckets[0]!;
  const b = buckets[1]!;
  const aSet = new Set(a);
  const bSet = new Set(b);
  for (let i = 2; i < buckets.length; i++) {
    // Distribute by which of {a, b} the bucket would have merged with first
    // — proxied by simple size affinity: smaller bucket folds into smaller side.
    // (Keeps things deterministic without another tree walk.)
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
  const rawClusters = [
    {
      id: `split_${ts}_${rand()}`,
      images: a.slice().sort(),
      confirmed_group: pickConfirmed(a, fnToGroup),
    },
    {
      id: `split_${ts}_${rand()}`,
      images: b.slice().sort(),
      confirmed_group: pickConfirmed(b, fnToGroup),
    },
  ];
  let named: ClusterResultData[];
  try {
    named = computeAutoNames(targetDir, rawClusters);
  } catch {
    named = rawClusters.map((c, i) => ({
      id: c.id,
      autoName: c.confirmed_group?.name ?? `Split ${i + 1}`,
      autoTags: [],
      images: c.images,
      confirmedGroup: c.confirmed_group,
    }));
  }
  return { childA: named[0]!, childB: named[1]! };
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

/** Compute distance from source centroid to every image outside `sourceImages`. */
export function expandCandidates(
  targetDir: string,
  sourceImages: string[],
  weights: WeightConfig | undefined,
): ExpandResult {
  const mapping = cachedHashMapping(targetDir);
  const fnToIdx = new Map(mapping.filenames.map((f, i) => [f, i]));
  const models = loadActiveModels(targetDir, weights);

  const sourceSet = new Set(sourceImages);
  const sourceIdx = sourceImages
    .map((f) => fnToIdx.get(f))
    .filter((x): x is number => x !== undefined);
  if (sourceIdx.length === 0) {
    return { candidates: [], p90Intra: 0, maxDistance: 0 };
  }
  const centroids = buildCentroids(models, sourceIdx);

  // P90 of intra-pair distance within source — used as 1× threshold reference.
  const p90Intra = computeP90IntraPair(models, sourceIdx);

  // Distance to centroid for every image not in source
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
