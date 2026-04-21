/**
 * Nearest-neighbor queries over cached image embeddings.
 *
 * Same weighted-cosine distance as the Rust cluster-tool, with two query
 * aggregations across multi-image queries:
 *   - "centroid": mean of query vectors per model (renormalized), cosine to all
 *   - "min":      per candidate, min distance across query images
 */

import type { NNAggregation, NNFilter, NNResult, WeightConfig } from "./client/types.ts";
import {
  cachedHashMapping,
  loadGroups,
  loadModelEmbedding,
  MODEL_KEYS,
  type ModelEmbedding,
  type ModelKey,
} from "./cluster.ts";

export { ModelMissingError } from "./cluster.ts";

export interface NNQueryOpts {
  topN: number;
  filter: NNFilter;
  aggregation: NNAggregation;
  excludeQuery: boolean;
  weights: WeightConfig;
  usePatches: boolean;
  restrictToFilenames?: string[];
  downsampleQueryTo?: number;
}

export interface NNQueryResult {
  results: NNResult[];
  usedModels: string[];
  queryCount: number;
  patchesBlended: boolean;
}

function activeModelsFromWeights(weights: WeightConfig): { key: ModelKey; weight: number }[] {
  const known = new Set<string>(MODEL_KEYS);
  const out: { key: ModelKey; weight: number }[] = [];
  for (const [key, val] of Object.entries(weights)) {
    const w = val ?? 0;
    if (w > 0 && known.has(key)) out.push({ key: key as ModelKey, weight: w });
  }
  return out;
}

function l2Norm(row: Float32Array | Float64Array): number {
  let s = 0;
  for (let i = 0; i < row.length; i++) s += row[i]! * row[i]!;
  return Math.sqrt(s);
}

function perModelDistances(
  emb: ModelEmbedding,
  queryIndices: number[],
  aggregation: NNAggregation,
): Float64Array {
  const { data, dim, normalized, filenames } = emb;
  const n = filenames.length;
  const distances = new Float64Array(n);

  if (aggregation === "centroid") {
    const centroid = new Float64Array(dim);
    for (const qi of queryIndices) {
      const row = data.subarray(qi * dim, (qi + 1) * dim);
      if (normalized) {
        for (let d = 0; d < dim; d++) centroid[d] = centroid[d]! + row[d]!;
      } else {
        const nrm = l2Norm(row) || 1e-10;
        for (let d = 0; d < dim; d++) centroid[d] = centroid[d]! + row[d]! / nrm;
      }
    }
    for (let d = 0; d < dim; d++) centroid[d] = centroid[d]! / queryIndices.length;
    const cNorm = l2Norm(centroid) || 1e-10;
    for (let d = 0; d < dim; d++) centroid[d] = centroid[d]! / cNorm;

    for (let i = 0; i < n; i++) {
      const row = data.subarray(i * dim, (i + 1) * dim);
      const rowNorm = normalized ? 1 : l2Norm(row) || 1e-10;
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += centroid[d]! * row[d]!;
      dot /= rowNorm;
      distances[i] = Math.max(0, 1 - dot);
    }
    return distances;
  }

  // aggregation === "min"
  const queryNorms = new Float64Array(queryIndices.length);
  for (let qi = 0; qi < queryIndices.length; qi++) {
    const q = queryIndices[qi]!;
    queryNorms[qi] = normalized ? 1 : l2Norm(data.subarray(q * dim, (q + 1) * dim)) || 1e-10;
  }
  for (let i = 0; i < n; i++) {
    const row = data.subarray(i * dim, (i + 1) * dim);
    const rowNorm = normalized ? 1 : l2Norm(row) || 1e-10;
    let minD = Infinity;
    for (let qi = 0; qi < queryIndices.length; qi++) {
      const q = queryIndices[qi]!;
      const qrow = data.subarray(q * dim, (q + 1) * dim);
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += qrow[d]! * row[d]!;
      dot /= rowNorm * queryNorms[qi]!;
      const dist = Math.max(0, 1 - dot);
      if (dist < minD) minD = dist;
    }
    distances[i] = minD;
  }
  return distances;
}

function condensedIdx(i: number, j: number, n: number): number {
  // i < j required; formula: i*n - i*(i+1)/2 + j - i - 1
  return i * n - (i * (i + 1)) / 2 + j - i - 1;
}

function aggregatePatchDistances(
  n: number,
  queryIndices: number[],
  condensed: Float64Array,
  aggregation: NNAggregation,
): Float64Array {
  const out = new Float64Array(n);
  const getDist = (i: number, j: number): number => {
    if (i === j) return 0;
    const [a, b] = i < j ? [i, j] : [j, i];
    return condensed[condensedIdx(a, b, n)]!;
  };

  if (aggregation === "centroid") {
    // Patches are pairwise — mean-over-query is the centroid analogue.
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (const q of queryIndices) sum += getDist(i, q);
      out[i] = sum / queryIndices.length;
    }
  } else {
    for (let i = 0; i < n; i++) {
      let minD = Infinity;
      for (const q of queryIndices) {
        const d = getDist(i, q);
        if (d < minD) minD = d;
      }
      out[i] = minD;
    }
  }
  return out;
}

function downsampleIndices(sorted: number[], maxCount: number): number[] {
  if (sorted.length <= maxCount) return sorted;
  const out: number[] = [];
  for (let i = 0; i < maxCount; i++) {
    const idx = Math.floor((i * sorted.length) / maxCount);
    out.push(sorted[idx]!);
  }
  return out;
}

export function findNearestNeighbors(
  targetDir: string,
  queryFilenames: string[],
  opts: NNQueryOpts,
  patchDistances: Float64Array | null,
): NNQueryResult {
  const mapping = cachedHashMapping(targetDir);
  const n = mapping.nImages;
  const fnToIdx = new Map(mapping.filenames.map((f, i) => [f, i]));

  const queryDedupe = [...new Set(queryFilenames)];
  const rawQueryIdx: number[] = [];
  for (const f of queryDedupe) {
    const idx = fnToIdx.get(f);
    if (idx !== undefined) rawQueryIdx.push(idx);
  }
  if (rawQueryIdx.length === 0) {
    throw new Error(
      "None of the query filenames have embeddings. Run extraction, or verify the filenames exist in content_hashes.json.",
    );
  }
  rawQueryIdx.sort((a, b) => a - b);

  const queryIndices = downsampleIndices(rawQueryIdx, opts.downsampleQueryTo ?? 64);

  let models = activeModelsFromWeights(opts.weights);
  if (models.length === 0) models = [{ key: "clip", weight: 1.0 }];

  const blended = new Float64Array(n);
  let totalWeight = 0;
  const usedModels: string[] = [];
  for (const { key, weight } of models) {
    const emb = loadModelEmbedding(targetDir, key);
    const d = perModelDistances(emb, queryIndices, opts.aggregation);
    for (let i = 0; i < n; i++) blended[i] = blended[i]! + weight * d[i]!;
    totalWeight += weight;
    usedModels.push(key);
  }

  let patchesBlended = false;
  if (opts.usePatches && patchDistances) {
    const patchAgg = aggregatePatchDistances(n, queryIndices, patchDistances, opts.aggregation);
    const patchWeight = 1.0;
    for (let i = 0; i < n; i++) blended[i] = blended[i]! + patchWeight * patchAgg[i]!;
    totalWeight += patchWeight;
    patchesBlended = true;
  }

  if (totalWeight > 0) {
    for (let i = 0; i < n; i++) blended[i] = blended[i]! / totalWeight;
  }

  const queryFilenameSet = new Set(queryDedupe);
  const excludeQuery = opts.excludeQuery !== false;
  const restrictSet = opts.restrictToFilenames ? new Set(opts.restrictToFilenames) : null;

  const fnToGroup = new Map<string, { id: string; name: string }>();
  for (const g of loadGroups(targetDir)) {
    for (const f of g.images) fnToGroup.set(f, { id: g.id, name: g.name });
  }

  const topN = Math.max(1, Math.min(opts.topN, 500));
  const candidates: { idx: number; dist: number }[] = [];
  for (let i = 0; i < n; i++) {
    const fn = mapping.filenames[i]!;
    if (excludeQuery && queryFilenameSet.has(fn)) continue;
    if (restrictSet && !restrictSet.has(fn)) continue;
    const group = fnToGroup.get(fn);
    if (opts.filter === "in-group" && !group) continue;
    if (opts.filter === "not-in-group" && group) continue;
    candidates.push({ idx: i, dist: blended[i]! });
  }

  candidates.sort((a, b) => a.dist - b.dist);
  const top = candidates.slice(0, topN);

  const results: NNResult[] = top.map((c) => {
    const fn = mapping.filenames[c.idx]!;
    const group = fnToGroup.get(fn);
    return {
      filename: fn,
      distance: c.dist,
      inGroupId: group?.id ?? null,
      inGroupName: group?.name ?? null,
    };
  });

  return { results, usedModels, queryCount: queryIndices.length, patchesBlended };
}
