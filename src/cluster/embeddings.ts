// Image-model embedding loaders and the in-memory caches that keep them
// hot across requests. Drives tree-navigation and nearest-neighbor search.

import { readFileSync, statSync } from "node:fs";
import {
  type HashMapping,
  loadHashMapping,
  parseNpyFromNpz,
  reindexToFilenameOrder,
  resolveHashCachePath,
} from "../cache-utils.ts";
import { cacheDir, contentHashesPath } from "../fs/paths.ts";
import { log } from "../log.ts";
import type { WeightConfig } from "../shared/types.ts";

export const MODEL_KEYS = [
  "dinov3",
  "pecore_g",
  "color",
  "learned_proj",
  "learned_proj_peg",
  "learned_proj_color",
] as const;
export type ModelKey = (typeof MODEL_KEYS)[number];

/** The models with positive weight, restricted to known MODEL_KEYS. */
export function activeModelsFromWeights(
  weights: WeightConfig,
): { key: ModelKey; weight: number }[] {
  const known = new Set<string>(MODEL_KEYS);
  const out: { key: ModelKey; weight: number }[] = [];
  for (const [key, val] of Object.entries(weights)) {
    const w = val ?? 0;
    if (w > 0 && known.has(key)) out.push({ key: key as ModelKey, weight: w });
  }
  return out;
}

export function l2Norm(row: Float32Array | Float64Array): number {
  let s = 0;
  for (let i = 0; i < row.length; i++) s += row[i]! * row[i]!;
  return Math.sqrt(s);
}

export class ModelMissingError extends Error {
  constructor(public modelKey: string) {
    super(`Model '${modelKey}' embedding not found in hash cache — run extraction first`);
  }
}

export interface ModelEmbedding {
  filenames: string[];
  data: Float32Array;
  dim: number;
  normalized: boolean; // color is not L2-normalized
}

const _modelEmbCaches = new Map<string, ModelEmbedding>();
let _hashMappingCache: { targetDir: string; mtime: number; mapping: HashMapping } | null = null;

/** mtime-keyed wrapper around loadHashMapping — content_hashes.json changes only on extraction/rename. */
export function cachedHashMapping(targetDir: string): HashMapping {
  const path = contentHashesPath(targetDir);
  const mtime = statSync(path).mtimeMs;
  if (
    _hashMappingCache &&
    _hashMappingCache.targetDir === targetDir &&
    _hashMappingCache.mtime === mtime
  ) {
    return _hashMappingCache.mapping;
  }
  const mapping = loadHashMapping(cacheDir(targetDir));
  _hashMappingCache = { targetDir, mtime, mapping };
  return mapping;
}

export function loadModelEmbedding(targetDir: string, modelKey: ModelKey): ModelEmbedding {
  const key = `${targetDir}::${modelKey}`;
  const cached = _modelEmbCaches.get(key);
  if (cached) return cached;

  const mapping = cachedHashMapping(targetDir);
  const npzBuf = readFileSync(resolveHashCachePath(cacheDir(targetDir))) as Buffer;
  let hashOrdered: Float32Array;
  try {
    hashOrdered = parseNpyFromNpz(npzBuf, `${modelKey}.npy`);
  } catch {
    throw new ModelMissingError(modelKey);
  }
  const dim = hashOrdered.length / mapping.hashOrder.length;
  const data = reindexToFilenameOrder(hashOrdered, dim, mapping);
  // Some extractions write NaN dimensions (observed in learned_proj heads:
  // a handful of dims NaN for every image). NaN poisons every dot product it
  // touches, silently flattening all downstream distances — zero them so the
  // remaining dimensions still rank.
  let sanitized = 0;
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i]!)) {
      data[i] = 0;
      sanitized++;
    }
  }
  if (sanitized > 0) {
    log("embeddings", `Sanitized ${sanitized} non-finite values in ${modelKey} embeddings`);
  }
  const emb: ModelEmbedding = {
    filenames: mapping.filenames,
    data,
    dim,
    normalized: modelKey !== "color",
  };
  _modelEmbCaches.set(key, emb);
  return emb;
}

export function clearEmbeddingsCache(): void {
  _modelEmbCaches.clear();
  _hashMappingCache = null;
}
