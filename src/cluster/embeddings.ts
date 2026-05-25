// Image-model embedding loaders and the in-memory caches that keep them
// hot across requests. Drives auto-naming (TF-IDF), tree-navigation, and
// nearest-neighbor search.

import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import {
  type HashMapping,
  loadHashMapping,
  parseNpyFromNpz,
  reindexToFilenameOrder,
} from "../cache-utils.ts";
import { cacheDir, contentHashesPath, HASH_CACHE_FILE } from "../fs/paths.ts";

export const MODEL_KEYS = [
  "clip",
  "dino",
  "dinov3",
  "pecore_l",
  "pecore_g",
  "color",
  "learned_proj",
] as const;
export type ModelKey = (typeof MODEL_KEYS)[number];

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
  const npzBuf = readFileSync(join(cacheDir(targetDir), HASH_CACHE_FILE)) as Buffer;
  let hashOrdered: Float32Array;
  try {
    hashOrdered = parseNpyFromNpz(npzBuf, `${modelKey}.npy`);
  } catch {
    throw new ModelMissingError(modelKey);
  }
  const dim = hashOrdered.length / mapping.hashOrder.length;
  const data = reindexToFilenameOrder(hashOrdered, dim, mapping);
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
