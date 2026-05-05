// TF-IDF cluster auto-naming using cached CLIP image embeddings against a
// fixed text-embedding vocabulary (`text_embeddings.json`).
//
// Also owns the on-disk text-embeddings file (precompute_text_embeddings.py),
// which is precomputed once per machine.

import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { cacheDir, HASH_CACHE_FILE, textEmbeddingsPath } from "../fs/paths.ts";
import { log } from "../log.ts";
import type { ClusterResultData } from "../shared/types.ts";
import { PYTHON, SCRIPTS_DIR } from "./binaries.ts";
import { loadModelEmbedding } from "./embeddings.ts";
import { spawn } from "./subprocess.ts";

interface TextEmbeddingsRaw {
  terms: string[];
  embeddings: number[][]; // [n_terms][512]
}

interface TextEmbeddings {
  terms: string[];
  flat: Float32Array; // flattened [n_terms * dim]
  dim: number;
}

let _textEmbCache: TextEmbeddings | null = null;
// Keyed by NPZ mtime: stats depend only on CLIP embedding values, which only
// change when the hash-cache NPZ is rewritten by extraction.
let _tfidfStatsCache: {
  targetDir: string;
  npzMtime: number;
  globalAvg: Float64Array;
  globalStd: Float64Array;
} | null = null;

export async function ensureTextEmbeddings(targetDir: string): Promise<string> {
  const path = textEmbeddingsPath(targetDir);
  if (existsSync(path)) return path;

  const script = join(SCRIPTS_DIR, "precompute_text_embeddings.py");
  log("cluster", `Precomputing text embeddings...`);
  await spawn([PYTHON, script, path], { label: "text-embeddings", progressPrefix: null });
  return path;
}

function loadTextEmbeddings(targetDir: string): TextEmbeddings {
  if (_textEmbCache) return _textEmbCache;
  const path = textEmbeddingsPath(targetDir);
  const raw: TextEmbeddingsRaw = JSON.parse(readFileSync(path, "utf-8"));
  const nTerms = raw.terms.length;
  if (nTerms === 0 || raw.embeddings.length === 0) {
    throw new Error("Text embeddings file is empty");
  }
  const dim = raw.embeddings[0]!.length;
  const flat = new Float32Array(nTerms * dim);
  for (let t = 0; t < nTerms; t++) {
    const row = raw.embeddings[t]!;
    for (let d = 0; d < dim; d++) {
      flat[t * dim + d] = row[d]!;
    }
  }
  _textEmbCache = { terms: raw.terms, flat, dim };
  return _textEmbCache;
}

export function clearTfidfCache(): void {
  _textEmbCache = null;
  _tfidfStatsCache = null;
}

/** Shape of cluster passed in from the Rust cluster-tool output. */
export interface NamedClusterInput {
  id: string;
  images: string[];
  confirmedGroup: { id: string; name: string; images: string[] } | null;
}

/** Fallback when CLIP embeddings aren't available — uses confirmed group names or generic labels. */
export function clustersWithoutAutoNames(clusters: NamedClusterInput[]): ClusterResultData[] {
  return clusters.map((c, i) => ({
    id: c.id,
    autoName: c.confirmedGroup?.name ?? `Cluster ${i + 1}`,
    autoTags: [],
    images: c.images,
    confirmedGroup: c.confirmedGroup,
  }));
}

export function computeAutoNames(
  targetDir: string,
  clusters: NamedClusterInput[],
): ClusterResultData[] {
  const { terms, flat: textFlat, dim: textDim } = loadTextEmbeddings(targetDir);
  const { filenames, data: clip, dim } = loadModelEmbedding(targetDir, "clip");
  const nImages = filenames.length;
  const fnToIdx = new Map(filenames.map((f, i) => [f, i]));

  const nTerms = terms.length;

  const npzMtime = statSync(join(cacheDir(targetDir), HASH_CACHE_FILE)).mtimeMs;
  if (
    !_tfidfStatsCache ||
    _tfidfStatsCache.targetDir !== targetDir ||
    _tfidfStatsCache.npzMtime !== npzMtime
  ) {
    // All typed array accesses below are within bounds: t < nTerms, d < dim, img < nImages
    const globalAvg = new Float64Array(nTerms);
    const globalSumSq = new Float64Array(nTerms);
    for (let img = 0; img < nImages; img++) {
      for (let t = 0; t < nTerms; t++) {
        let dot = 0;
        for (let d = 0; d < dim; d++) {
          dot += clip[img * dim + d]! * textFlat[t * textDim + d]!;
        }
        globalAvg[t] = globalAvg[t]! + dot;
        globalSumSq[t] = globalSumSq[t]! + dot * dot;
      }
    }
    const globalStd = new Float64Array(nTerms);
    for (let t = 0; t < nTerms; t++) {
      globalAvg[t] = globalAvg[t]! / nImages;
      const variance = globalSumSq[t]! / nImages - globalAvg[t]! * globalAvg[t]!;
      globalStd[t] = Math.sqrt(Math.max(0, variance));
    }
    _tfidfStatsCache = { targetDir, npzMtime, globalAvg, globalStd };
  }
  const { globalAvg, globalStd } = _tfidfStatsCache;

  return clusters.map((c) => {
    // If this cluster has a confirmed group, use its name
    if (c.confirmedGroup && c.images.length === c.confirmedGroup.images.length) {
      return {
        id: c.id,
        autoName: c.confirmedGroup.name,
        autoTags: [],
        images: c.images,
        confirmedGroup: c.confirmedGroup,
      };
    }

    // Compute cluster centroid in CLIP space
    const centroid = new Float64Array(dim);
    let count = 0;
    for (const f of c.images) {
      const idx = fnToIdx.get(f);
      if (idx === undefined) continue;
      for (let d = 0; d < dim; d++) {
        centroid[d] = centroid[d]! + clip[idx * dim + d]!;
      }
      count++;
    }
    if (count > 0) {
      let norm = 0;
      for (let d = 0; d < dim; d++) {
        centroid[d] = centroid[d]! / count;
        norm += centroid[d]! * centroid[d]!;
      }
      norm = Math.sqrt(norm);
      if (norm > 1e-10) {
        for (let d = 0; d < dim; d++) centroid[d] = centroid[d]! / norm;
      }
    }

    const zScores: { term: string; z: number }[] = [];
    for (let t = 0; t < nTerms; t++) {
      let dot = 0;
      for (let d = 0; d < dim; d++) {
        dot += centroid[d]! * textFlat[t * textDim + d]!;
      }
      const z = globalStd[t]! > 1e-10 ? (dot - globalAvg[t]!) / globalStd[t]! : 0;
      zScores.push({ term: terms[t]!, z });
    }
    zScores.sort((a, b) => b.z - a.z);

    // Build name from top 3 non-redundant terms
    const parts: string[] = [];
    for (const { term } of zScores) {
      if (parts.length >= 3) break;
      const short = term.split(" ")[0]!.toLowerCase();
      if (
        !parts.some(
          (p) =>
            p.toLowerCase().startsWith(short) || short.startsWith(p.toLowerCase().split("_")[0]!),
        )
      ) {
        parts.push(term.split(" ").slice(0, 2).join("_"));
      }
    }
    const autoName = c.confirmedGroup ? c.confirmedGroup.name : parts.join(" · ");

    return {
      id: c.id,
      autoName,
      autoTags: zScores.slice(0, 8),
      images: c.images,
      confirmedGroup: c.confirmedGroup,
    };
  });
}
