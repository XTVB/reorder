// Distance matrices used by the clustering pipeline:
//   - rerank_dist_matrix.bin — k-reciprocal re-ranked distances
//   - patch_dist_matrix.bin  — DINOv3 patch-match distances
//
// Both use signature-based cache invalidation via the sidecar `.meta.json`
// file (content-hash + model-version + algo-version) so we don't false-
// invalidate when content_hashes.json is rewritten on every extract run, and
// don't false-validate when images are deleted.

import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import {
  computeCacheSignature,
  readSidecarSignature,
  writeSidecarSignature,
} from "../cache-utils.ts";
import {
  cacheDir,
  contentHashesPath,
  DINOV3_PATCHES_FILE,
  DINOV3_PATCHES_HASHES_FILE,
  HASH_CACHE_FILE,
  patchDistMatrixPath,
  rerankDistMatrixPath,
} from "../fs/paths.ts";
import { log } from "../log.ts";
import type { WeightConfig } from "../shared/types.ts";
import { GROUP_SIM_BINARY, PYTHON, SCRIPTS_DIR } from "./binaries.ts";
import { spawn } from "./subprocess.ts";

// k-reciprocal re-ranking parameters. Hardcoded — the benchmark sweep showed
// these are robust across datasets (ARI plateau over k1∈[60,80], k2∈[4,8]).
const RERANK_K1 = 65;
const RERANK_K2 = 4;
const RERANK_VERSION = "v1-jaccard-kernel"; // bump to invalidate all caches

// Bump to invalidate all existing patch_dist_matrix caches (e.g. after an
// algorithmic change in rust/group-similarity).
const PATCH_DIST_VERSION = "v1";

export { patchDistMatrixPath, rerankDistMatrixPath };

function sidecarPath(matrixPath: string): string {
  return `${matrixPath}.meta.json`;
}

/** Ensure the k-reciprocal re-ranking distance matrix exists and is current. */
export async function ensureRerankDistMatrix(
  targetDir: string,
  weights: WeightConfig,
  onProgress?: (line: string) => void,
): Promise<string> {
  const cache = cacheDir(targetDir);
  const hashCachePath = join(cache, HASH_CACHE_FILE);
  const matrixPath = rerankDistMatrixPath(targetDir);
  const sidecar = sidecarPath(matrixPath);

  if (!existsSync(hashCachePath)) {
    throw new Error("Embedding cache not found. Run feature extraction first.");
  }
  if (!existsSync(PYTHON)) {
    throw new Error(`Python not found at ${PYTHON}. See README for venv setup.`);
  }

  const currentSignature = computeCacheSignature({
    cacheDir: cache,
    weights: weights as Record<string, number | undefined>,
    extra: { k1: RERANK_K1, k2: RERANK_K2, rerank_version: RERANK_VERSION },
  });

  if (existsSync(matrixPath) && readSidecarSignature(sidecar) === currentSignature) {
    log("cluster", "Using cached re-rank distance matrix");
    return matrixPath;
  }

  log("cluster", `Precomputing re-rank distance matrix (k1=${RERANK_K1}, k2=${RERANK_K2})...`);
  onProgress?.("Computing re-rank distance matrix...");

  const script = join(SCRIPTS_DIR, "precompute_rerank_distance.py");
  await spawn(
    [
      PYTHON,
      script,
      "--cache-dir",
      cache,
      "--output",
      matrixPath,
      "--weights",
      JSON.stringify(weights ?? {}),
      "--k1",
      String(RERANK_K1),
      "--k2",
      String(RERANK_K2),
    ],
    { label: "rerank-dist-matrix", onProgress },
  );
  writeSidecarSignature(sidecar, currentSignature);
  return matrixPath;
}

/** Ensure the DINOv3 patch distance matrix exists; recompute via group-similarity if stale/missing. */
export async function ensurePatchDistMatrix(
  targetDir: string,
  onProgress?: (line: string) => void,
): Promise<string> {
  const cache = cacheDir(targetDir);
  const patchesCachePath = join(cache, DINOV3_PATCHES_FILE);
  const patchesHashesPath = join(cache, DINOV3_PATCHES_HASHES_FILE);
  const distMatrixPath = patchDistMatrixPath(targetDir);
  const sidecar = sidecarPath(distMatrixPath);

  if (!existsSync(patchesCachePath)) {
    throw new Error(
      "DINOv3 patches cache not found. Run feature extraction with --required dinov3 first.",
    );
  }
  if (!existsSync(GROUP_SIM_BINARY)) {
    throw new Error(
      `group-similarity binary not found. Build with: cd rust/group-similarity && cargo build --release`,
    );
  }

  // Patches use only DINOv3 features, so we mark dinov3 as the active model
  // for the signature — this picks up its `_v_dinov3` version string so a
  // re-extracted DINOv3 invalidates the matrix.
  const currentSignature = computeCacheSignature({
    cacheDir: cache,
    weights: { dinov3: 1.0 },
    extra: { algo: "patch-dist-matrix", patch_dist_version: PATCH_DIST_VERSION },
  });

  if (existsSync(distMatrixPath) && readSidecarSignature(sidecar) === currentSignature) {
    log("cluster", "Using cached patch-based distance matrix");
    return distMatrixPath;
  }

  log("cluster", "Precomputing patch-based distance matrix...");
  onProgress?.("Computing patch distance matrix...");
  await spawn(
    [
      GROUP_SIM_BINARY,
      "--patches-cache",
      patchesCachePath,
      "--content-hashes",
      contentHashesPath(targetDir),
      "--patches-hashes",
      patchesHashesPath,
      "--groups",
      "",
      "--mode",
      "dist-matrix",
      "--output",
      distMatrixPath,
    ],
    { label: "patch-dist-matrix", onProgress },
  );
  writeSidecarSignature(sidecar, currentSignature);
  return distMatrixPath;
}

let _patchDistMatrixCache: {
  targetDir: string;
  mtime: number;
  n: number;
  distances: Float64Array;
} | null = null;

/** Read patch_dist_matrix.bin: [u64 n][f64×n*(n-1)/2] condensed upper triangle. mtime-cached. */
export function loadPatchDistMatrix(targetDir: string): { n: number; distances: Float64Array } {
  const path = patchDistMatrixPath(targetDir);
  const mtime = statSync(path).mtimeMs;
  if (
    _patchDistMatrixCache &&
    _patchDistMatrixCache.targetDir === targetDir &&
    _patchDistMatrixCache.mtime === mtime
  ) {
    return { n: _patchDistMatrixCache.n, distances: _patchDistMatrixCache.distances };
  }
  const buf = readFileSync(path) as Buffer;
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  // n stored as u64 LE — low 32 bits suffice for our scale
  const n = view.getUint32(0, true);
  const nHi = view.getUint32(4, true);
  if (nHi !== 0) throw new Error("Patch dist matrix: n > 2^32 not supported");
  const nPairs = (n * (n - 1)) / 2;
  const distances = new Float64Array(buf.buffer, buf.byteOffset + 8, nPairs);
  _patchDistMatrixCache = { targetDir, mtime, n, distances };
  return { n, distances };
}

export function clearPatchDistMatrixCache(): void {
  _patchDistMatrixCache = null;
}
