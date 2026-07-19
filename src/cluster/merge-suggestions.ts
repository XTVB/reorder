// Merge suggestions: pairwise group similarity. Two methods, both computed by
// the Rust group-similarity binary and sharing one wire shape (GroupPairResult):
//   - "patches"    — DINOv3 patch matching ("merge-suggestions" mode).
//   - "embeddings" — weighted blend of CLS embeddings ("embeddings" mode),
//                    reusing the same per-model weights as the cluster pipeline.
// Cached in merge_suggestions2{_full|_emb_<sig>}{_mN}.json keyed by the relevant
// inputs' mtimes. The cache always holds the full unfiltered pair set; rejected
// merge pairs are filtered at return time (so a rejection never invalidates the
// cache, and ordering callers can ask for the real scores via includeRejected).

import { existsSync } from "node:fs";
import { stat } from "node:fs/promises";
import { join } from "node:path";
import { ensureHashOrderJson, resolveHashCachePath } from "../cache-utils.ts";
import {
  cacheDir,
  contentHashesPath,
  DINOV3_PATCHES_FILE,
  DINOV3_PATCHES_FULL_FILE,
  DINOV3_PATCHES_HASHES_FILE,
  groupsPath,
  HASH_ORDER_FILE,
} from "../fs/paths.ts";
import { log } from "../log.ts";
import type { WeightConfig } from "../shared/types.ts";
import { GROUP_SIM_BINARY } from "./binaries.ts";
import { loadConstraints, mergePairKey } from "./constraints.ts";
import { availableLearnedKeys, rescaleLearnedProjWeight } from "./pipeline.ts";
import { spawn } from "./subprocess.ts";

export type MergeMethod = "patches" | "embeddings";

/** Stable short signature of the rescaled weights, for the embeddings cache filename. */
function weightSignature(weights: WeightConfig, availableLearned?: ReadonlySet<string>): string {
  const rescaled = rescaleLearnedProjWeight(weights, availableLearned);
  // Deterministic key order; round so float noise doesn't churn the cache.
  const parts = (
    [
      "pecore_g",
      "dinov3",
      "color",
      "learned_proj",
      "learned_proj_peg",
      "learned_proj_color",
    ] as const
  )
    .map((k) => `${k}:${(rescaled[k] ?? 0).toFixed(4)}`)
    .join("|");
  // djb2 → base36 keeps the filename short and filesystem-safe.
  let h = 5381;
  for (let i = 0; i < parts.length; i++) h = (h * 33) ^ parts.charCodeAt(i);
  return (h >>> 0).toString(36);
}

export interface GroupPairResult {
  groupA: string;
  groupB: string;
  sizeA: number;
  sizeB: number;
  patchMedian: number;
  patchP75: number;
  patchBest: number;
  closestPair: [string, string];
}

export async function computeMergeSuggestions(
  targetDir: string,
  minScore = 0.55,
  options?: {
    method?: MergeMethod;
    fullResolution?: boolean;
    maxCombinedSize?: number;
    weights?: WeightConfig;
    /**
     * Keep pairs the user rejected on the merge page. A rejection means "don't
     * combine these groups", not "these aren't similar" — ordering callers need
     * the real scores, otherwise a rejected pair reads as maximally dissimilar.
     */
    includeRejected?: boolean;
    onProgress?: (msg: string) => void;
  },
): Promise<GroupPairResult[]> {
  const cache = cacheDir(targetDir);
  const method: MergeMethod = options?.method ?? "embeddings";
  const fullRes = options?.fullResolution ?? false;
  const maxCombinedSize = Math.max(0, Math.floor(options?.maxCombinedSize ?? 0));
  const contentHashesP = contentHashesPath(targetDir);
  const groupsP = groupsPath(targetDir);
  const sizeSuffix = maxCombinedSize > 0 ? `_m${maxCombinedSize}` : "";

  if (!existsSync(GROUP_SIM_BINARY)) {
    throw new Error(
      `group-similarity binary not found at ${GROUP_SIM_BINARY}. Build with: cd rust/group-similarity && cargo build --release`,
    );
  }

  // Per-method setup: the primary input file (whose mtime gates the cache), the
  // result cache path, the extra subprocess args, and a progress label.
  let primaryInputPath: string;
  let resultCachePath: string;
  const args = [GROUP_SIM_BINARY, "--content-hashes", contentHashesP, "--groups", groupsP];
  let label: string;
  let loadingMsg: string;

  if (method === "embeddings") {
    const hashCachePath = resolveHashCachePath(cache);
    if (!existsSync(hashCachePath)) {
      throw new Error(
        "Embeddings cache not found. Run feature extraction (Compute in Cluster mode) first.",
      );
    }
    const weights = options?.weights ?? {};
    if (!Object.values(weights).some((v) => (v ?? 0) > 0)) {
      throw new Error("Embeddings merge mode requires at least one positive model weight.");
    }
    ensureHashOrderJson(cache); // regenerate the hash_cache_order.json sidecar if stale
    const hashOrderPath = join(cache, HASH_ORDER_FILE);
    primaryInputPath = hashCachePath;
    const availableLearned = availableLearnedKeys(hashCachePath);
    resultCachePath = join(
      cache,
      `merge_suggestions2_emb_${weightSignature(weights, availableLearned)}${sizeSuffix}.json`,
    );
    args.push("--mode", "embeddings", "--hash-cache", hashCachePath, "--hash-order", hashOrderPath);
    // Rescale learned_proj the same way the cluster pipeline does, then pass each weight.
    const rescaled = rescaleLearnedProjWeight(weights, availableLearned);
    for (const [key, val] of Object.entries(rescaled)) {
      if (val !== undefined) args.push(`--${key.replace(/_/g, "-")}-weight`, String(val));
    }
    label = "merge-suggestions-emb";
    loadingMsg = "Loading embeddings...";
  } else {
    const patchesCachePath = join(cache, fullRes ? DINOV3_PATCHES_FULL_FILE : DINOV3_PATCHES_FILE);
    if (!existsSync(patchesCachePath)) {
      throw new Error(
        fullRes
          ? "Full-resolution DINOv3 patches cache not found. Re-run feature extraction with --required dinov3 to generate it."
          : "DINOv3 patches cache not found. Run feature extraction with --required dinov3 first.",
      );
    }
    const patchesHashesPath = join(cache, DINOV3_PATCHES_HASHES_FILE);
    primaryInputPath = patchesCachePath;
    resultCachePath = join(cache, `merge_suggestions2${fullRes ? "_full" : ""}${sizeSuffix}.json`);
    args.push("--patches-cache", patchesCachePath, "--patches-hashes", patchesHashesPath);
    label = fullRes ? "merge-suggestions-full" : "merge-suggestions";
    loadingMsg = `Loading ${fullRes ? "14x14 full-res" : "7x7 pooled"} patches...`;
  }

  const applyFilters = (rows: GroupPairResult[]) => {
    let out = rows;
    if (!options?.includeRejected) {
      const rejected = new Set(
        loadConstraints(targetDir).rejectedMergePairs.map((p) => mergePairKey(p.groupA, p.groupB)),
      );
      if (rejected.size > 0) {
        out = out.filter((r) => !rejected.has(mergePairKey(r.groupA, r.groupB)));
      }
    }
    if (maxCombinedSize > 0) {
      out = out.filter((r) => r.sizeA + r.sizeB <= maxCombinedSize);
    }
    if (minScore > 0) {
      out = out.filter((r) => r.patchMedian >= minScore);
    }
    return out;
  };

  // Disk cache is valid if newer than the groups file, the primary input
  // (patches cache or embeddings NPZ), and content_hashes. Rejected-pair
  // changes never invalidate it: rejection filtering happens in applyFilters.
  try {
    const [cacheStat, groupsStat, inputStat, hashesStat] = await Promise.all([
      stat(resultCachePath),
      stat(groupsP),
      stat(primaryInputPath),
      stat(contentHashesP),
    ]);
    if (
      cacheStat.mtimeMs > groupsStat.mtimeMs &&
      cacheStat.mtimeMs > inputStat.mtimeMs &&
      cacheStat.mtimeMs > hashesStat.mtimeMs
    ) {
      const cached = (await Bun.file(resultCachePath).json()) as GroupPairResult[];
      log(label, "Using cached results");
      options?.onProgress?.("Using cached results");
      return applyFilters(cached);
    }
  } catch {}

  // Compute unfiltered so the cache can serve any threshold; TS re-filters on return.
  args.push("--min-score", "0");
  if (maxCombinedSize > 0) {
    args.push("--max-combined-size", String(maxCombinedSize));
  }

  log(label, `Running group-similarity: ${args.join(" ")}`);
  options?.onProgress?.(loadingMsg);

  const { stdout } = await spawn(args, {
    label,
    onProgress: (line) => options?.onProgress?.(line),
  });
  const allResults: GroupPairResult[] = JSON.parse(stdout);

  await Bun.write(resultCachePath, stdout);
  log(label, `Cached ${allResults.length} results to ${resultCachePath}`);

  return applyFilters(allResults);
}
