// Merge suggestions: DINOv3 patch-based group similarity. Computed by the
// Rust group-similarity binary in "merge-suggestions" mode. Cached in
// merge_suggestions[_full][_mN].json keyed by groups+patches+content_hashes
// mtimes; one-time invalidation if the cache is in pre-camelCase format.

import { existsSync } from "node:fs";
import { stat } from "node:fs/promises";
import { join } from "node:path";
import {
  cacheDir,
  contentHashesPath,
  DINOV3_PATCHES_FILE,
  DINOV3_PATCHES_FULL_FILE,
  DINOV3_PATCHES_HASHES_FILE,
  groupsPath,
} from "../fs/paths.ts";
import { log } from "../log.ts";
import { GROUP_SIM_BINARY } from "./binaries.ts";
import { writeResolvedRejectedPairsFile } from "./constraints.ts";
import { spawn } from "./subprocess.ts";

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
    fullResolution?: boolean;
    maxCombinedSize?: number;
    onProgress?: (msg: string) => void;
  },
): Promise<GroupPairResult[]> {
  const cache = cacheDir(targetDir);
  const fullRes = options?.fullResolution ?? false;
  const maxCombinedSize = Math.max(0, Math.floor(options?.maxCombinedSize ?? 0));
  const patchesCachePath = join(cache, fullRes ? DINOV3_PATCHES_FULL_FILE : DINOV3_PATCHES_FILE);
  const patchesHashesPath = join(cache, DINOV3_PATCHES_HASHES_FILE);
  const contentHashesP = contentHashesPath(targetDir);
  const groupsP = groupsPath(targetDir);
  const resSuffix = fullRes ? "_full" : "";
  const sizeSuffix = maxCombinedSize > 0 ? `_m${maxCombinedSize}` : "";
  const resultCachePath = join(cache, `merge_suggestions${resSuffix}${sizeSuffix}.json`);

  if (!existsSync(patchesCachePath)) {
    throw new Error(
      fullRes
        ? "Full-resolution DINOv3 patches cache not found. Re-run feature extraction with --required dinov3 to generate it."
        : "DINOv3 patches cache not found. Run feature extraction with --required dinov3 first.",
    );
  }
  if (!existsSync(GROUP_SIM_BINARY)) {
    throw new Error(
      `group-similarity binary not found at ${GROUP_SIM_BINARY}. Build with: cd rust/group-similarity && cargo build --release`,
    );
  }

  const applyFilters = (rows: GroupPairResult[]) => {
    let out = rows;
    if (maxCombinedSize > 0) {
      out = out.filter((r) => r.sizeA + r.sizeB <= maxCombinedSize);
    }
    if (minScore > 0) {
      out = out.filter((r) => r.patchMedian >= minScore);
    }
    return out;
  };

  // No-ops if rejected-pairs content is unchanged, so cache mtime check stays valid.
  const rejectedPairsPath = await writeResolvedRejectedPairsFile(targetDir);

  // Disk cache is valid if newer than the groups file, patches cache,
  // content_hashes, and (if present) the resolved rejected-pairs file.
  try {
    const [cacheStat, groupsStat, patchesStat, hashesStat, rejectedStat] = await Promise.all([
      stat(resultCachePath),
      stat(groupsP),
      stat(patchesCachePath),
      stat(contentHashesP),
      rejectedPairsPath ? stat(rejectedPairsPath) : Promise.resolve(null),
    ]);
    if (
      cacheStat.mtimeMs > groupsStat.mtimeMs &&
      cacheStat.mtimeMs > patchesStat.mtimeMs &&
      cacheStat.mtimeMs > hashesStat.mtimeMs &&
      (rejectedStat === null || cacheStat.mtimeMs > rejectedStat.mtimeMs)
    ) {
      const cached = (await Bun.file(resultCachePath).json()) as unknown[];
      // Detect snake_case shape from before the camelCase wire-format migration.
      if (
        Array.isArray(cached) &&
        cached.length > 0 &&
        cached[0] &&
        "group_a" in (cached[0] as object)
      ) {
        log("merge-suggestions", "Old-format cache detected, recomputing");
      } else {
        log("merge-suggestions", `Using cached results (${fullRes ? "full-res" : "pooled"})`);
        options?.onProgress?.("Using cached results");
        return applyFilters(cached as GroupPairResult[]);
      }
    }
  } catch {}

  const args = [
    GROUP_SIM_BINARY,
    "--patches-cache",
    patchesCachePath,
    "--content-hashes",
    contentHashesP,
    "--patches-hashes",
    patchesHashesPath,
    "--groups",
    groupsP,
    // Compute unfiltered so the cache can serve any threshold; TS re-filters on return.
    "--min-score",
    "0",
  ];
  if (maxCombinedSize > 0) {
    args.push("--max-combined-size", String(maxCombinedSize));
  }
  if (rejectedPairsPath) {
    args.push("--rejected-pairs", rejectedPairsPath);
  }

  const label = fullRes ? "merge-suggestions-full" : "merge-suggestions";
  log(label, `Running group-similarity: ${args.join(" ")}`);
  options?.onProgress?.(`Loading ${fullRes ? "14x14 full-res" : "7x7 pooled"} patches...`);

  const { stdout } = await spawn(args, {
    label,
    onProgress: (line) => options?.onProgress?.(line),
  });
  const allResults: GroupPairResult[] = JSON.parse(stdout);

  await Bun.write(resultCachePath, stdout);
  log(label, `Cached ${allResults.length} results to ${resultCachePath}`);

  return applyFilters(allResults);
}
