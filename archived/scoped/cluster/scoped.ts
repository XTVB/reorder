// Scoped clustering: runs the linkage pipeline against a filename subset
// (the union of selected groups' images) and stores its outputs under
// .reorder-cache/scoped/<scopeKey>/. Re-cuts of a scoped tree happen entirely
// in TS without re-spawning Rust.

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { ensureHashOrderJson } from "../cache-utils.ts";
import { writeJsonAtomic } from "../fs/atomic-json.ts";
import { loadGroups } from "../fs/groups.ts";
import {
  cacheDir,
  contentHashesPath,
  groupsPath,
  HASH_CACHE_FILE,
  scopedDir,
} from "../fs/paths.ts";
import { log } from "../log.ts";
import type {
  ClusterData,
  ClusterResultData,
  ClusterScope,
  DistanceProfile,
  WeightConfig,
} from "../shared/types.ts";
import { RUST_BINARY } from "./binaries.ts";
import { writeResolvedConstraintFiles } from "./constraints.ts";
import { cutTree, distanceProfileFromTree, type LinkageTree, parseLinkageTree } from "./linkage.ts";
import type { RustOutput } from "./pipeline.ts";
import { buildClustersFromLabels, computeSuggestedCounts } from "./pipeline.ts";
import { spawnJSON } from "./subprocess.ts";
import { clustersWithoutAutoNames, computeAutoNames } from "./tfidf.ts";

interface ScopedMeta {
  groupIds: string[];
  groupNames: string[];
  subsetFilenames: string[];
}

export function computeScopeKey(groupIds: string[]): string {
  const canonical = [...groupIds].sort().join("\n");
  return createHash("sha256").update(canonical).digest("hex").slice(0, 16);
}

export function scopedCacheDir(targetDir: string, scopeKey: string): string {
  return join(scopedDir(targetDir), scopeKey);
}

function scopedTreePath(targetDir: string, scopeKey: string): string {
  return join(scopedCacheDir(targetDir, scopeKey), "linkage_tree.bin");
}

function scopedSubsetPath(targetDir: string, scopeKey: string): string {
  return join(scopedCacheDir(targetDir, scopeKey), "subset_filenames.json");
}

function scopedMetaPath(targetDir: string, scopeKey: string): string {
  return join(scopedCacheDir(targetDir, scopeKey), "meta.json");
}

export async function unionGroupFilenames(
  targetDir: string,
  groupIds: string[],
): Promise<{ filenames: string[]; groupNames: string[] }> {
  const groups = loadGroups(targetDir);
  if (groups.length === 0) {
    throw new Error("No groups found — create groups first");
  }
  const byId = new Map(groups.map((g) => [g.id, g]));

  const groupNames: string[] = [];
  const fnSet = new Set<string>();
  for (const gid of groupIds) {
    const g = byId.get(gid);
    if (!g) throw new Error(`Group not found: ${gid}`);
    groupNames.push(g.name);
    for (const f of g.images) fnSet.add(f);
  }

  const path = contentHashesPath(targetDir);
  if (!existsSync(path)) {
    throw new Error("content_hashes.json missing — run feature extraction first");
  }
  const contentHashes: Record<string, string> = JSON.parse(readFileSync(path, "utf-8"));

  const filenames = [...fnSet].filter((f) => f in contentHashes).sort();
  return { filenames, groupNames };
}

export async function runScopedLinkage(
  targetDir: string,
  groupIds: string[],
  nClusters: number,
  weights?: WeightConfig,
  onProgress?: (line: string) => void,
): Promise<{ rustOutput: RustOutput; scopeKey: string; meta: ScopedMeta }> {
  const scopeKey = computeScopeKey(groupIds);
  const scopeDir = scopedCacheDir(targetDir, scopeKey);
  mkdirSync(scopeDir, { recursive: true });

  const { filenames: subsetFilenames, groupNames } = await unionGroupFilenames(targetDir, groupIds);
  if (subsetFilenames.length < 2) {
    throw new Error("Scoped clustering needs at least 2 images across the selected groups");
  }

  const meta: ScopedMeta = { groupIds, groupNames, subsetFilenames };
  await Promise.all([
    writeJsonAtomic(scopedSubsetPath(targetDir, scopeKey), subsetFilenames, { pretty: false }),
    writeJsonAtomic(scopedMetaPath(targetDir, scopeKey), meta, { pretty: false }),
  ]);

  const cache = cacheDir(targetDir);
  const hashCachePath = join(cache, HASH_CACHE_FILE);
  const contentHashesP = contentHashesPath(targetDir);
  const hashOrderPath = join(cache, "hash_cache_order.json");
  const groupsFile = groupsPath(targetDir);
  const treePath = scopedTreePath(targetDir, scopeKey);

  ensureHashOrderJson(cache);

  // Scoped clustering operates on a filename subset (--filenames). The Rust
  // tool disallows combining that with --dist-matrix, so re-ranking isn't
  // available here in v1. Use ward linkage on raw cosine distances.
  const args = [
    RUST_BINARY,
    "--hash-cache",
    hashCachePath,
    "--content-hashes",
    contentHashesP,
    "--hash-order",
    hashOrderPath,
    "--filenames",
    scopedSubsetPath(targetDir, scopeKey),
    "--n-clusters",
    String(nClusters),
    "--output-tree",
    treePath,
    "--linkage",
    "ward",
  ];
  if (existsSync(groupsFile)) args.push("--groups", groupsFile);
  const scopedConstraints = await writeResolvedConstraintFiles(targetDir, {
    allowedFilenames: new Set(subsetFilenames),
  });
  if (scopedConstraints.cannotLinkPath) {
    args.push("--cannot-link", scopedConstraints.cannotLinkPath);
  }
  if (scopedConstraints.lockedGroupsPath) {
    args.push("--locked-groups", scopedConstraints.lockedGroupsPath);
  }
  if (weights) {
    for (const [key, val] of Object.entries(weights)) {
      if (val !== undefined) args.push(`--${key.replace(/_/g, "-")}-weight`, String(val));
    }
  }

  if (!existsSync(RUST_BINARY)) {
    throw new Error(
      `Rust binary not found at ${RUST_BINARY}. Build with: cd rust/cluster-tool && cargo build --release`,
    );
  }

  log("cluster", `Running scoped linkage (${subsetFilenames.length} images): ${args.join(" ")}`);
  const rustOutput = await spawnJSON<RustOutput>(args, { label: "scoped-cluster", onProgress });
  return { rustOutput, scopeKey, meta };
}

function loadScopedMeta(targetDir: string, scopeKey: string): ScopedMeta {
  const path = scopedMetaPath(targetDir, scopeKey);
  if (!existsSync(path)) {
    throw new Error(`Scope ${scopeKey} not found — enter scope again`);
  }
  return JSON.parse(readFileSync(path, "utf-8"));
}

function loadScopedTree(targetDir: string, scopeKey: string): LinkageTree {
  const path = scopedTreePath(targetDir, scopeKey);
  if (!existsSync(path)) {
    throw new Error(`Scoped tree for ${scopeKey} missing — run scoped clustering first`);
  }
  return parseLinkageTree(path);
}

export async function runScopedFull(
  targetDir: string,
  groupIds: string[],
  nClusters: number,
  weights?: WeightConfig,
  onProgress?: (line: string) => void,
): Promise<ClusterData> {
  const { rustOutput, scopeKey, meta } = await runScopedLinkage(
    targetDir,
    groupIds,
    nClusters,
    weights,
    onProgress,
  );

  let clusters: ClusterResultData[];
  try {
    clusters = computeAutoNames(targetDir, rustOutput.clusters);
  } catch {
    clusters = clustersWithoutAutoNames(rustOutput.clusters);
  }

  const scope: ClusterScope = {
    scopeKey,
    groupIds,
    groupNames: meta.groupNames,
    nImages: meta.subsetFilenames.length,
    subsetFilenames: meta.subsetFilenames,
  };

  return {
    clusters,
    suggestedCounts: computeSuggestedCounts(meta.subsetFilenames.length),
    nClusters,
    distanceProfile: distanceProfileFromTree(loadScopedTree(targetDir, scopeKey)),
    scope,
  };
}

export async function runScopedRecut(
  targetDir: string,
  scopeKey: string,
  params: { nClusters?: number; threshold?: number; minClusterSize?: number },
): Promise<ClusterData> {
  const meta = loadScopedMeta(targetDir, scopeKey);
  const tree = loadScopedTree(targetDir, scopeKey);
  const nAfterPremerge = tree.nImages - tree.nPreMerges;

  let labels: number[];
  let resultN: number;

  if (params.threshold != null) {
    const { nPreMerges, steps } = tree;
    let mainMerges = 0;
    for (let i = nPreMerges; i < steps.length; i++) {
      if (steps[i]!.distance >= params.threshold) break;
      mainMerges++;
    }
    labels = cutTree(tree, mainMerges);
    resultN = nAfterPremerge - mainMerges;
  } else if (params.minClusterSize != null) {
    // Scoped adaptive: approximate by mapping minClusterSize to a target N.
    const target = Math.max(2, Math.floor(meta.subsetFilenames.length / params.minClusterSize));
    const minClusters = Math.max(target, tree.nGroups);
    const mainMergesNeeded = Math.max(0, nAfterPremerge - minClusters);
    labels = cutTree(tree, mainMergesNeeded);
    resultN = nAfterPremerge - mainMergesNeeded;
  } else {
    const n = params.nClusters ?? Math.max(10, Math.floor(meta.subsetFilenames.length / 20));
    const minClusters = Math.max(n, tree.nGroups);
    const mainMergesNeeded = Math.max(0, nAfterPremerge - minClusters);
    labels = cutTree(tree, mainMergesNeeded);
    resultN = nAfterPremerge - mainMergesNeeded;
  }

  return buildScopedRecutResult(
    targetDir,
    scopeKey,
    meta,
    labels,
    resultN,
    distanceProfileFromTree(tree),
  );
}

async function buildScopedRecutResult(
  targetDir: string,
  scopeKey: string,
  meta: ScopedMeta,
  labels: number[],
  nClusters: number,
  distanceProfile: DistanceProfile,
): Promise<ClusterData> {
  const filenames = meta.subsetFilenames;
  const clusters = buildClustersFromLabels(targetDir, filenames, labels, {
    idPrefix: "scoped_cluster_",
  });

  const scope: ClusterScope = {
    scopeKey,
    groupIds: meta.groupIds,
    groupNames: meta.groupNames,
    nImages: meta.subsetFilenames.length,
    subsetFilenames: meta.subsetFilenames,
  };

  return {
    clusters,
    suggestedCounts: computeSuggestedCounts(filenames.length),
    nClusters,
    distanceProfile,
    scope,
  };
}

export function clearScopedCache(targetDir: string): void {
  const path = scopedDir(targetDir);
  if (existsSync(path)) {
    rmSync(path, { recursive: true, force: true });
  }
}
