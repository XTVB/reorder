// Cluster orchestration: extraction → linkage → name assignment. The cluster-tool
// Rust binary handles linkage; we drive it from here and weave in re-rank or
// patch matrices when requested.

import { existsSync } from "node:fs";
import { rename } from "node:fs/promises";
import { join } from "node:path";
import { ensureHashOrderJson, resolveHashCachePath } from "../cache-utils.ts";
import { loadGroups } from "../fs/groups.ts";
import { withRenameLock } from "../fs/lock.ts";
import {
  cacheDir,
  contentHashesPath,
  contentHashesTmpPath,
  groupsPath,
  linkageTreePath,
} from "../fs/paths.ts";
import { log } from "../log.ts";
import type {
  ClusterData,
  ClusterResultData,
  DistanceProfile,
  ImageGroup,
  LinkageMethod,
  WeightConfig,
} from "../shared/types.ts";
import { PYTHON, RUST_BINARY, SCRIPTS_DIR } from "./binaries.ts";
import { writeResolvedConstraintFiles } from "./constraints.ts";
import { ensurePatchDistMatrix, ensureRerankDistMatrix } from "./distance-matrices.ts";
import { cachedHashMapping, MODEL_KEYS } from "./embeddings.ts";
import { getClusterAbortSignal } from "./job-mutex.ts";
import {
  getDistanceProfile,
  recutTree,
  recutTreeAdaptive,
  recutTreeByThreshold,
} from "./linkage.ts";
import { spawnJSON } from "./subprocess.ts";

/** Shape of cluster passed in from the Rust cluster-tool output. */
interface RawCluster {
  id: string;
  images: string[];
  confirmedGroup: { id: string; name: string; images: string[] } | null;
}

// Internal type matching the cluster-tool stdout shape.
export interface RustOutput {
  clusters: RawCluster[];
  nClusters: number;
  treePath: string;
}

/** Assign names from confirmed-group names with a generic fallback. */
function namedClusters(clusters: RawCluster[]): ClusterResultData[] {
  return clusters.map((c, i) => ({
    id: c.id,
    name: c.confirmedGroup?.name ?? `Cluster ${i + 1}`,
    images: c.images,
    confirmedGroup: c.confirmedGroup,
  }));
}

/**
 * Run feature extraction (Python). The pipeline writes content_hashes.json
 * via a temp-then-rename atomic step under withRenameLock so concurrent
 * /api/save calls can't observe a half-rewritten file.
 */
export async function extractFeatures(
  targetDir: string,
  onProgress?: (line: string) => void,
  opts?: { force?: string[]; required?: string[]; signal?: AbortSignal },
): Promise<{ total: number; cached: number; extracted: number }> {
  const script = join(SCRIPTS_DIR, "extract_features.py");
  const cache = cacheDir(targetDir);

  if (!existsSync(PYTHON)) {
    throw new Error(
      `Python not found at ${PYTHON}. Create venv with: uv venv ~/.venvs/imgcluster-env && source ~/.venvs/imgcluster-env/bin/activate && uv pip install torch torchvision open-clip-torch pillow numpy transformers`,
    );
  }

  const args = [PYTHON, script, targetDir, "--cache-dir", cache];
  if (opts?.force && opts.force.length > 0) {
    args.push("--models", opts.force.join(","));
  }
  if (opts?.required && opts.required.length > 0) {
    args.push("--required", opts.required.join(","));
  }

  log("cluster", `Extracting features: ${args.join(" ")}`);
  const result = await spawnJSON<{
    total: number;
    cached: number;
    extracted: number;
    interrupted?: boolean;
    cachePath?: string;
  }>(args, {
    label: "extract-features",
    progressPrefix: "", // Python emits free-form stderr; every non-empty line is progress
    onProgress,
    signal: opts?.signal,
  });

  if (result.interrupted) {
    throw new Error(
      "Feature extraction was interrupted. Partial results were saved — re-run to continue from where it left off.",
    );
  }

  // Atomically promote content_hashes.json.tmp → content_hashes.json under the
  // FS lock. extract_features.py writes the tmp file, and we rename it to the
  // final path here so concurrent /api/save calls can't observe a half-written
  // cache mid-rewrite.
  const tmpPath = contentHashesTmpPath(targetDir);
  if (existsSync(tmpPath)) {
    await withRenameLock(async () => {
      await rename(tmpPath, contentHashesPath(targetDir));
    });
  }

  return result;
}

const DEFAULT_RERANK_BLEND = 0.7;

/**
 * Re-interpret the `learned_proj` weight as "target fraction of the final cosine
 * signal" rather than as a raw concat-multiplier.
 *
 * For unit-norm sub-vectors fed into rust's concat-then-cosine pipeline, each
 * component's contribution to the final cosine is wₖ² / Σwⱼ². So to make
 * learned_proj contribute exactly `b` of the total, we set its raw weight to
 *   w_learned = √(b · S / (1 − b))    where S = Σ wⱼ² for j ≠ learned_proj.
 *
 * Edge cases:
 *  - b ≤ 0: pass through (no learned head contribution).
 *  - b ≥ 1, or S == 0: zero out the other weights and set learned_proj to 1.
 *
 * This lets the UI slider (and any caller) treat the learned_proj weight as a
 * percentage of the final signal, independent of how the other model weights
 * are set.
 */
export function rescaleLearnedProjWeight(weights: WeightConfig): WeightConfig {
  const b = weights.learned_proj ?? 0;
  if (b <= 0) return weights;

  let s = 0;
  for (const [key, val] of Object.entries(weights)) {
    if (key === "learned_proj") continue;
    const v = val ?? 0;
    if (v > 0) s += v * v;
  }

  if (b >= 1 || s === 0) {
    // 100% learned head — zero out the other components.
    const out: WeightConfig = { learned_proj: 1 };
    for (const key of Object.keys(weights)) {
      if (key !== "learned_proj") (out as Record<string, number>)[key] = 0;
    }
    return out;
  }

  const learnedActual = Math.sqrt((b * s) / (1 - b));
  return { ...weights, learned_proj: learnedActual };
}

export interface LinkageOptions {
  usePatches?: boolean;
  useRerank?: boolean;
  /** Blend strength for re-rank matrix vs raw cosine. 0=cosine only, 1=rerank only. Default 0.7. */
  rerankBlend?: number;
  /** Explicit linkage method; defaults to ward when omitted.
   * Average/complete can beat ward on datasets with few large or uneven-sized clusters. */
  linkage?: LinkageMethod;
}

export async function runLinkage(
  targetDir: string,
  nClusters: number,
  weights?: WeightConfig,
  options?: LinkageOptions,
  onProgress?: (line: string) => void,
): Promise<RustOutput> {
  const cache = cacheDir(targetDir);
  const hashCachePath = resolveHashCachePath(cache);
  const contentHashesP = contentHashesPath(targetDir);
  const hashOrderPath = join(cache, "hash_cache_order.json");
  const groupsFile = groupsPath(targetDir);
  const treePath = linkageTreePath(targetDir);

  // Re-rank takes precedence over patches when both are enabled.
  const useRerank = options?.useRerank ?? false;
  const usePatches = !useRerank && (options?.usePatches ?? false);
  const rerankBlend = options?.rerankBlend ?? DEFAULT_RERANK_BLEND;

  let distMatrixPath: string | null = null;
  let distMatrixWeight: number | null = null;
  if (useRerank) {
    distMatrixPath = await ensureRerankDistMatrix(targetDir, weights ?? {}, onProgress);
    distMatrixWeight = rerankBlend;
  } else if (usePatches) {
    distMatrixPath = await ensurePatchDistMatrix(targetDir);
    const hasEmbWeights = weights && Object.values(weights).some((v) => (v ?? 0) > 0);
    distMatrixWeight = hasEmbWeights ? 0.5 : 1.0;
  }

  // Ensure the JSON sidecar exists (regenerate from NPZ if needed)
  ensureHashOrderJson(cache);

  // Linkage method: explicit override wins; ward is the default (it beat average
  // in the LOMO even with re-rank on — see LEARNED_HEAD.md).
  const linkageMethod: LinkageMethod = options?.linkage ?? "ward";

  const args = [
    RUST_BINARY,
    "--hash-cache",
    hashCachePath,
    "--content-hashes",
    contentHashesP,
    "--hash-order",
    hashOrderPath,
    "--n-clusters",
    String(nClusters),
    "--output-tree",
    treePath,
    "--linkage",
    linkageMethod,
  ];
  if (existsSync(groupsFile)) {
    args.push("--groups", groupsFile);
  }
  if (distMatrixPath && distMatrixWeight != null) {
    args.push("--dist-matrix", distMatrixPath);
    args.push("--dist-matrix-weight", String(distMatrixWeight));
  }
  const constraintFiles = await writeResolvedConstraintFiles(targetDir);
  if (constraintFiles.cannotLinkPath) {
    args.push("--cannot-link", constraintFiles.cannotLinkPath);
  }
  if (constraintFiles.lockedGroupsPath) {
    args.push("--locked-groups", constraintFiles.lockedGroupsPath);
  }
  if (weights) {
    const rescaled = rescaleLearnedProjWeight(weights);
    for (const [key, val] of Object.entries(rescaled)) {
      if (val !== undefined) args.push(`--${key.replace(/_/g, "-")}-weight`, String(val));
    }
  }

  if (!existsSync(RUST_BINARY)) {
    throw new Error(
      `Rust binary not found at ${RUST_BINARY}. Build with: cd rust/cluster-tool && cargo build --release`,
    );
  }

  log("cluster", `Running ${linkageMethod} linkage: ${args.join(" ")}`);
  return spawnJSON<RustOutput>(args, { label: "cluster-tool" });
}

function suggestedCounts(nImages: number): number[] {
  return [
    ...new Set([
      Math.max(10, Math.floor(nImages / 100)),
      Math.max(20, Math.floor(nImages / 50)),
      Math.max(50, Math.floor(nImages / 30)),
      Math.max(75, Math.floor(nImages / 20)),
      100,
      150,
      200,
      300,
    ]),
  ].sort((a, b) => a - b);
}

export { suggestedCounts as computeSuggestedCounts };

/** Derive the set of model keys needed for a given weight config.
 * Only models explicitly given a positive weight are extracted — missing keys
 * mean "don't extract", so CLIP etc. are never pulled unless the user asked for them.
 * learned_proj is derived from pecore_g + color at extraction time, so it implies
 * both of those as upstream dependencies. */
export function modelsForWeights(weights?: WeightConfig): string[] | undefined {
  if (!weights) return undefined; // no config → extract all (auto mode)
  const out = new Set(MODEL_KEYS.filter((k) => (weights[k] ?? 0) > 0));
  if (out.has("learned_proj")) {
    out.add("pecore_g");
    out.add("color");
  }
  return Array.from(out);
}

export async function runFullCluster(
  targetDir: string,
  nClusters: number,
  onProgress?: (line: string) => void,
  weights?: WeightConfig,
  options?: LinkageOptions,
): Promise<ClusterData> {
  const required = modelsForWeights(weights);
  const usePatches = options?.usePatches ?? false;
  if (usePatches && required && !required.includes("dinov3")) {
    required.push("dinov3");
  }
  const signal = getClusterAbortSignal();
  const extraction = await extractFeatures(
    targetDir,
    onProgress,
    required ? { required, signal } : { signal },
  );
  log("cluster", `Extraction: ${extraction.extracted} new, ${extraction.cached} cached`);

  const rustOutput = await runLinkage(targetDir, nClusters, weights, options, onProgress);
  log("cluster", `Linkage complete: ${rustOutput.clusters.length} clusters`);

  const clusters = namedClusters(rustOutput.clusters);

  const nImages = clusters.reduce((n, c) => n + c.images.length, 0);
  const distanceProfile = getDistanceProfile(targetDir);
  return { clusters, suggestedCounts: suggestedCounts(nImages), nClusters, distanceProfile };
}

export async function runLinkageOnly(
  targetDir: string,
  nClusters: number,
  weights?: WeightConfig,
  options?: LinkageOptions,
): Promise<ClusterData> {
  const rustOutput = await runLinkage(targetDir, nClusters, weights, options);
  const clusters = namedClusters(rustOutput.clusters);
  const nImages = clusters.reduce((n, c) => n + c.images.length, 0);
  return { clusters, suggestedCounts: suggestedCounts(nImages), nClusters };
}

export async function runRecut(targetDir: string, nClusters: number): Promise<ClusterData> {
  const { labels, distanceProfile } = recutTree(targetDir, nClusters);
  return buildRecutResult(targetDir, labels, nClusters, distanceProfile);
}

export async function runRecutByThreshold(
  targetDir: string,
  threshold: number,
): Promise<ClusterData> {
  const { labels, nClusters, distanceProfile } = recutTreeByThreshold(targetDir, threshold);
  return buildRecutResult(targetDir, labels, nClusters, distanceProfile);
}

export async function runRecutAdaptive(
  targetDir: string,
  minClusterSize: number,
): Promise<ClusterData> {
  const { labels, nClusters, distanceProfile } = recutTreeAdaptive(targetDir, minClusterSize);
  return buildRecutResult(targetDir, labels, nClusters, distanceProfile);
}

/** Group labels[] into clusters, attach confirmed-group info, auto-name, sort by size. */
export function buildClustersFromLabels(
  targetDir: string,
  filenames: string[],
  labels: number[],
  opts: { idPrefix: string },
): ClusterResultData[] {
  const imgToGroup = new Map<string, ImageGroup>();
  for (const g of loadGroups(targetDir)) {
    for (const f of g.images) imgToGroup.set(f, g);
  }

  const clusterMembers = new Map<number, string[]>();
  for (let i = 0; i < labels.length; i++) {
    const label = labels[i]!;
    if (!clusterMembers.has(label)) clusterMembers.set(label, []);
    clusterMembers.get(label)!.push(filenames[i]!);
  }

  const rawClusters = [...clusterMembers.entries()]
    .sort((a, b) => b[1].length - a[1].length)
    .map(([, images], ci): RawCluster => {
      const confirmed = images.find((f) => imgToGroup.has(f));
      const group = confirmed ? (imgToGroup.get(confirmed) ?? null) : null;
      return {
        id: `${opts.idPrefix}${ci}`,
        images: images.sort(),
        confirmedGroup: group ? { id: group.id, name: group.name, images: group.images } : null,
      };
    });

  return namedClusters(rawClusters);
}

async function buildRecutResult(
  targetDir: string,
  labels: number[],
  nClusters: number,
  distanceProfile: DistanceProfile,
): Promise<ClusterData> {
  const { filenames } = cachedHashMapping(targetDir);
  // labels are indexed by position in the sorted filename list used at
  // tree-build time. If the file set has changed since (e.g. deletes), the
  // cached tree is stale and any mapping back to filenames would be wrong.
  if (labels.length !== filenames.length) {
    throw new Error(
      `Cached linkage tree is stale (tree has ${labels.length} images, ` +
        `content_hashes.json has ${filenames.length}). Re-run clustering.`,
    );
  }
  const clusters = buildClustersFromLabels(targetDir, filenames, labels, { idPrefix: "cluster_" });
  return {
    clusters,
    suggestedCounts: suggestedCounts(filenames.length),
    nClusters,
    distanceProfile,
  };
}
