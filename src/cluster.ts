/**
 * Server-side clustering support:
 * - Spawns Python for feature extraction
 * - Spawns Rust for Ward's linkage
 * - Parses linkage tree binary for re-cuts
 * - Computes TF-IDF auto-names from cached CLIP embeddings
 * - Generates contact sheets via Sharp
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { unlink } from "node:fs/promises";
import { dirname, join } from "node:path";
import sharp from "sharp";
import {
  ensureHashOrderJson,
  type HashMapping,
  loadHashMapping,
  parseNpyFromNpz,
  reindexToFilenameOrder,
} from "./cache-utils.ts";
import type {
  ClusterData,
  ClusterResultData,
  ClusterScope,
  DistanceProfile,
  ImageGroup,
  ImportClusterInput,
  WeightConfig,
} from "./client/types.ts";
import { log } from "./log.ts";

export type { ImportClusterInput, WeightConfig };

// ── Shared helpers ───────────────────────────────────────────────────────────

/** Read .reorder-groups.json, tolerating both array and `{groups: [...]}` shapes. */
export function loadGroups(targetDir: string): ImageGroup[] {
  const path = join(targetDir, ".reorder-groups.json");
  if (!existsSync(path)) return [];
  try {
    const raw = JSON.parse(readFileSync(path, "utf-8"));
    if (Array.isArray(raw)) return raw;
    if (Array.isArray(raw?.groups)) return raw.groups;
  } catch {}
  return [];
}

// ── Paths ────────────────────────────────────────────────────────────────────

const SCRIPTS_DIR = join(dirname(import.meta.dir), "scripts");
const RUST_BINARY = join(
  dirname(import.meta.dir),
  "rust",
  "cluster-tool",
  "target",
  "release",
  "cluster-tool",
);
const GROUP_SIM_BINARY = join(
  dirname(import.meta.dir),
  "rust",
  "group-similarity",
  "target",
  "release",
  "group-similarity",
);
const PYTHON =
  process.env.CLUSTER_PYTHON || `${process.env.HOME}/.venvs/imgcluster-env/bin/python3`;

function cacheDir(targetDir: string) {
  return join(targetDir, ".reorder-cache");
}

interface RustOutput {
  clusters: {
    id: string;
    images: string[];
    confirmed_group: { id: string; name: string; images: string[] } | null;
  }[];
  n_clusters: number;
  tree_path: string;
}

interface LinkageTree {
  nImages: number;
  nPreMerges: number;
  nGroups: number;
  steps: { clusterA: number; clusterB: number; distance: number; newSize: number }[];
}

interface TextEmbeddingsRaw {
  terms: string[];
  embeddings: number[][]; // [n_terms][512]
}

interface TextEmbeddings {
  terms: string[];
  flat: Float32Array; // flattened [n_terms * dim]
  dim: number;
}

// ── Rust subprocess helper ───────────────────────────────────────────────────
// Drains stdout/stderr concurrently with awaiting exit to avoid pipe-buffer
// deadlock if the child emits more than the pipe can hold before exiting.

async function runRustBinary(
  binary: string,
  args: string[],
  label: string,
  onProgress?: (line: string) => void,
): Promise<string> {
  const proc = Bun.spawn([binary, ...args], { stdout: "pipe", stderr: "pipe" });

  let stderrPromise: Promise<string>;
  if (onProgress) {
    // Stream stderr line-by-line, forwarding "progress:" lines to the callback
    const stderrLines: string[] = [];
    stderrPromise = (async () => {
      const reader = proc.stderr.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() || "";
        for (const line of lines) {
          stderrLines.push(line);
          if (line.startsWith("progress:")) {
            onProgress(line.slice(10).trim());
          }
        }
      }
      if (buf) {
        stderrLines.push(buf);
        if (buf.startsWith("progress:")) {
          onProgress(buf.slice(10).trim());
        }
      }
      return stderrLines.join("\n");
    })();
  } else {
    stderrPromise = new Response(proc.stderr).text();
  }

  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    stderrPromise,
    proc.exited,
  ]);
  if (stderr) log(label, stderr.trim());
  if (exitCode !== 0) {
    throw new Error(`${label} failed (exit ${exitCode}): ${stderr}`);
  }
  return stdout;
}

// ── Feature extraction (Python) ──────────────────────────────────────────────

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
  const proc = Bun.spawn(args, {
    stdout: "pipe",
    stderr: "pipe",
  });

  // Allow callers to abort the extraction (sends SIGINT so Python saves checkpoint)
  const onAbort = () => {
    log("cluster", "Sending SIGINT to extraction subprocess for graceful shutdown...");
    proc.kill("SIGINT");
  };
  opts?.signal?.addEventListener("abort", onAbort, { once: true });

  const stderrReader = proc.stderr.getReader();
  const decoder = new TextDecoder();
  let stderrText = "";
  (async () => {
    while (true) {
      const { done, value } = await stderrReader.read();
      if (done) break;
      const text = decoder.decode(value);
      stderrText += text;
      if (onProgress) {
        for (const line of text.split("\n").filter(Boolean)) {
          onProgress(line.trim());
        }
      }
    }
  })();

  const exitCode = await proc.exited;
  opts?.signal?.removeEventListener("abort", onAbort);

  if (exitCode !== 0) {
    throw new Error(`Feature extraction failed (exit ${exitCode}): ${stderrText}`);
  }

  const stdout = await new Response(proc.stdout).text();
  const result = JSON.parse(stdout);
  if (result.interrupted) {
    throw new Error(
      "Feature extraction was interrupted. Partial results were saved — re-run to continue from where it left off.",
    );
  }
  return result;
}

// ── Ensure text embeddings exist ─────────────────────────────────────────────

export async function ensureTextEmbeddings(targetDir: string): Promise<string> {
  const path = join(cacheDir(targetDir), "text_embeddings.json");
  if (existsSync(path)) return path;

  const script = join(SCRIPTS_DIR, "precompute_text_embeddings.py");
  log("cluster", `Precomputing text embeddings...`);
  const proc = Bun.spawn([PYTHON, script, path], { stderr: "pipe" });
  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    const err = await new Response(proc.stderr).text();
    throw new Error(`Text embedding precomputation failed: ${err}`);
  }
  return path;
}

// ── Rust linkage ─────────────────────────────────────────────────────────────

export function patchDistMatrixPath(targetDir: string): string {
  return join(cacheDir(targetDir), "patch_dist_matrix.bin");
}

/** Ensure the DINOv3 patch distance matrix exists; recompute via group-similarity if stale/missing. */
export async function ensurePatchDistMatrix(
  targetDir: string,
  onProgress?: (line: string) => void,
): Promise<string> {
  const cache = cacheDir(targetDir);
  const contentHashesPath = join(cache, "content_hashes.json");
  const patchesCachePath = join(cache, "dinov3_patches_hash_cache.npy");
  const patchesHashesPath = join(cache, "dinov3_patches_hashes.json");
  const distMatrixPath = patchDistMatrixPath(targetDir);

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

  let cacheValid = false;
  if (existsSync(distMatrixPath)) {
    const matMtime = Bun.file(distMatrixPath).lastModified;
    if (
      matMtime > Bun.file(patchesCachePath).lastModified &&
      matMtime > Bun.file(contentHashesPath).lastModified
    ) {
      cacheValid = true;
    }
  }

  if (cacheValid) {
    log("cluster", "Using cached patch-based distance matrix");
    return distMatrixPath;
  }

  log("cluster", "Precomputing patch-based distance matrix...");
  onProgress?.("Computing patch distance matrix...");
  await runRustBinary(
    GROUP_SIM_BINARY,
    [
      "--patches-cache",
      patchesCachePath,
      "--content-hashes",
      contentHashesPath,
      "--patches-hashes",
      patchesHashesPath,
      "--groups",
      "",
      "--mode",
      "dist-matrix",
      "--output",
      distMatrixPath,
    ],
    "patch-dist-matrix",
    onProgress,
  );
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

export async function runLinkage(
  targetDir: string,
  nClusters: number,
  weights?: WeightConfig,
  usePatches?: boolean,
): Promise<RustOutput> {
  const cache = cacheDir(targetDir);
  const hashCachePath = join(cache, "clip_hash_cache.npz");
  const contentHashesPath = join(cache, "content_hashes.json");
  const hashOrderPath = join(cache, "hash_cache_order.json");
  const groupsFile = join(targetDir, ".reorder-groups.json");
  const treePath = join(cache, "linkage_tree.bin");
  const distMatrixPath = patchDistMatrixPath(targetDir);

  if (usePatches) {
    await ensurePatchDistMatrix(targetDir);
  }

  // Ensure the JSON sidecar exists (regenerate from NPZ if needed)
  ensureHashOrderJson(cache);

  const args = [
    "--hash-cache",
    hashCachePath,
    "--content-hashes",
    contentHashesPath,
    "--hash-order",
    hashOrderPath,
    "--n-clusters",
    String(nClusters),
    "--output-tree",
    treePath,
  ];
  if (existsSync(groupsFile)) {
    args.push("--groups", groupsFile);
  }
  if (usePatches) {
    args.push("--dist-matrix", distMatrixPath);
    const hasEmbWeights = weights && Object.values(weights).some((v) => (v ?? 0) > 0);
    if (hasEmbWeights) {
      args.push("--dist-matrix-weight", "0.5");
    }
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

  log("cluster", `Running Rust linkage: ${RUST_BINARY} ${args.join(" ")}`);
  const stdout = await runRustBinary(RUST_BINARY, args, "cluster-tool");
  return JSON.parse(stdout);
}

// ── Linkage tree parsing (for Bun-side re-cuts) ─────────────────────────────

function parseLinkageTree(path: string): LinkageTree {
  const buf = readFileSync(path);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);

  let offset = 0;
  const nImages = view.getUint32(offset, true);
  offset += 4;
  const nPreMerges = view.getUint32(offset, true);
  offset += 4;
  const nGroups = view.getUint32(offset, true);
  offset += 4;
  const nSteps = view.getUint32(offset, true);
  offset += 4;

  const steps = [];
  for (let i = 0; i < nSteps; i++) {
    const clusterA = view.getUint32(offset, true);
    offset += 4;
    const clusterB = view.getUint32(offset, true);
    offset += 4;
    const distance = view.getFloat32(offset, true);
    offset += 4;
    const newSize = view.getUint32(offset, true);
    offset += 4;
    steps.push({ clusterA, clusterB, distance, newSize });
  }

  return { nImages, nPreMerges, nGroups, steps };
}

// ── Re-cut (Bun-side, instant) ───────────────────────────────────────────────

function loadTree(targetDir: string): LinkageTree {
  const treePath = join(cacheDir(targetDir), "linkage_tree.bin");
  if (!existsSync(treePath)) {
    throw new Error("No linkage tree found. Run full clustering first.");
  }
  return parseLinkageTree(treePath);
}

/** Apply pre-merges, then main merges up to `maxMainMerges`, return labels. */
function cutTree(tree: LinkageTree, maxMainMerges: number): number[] {
  const { nImages, nPreMerges, steps } = tree;

  const parent = new Int32Array(nImages);
  for (let i = 0; i < nImages; i++) parent[i] = i;

  // Path-halving union-find: each step skips to grandparent, flattening the tree
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

  for (let i = 0; i < nPreMerges; i++) {
    const s = steps[i]!;
    const ra = find(s.clusterA);
    const rb = find(s.clusterB);
    if (ra !== rb) parent[ra] = rb;
  }

  for (let i = 0; i < maxMainMerges; i++) {
    const s = steps[nPreMerges + i]!;
    const ra = find(s.clusterA);
    const rb = find(s.clusterB);
    if (ra !== rb) parent[ra] = rb;
  }

  const remap = new Map<number, number>();
  let nextLabel = 0;
  const labels = new Array<number>(nImages);
  for (let i = 0; i < nImages; i++) {
    const r = find(i);
    if (!remap.has(r)) remap.set(r, nextLabel++);
    labels[i] = remap.get(r)!;
  }

  return labels;
}

function distanceProfileFromTree(tree: LinkageTree): DistanceProfile {
  const distances = tree.steps
    .slice(tree.nPreMerges)
    .map((s) => s.distance)
    .filter((d) => d < 1e10);
  return {
    distances,
    nAfterPremerge: tree.nImages - tree.nPreMerges,
    nGroups: tree.nGroups,
  };
}

export function recutTree(
  targetDir: string,
  nClusters: number,
): { labels: number[]; nClusters: number; distanceProfile: DistanceProfile } {
  const tree = loadTree(targetDir);
  const nAfterPremerge = tree.nImages - tree.nPreMerges;
  // Never go below nGroups clusters — confirmed groups must stay separate
  const minClusters = Math.max(nClusters, tree.nGroups);
  const mainMergesNeeded = Math.max(0, nAfterPremerge - minClusters);
  const labels = cutTree(tree, mainMergesNeeded);
  return {
    labels,
    nClusters: nAfterPremerge - mainMergesNeeded,
    distanceProfile: distanceProfileFromTree(tree),
  };
}

export function recutTreeByThreshold(
  targetDir: string,
  threshold: number,
): { labels: number[]; nClusters: number; distanceProfile: DistanceProfile } {
  const tree = loadTree(targetDir);
  const { nPreMerges, steps } = tree;
  const nAfterPremerge = tree.nImages - nPreMerges;

  // Count main merges below threshold (steps are sorted by distance;
  // group-to-group sentinel distances ~1e18 are naturally excluded)
  let mainMerges = 0;
  for (let i = nPreMerges; i < steps.length; i++) {
    if (steps[i]!.distance >= threshold) break;
    mainMerges++;
  }

  const labels = cutTree(tree, mainMerges);
  return {
    labels,
    nClusters: nAfterPremerge - mainMerges,
    distanceProfile: distanceProfileFromTree(tree),
  };
}

/**
 * HDBSCAN-style stability-based cluster extraction with condensed tree.
 *
 * Instead of a global threshold, scores each potential cluster by how long it
 * persists in the hierarchy. Clusters that exist over a wide range of distances
 * (high stability) are "real" groups — they naturally handle variable sizes.
 *
 * Algorithm:
 * 1. Build the full binary tree from the linkage steps (each merge creates a node)
 * 2. Condense: remove nodes smaller than minClusterSize, propagating their
 *    members up to the parent (these become "noise" that falls out of small clusters)
 * 3. Compute stability for each condensed node: Σ (1/λ_birth - 1/λ_death) per point,
 *    where λ = distance (using distance directly since Ward distances are monotonic)
 * 4. Bottom-up selection: for each node, if its own stability > sum of children's
 *    selected stability, select it and deselect children. Otherwise propagate.
 *
 * minClusterSize controls granularity: higher = fewer, larger clusters.
 * Typical range: 3–15.
 */
export function recutTreeAdaptive(
  targetDir: string,
  minClusterSize: number,
): { labels: number[]; nClusters: number; distanceProfile: DistanceProfile } {
  const tree = loadTree(targetDir);
  const { nImages, steps } = tree;

  const ufParent = new Int32Array(nImages);
  const ufSize = new Int32Array(nImages).fill(1);
  for (let i = 0; i < nImages; i++) ufParent[i] = i;
  function find(x: number): number {
    while (ufParent[x]! !== x) {
      ufParent[x] = ufParent[ufParent[x]!]!;
      x = ufParent[x]!;
    }
    return x;
  }

  // ── Build condensed tree ──────────────────────────────────────────────
  interface CNode {
    id: number;
    birthDist: number;
    deathDist: number;
    size: number;
    children: number[];
    parentId: number;
  }
  const cnodes: CNode[] = [];
  let nextCid = 0;
  const repCnode = new Map<number, number>();
  const imgCnode = new Int32Array(nImages).fill(-1);

  for (const s of steps) {
    if (s.distance >= 1e10) break;
    const ra = find(s.clusterA),
      rb = find(s.clusterB);
    if (ra === rb) continue;
    const sA = ufSize[ra]!,
      sB = ufSize[rb]!;
    const cA = repCnode.get(ra),
      cB = repCnode.get(rb);

    if (sA >= minClusterSize && sB >= minClusterSize) {
      if (cA !== undefined) cnodes[cA]!.deathDist = s.distance;
      if (cB !== undefined) cnodes[cB]!.deathDist = s.distance;
      const pid = nextCid++;
      cnodes.push({
        id: pid,
        birthDist: s.distance,
        deathDist: Infinity,
        size: sA + sB,
        children: [],
        parentId: -1,
      });
      if (cA !== undefined) {
        cnodes[pid]!.children.push(cA);
        cnodes[cA]!.parentId = pid;
      }
      if (cB !== undefined) {
        cnodes[pid]!.children.push(cB);
        cnodes[cB]!.parentId = pid;
      }
      ufParent[ra] = rb;
      ufSize[rb] = sA + sB;
      repCnode.delete(ra);
      repCnode.set(rb, pid);
    } else {
      ufParent[ra] = rb;
      ufSize[rb] = sA + sB;
      const ms = ufSize[rb]!;
      if (!repCnode.has(rb) && ms >= minClusterSize) {
        const id = nextCid++;
        cnodes.push({
          id,
          birthDist: s.distance,
          deathDist: Infinity,
          size: ms,
          children: [],
          parentId: -1,
        });
        repCnode.set(rb, id);
        for (let j = 0; j < nImages; j++) {
          if (find(j) === rb && imgCnode[j] === -1) imgCnode[j] = id;
        }
      }
      if (cA !== undefined && repCnode.has(rb)) {
        const pc = repCnode.get(rb)!;
        cnodes[cA]!.deathDist = s.distance;
        cnodes[pc]!.children.push(cA);
        cnodes[cA]!.parentId = pc;
      }
      repCnode.delete(ra);
    }
  }

  // Remaining unmapped images get their root's condensed node
  for (let i = 0; i < nImages; i++) {
    if (imgCnode[i] === -1) {
      const root = find(i);
      const cid = repCnode.get(root);
      if (cid !== undefined) imgCnode[i] = cid;
    }
  }

  if (cnodes.length === 0) {
    return recutTree(targetDir, Math.max(50, Math.floor(nImages / 30)));
  }

  // ── Stability ─────────────────────────────────────────────────────────
  const cstab = new Float64Array(cnodes.length);
  for (let i = 0; i < cnodes.length; i++) {
    const c = cnodes[i]!;
    const lb = c.birthDist > 0 ? 1 / c.birthDist : 0;
    const ld = c.deathDist < Infinity && c.deathDist > 0 ? 1 / c.deathDist : 0;
    cstab[i] = c.size * Math.max(0, lb - ld);
  }

  // ── Bottom-up selection (topological order: children before parents) ──
  const topoOrder: number[] = [];
  const vis = new Uint8Array(cnodes.length);
  function visit(id: number) {
    if (vis[id]) return;
    vis[id] = 1;
    for (const ch of cnodes[id]!.children) visit(ch);
    topoOrder.push(id);
  }
  for (let i = 0; i < cnodes.length; i++) visit(i);

  const csel = new Uint8Array(cnodes.length);
  const cbest = new Float64Array(cnodes.length);
  for (const i of topoOrder) {
    const c = cnodes[i]!;
    let childSum = 0;
    for (const ch of c.children) childSum += cbest[ch]!;
    if (cstab[i]! > childSum || c.children.length === 0) {
      csel[i] = 1;
      cbest[i] = cstab[i]!;
      const stk = [...c.children];
      while (stk.length) {
        const d = stk.pop()!;
        csel[d] = 0;
        stk.push(...cnodes[d]!.children);
      }
    } else {
      cbest[i] = childSum;
    }
  }

  // ── Label assignment: walk up condensed tree from each image ──────────
  const labels = new Int32Array(nImages).fill(-1);
  const clusterMap = new Map<number, number>();
  let nextLabel = 0;

  for (let i = 0; i < nImages; i++) {
    let cid = imgCnode[i]!;
    while (cid >= 0) {
      if (csel[cid]) {
        if (!clusterMap.has(cid)) clusterMap.set(cid, nextLabel++);
        labels[i] = clusterMap.get(cid)!;
        break;
      }
      cid = cnodes[cid]!.parentId;
    }
  }

  // ── Orphan assignment via merge-step nearest-neighbor ─────────────────
  // For each orphan, replay merge steps with linked-list cluster tracking.
  // When an orphan's cluster merges with one containing labeled images,
  // assign the orphan that label (nearest neighbor in the merge tree).
  const orphanSet = new Set<number>();
  for (let i = 0; i < nImages; i++) if (labels[i]! < 0) orphanSet.add(i);

  if (orphanSet.size > 0) {
    for (let i = 0; i < nImages; i++) {
      ufParent[i] = i;
      ufSize[i] = 1;
    }
    const next = new Int32Array(nImages).fill(-1);
    const head = new Int32Array(nImages);
    const tail = new Int32Array(nImages);
    for (let i = 0; i < nImages; i++) {
      head[i] = i;
      tail[i] = i;
    }

    for (const s of steps) {
      if (s.distance >= 1e10 || orphanSet.size === 0) break;
      const ra = find(s.clusterA),
        rb = find(s.clusterB);
      if (ra === rb) continue;

      let labelA = -1,
        labelB = -1;
      for (let j = head[ra]!; j >= 0; j = next[j]!) {
        if (labels[j]! >= 0) {
          labelA = labels[j]!;
          break;
        }
      }
      for (let j = head[rb]!; j >= 0; j = next[j]!) {
        if (labels[j]! >= 0) {
          labelB = labels[j]!;
          break;
        }
      }

      if (labelA >= 0 && labelB < 0) {
        for (let j = head[rb]!; j >= 0; j = next[j]!) {
          if (labels[j]! < 0) {
            labels[j] = labelA;
            orphanSet.delete(j);
          }
        }
      } else if (labelB >= 0 && labelA < 0) {
        for (let j = head[ra]!; j >= 0; j = next[j]!) {
          if (labels[j]! < 0) {
            labels[j] = labelB;
            orphanSet.delete(j);
          }
        }
      }

      ufParent[ra] = rb;
      ufSize[rb] = ufSize[ra]! + ufSize[rb]!;
      next[tail[rb]!] = head[ra]!;
      tail[rb] = tail[ra]!;
    }

    // Any remaining orphans: walk cluster list to find a labeled neighbor
    for (const i of orphanSet) {
      const root = find(i);
      for (let j = head[root]!; j >= 0; j = next[j]!) {
        if (labels[j]! >= 0) {
          labels[i] = labels[j]!;
          break;
        }
      }
      if (labels[i]! < 0) labels[i] = nextLabel++;
    }
  }

  return {
    labels: Array.from(labels),
    nClusters: nextLabel,
    distanceProfile: distanceProfileFromTree(tree),
  };
}

export function getDistanceProfile(targetDir: string): DistanceProfile {
  return distanceProfileFromTree(loadTree(targetDir));
}

// ── Embedding loading (shared between auto-naming + NN query) ───────────────

export const MODEL_KEYS = ["clip", "dino", "dinov3", "pecore_l", "pecore_g", "color"] as const;
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

let _textEmbCache: TextEmbeddings | null = null;
const _modelEmbCaches = new Map<string, ModelEmbedding>();
let _tfidfStatsCache: { globalAvg: Float64Array; globalStd: Float64Array } | null = null;
let _hashMappingCache: { targetDir: string; mtime: number; mapping: HashMapping } | null = null;

function loadTextEmbeddings(targetDir: string): TextEmbeddings {
  if (_textEmbCache) return _textEmbCache;
  const path = join(cacheDir(targetDir), "text_embeddings.json");
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

/** mtime-keyed wrapper around loadHashMapping — content_hashes.json changes only on extraction/rename. */
export function cachedHashMapping(targetDir: string): HashMapping {
  const cache = cacheDir(targetDir);
  const hashesPath = join(cache, "content_hashes.json");
  const mtime = statSync(hashesPath).mtimeMs;
  if (
    _hashMappingCache &&
    _hashMappingCache.targetDir === targetDir &&
    _hashMappingCache.mtime === mtime
  ) {
    return _hashMappingCache.mapping;
  }
  const mapping = loadHashMapping(cache);
  _hashMappingCache = { targetDir, mtime, mapping };
  return mapping;
}

export function loadModelEmbedding(targetDir: string, modelKey: ModelKey): ModelEmbedding {
  const key = `${targetDir}::${modelKey}`;
  const cached = _modelEmbCaches.get(key);
  if (cached) return cached;

  const mapping = cachedHashMapping(targetDir);
  const npzBuf = readFileSync(join(cacheDir(targetDir), "clip_hash_cache.npz")) as Buffer;
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

/** Back-compat wrapper for computeAutoNames / buildRecutResult which use `clip` as the field name. */
function loadClipEmbeddings(targetDir: string): {
  filenames: string[];
  clip: Float32Array;
  nImages: number;
  dim: number;
} {
  const emb = loadModelEmbedding(targetDir, "clip");
  return { filenames: emb.filenames, clip: emb.data, nImages: emb.filenames.length, dim: emb.dim };
}

/** Fallback when CLIP embeddings aren't available — uses confirmed group names or generic labels. */
function clustersWithoutAutoNames(clusters: RustOutput["clusters"]): ClusterResultData[] {
  return clusters.map((c, i) => ({
    id: c.id,
    autoName: c.confirmed_group?.name ?? `Cluster ${i + 1}`,
    autoTags: [],
    images: c.images,
    confirmedGroup: c.confirmed_group,
  }));
}

export function computeAutoNames(
  targetDir: string,
  clusters: RustOutput["clusters"],
): ClusterResultData[] {
  const { terms, flat: textFlat, dim: textDim } = loadTextEmbeddings(targetDir);
  const { filenames, clip, nImages, dim } = loadClipEmbeddings(targetDir);
  const fnToIdx = new Map(filenames.map((f, i) => [f, i]));

  const nTerms = terms.length;

  // Compute global stats (cached — only depends on embeddings, not on cluster cut)
  if (!_tfidfStatsCache) {
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
    _tfidfStatsCache = { globalAvg, globalStd };
  }
  const { globalAvg, globalStd } = _tfidfStatsCache;

  return clusters.map((c) => {
    // If this cluster has a confirmed group, use its name
    if (c.confirmed_group && c.images.length === c.confirmed_group.images.length) {
      return {
        id: c.id,
        autoName: c.confirmed_group.name,
        autoTags: [],
        images: c.images,
        confirmedGroup: c.confirmed_group,
      };
    }

    // Compute cluster centroid in CLIP space
    // All typed array accesses below are within bounds
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
    const autoName = c.confirmed_group ? c.confirmed_group.name : parts.join(" · ");

    return {
      id: c.id,
      autoName,
      autoTags: zScores.slice(0, 8),
      images: c.images,
      confirmedGroup: c.confirmed_group,
    };
  });
}

// ── Contact sheet generation ─────────────────────────────────────────────────

function escapeSvgText(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function truncateMiddle(s: string, max: number): string {
  if (s.length <= max) return s;
  const keep = max - 1;
  const left = Math.ceil(keep * 0.55);
  const right = keep - left;
  return `${s.slice(0, left)}…${s.slice(s.length - right)}`;
}

interface LabelCell {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
}

function buildLabelOverlay(cells: LabelCell[], width: number, height: number): Buffer {
  // Fixed sizes — labels just need to be legible, not scale with the thumb.
  const labelH = 34;
  const fontSize = 16;
  const padX = 10;
  const parts: string[] = [];
  for (const c of cells) {
    const maxChars = Math.max(12, Math.floor((c.width - padX * 2) / 9));
    const text = escapeSvgText(truncateMiddle(c.label, maxChars));
    parts.push(
      `<rect x="${c.x}" y="${c.y + c.height - labelH}" width="${c.width}" height="${labelH}" fill="black" fill-opacity="0.72"/>`,
    );
    parts.push(
      `<text x="${c.x + padX}" y="${c.y + c.height - 11}" font-family="Arial, sans-serif" font-size="${fontSize}" font-weight="600" fill="white">${text}</text>`,
    );
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${parts.join("")}</svg>`;
  return Buffer.from(svg);
}

// Grid for N images (index = N-1). Minimises empty cells while keeping the
// layout roughly square; N=10 keeps 4×3 with 2 blanks for aspect consistency.
const CONTACT_SHEET_GRID: [cols: number, rows: number][] = [
  [1, 1], // 1
  [2, 1], // 2
  [3, 1], // 3
  [2, 2], // 4
  [3, 2], // 5 (1 blank)
  [3, 2], // 6
  [4, 2], // 7 (1 blank)
  [4, 2], // 8
  [3, 3], // 9
  [4, 3], // 10 (2 blanks)
  [4, 3], // 11 (1 blank)
  [4, 3], // 12
];

export async function generateContactSheet(
  targetDir: string,
  filenames: string[],
  clusterName: string,
  withLabels = false,
): Promise<string> {
  const outDir = join(cacheDir(targetDir), "contact_sheets");
  mkdirSync(outDir, { recursive: true });

  // Pick up to 12 evenly spaced images
  const maxImages = 12;
  let selected: string[];
  if (filenames.length <= maxImages) {
    selected = filenames;
  } else {
    selected = [];
    for (let i = 0; i < maxImages; i++) {
      const idx = Math.floor((i * filenames.length) / maxImages);
      selected.push(filenames[idx]!); // idx < filenames.length by construction
    }
  }

  const [cols, rows] = CONTACT_SHEET_GRID[selected.length - 1] ?? [4, 3];

  // Justified-row layout: per-image widths sum exactly to canvasW so no letterbox
  // bars appear around real images; rows with blanks pad the width budget with
  // avgAspect so non-full rows scale the same as full ones.
  const paths = selected.map((f) => join(targetDir, f));
  const aspects = await Promise.all(
    paths.map(async (p) => {
      const m = await sharp(p).metadata();
      return (m.width ?? 1) / (m.height ?? 1);
    }),
  );
  const avgAspect = aspects.reduce((s, a) => s + a, 0) / aspects.length;

  const canvasW = 2000;
  interface Cell {
    x: number;
    y: number;
    width: number;
    height: number;
  }
  const cells: Cell[] = [];
  let yCursor = 0;
  for (let r = 0; r < rows; r++) {
    const start = r * cols;
    const end = Math.min(start + cols, selected.length);
    const rowAspects = aspects.slice(start, end);
    const blanks = cols - rowAspects.length;
    const effectiveSum = rowAspects.reduce((s, a) => s + a, 0) + blanks * avgAspect;
    const rowH = Math.round(canvasW / effectiveSum);

    let x = 0;
    for (let i = 0; i < rowAspects.length; i++) {
      const isLastInFullRow = blanks === 0 && i === rowAspects.length - 1;
      const right = isLastInFullRow ? canvasW : Math.round(x + rowH * rowAspects[i]!);
      cells.push({ x, y: yCursor, width: right - x, height: rowH });
      x = right;
    }
    yCursor += rowH;
  }
  const canvasH = yCursor;

  const thumbnails = await Promise.all(
    cells.map((c, i) =>
      sharp(paths[i]!).resize(c.width, c.height, { fit: "cover" }).jpeg({ quality: 85 }).toBuffer(),
    ),
  );

  // Label overlay composited last so text sits above thumbnails.
  const composites: { input: Buffer; left: number; top: number }[] = cells.map((c, i) => ({
    input: thumbnails[i]!,
    left: c.x,
    top: c.y,
  }));
  if (withLabels) {
    const labelCells = cells.map((c, i) => ({ ...c, label: selected[i]! }));
    composites.push({
      input: buildLabelOverlay(labelCells, canvasW, canvasH),
      left: 0,
      top: 0,
    });
  }

  const safeName = clusterName.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 80);
  const outPath = join(outDir, `${safeName}.jpg`);

  await sharp({
    create: {
      width: canvasW,
      height: canvasH,
      channels: 3,
      background: { r: 26, g: 26, b: 46 },
    },
  })
    .composite(composites)
    .jpeg({ quality: 85 })
    .toFile(outPath);

  return outPath;
}

function computeSuggestedCounts(nImages: number): number[] {
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

// ── Full pipeline ────────────────────────────────────────────────────────────

/** Derive the set of model keys needed for a given weight config.
 * Only models explicitly given a positive weight are extracted — missing keys
 * mean "don't extract", so CLIP etc. are never pulled unless the user asked for them. */
function modelsForWeights(weights?: WeightConfig): string[] | undefined {
  if (!weights) return undefined; // no config → extract all (auto mode)
  const keys: (keyof WeightConfig)[] = ["clip", "color", "dino", "pecore_l", "pecore_g", "dinov3"];
  return keys.filter((k) => (weights[k] ?? 0) > 0);
}

export async function runFullCluster(
  targetDir: string,
  nClusters: number,
  onProgress?: (line: string) => void,
  weights?: WeightConfig,
  usePatches?: boolean,
): Promise<ClusterData> {
  const required = modelsForWeights(weights);
  if (usePatches && required && !required.includes("dinov3")) {
    required.push("dinov3");
  }
  const signal = getClusterAbortSignal();
  const hasClip = !required || required.includes("clip");
  const extractionPromises: Promise<unknown>[] = [
    extractFeatures(targetDir, onProgress, required ? { required, signal } : { signal }),
  ];
  if (hasClip) extractionPromises.push(ensureTextEmbeddings(targetDir));
  const [extraction] = (await Promise.all(extractionPromises)) as [
    Awaited<ReturnType<typeof extractFeatures>>,
    ...unknown[],
  ];
  log("cluster", `Extraction: ${extraction.extracted} new, ${extraction.cached} cached`);

  const rustOutput = await runLinkage(targetDir, nClusters, weights, usePatches);
  log("cluster", `Linkage complete: ${rustOutput.clusters.length} clusters`);

  const clusters = hasClip
    ? computeAutoNames(targetDir, rustOutput.clusters)
    : clustersWithoutAutoNames(rustOutput.clusters);

  const nImages = clusters.reduce((n, c) => n + c.images.length, 0);
  const distanceProfile = getDistanceProfile(targetDir);
  return { clusters, suggestedCounts: computeSuggestedCounts(nImages), nClusters, distanceProfile };
}

export async function runLinkageOnly(
  targetDir: string,
  nClusters: number,
  weights?: WeightConfig,
  usePatches?: boolean,
): Promise<ClusterData> {
  const rustOutput = await runLinkage(targetDir, nClusters, weights, usePatches);
  const clusters = computeAutoNames(targetDir, rustOutput.clusters);
  const nImages = clusters.reduce((n, c) => n + c.images.length, 0);
  return { clusters, suggestedCounts: computeSuggestedCounts(nImages), nClusters };
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
function buildClustersFromLabels(
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
    .map(([, images], ci) => {
      const confirmed = images.find((f) => imgToGroup.has(f));
      const group = confirmed ? (imgToGroup.get(confirmed) ?? null) : null;
      return {
        id: `${opts.idPrefix}${ci}`,
        images: images.sort(),
        confirmed_group: group ? { id: group.id, name: group.name, images: group.images } : null,
      };
    });

  try {
    return computeAutoNames(targetDir, rawClusters);
  } catch {
    return clustersWithoutAutoNames(rawClusters);
  }
}

async function buildRecutResult(
  targetDir: string,
  labels: number[],
  nClusters: number,
  distanceProfile: DistanceProfile,
): Promise<ClusterData> {
  const { filenames } = cachedHashMapping(targetDir);
  const clusters = buildClustersFromLabels(targetDir, filenames, labels, { idPrefix: "cluster_" });
  return {
    clusters,
    suggestedCounts: computeSuggestedCounts(filenames.length),
    nClusters,
    distanceProfile,
  };
}

// ── Imported clusters (JSON import bypass) ──────────────────────────────────

const IMPORTED_CLUSTERS_FILENAME = "imported_clusters.json";

export function importedClustersPath(targetDir: string): string {
  return join(cacheDir(targetDir), IMPORTED_CLUSTERS_FILENAME);
}

export function hasImportedClusters(targetDir: string): boolean {
  return existsSync(importedClustersPath(targetDir));
}

export async function loadImportedClusters(targetDir: string): Promise<ClusterData | null> {
  try {
    return (await Bun.file(importedClustersPath(targetDir)).json()) as ClusterData;
  } catch {
    return null;
  }
}

export async function saveImportedClusters(targetDir: string, data: ClusterData): Promise<void> {
  mkdirSync(cacheDir(targetDir), { recursive: true });
  await Bun.write(importedClustersPath(targetDir), JSON.stringify(data, null, 2));
}

export async function clearImportedClusters(targetDir: string): Promise<void> {
  await unlink(importedClustersPath(targetDir)).catch(() => {});
}

export async function buildImportedResult(
  targetDir: string,
  input: ImportClusterInput[],
): Promise<ClusterData> {
  const imgToGroup = new Map<string, ImageGroup>();
  for (const g of loadGroups(targetDir)) {
    for (const f of g.images) imgToGroup.set(f, g);
  }

  const clusters: ClusterResultData[] = input.map((c, i) => {
    const sortedImages = [...c.images].sort();
    const confirmed = sortedImages.find((f) => imgToGroup.has(f));
    const group = confirmed ? (imgToGroup.get(confirmed) ?? null) : null;
    return {
      id: `imported_${i}`,
      autoName: c.name,
      autoTags: [],
      images: sortedImages,
      confirmedGroup: group ? { id: group.id, name: group.name, images: group.images } : null,
    };
  });

  const totalImages = clusters.reduce((n, c) => n + c.images.length, 0);
  return {
    clusters,
    suggestedCounts: computeSuggestedCounts(totalImages),
    nClusters: clusters.length,
  };
}

// ── Cache invalidation ───────────────────────────────────────────────────────

export function invalidateClusterCache() {
  _textEmbCache = null;
  _modelEmbCaches.clear();
  _tfidfStatsCache = null;
  _hashMappingCache = null;
  _patchDistMatrixCache = null;
}

// ── Job guard ───────────────────────────────────────────────────────────────

let _clusterAbort: AbortController | null = null;
export function isClusterJobRunning() {
  return _clusterAbort !== null;
}
export function setClusterJobRunning(v: boolean) {
  _clusterAbort = v ? new AbortController() : null;
}
export function getClusterAbortSignal(): AbortSignal | undefined {
  return _clusterAbort?.signal;
}
export function cancelClusterJob() {
  if (_clusterAbort) {
    _clusterAbort.abort();
    log("cluster", "Cluster job cancellation requested");
  }
}

// ── Scoped clustering (subset of groups) ────────────────────────────────────

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
  return join(cacheDir(targetDir), "scoped", scopeKey);
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

  const contentHashesPath = join(cacheDir(targetDir), "content_hashes.json");
  if (!existsSync(contentHashesPath)) {
    throw new Error("content_hashes.json missing — run feature extraction first");
  }
  const contentHashes: Record<string, string> = JSON.parse(
    readFileSync(contentHashesPath, "utf-8"),
  );

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
  await Bun.write(scopedSubsetPath(targetDir, scopeKey), JSON.stringify(subsetFilenames));
  await Bun.write(scopedMetaPath(targetDir, scopeKey), JSON.stringify(meta));

  const cache = cacheDir(targetDir);
  const hashCachePath = join(cache, "clip_hash_cache.npz");
  const contentHashesPath = join(cache, "content_hashes.json");
  const hashOrderPath = join(cache, "hash_cache_order.json");
  const groupsFile = join(targetDir, ".reorder-groups.json");
  const treePath = scopedTreePath(targetDir, scopeKey);

  ensureHashOrderJson(cache);

  const args = [
    "--hash-cache",
    hashCachePath,
    "--content-hashes",
    contentHashesPath,
    "--hash-order",
    hashOrderPath,
    "--filenames",
    scopedSubsetPath(targetDir, scopeKey),
    "--n-clusters",
    String(nClusters),
    "--output-tree",
    treePath,
  ];
  if (existsSync(groupsFile)) args.push("--groups", groupsFile);
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
  const stdout = await runRustBinary(RUST_BINARY, args, "scoped-cluster", onProgress);
  const rustOutput: RustOutput = JSON.parse(stdout);
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
  // Tree staleness is surfaced to the client (treeStale flag → "Re-run scoped"
  // banner), matching how the global tree handles it. The server trusts the
  // caller and just re-cuts from disk.
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
  const scopedDir = join(cacheDir(targetDir), "scoped");
  if (existsSync(scopedDir)) {
    rmSync(scopedDir, { recursive: true, force: true });
  }
}

// ── Merge suggestions: DINOv3 patch-based group similarity ──────────────────

export interface GroupPairResult {
  group_a: string;
  group_b: string;
  size_a: number;
  size_b: number;
  patch_median: number;
  patch_p75: number;
  patch_best: number;
  closest_pair: [string, string];
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
  const patchesCachePath = join(
    cache,
    fullRes ? "dinov3_patches_full_hash_cache.npy" : "dinov3_patches_hash_cache.npy",
  );
  const patchesHashesPath = join(cache, "dinov3_patches_hashes.json");
  const contentHashesPath = join(cache, "content_hashes.json");
  const groupsPath = join(targetDir, ".reorder-groups.json");
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
      out = out.filter((r) => r.size_a + r.size_b <= maxCombinedSize);
    }
    if (minScore > 0) {
      out = out.filter((r) => r.patch_median >= minScore);
    }
    return out;
  };

  // Disk cache is valid if newer than both the groups file and patches cache.
  try {
    const cacheFile = Bun.file(resultCachePath);
    const cacheTime = cacheFile.lastModified;
    if (
      cacheTime > Bun.file(groupsPath).lastModified &&
      cacheTime > Bun.file(patchesCachePath).lastModified &&
      cacheTime > Bun.file(contentHashesPath).lastModified
    ) {
      log("merge-suggestions", `Using cached results (${fullRes ? "full-res" : "pooled"})`);
      options?.onProgress?.("Using cached results");
      const cached: GroupPairResult[] = await cacheFile.json();
      return applyFilters(cached);
    }
  } catch {}

  const args = [
    "--patches-cache",
    patchesCachePath,
    "--content-hashes",
    contentHashesPath,
    "--patches-hashes",
    patchesHashesPath,
    "--groups",
    groupsPath,
    // Compute unfiltered so the cache can serve any threshold; TS re-filters on return.
    "--min-score",
    "0",
  ];
  if (maxCombinedSize > 0) {
    args.push("--max-combined-size", String(maxCombinedSize));
  }

  const label = fullRes ? "merge-suggestions-full" : "merge-suggestions";
  log(label, `Running group-similarity: ${args.join(" ")}`);
  options?.onProgress?.(`Loading ${fullRes ? "14x14 full-res" : "7x7 pooled"} patches...`);

  const stdout = await runRustBinary(GROUP_SIM_BINARY, args, label, (line) => {
    options?.onProgress?.(line);
  });
  const allResults: GroupPairResult[] = JSON.parse(stdout);

  await Bun.write(resultCachePath, stdout);
  log(label, `Cached ${allResults.length} results to ${resultCachePath}`);

  return applyFilters(allResults);
}

// Progress broadcasting — allows reconnecting to an in-progress job
let _lastProgress = "";
const _progressListeners = new Set<(msg: string) => void>();

export function broadcastProgress(msg: string) {
  _lastProgress = msg;
  for (const listener of _progressListeners) listener(msg);
}

export function getLastProgress() {
  return _lastProgress;
}

export function subscribeProgress(listener: (msg: string) => void): () => void {
  _progressListeners.add(listener);
  return () => _progressListeners.delete(listener);
}
