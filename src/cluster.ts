/**
 * Server-side clustering support:
 * - Spawns Python for feature extraction
 * - Spawns Rust for Ward's linkage
 * - Parses linkage tree binary for re-cuts
 * - Computes TF-IDF auto-names from cached CLIP embeddings
 * - Generates contact sheets via Sharp
 */

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import sharp from "sharp";
import type {
  ClusterData,
  ClusterResultData,
  DistanceProfile,
  WeightConfig,
} from "./client/types.ts";
import { log } from "./log.ts";

export type { WeightConfig };

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
const PYTHON = process.env.CLUSTER_PYTHON || "/tmp/imgcluster-env/bin/python3";

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

// ── Feature extraction (Python) ──────────────────────────────────────────────

export async function extractFeatures(
  targetDir: string,
  onProgress?: (line: string) => void,
  opts?: { force?: string[]; required?: string[] },
): Promise<{ total: number; cached: number; extracted: number }> {
  const script = join(SCRIPTS_DIR, "extract_features.py");
  const cache = cacheDir(targetDir);

  if (!existsSync(PYTHON)) {
    throw new Error(
      `Python not found at ${PYTHON}. Create venv with: uv venv /tmp/imgcluster-env && source /tmp/imgcluster-env/bin/activate && uv pip install torch torchvision open-clip-torch pillow numpy`,
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
  if (exitCode !== 0) {
    throw new Error(`Feature extraction failed (exit ${exitCode}): ${stderrText}`);
  }

  const stdout = await new Response(proc.stdout).text();
  return JSON.parse(stdout);
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

export async function runLinkage(
  targetDir: string,
  nClusters: number,
  weights?: WeightConfig,
): Promise<RustOutput> {
  const cache = cacheDir(targetDir);
  const embeddings = join(cache, "clip_embeddings.npz");
  const groupsFile = join(targetDir, ".reorder-groups.json");
  const treePath = join(cache, "linkage_tree.bin");

  const args = [
    "--embeddings",
    embeddings,
    "--n-clusters",
    String(nClusters),
    "--output-tree",
    treePath,
  ];
  if (existsSync(groupsFile)) {
    args.push("--groups", groupsFile);
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
  const proc = Bun.spawn([RUST_BINARY, ...args], {
    stdout: "pipe",
    stderr: "pipe",
  });

  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    const err = await new Response(proc.stderr).text();
    throw new Error(`Rust linkage failed (exit ${exitCode}): ${err}`);
  }

  const stdout = await new Response(proc.stdout).text();
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
): { labels: number[]; distanceProfile: DistanceProfile } {
  const tree = loadTree(targetDir);
  const nAfterPremerge = tree.nImages - tree.nPreMerges;
  // Never go below nGroups clusters — confirmed groups must stay separate
  const minClusters = Math.max(nClusters, tree.nGroups);
  const mainMergesNeeded = Math.max(0, nAfterPremerge - minClusters);
  const labels = cutTree(tree, mainMergesNeeded);
  return { labels, distanceProfile: distanceProfileFromTree(tree) };
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
  const { nImages, nPreMerges, steps } = tree;

  const ufParent = new Int32Array(nImages);
  const ufSize = new Int32Array(nImages).fill(1);
  for (let i = 0; i < nImages; i++) ufParent[i] = i;
  function find(x: number): number {
    while (ufParent[x]! !== x) { ufParent[x] = ufParent[ufParent[x]!]!; x = ufParent[x]!; }
    return x;
  }

  // ── Build condensed tree ──────────────────────────────────────────────
  interface CNode {
    id: number; birthDist: number; deathDist: number; size: number;
    children: number[]; parentId: number;
  }
  const cnodes: CNode[] = [];
  let nextCid = 0;
  const repCnode = new Map<number, number>();
  const imgCnode = new Int32Array(nImages).fill(-1);

  for (const s of steps) {
    if (s.distance >= 1e10) break;
    const ra = find(s.clusterA), rb = find(s.clusterB);
    if (ra === rb) continue;
    const sA = ufSize[ra]!, sB = ufSize[rb]!;
    const cA = repCnode.get(ra), cB = repCnode.get(rb);

    if (sA >= minClusterSize && sB >= minClusterSize) {
      if (cA !== undefined) cnodes[cA]!.deathDist = s.distance;
      if (cB !== undefined) cnodes[cB]!.deathDist = s.distance;
      const pid = nextCid++;
      cnodes.push({ id: pid, birthDist: s.distance, deathDist: Infinity,
        size: sA + sB, children: [], parentId: -1 });
      if (cA !== undefined) { cnodes[pid]!.children.push(cA); cnodes[cA]!.parentId = pid; }
      if (cB !== undefined) { cnodes[pid]!.children.push(cB); cnodes[cB]!.parentId = pid; }
      ufParent[ra] = rb; ufSize[rb] = sA + sB;
      repCnode.delete(ra); repCnode.set(rb, pid);
    } else {
      ufParent[ra] = rb; ufSize[rb] = sA + sB;
      const ms = ufSize[rb]!;
      if (!repCnode.has(rb) && ms >= minClusterSize) {
        const id = nextCid++;
        cnodes.push({ id, birthDist: s.distance, deathDist: Infinity,
          size: ms, children: [], parentId: -1 });
        repCnode.set(rb, id);
        for (let j = 0; j < nImages; j++) {
          if (find(j) === rb && imgCnode[j] === -1) imgCnode[j] = id;
        }
      }
      if (cA !== undefined && repCnode.has(rb)) {
        const pc = repCnode.get(rb)!;
        cnodes[cA]!.deathDist = s.distance;
        cnodes[pc]!.children.push(cA); cnodes[cA]!.parentId = pc;
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
    if (vis[id]) return; vis[id] = 1;
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
        const d = stk.pop()!; csel[d] = 0;
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
  for (let i = 0; i < nImages; i++) if (labels[i] < 0) orphanSet.add(i);

  if (orphanSet.size > 0) {
    for (let i = 0; i < nImages; i++) { ufParent[i] = i; ufSize[i] = 1; }
    const next = new Int32Array(nImages).fill(-1);
    const head = new Int32Array(nImages);
    const tail = new Int32Array(nImages);
    for (let i = 0; i < nImages; i++) { head[i] = i; tail[i] = i; }

    for (const s of steps) {
      if (s.distance >= 1e10 || orphanSet.size === 0) break;
      const ra = find(s.clusterA), rb = find(s.clusterB);
      if (ra === rb) continue;

      let labelA = -1, labelB = -1;
      for (let j = head[ra]!; j >= 0; j = next[j]!) {
        if (labels[j] >= 0) { labelA = labels[j]!; break; }
      }
      for (let j = head[rb]!; j >= 0; j = next[j]!) {
        if (labels[j] >= 0) { labelB = labels[j]!; break; }
      }

      if (labelA >= 0 && labelB < 0) {
        for (let j = head[rb]!; j >= 0; j = next[j]!) {
          if (labels[j] < 0) { labels[j] = labelA; orphanSet.delete(j); }
        }
      } else if (labelB >= 0 && labelA < 0) {
        for (let j = head[ra]!; j >= 0; j = next[j]!) {
          if (labels[j] < 0) { labels[j] = labelB; orphanSet.delete(j); }
        }
      }

      ufParent[ra] = rb; ufSize[rb] = ufSize[ra]! + ufSize[rb]!;
      next[tail[rb]!] = head[ra]!;
      tail[rb] = tail[ra]!;
    }

    // Any remaining orphans: walk cluster list to find a labeled neighbor
    for (const i of orphanSet) {
      const root = find(i);
      for (let j = head[root]!; j >= 0; j = next[j]!) {
        if (labels[j] >= 0) { labels[i] = labels[j]!; break; }
      }
      if (labels[i] < 0) labels[i] = nextLabel++;
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

// ── Auto-naming (TF-IDF) ────────────────────────────────────────────────────

let _textEmbCache: TextEmbeddings | null = null;
let _clipEmbCache: {
  filenames: string[];
  clip: Float32Array;
  nImages: number;
  dim: number;
} | null = null;
let _tfidfStatsCache: { globalAvg: Float64Array; globalStd: Float64Array } | null = null;

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

function loadClipEmbeddings(targetDir: string) {
  if (_clipEmbCache) return _clipEmbCache;

  // Read filenames
  const fnPath = join(cacheDir(targetDir), "clip_embeddings.filenames.json");
  const filenames: string[] = JSON.parse(readFileSync(fnPath, "utf-8"));

  // Read npz — we only need the clip array (512-dim, float32)
  // npz is a zip of .npy files. Use a simple parser.
  const npzBuf = readFileSync(join(cacheDir(targetDir), "clip_embeddings.npz"));
  const clip = parseNpyFromNpz(npzBuf, "clip.npy");
  const nImages = filenames.length;
  const dim = clip.length / nImages;

  _clipEmbCache = { filenames, clip, nImages, dim };
  return _clipEmbCache;
}

function parseNpyFromNpz(npzBuf: Buffer, entryName: string): Float32Array {
  // npz is a ZIP file. Find the entry by scanning local file headers.
  let offset = 0;
  while (offset < npzBuf.length - 4) {
    const sig = npzBuf.readUInt32LE(offset);
    if (sig !== 0x04034b50) break; // PK\x03\x04
    const compMethod = npzBuf.readUInt16LE(offset + 8);
    const compSize = npzBuf.readUInt32LE(offset + 18);
    const uncompSize = npzBuf.readUInt32LE(offset + 22);
    const fnLen = npzBuf.readUInt16LE(offset + 26);
    const extraLen = npzBuf.readUInt16LE(offset + 28);
    const fn = npzBuf.subarray(offset + 30, offset + 30 + fnLen).toString("utf-8");
    const dataStart = offset + 30 + fnLen + extraLen;

    if (fn === entryName) {
      let data: Buffer;
      if (compMethod === 0) {
        data = npzBuf.subarray(dataStart, dataStart + uncompSize);
      } else {
        // Deflate
        data = Buffer.from(
          Bun.inflateSync(
            npzBuf.subarray(dataStart, dataStart + compSize) as Uint8Array<ArrayBuffer>,
          ),
        );
      }
      // Parse .npy header
      // Magic: \x93NUMPY, version, header_len, then header string, then data
      const headerLen = data.readUInt16LE(8);
      const arrayData = data.subarray(10 + headerLen);
      return new Float32Array(arrayData.buffer, arrayData.byteOffset, arrayData.byteLength / 4);
    }

    offset = dataStart + compSize;
  }
  throw new Error(`Entry ${entryName} not found in npz`);
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

  return clusters.map((c, ci) => {
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

export async function generateContactSheet(
  targetDir: string,
  filenames: string[],
  clusterName: string,
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

  const thumbSize = 400;
  const cols = 4;
  const rows = Math.ceil(selected.length / cols);
  const width = cols * thumbSize;
  const height = rows * thumbSize;

  // Generate thumbnails
  const thumbnails = await Promise.all(
    selected.map(async (f) => {
      const path = join(targetDir, f);
      return sharp(path)
        .resize(thumbSize, thumbSize, { fit: "cover" })
        .jpeg({ quality: 85 })
        .toBuffer();
    }),
  );

  // Compose grid
  const composites = thumbnails.map((buf, i) => ({
    input: buf,
    left: (i % cols) * thumbSize,
    top: Math.floor(i / cols) * thumbSize,
  }));

  const safeName = clusterName.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 80);
  const outPath = join(outDir, `${safeName}.jpg`);

  await sharp({
    create: {
      width,
      height,
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

/** Derive the set of model keys needed for a given weight config. */
function modelsForWeights(weights?: WeightConfig): string[] | undefined {
  if (!weights) return undefined; // no config → extract all (auto mode)
  // Rust defaults: clip=1.0, color=0.5, others=0.0
  const defaults: Record<string, number> = { clip: 1.0, color: 0.5, dino: 0.0, pecore_l: 0.0, pecore_g: 0.0 };
  const needed: string[] = [];
  for (const [key, defaultVal] of Object.entries(defaults)) {
    const val = weights[key as keyof WeightConfig] ?? defaultVal;
    if (val > 0) needed.push(key);
  }
  // CLIP is always needed for TF-IDF auto-naming
  if (!needed.includes("clip")) needed.push("clip");
  return needed;
}

export async function runFullCluster(
  targetDir: string,
  nClusters: number,
  onProgress?: (line: string) => void,
  weights?: WeightConfig,
): Promise<ClusterData> {
  const required = modelsForWeights(weights);
  const [extraction] = await Promise.all([
    extractFeatures(targetDir, onProgress, required ? { required } : undefined),
    ensureTextEmbeddings(targetDir),
  ]);
  log("cluster", `Extraction: ${extraction.extracted} new, ${extraction.cached} cached`);

  const rustOutput = await runLinkage(targetDir, nClusters, weights);
  log("cluster", `Linkage complete: ${rustOutput.clusters.length} clusters`);

  const clusters = computeAutoNames(targetDir, rustOutput.clusters);

  const nImages = clusters.reduce((n, c) => n + c.images.length, 0);
  const distanceProfile = getDistanceProfile(targetDir);
  return { clusters, suggestedCounts: computeSuggestedCounts(nImages), nClusters, distanceProfile };
}

export async function runLinkageOnly(
  targetDir: string,
  nClusters: number,
  weights?: WeightConfig,
): Promise<ClusterData> {
  const rustOutput = await runLinkage(targetDir, nClusters, weights);
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

export async function runRecutAdaptive(targetDir: string, minClusterSize: number): Promise<ClusterData> {
  const { labels, nClusters, distanceProfile } = recutTreeAdaptive(targetDir, minClusterSize);
  return buildRecutResult(targetDir, labels, nClusters, distanceProfile);
}

async function buildRecutResult(
  targetDir: string,
  labels: number[],
  nClusters: number,
  distanceProfile: DistanceProfile,
): Promise<ClusterData> {
  const { filenames } = loadClipEmbeddings(targetDir);

  // Load groups for confirmed_group info
  const groupsPath = join(targetDir, ".reorder-groups.json");
  let groups: { id: string; name: string; images: string[] }[] = [];
  try {
    groups = await Bun.file(groupsPath).json();
  } catch {}
  const imgToGroup = new Map<string, (typeof groups)[0]>();
  for (const g of groups) {
    for (const f of g.images) imgToGroup.set(f, g);
  }

  // Group by label
  const clusterMembers = new Map<number, string[]>();
  for (let i = 0; i < labels.length; i++) {
    const label = labels[i]!;
    if (!clusterMembers.has(label)) clusterMembers.set(label, []);
    clusterMembers.get(label)!.push(filenames[i]!);
  }

  const rawClusters = [...clusterMembers.entries()]
    .sort((a, b) => b[1].length - a[1].length)
    .map(([label, images], ci) => {
      const confirmed = images.find((f) => imgToGroup.has(f));
      const group = confirmed ? (imgToGroup.get(confirmed) ?? null) : null;
      return {
        id: `cluster_${ci}`,
        images: images.sort(),
        confirmed_group: group ? { id: group.id, name: group.name, images: group.images } : null,
      };
    });

  const clusters = computeAutoNames(targetDir, rawClusters);

  return {
    clusters,
    suggestedCounts: computeSuggestedCounts(filenames.length),
    nClusters,
    distanceProfile,
  };
}

// ── Cache invalidation ───────────────────────────────────────────────────────

export function invalidateClusterCache() {
  _textEmbCache = null;
  _clipEmbCache = null;
  _tfidfStatsCache = null;
}

// ── Job guard ───────────────────────────────────────────────────────────────

let _clusterJobRunning = false;
export function isClusterJobRunning() {
  return _clusterJobRunning;
}
export function setClusterJobRunning(v: boolean) {
  _clusterJobRunning = v;
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
