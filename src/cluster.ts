/**
 * Server-side clustering support:
 * - Spawns Python for feature extraction
 * - Spawns Rust for Ward's linkage
 * - Parses linkage tree binary for re-cuts
 * - Computes TF-IDF auto-names from cached CLIP embeddings
 * - Generates contact sheets via Sharp
 */

import { join, dirname } from "node:path";
import { readFileSync, existsSync, mkdirSync } from "node:fs";
import { log } from "./log.ts";
import sharp from "sharp";

// ── Paths ────────────────────────────────────────────────────────────────────

const SCRIPTS_DIR = join(dirname(import.meta.dir), "scripts");
const RUST_BINARY = join(dirname(import.meta.dir), "rust", "cluster-tool", "target", "release", "cluster-tool");
const PYTHON = process.env.CLUSTER_PYTHON || "/tmp/imgcluster-env/bin/python3";

function cacheDir(targetDir: string) {
  return join(targetDir, ".reorder-cache");
}

// ── Types ────────────────────────────────────────────────────────────────────

export interface ClusterResult {
  id: string;
  autoName: string;
  autoTags: { term: string; z: number }[];
  images: string[];
  confirmedGroup: { id: string; name: string; images: string[] } | null;
}

export interface ClusterData {
  clusters: ClusterResult[];
  suggestedCounts: number[];
  nClusters: number;
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
  steps: { clusterA: number; clusterB: number; distance: number; newSize: number }[];
}

interface TextEmbeddingsRaw {
  terms: string[];
  embeddings: number[][];  // [n_terms][512]
}

interface TextEmbeddings {
  terms: string[];
  flat: Float32Array;  // flattened [n_terms * dim]
  dim: number;
}

// ── Feature extraction (Python) ──────────────────────────────────────────────

export async function extractFeatures(
  targetDir: string,
  onProgress?: (line: string) => void,
): Promise<{ total: number; cached: number; extracted: number }> {
  const script = join(SCRIPTS_DIR, "extract_features.py");
  const cache = cacheDir(targetDir);

  if (!existsSync(PYTHON)) {
    throw new Error(`Python not found at ${PYTHON}. Create venv with: uv venv /tmp/imgcluster-env && source /tmp/imgcluster-env/bin/activate && uv pip install torch torchvision open-clip-torch pillow numpy`);
  }

  log("cluster", `Extracting features: ${PYTHON} ${script} ${targetDir}`);
  const proc = Bun.spawn([PYTHON, script, targetDir, "--cache-dir", cache], {
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

async function ensureTextEmbeddings(targetDir: string): Promise<string> {
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
): Promise<RustOutput> {
  const cache = cacheDir(targetDir);
  const embeddings = join(cache, "clip_embeddings.npz");
  const groupsFile = join(targetDir, ".reorder-groups.json");
  const treePath = join(cache, "linkage_tree.bin");

  const args = [
    "--embeddings", embeddings,
    "--n-clusters", String(nClusters),
    "--output-tree", treePath,
  ];
  if (existsSync(groupsFile)) {
    args.push("--groups", groupsFile);
  }

  if (!existsSync(RUST_BINARY)) {
    throw new Error(`Rust binary not found at ${RUST_BINARY}. Build with: cd rust/cluster-tool && cargo build --release`);
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
  const nImages = view.getUint32(offset, true); offset += 4;
  const nPreMerges = view.getUint32(offset, true); offset += 4;
  const nSteps = view.getUint32(offset, true); offset += 4;

  const steps = [];
  for (let i = 0; i < nSteps; i++) {
    const clusterA = view.getUint32(offset, true); offset += 4;
    const clusterB = view.getUint32(offset, true); offset += 4;
    const distance = view.getFloat32(offset, true); offset += 4;
    const newSize = view.getUint32(offset, true); offset += 4;
    steps.push({ clusterA, clusterB, distance, newSize });
  }

  return { nImages, nPreMerges, steps };
}

// ── Re-cut (Bun-side, instant) ───────────────────────────────────────────────

export function recutTree(
  targetDir: string,
  nClusters: number,
): { labels: number[]; nInitialAfterPremerge: number } {
  const treePath = join(cacheDir(targetDir), "linkage_tree.bin");
  if (!existsSync(treePath)) {
    throw new Error("No linkage tree found. Run full clustering first.");
  }

  const tree = parseLinkageTree(treePath);
  const { nImages, nPreMerges, steps } = tree;
  const nAfterPremerge = nImages - nPreMerges;

  const parent = new Int32Array(nImages);
  for (let i = 0; i < nImages; i++) parent[i] = i;

  function find(x: number): number {
    // Path-halving union-find; all indices are in [0, nImages)
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
    const s = steps[i]!; // bounded by parseLinkageTree's nSteps
    const ra = find(s.clusterA);
    const rb = find(s.clusterB);
    if (ra !== rb) parent[ra] = rb;
  }

  // Main steps are already sorted by distance (Rust sorts before saving)
  const mainMergesNeeded = Math.max(0, nAfterPremerge - nClusters);
  for (let i = 0; i < mainMergesNeeded; i++) {
    const s = steps[nPreMerges + i]!;
    const ra = find(s.clusterA);
    const rb = find(s.clusterB);
    if (ra !== rb) parent[ra] = rb;
  }

  const roots = new Int32Array(nImages);
  for (let i = 0; i < nImages; i++) roots[i] = find(i);

  const remap = new Map<number, number>();
  let nextLabel = 0;
  const labels = new Array<number>(nImages);
  for (let i = 0; i < nImages; i++) {
    const r = roots[i]!; // typed array, bounded
    if (!remap.has(r)) remap.set(r, nextLabel++);
    labels[i] = remap.get(r)!; // just set above
  }

  return { labels, nInitialAfterPremerge: nAfterPremerge };
}

// ── Auto-naming (TF-IDF) ────────────────────────────────────────────────────

let _textEmbCache: TextEmbeddings | null = null;
let _clipEmbCache: { filenames: string[]; clip: Float32Array; nImages: number; dim: number } | null = null;
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
        data = Buffer.from(Bun.inflateSync(npzBuf.subarray(dataStart, dataStart + compSize) as Uint8Array<ArrayBuffer>));
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
): ClusterResult[] {
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
      if (!parts.some(p => p.toLowerCase().startsWith(short) || short.startsWith(p.toLowerCase().split("_")[0]!))) {
        parts.push(term.split(" ").slice(0, 2).join("_"));
      }
    }
    const autoName = c.confirmed_group
      ? c.confirmed_group.name
      : parts.join(" · ");

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
      const idx = Math.floor(i * filenames.length / maxImages);
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
    })
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
  return [...new Set([
    Math.max(10, Math.floor(nImages / 100)),
    Math.max(20, Math.floor(nImages / 50)),
    Math.max(50, Math.floor(nImages / 30)),
    Math.max(75, Math.floor(nImages / 20)),
    100, 150, 200, 300,
  ])].sort((a, b) => a - b);
}

// ── Full pipeline ────────────────────────────────────────────────────────────

export async function runFullCluster(
  targetDir: string,
  nClusters: number,
  onProgress?: (line: string) => void,
): Promise<ClusterData> {
  const [extraction] = await Promise.all([
    extractFeatures(targetDir, onProgress),
    ensureTextEmbeddings(targetDir),
  ]);
  log("cluster", `Extraction: ${extraction.extracted} new, ${extraction.cached} cached`);

  const rustOutput = await runLinkage(targetDir, nClusters);
  log("cluster", `Linkage complete: ${rustOutput.clusters.length} clusters`);

  const clusters = computeAutoNames(targetDir, rustOutput.clusters);

  const nImages = clusters.reduce((n, c) => n + c.images.length, 0);
  return { clusters, suggestedCounts: computeSuggestedCounts(nImages), nClusters };
}

export async function runRecut(
  targetDir: string,
  nClusters: number,
): Promise<ClusterData> {
  const { labels } = recutTree(targetDir, nClusters);
  const { filenames } = loadClipEmbeddings(targetDir);

  // Load groups for confirmed_group info
  const groupsPath = join(targetDir, ".reorder-groups.json");
  let groups: { id: string; name: string; images: string[] }[] = [];
  try { groups = await Bun.file(groupsPath).json(); } catch {}
  const imgToGroup = new Map<string, typeof groups[0]>();
  for (const g of groups) {
    for (const f of g.images) imgToGroup.set(f, g);
  }

  // Group by label
  const clusterMembers = new Map<number, string[]>();
  for (let i = 0; i < labels.length; i++) {
    const label = labels[i]!; // bounded by recutTree output
    if (!clusterMembers.has(label)) clusterMembers.set(label, []);
    clusterMembers.get(label)!.push(filenames[i]!); // same length as labels
  }

  const rawClusters = [...clusterMembers.entries()]
    .sort((a, b) => b[1].length - a[1].length)
    .map(([label, images], ci) => {
      const confirmed = images.find(f => imgToGroup.has(f));
      const group = confirmed ? imgToGroup.get(confirmed) ?? null : null;
      return {
        id: `cluster_${ci}`,
        images: images.sort(),
        confirmed_group: group ? { id: group.id, name: group.name, images: group.images } : null,
      };
    });

  const clusters = computeAutoNames(targetDir, rawClusters);

  return { clusters, suggestedCounts: computeSuggestedCounts(filenames.length), nClusters };
}

// ── Cache invalidation ───────────────────────────────────────────────────────

export function invalidateClusterCache() {
  _textEmbCache = null;
  _clipEmbCache = null;
  _tfidfStatsCache = null;
}

// ── Job guard ───────────────────────────────────────────────────────────────

let _clusterJobRunning = false;
export function isClusterJobRunning() { return _clusterJobRunning; }
export function setClusterJobRunning(v: boolean) { _clusterJobRunning = v; }
