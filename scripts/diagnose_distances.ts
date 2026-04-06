/**
 * Diagnostic script: compute pairwise distances between test group pairs
 * to understand what metrics separate "same set" from "never match".
 *
 * Usage: bun scripts/diagnose_distances.ts path
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { loadHashMapping, parseNpyFromNpz, reindexToFilenameOrder } from "../src/cache-utils.ts";

const targetDir = process.argv[2]!;
const cacheDirPath = join(targetDir, ".reorder-cache");
const npzPath = join(cacheDirPath, "clip_hash_cache.npz");
const groupsPath = join(targetDir, ".reorder-groups.json");

// ── Load data ────────────────────────────────────────────────────────────────
console.log("Loading data...");
const mapping = loadHashMapping(cacheDirPath);
const { contentHashes, filenames, nImages } = mapping;
const fnToIdx = new Map(filenames.map((f, i) => [f, i]));
const groups: { id: string; name: string; images: string[] }[] = JSON.parse(readFileSync(groupsPath, "utf-8"));
const groupMap = new Map(groups.map(g => [g.id, g]));

const npzBuf = readFileSync(npzPath);

// Load all available embeddings
const embTypes: { key: string; dim: number; data: Float32Array; needsNorm: boolean }[] = [];
for (const key of ["clip", "dino", "pecore_l", "pecore_g", "color"]) {
  try {
    const hashArr = parseNpyFromNpz(npzBuf, `${key}.npy`);
    const dim = hashArr.length / mapping.hashOrder.length;
    const arr = reindexToFilenameOrder(hashArr, dim, mapping);
    embTypes.push({ key, dim, data: arr, needsNorm: key === "color" });
    console.log(`  ${key}: ${dim}d`);
  } catch {
    console.log(`  ${key}: not found`);
  }
}

// ── Distance functions ───────────────────────────────────────────────────────

function cosineDistRaw(data: Float32Array, dim: number, i: number, j: number): number {
  let dot = 0, normI = 0, normJ = 0;
  const oi = i * dim, oj = j * dim;
  for (let d = 0; d < dim; d++) {
    const a = data[oi + d]!, b = data[oj + d]!;
    dot += a * b;
    normI += a * a;
    normJ += b * b;
  }
  const denom = Math.sqrt(normI) * Math.sqrt(normJ);
  if (denom < 1e-20) return 1.0;
  return Math.max(0, 1 - dot / denom);
}

// Build combined features for a given weight config
function buildCombined(weights: Record<string, number>) {
  const active = embTypes.filter(e => (weights[e.key] ?? 0) > 0);
  const combinedDim = active.reduce((s, e) => s + e.dim, 0);
  const features = new Float32Array(nImages * combinedDim);
  const norms = new Float64Array(nImages);

  for (let i = 0; i < nImages; i++) {
    let offset = 0;
    for (const { data, dim, needsNorm, key } of active) {
      const w = weights[key] ?? 1;
      const rowStart = i * dim;
      if (needsNorm) {
        let ns = 0;
        for (let d = 0; d < dim; d++) ns += data[rowStart + d]! ** 2;
        const n = Math.sqrt(ns) || 1e-10;
        for (let d = 0; d < dim; d++)
          features[i * combinedDim + offset + d] = (data[rowStart + d]! / n) * w;
      } else {
        for (let d = 0; d < dim; d++)
          features[i * combinedDim + offset + d] = data[rowStart + d]! * w;
      }
      offset += dim;
    }
    let s = 0;
    for (let d = 0; d < combinedDim; d++) s += features[i * combinedDim + d]! ** 2;
    norms[i] = Math.sqrt(s);
  }
  return { features, norms, dim: combinedDim };
}

function cosineDist(features: Float32Array, norms: Float64Array, dim: number, i: number, j: number): number {
  let dot = 0;
  const oi = i * dim, oj = j * dim;
  for (let d = 0; d < dim; d++) dot += features[oi + d]! * features[oj + d]!;
  const denom = norms[i]! * norms[j]!;
  if (denom < 1e-20) return 1.0;
  return Math.max(0, 1 - dot / denom);
}

// ── Group pair metrics ───────────────────────────────────────────────────────

interface GroupMetrics {
  minDist: number;
  avgDist: number;
  p10Dist: number; // 10th percentile distance
  p25Dist: number; // 25th percentile
  medianDist: number;
  closestPair: [string, string];
  // Fraction of images in A whose nearest neighbor is in B (and vice versa)
  nnFractionAB: number;
  nnFractionBA: number;
}

function computeGroupMetrics(
  features: Float32Array, norms: Float64Array, dim: number,
  groupA: { id: string; images: string[] },
  groupB: { id: string; images: string[] },
  allGroups: { id: string; images: string[] }[]
): GroupMetrics {
  const idxA = groupA.images.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);
  const idxB = groupB.images.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);

  // All pairwise distances
  const dists: number[] = [];
  let minDist = Infinity, bestA = 0, bestB = 0;
  for (const ia of idxA) {
    for (const ib of idxB) {
      const d = cosineDist(features, norms, dim, ia, ib);
      dists.push(d);
      if (d < minDist) { minDist = d; bestA = ia; bestB = ib; }
    }
  }
  dists.sort((a, b) => a - b);
  const avgDist = dists.reduce((s, d) => s + d, 0) / dists.length;
  const p10Dist = dists[Math.floor(dists.length * 0.1)] ?? minDist;
  const p25Dist = dists[Math.floor(dists.length * 0.25)] ?? minDist;
  const medianDist = dists[Math.floor(dists.length * 0.5)] ?? minDist;

  // NN fraction: for each image in A, is its global NN in B?
  // Collect all group image indices
  const allGroupIdx = new Map<number, string>();
  for (const g of allGroups) {
    for (const img of g.images) {
      const idx = fnToIdx.get(img);
      if (idx !== undefined) allGroupIdx.set(idx, g.id);
    }
  }

  let nnHitsAB = 0;
  for (const ia of idxA) {
    let bestD = Infinity, bestG = "";
    for (const [idx, gid] of allGroupIdx) {
      if (gid === groupA.id) continue;
      const d = cosineDist(features, norms, dim, ia, idx);
      if (d < bestD) { bestD = d; bestG = gid; }
    }
    if (bestG === groupB.id) nnHitsAB++;
  }

  let nnHitsBA = 0;
  for (const ib of idxB) {
    let bestD = Infinity, bestG = "";
    for (const [idx, gid] of allGroupIdx) {
      if (gid === groupB.id) continue;
      const d = cosineDist(features, norms, dim, ib, idx);
      if (d < bestD) { bestD = d; bestG = gid; }
    }
    if (bestG === groupA.id) nnHitsBA++;
  }

  return {
    minDist, avgDist, p10Dist, p25Dist, medianDist,
    closestPair: [filenames[bestA]!, filenames[bestB]!],
    nnFractionAB: idxA.length > 0 ? nnHitsAB / idxA.length : 0,
    nnFractionBA: idxB.length > 0 ? nnHitsBA / idxB.length : 0,
  };
}

// ── Test pairs ───────────────────────────────────────────────────────────────

const testPairs: { category: string; a: string; b: string }[] = [
  // Same set
  { category: "SAME", a: "d0cbe1d7-ea6d-4678-95d2-324b1c11684d", b: "ac74c692-ecb6-42e7-82c2-e8ecc1be7522" },
  { category: "SAME", a: "a6d1c65c-cbea-4105-af74-e0b9f9b95e51", b: "02e70989-426d-4805-b409-6215616f2a34" },
  { category: "SAME", a: "1e2cee94-9dc6-4f2e-94f9-9146950d109b", b: "8eac2757-b071-44a6-a91d-b772fddde592" },
  { category: "SAME", a: "01a12d5a-0e1f-40b3-8935-30e8d5d8c102", b: "2f66645e-504b-42c9-8b39-3e4e779105c7" },
  // Similar
  { category: "SIMILAR", a: "de990bc6-d3a7-4f12-b522-075263401ce3", b: "2a679639-574d-4c46-b5ca-1aed37de7fdb" },
  { category: "SIMILAR", a: "11298375-d7ed-490d-a681-29053f560a85", b: "3063db5d-737e-406e-9f54-8a3962925e41" },
  { category: "SIMILAR", a: "7b065238-5677-46b9-a9cc-adbf97fbb542", b: "5944be9e-d18e-4179-9b35-10abc8ed1661" },
  { category: "SIMILAR", a: "69e87ffe-8124-42ca-bc48-50a81c263c4e", b: "d0cbe1d7-ea6d-4678-95d2-324b1c11684d" },
  // Never
  { category: "NEVER", a: "01803486-5fe0-4ea2-8b6e-a22fabf13653", b: "40e5c93e-0490-4767-9c3f-29caaccce340" },
  { category: "NEVER", a: "e1853e6f-9458-416b-83f7-02f5bf9b0d29", b: "547c6d66-243d-43dc-9ec3-7d416f96a916" },
  { category: "NEVER", a: "d0cbe1d7-ea6d-4678-95d2-324b1c11684d", b: "dbf6d6a2-a543-4e6d-a4ca-a3022f79351e" },
  { category: "NEVER", a: "70928dcf-5f8b-4396-bb8a-e5b2c4b9f9f0", b: "233e9056-030d-4d38-b018-11ef40038979" },
];

// ── Run analysis ─────────────────────────────────────────────────────────────

const weightConfigs: { name: string; weights: Record<string, number> }[] = [
  { name: "PE-G+color", weights: { pecore_g: 1.0, color: 0.5 } },
  { name: "PE-G only", weights: { pecore_g: 1.0 } },
  { name: "color only", weights: { color: 1.0 } },
  { name: "CLIP+color", weights: { clip: 1.0, color: 0.5 } },
  { name: "PE-G+CLIP+color", weights: { pecore_g: 1.0, clip: 1.0, color: 0.5 } },
];

// Also compute per-embedding-type raw distances for each pair
console.log("\n=== Per-embedding-type min/avg distances ===\n");
console.log("Category | Pair | " + embTypes.map(e => `${e.key} min`).join(" | ") + " | " + embTypes.map(e => `${e.key} avg`).join(" | "));
console.log("---|---|" + embTypes.map(() => "---").join("|") + "|" + embTypes.map(() => "---").join("|"));

for (const tp of testPairs) {
  const gA = groupMap.get(tp.a)!;
  const gB = groupMap.get(tp.b)!;
  const idxA = gA.images.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);
  const idxB = gB.images.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);

  const mins: string[] = [];
  const avgs: string[] = [];
  for (const emb of embTypes) {
    let minD = Infinity, sumD = 0, count = 0;
    for (const ia of idxA) {
      for (const ib of idxB) {
        const d = cosineDistRaw(emb.data, emb.dim, ia, ib);
        if (d < minD) minD = d;
        sumD += d;
        count++;
      }
    }
    mins.push(minD.toFixed(4));
    avgs.push((sumD / count).toFixed(4));
  }
  console.log(`${tp.category} | ${gA.name} × ${gB.name} | ${mins.join(" | ")} | ${avgs.join(" | ")}`);
}

// ── Additional analysis: what fraction of cross-group pairs are below
//    various percentile thresholds of the INTRA-group distance distribution?
console.log("\n=== Overlap analysis: color-only ===");
console.log("For each pair, compute what fraction of cross-group image pairs");
console.log("have color distance below the median intra-group color distance.\n");

const colorEmb = embTypes.find(e => e.key === "color")!;

function intraGroupMedianDist(emb: { data: Float32Array; dim: number }, groupImages: string[]): number {
  const idxs = groupImages.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);
  if (idxs.length < 2) return 0;
  const dists: number[] = [];
  for (let i = 0; i < idxs.length; i++) {
    for (let j = i + 1; j < idxs.length; j++) {
      dists.push(cosineDistRaw(emb.data, emb.dim, idxs[i]!, idxs[j]!));
    }
  }
  dists.sort((a, b) => a - b);
  return dists[Math.floor(dists.length / 2)] ?? 0;
}

function crossGroupOverlapFraction(emb: { data: Float32Array; dim: number }, gA: string[], gB: string[], threshold: number): number {
  const idxA = gA.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);
  const idxB = gB.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);
  let below = 0, total = 0;
  for (const ia of idxA) {
    for (const ib of idxB) {
      if (cosineDistRaw(emb.data, emb.dim, ia, ib) < threshold) below++;
      total++;
    }
  }
  return total > 0 ? below / total : 0;
}

console.log("Category | Pair | intraA | intraB | crossMin | crossAvg | overlap@medA | overlap@medB | overlap@0.05 | overlap@0.10");
console.log("---|---|---|---|---|---|---|---|---|---");

for (const tp of testPairs) {
  const gA = groupMap.get(tp.a)!;
  const gB = groupMap.get(tp.b)!;
  const intraA = intraGroupMedianDist(colorEmb, gA.images);
  const intraB = intraGroupMedianDist(colorEmb, gB.images);

  const idxA = gA.images.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);
  const idxB = gB.images.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);
  let minD = Infinity, sumD = 0, count = 0;
  for (const ia of idxA) {
    for (const ib of idxB) {
      const d = cosineDistRaw(colorEmb.data, colorEmb.dim, ia, ib);
      if (d < minD) minD = d;
      sumD += d;
      count++;
    }
  }

  const overlapA = crossGroupOverlapFraction(colorEmb, gA.images, gB.images, intraA);
  const overlapB = crossGroupOverlapFraction(colorEmb, gA.images, gB.images, intraB);
  const overlap005 = crossGroupOverlapFraction(colorEmb, gA.images, gB.images, 0.05);
  const overlap010 = crossGroupOverlapFraction(colorEmb, gA.images, gB.images, 0.10);

  console.log(
    `${tp.category} | ${gA.name.slice(0,20)} × ${gB.name.slice(0,20)} | ${intraA.toFixed(4)} | ${intraB.toFixed(4)} | ${minD.toFixed(4)} | ${(sumD/count).toFixed(4)} | ${overlapA.toFixed(2)} | ${overlapB.toFixed(2)} | ${overlap005.toFixed(2)} | ${overlap010.toFixed(2)}`
  );
}

// ── Key insight test: use a COMPOSITE score ──────────────────────────────────
// Idea: combine color overlap + semantic distance in a way that requires BOTH
// to be similar (not just one). Same-set groups share wardrobe+setting (color)
// AND similar poses/framing (semantic). Different-set groups of the same model
// may match on semantic but not color, or color but not semantic.

console.log("\n=== Composite scores (experimental) ===");
console.log("Score = (1 - colorAvg) * (1 - semanticAvg) — requires both to be close\n");

for (const embKey of ["pecore_g", "clip"] as const) {
  const semEmb = embTypes.find(e => e.key === embKey)!;
  if (!semEmb) continue;

  console.log(`\nUsing ${embKey} for semantic component:`);
  console.log("Category | Pair | colorAvg | semAvg | composite | colorMin | semMin");
  console.log("---|---|---|---|---|---|---");

  for (const tp of testPairs) {
    const gA = groupMap.get(tp.a)!;
    const gB = groupMap.get(tp.b)!;
    const idxA = gA.images.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);
    const idxB = gB.images.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);

    let colorSum = 0, semSum = 0, colorMin = Infinity, semMin = Infinity, count = 0;
    for (const ia of idxA) {
      for (const ib of idxB) {
        const cd = cosineDistRaw(colorEmb.data, colorEmb.dim, ia, ib);
        const sd = cosineDistRaw(semEmb.data, semEmb.dim, ia, ib);
        colorSum += cd;
        semSum += sd;
        if (cd < colorMin) colorMin = cd;
        if (sd < semMin) semMin = sd;
        count++;
      }
    }
    const colorAvg = colorSum / count;
    const semAvg = semSum / count;
    const composite = (1 - colorAvg) * (1 - semAvg);

    console.log(
      `${tp.category} | ${gA.name.slice(0,25)} × ${gB.name.slice(0,25)} | ${colorAvg.toFixed(4)} | ${semAvg.toFixed(4)} | ${composite.toFixed(4)} | ${colorMin.toFixed(4)} | ${semMin.toFixed(4)}`
    );
  }
}

// Full metrics for default config
console.log("\n=== Full metrics (PE-G + color) ===\n");
const { features, norms, dim } = buildCombined({ pecore_g: 1.0, color: 0.5 });

console.log("Category | Pair | sizes | min | p10 | p25 | median | avg | nnAB | nnBA");
console.log("---|---|---|---|---|---|---|---|---|---");

for (const tp of testPairs) {
  const gA = groupMap.get(tp.a)!;
  const gB = groupMap.get(tp.b)!;
  const m = computeGroupMetrics(features, norms, dim, gA, gB, groups);
  console.log(
    `${tp.category} | ${gA.name} × ${gB.name} | ${gA.images.length}×${gB.images.length} | ` +
    `${m.minDist.toFixed(4)} | ${m.p10Dist.toFixed(4)} | ${m.p25Dist.toFixed(4)} | ${m.medianDist.toFixed(4)} | ${m.avgDist.toFixed(4)} | ` +
    `${m.nnFractionAB.toFixed(2)} | ${m.nnFractionBA.toFixed(2)}`
  );
}
