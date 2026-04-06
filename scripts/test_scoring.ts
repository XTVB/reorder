/**
 * Fast scoring iteration script — loads raw Rust output and tests different
 * scoring formulas without re-running the expensive Rust computation.
 *
 * Also tests individual embedding types to find which ones actually discriminate.
 *
 * Usage: bun scripts/test_scoring.ts /path/to/target/dir
 */
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { loadHashMapping, parseNpyFromNpz, reindexToFilenameOrder } from "../src/cache-utils.ts";

const targetDir = process.argv[2]!;
const cacheDirPath = join(targetDir, ".reorder-cache");
const npzPath = join(cacheDirPath, "clip_hash_cache.npz");
const groupsPath = join(targetDir, ".reorder-groups.json");

// ── Load data ────────────────────────────────────────────────────────────────
const mapping = loadHashMapping(cacheDirPath);
const { contentHashes, filenames, nImages } = mapping;
const fnToIdx = new Map(filenames.map((f, i) => [f, i]));
const groups: { id: string; name: string; images: string[] }[] = JSON.parse(readFileSync(groupsPath, "utf-8"));
const groupMap = new Map(groups.map(g => [g.id, g]));
const npzBuf = readFileSync(npzPath);

// Load per-type embeddings
const embTypes: Record<string, { data: Float32Array; dim: number }> = {};
for (const key of ["clip", "dino", "pecore_l", "pecore_g", "color"]) {
  try {
    const hashArr = parseNpyFromNpz(npzBuf, `${key}.npy`);
    const dim = hashArr.length / mapping.hashOrder.length;
    const arr = reindexToFilenameOrder(hashArr, dim, mapping);
    embTypes[key] = { data: arr, dim };
  } catch {}
}

function cosineDistSingle(emb: { data: Float32Array; dim: number }, i: number, j: number): number {
  const { data, dim } = emb;
  let dot = 0, ni = 0, nj = 0;
  const oi = i * dim, oj = j * dim;
  for (let d = 0; d < dim; d++) {
    const a = data[oi + d]!, b = data[oj + d]!;
    dot += a * b; ni += a * a; nj += b * b;
  }
  const denom = Math.sqrt(ni) * Math.sqrt(nj);
  return denom < 1e-20 ? 1.0 : Math.max(0, 1 - dot / denom);
}

// ── Test pairs ───────────────────────────────────────────────────────────────
const testPairs = [
  { cat: "SAME", a: "1e2cee94-9dc6-4f2e-94f9-9146950d109b", b: "8eac2757-b071-44a6-a91d-b772fddde592", name: "PinkBikini x PinkBikini2" },
  { cat: "SAME", a: "a6d1c65c-cbea-4105-af74-e0b9f9b95e51", b: "02e70989-426d-4805-b409-6215616f2a34", name: "WhiteSnow x WinterWhite2" },
  { cat: "SAME", a: "01a12d5a-0e1f-40b3-8935-30e8d5d8c102", b: "2f66645e-504b-42c9-8b39-3e4e779105c7", name: "StrawRobe2 x StrawRobeStrip" },
  { cat: "SAME", a: "d0cbe1d7-ea6d-4678-95d2-324b1c11684d", b: "ac74c692-ecb6-42e7-82c2-e8ecc1be7522", name: "PinkChains x Makima" },
  { cat: "SIM", a: "de990bc6-d3a7-4f12-b522-075263401ce3", b: "2a679639-574d-4c46-b5ca-1aed37de7fdb", name: "WhiteBody x TransWhite" },
  { cat: "SIM", a: "11298375-d7ed-490d-a681-29053f560a85", b: "3063db5d-737e-406e-9f54-8a3962925e41", name: "OrangeBikini x RedBikini" },
  { cat: "SIM", a: "7b065238-5677-46b9-a9cc-adbf97fbb542", b: "5944be9e-d18e-4179-9b35-10abc8ed1661", name: "PalePink x WhitePoolside" },
  { cat: "SIM", a: "69e87ffe-8124-42ca-bc48-50a81c263c4e", b: "d0cbe1d7-ea6d-4678-95d2-324b1c11684d", name: "DevilPrep x PinkChains" },
  { cat: "NEVR", a: "01803486-5fe0-4ea2-8b6e-a22fabf13653", b: "40e5c93e-0490-4767-9c3f-29caaccce340", name: "ValentG x SnowElves" },
  { cat: "NEVR", a: "e1853e6f-9458-416b-83f7-02f5bf9b0d29", b: "547c6d66-243d-43dc-9ec3-7d416f96a916", name: "NeonLatex x SexyShower" },
  { cat: "NEVR", a: "d0cbe1d7-ea6d-4678-95d2-324b1c11684d", b: "dbf6d6a2-a543-4e6d-a4ca-a3022f79351e", name: "PinkChains x BlackDevil" },
  { cat: "NEVR", a: "70928dcf-5f8b-4396-bb8a-e5b2c4b9f9f0", b: "233e9056-030d-4d38-b018-11ef40038979", name: "Venom x NaturalWaves" },
];

// ── Approach: per-image "best match" with multiplied distances ───────────
// For each image in group A, find its best match in group B using
// SEPARATE distance metrics per embedding type. Then require that
// the best match is close in BOTH color AND semantic.

type PerImageMatch = {
  colorDist: number;
  semDist: number;
  combined: number;
};

function computePerImageMatches(
  gA: string[], gB: string[],
  colorKey: string, semKey: string,
): PerImageMatch[] {
  const colorEmb = embTypes[colorKey]!;
  const semEmb = embTypes[semKey]!;
  const idxA = gA.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);
  const idxB = gB.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);

  const matches: PerImageMatch[] = [];
  for (const ia of idxA) {
    let bestColor = Infinity, bestSem = Infinity, bestCombined = Infinity;
    for (const ib of idxB) {
      const cd = cosineDistSingle(colorEmb, ia, ib);
      const sd = cosineDistSingle(semEmb, ia, ib);
      // Multiplicative: requires BOTH to be low
      const combined = cd * sd;
      if (combined < bestCombined) {
        bestCombined = combined;
        bestColor = cd;
        bestSem = sd;
      }
    }
    matches.push({ colorDist: bestColor, semDist: bestSem, combined: bestCombined });
  }
  // Also B→A
  for (const ib of idxB) {
    let bestColor = Infinity, bestSem = Infinity, bestCombined = Infinity;
    for (const ia of idxA) {
      const cd = cosineDistSingle(colorEmb, ib, ia);
      const sd = cosineDistSingle(semEmb, ib, ia);
      const combined = cd * sd;
      if (combined < bestCombined) {
        bestCombined = combined;
        bestColor = cd;
        bestSem = sd;
      }
    }
    matches.push({ colorDist: bestColor, semDist: bestSem, combined: bestCombined });
  }
  return matches;
}

// Test multiplicative approach with different embedding combos
const combos: [string, string][] = [
  ["color", "pecore_g"],
  ["color", "clip"],
  ["color", "dino"],
  ["color", "pecore_l"],
];

for (const [colorKey, semKey] of combos) {
  if (!embTypes[colorKey] || !embTypes[semKey]) continue;
  console.log(`\n=== Multiplicative: ${colorKey} × ${semKey} ===`);
  console.log("cat  | pair                      | medCombined | medColor | medSem | p25Combined");
  console.log("-----|---------------------------|-------------|----------|--------|------------");

  for (const tp of testPairs) {
    const gA = groupMap.get(tp.a)!;
    const gB = groupMap.get(tp.b)!;
    const matches = computePerImageMatches(gA.images, gB.images, colorKey, semKey);

    const combineds = matches.map(m => m.combined).sort((a, b) => a - b);
    const colors = matches.map(m => m.colorDist).sort((a, b) => a - b);
    const sems = matches.map(m => m.semDist).sort((a, b) => a - b);

    const medCombined = combineds[Math.floor(combineds.length / 2)]!;
    const medColor = colors[Math.floor(colors.length / 2)]!;
    const medSem = sems[Math.floor(sems.length / 2)]!;
    const p25Combined = combineds[Math.floor(combineds.length / 4)]!;

    console.log(
      tp.cat.padEnd(4) + " | " + tp.name.padEnd(25) + " | " +
      medCombined.toFixed(5).padStart(11) + " | " +
      medColor.toFixed(4).padStart(8) + " | " +
      medSem.toFixed(4).padStart(6) + " | " +
      p25Combined.toFixed(5).padStart(11)
    );
  }
}

// ── New idea: Hausdorff-like with color gate ─────────────────────────────
// For each image in A, find the CLOSEST match in B by semantic distance
// BUT only count it if the color distance is also below a threshold.
// Score = fraction of images in A that have a "color-gated close match" in B.
console.log("\n=== Color-gated semantic matching ===");
console.log("For each image, find semantic NN in other group, but only count if color dist < threshold");

for (const colorThresh of [0.05, 0.08, 0.10, 0.15, 0.20]) {
  console.log(`\n  Color threshold: ${colorThresh}`);
  console.log("  cat  | pair                      | gatedFracAB | gatedFracBA | avg");
  console.log("  -----|---------------------------|-------------|-------------|-----");

  for (const tp of testPairs) {
    const gA = groupMap.get(tp.a)!;
    const gB = groupMap.get(tp.b)!;
    const colorEmb = embTypes["color"]!;
    const semEmb = embTypes["pecore_g"]!;

    const idxA = gA.images.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);
    const idxB = gB.images.map(f => fnToIdx.get(f)).filter((i): i is number => i !== undefined);

    // A→B: for each in A, find best semantic match in B where color < threshold
    let gatedA = 0;
    for (const ia of idxA) {
      let hasClosed = false;
      for (const ib of idxB) {
        const cd = cosineDistSingle(colorEmb, ia, ib);
        if (cd < colorThresh) {
          hasClosed = true;
          break;
        }
      }
      if (hasClosed) gatedA++;
    }

    let gatedB = 0;
    for (const ib of idxB) {
      let hasClosed = false;
      for (const ia of idxA) {
        const cd = cosineDistSingle(colorEmb, ib, ia);
        if (cd < colorThresh) {
          hasClosed = true;
          break;
        }
      }
      if (hasClosed) gatedB++;
    }

    const fracAB = idxA.length > 0 ? gatedA / idxA.length : 0;
    const fracBA = idxB.length > 0 ? gatedB / idxB.length : 0;
    const avg = (fracAB + fracBA) / 2;

    console.log(
      "  " + tp.cat.padEnd(4) + " | " + tp.name.padEnd(25) + " | " +
      fracAB.toFixed(3).padStart(11) + " | " +
      fracBA.toFixed(3).padStart(11) + " | " +
      avg.toFixed(3).padStart(5)
    );
  }
}
