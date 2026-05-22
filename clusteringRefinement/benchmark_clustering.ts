/**
 * Clustering benchmark — compute ARI / NMI of the current Rust linkage pipeline
 * (or a custom precomputed distance matrix) against `.reorder-groups.json` labels.
 *
 *   bun scripts/benchmark_clustering.ts <targetDir>
 *     [--n <int>]                      cluster count to cut at (default = #groups in labels)
 *     [--weights k=v,k=v]              embedding blend weights (default: pecore_g=1.0,color=0.8)
 *     [--dist-matrix <path>]           override embedding distances with a precomputed matrix
 *                                      (binary: u64 LE n_images followed by n*(n-1)/2 f64 LE distances)
 *     [--dist-matrix-weight <0..1>]    blend weight for dist matrix (default 1.0 = matrix only)
 *
 * Reuses the existing rust cluster-tool binary so the benchmark exercises the
 * same code path the production pipeline uses. Writes its linkage tree to
 * `.reorder-cache/linkage_tree.benchmark.bin` so it doesn't trample the user's
 * working tree.
 */

import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { ensureHashOrderJson } from "../src/cache-utils.ts";
import { RUST_BINARY } from "../src/cluster/binaries.ts";
import { loadGroups } from "../src/fs/groups.ts";
import { cacheDir, contentHashesPath, groupsPath, HASH_CACHE_FILE } from "../src/fs/paths.ts";

interface Args {
  targetDir: string;
  n?: number;
  weights: Record<string, number>;
  distMatrix?: string;
  distMatrixWeight: number;
  evalSubset?: string;
}

function parseArgs(argv: string[]): Args {
  const out: Args = {
    targetDir: "",
    weights: { pecore_g: 1.0, color: 0.8 },
    distMatrixWeight: 1.0,
  };
  let i = 2;
  if (!argv[i] || argv[i]!.startsWith("--")) {
    throw new Error("usage: bun scripts/benchmark_clustering.ts <targetDir> [flags]");
  }
  out.targetDir = argv[i++]!;
  while (i < argv.length) {
    const a = argv[i++]!;
    if (a === "--n") out.n = Number.parseInt(argv[i++]!, 10);
    else if (a === "--weights") {
      out.weights = {};
      for (const part of argv[i++]!.split(",")) {
        const [k, v] = part.split("=");
        if (!k || v === undefined) throw new Error(`bad --weights entry: ${part}`);
        out.weights[k] = Number.parseFloat(v);
      }
    } else if (a === "--dist-matrix") out.distMatrix = argv[i++]!;
    else if (a === "--dist-matrix-weight") out.distMatrixWeight = Number.parseFloat(argv[i++]!);
    else if (a === "--eval-subset") out.evalSubset = argv[i++]!;
    else throw new Error(`unknown arg: ${a}`);
  }
  return out;
}

/** True labels indexed by filename. Returns null for filenames not assigned to any group. */
function loadTrueLabels(targetDir: string): Map<string, number> {
  const groups = loadGroups(targetDir);
  if (groups.length === 0) throw new Error(`no groups in ${groupsPath(targetDir)}`);
  const m = new Map<string, number>();
  groups.forEach((g, gi) => {
    for (const fn of g.images) m.set(fn, gi);
  });
  return m;
}

/** Run cluster-tool, return its parsed JSON output. */
function runRustLinkage(args: Args): Promise<{
  clusters: { id: string; images: string[] }[];
  nClusters: number;
  treePath: string;
}> {
  const cache = cacheDir(args.targetDir);
  ensureHashOrderJson(cache);
  const treePath = join(cache, "linkage_tree.benchmark.bin");
  const groupsFile = groupsPath(args.targetDir);
  // Estimate cluster count if not provided
  const nClusters = args.n ?? loadGroups(args.targetDir).length;

  const argv = [
    "--hash-cache",
    join(cache, HASH_CACHE_FILE),
    "--content-hashes",
    contentHashesPath(args.targetDir),
    "--hash-order",
    join(cache, "hash_cache_order.json"),
    "--n-clusters",
    String(nClusters),
    "--output-tree",
    treePath,
    "--linkage",
    "ward",
  ];
  // NOTE: passing --groups would pre-seed the ground-truth groups as confirmed
  // clusters during NNC, which trivially inflates ARI. We deliberately omit it.
  if (args.distMatrix) {
    argv.push("--dist-matrix", args.distMatrix);
    argv.push("--dist-matrix-weight", String(args.distMatrixWeight));
  }
  for (const [k, v] of Object.entries(args.weights)) {
    if (v > 0) argv.push(`--${k.replace(/_/g, "-")}-weight`, String(v));
  }

  if (!existsSync(RUST_BINARY)) {
    throw new Error(`Rust binary not found at ${RUST_BINARY}. Build cluster-tool first.`);
  }

  return new Promise((resolve, reject) => {
    const child = spawn(RUST_BINARY, argv, { stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    let err = "";
    child.stdout.on("data", (b) => {
      out += b.toString();
    });
    child.stderr.on("data", (b) => {
      err += b.toString();
      process.stderr.write(b); // stream progress
    });
    child.on("close", (code) => {
      if (code !== 0) return reject(new Error(`cluster-tool exited ${code}: ${err}`));
      try {
        resolve(JSON.parse(out));
      } catch (e) {
        reject(new Error(`failed to parse cluster-tool output: ${e}`));
      }
    });
  });
}

// ── Metrics ──────────────────────────────────────────────────────────────────

function comb2(n: number): number {
  return (n * (n - 1)) / 2;
}

/** Adjusted Rand Index. Inputs are int-label arrays of equal length. */
function adjustedRandIndex(trueLab: number[], predLab: number[]): number {
  const n = trueLab.length;
  if (n === 0) return 0;
  // contingency table
  const ct = new Map<number, Map<number, number>>();
  const rowTot = new Map<number, number>();
  const colTot = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const t = trueLab[i]!,
      p = predLab[i]!;
    let row = ct.get(t);
    if (!row) {
      row = new Map();
      ct.set(t, row);
    }
    row.set(p, (row.get(p) ?? 0) + 1);
    rowTot.set(t, (rowTot.get(t) ?? 0) + 1);
    colTot.set(p, (colTot.get(p) ?? 0) + 1);
  }
  let sumNij = 0;
  for (const row of ct.values()) for (const v of row.values()) sumNij += comb2(v);
  let sumA = 0;
  for (const v of rowTot.values()) sumA += comb2(v);
  let sumB = 0;
  for (const v of colTot.values()) sumB += comb2(v);
  const totalPairs = comb2(n);
  const expected = (sumA * sumB) / totalPairs;
  const max = 0.5 * (sumA + sumB);
  return max === expected ? 0 : (sumNij - expected) / (max - expected);
}

/** Entropy of a label distribution. */
function entropy(labels: number[]): number {
  const counts = new Map<number, number>();
  for (const l of labels) counts.set(l, (counts.get(l) ?? 0) + 1);
  const n = labels.length;
  let h = 0;
  for (const c of counts.values()) {
    const p = c / n;
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

/** Mutual information between two labelings. */
function mutualInformation(trueLab: number[], predLab: number[]): number {
  const n = trueLab.length;
  const ct = new Map<string, number>();
  const a = new Map<number, number>();
  const b = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const key = `${trueLab[i]},${predLab[i]}`;
    ct.set(key, (ct.get(key) ?? 0) + 1);
    a.set(trueLab[i]!, (a.get(trueLab[i]!) ?? 0) + 1);
    b.set(predLab[i]!, (b.get(predLab[i]!) ?? 0) + 1);
  }
  let mi = 0;
  for (const [key, nij] of ct) {
    const [ti, pj] = key.split(",").map(Number) as [number, number];
    const ai = a.get(ti)!;
    const bj = b.get(pj)!;
    mi += (nij / n) * Math.log((n * nij) / (ai * bj));
  }
  return mi;
}

/** Normalized Mutual Information (arithmetic mean of entropies). */
function normalizedMutualInformation(trueLab: number[], predLab: number[]): number {
  const mi = mutualInformation(trueLab, predLab);
  const ht = entropy(trueLab);
  const hp = entropy(predLab);
  const denom = 0.5 * (ht + hp);
  return denom === 0 ? 0 : mi / denom;
}

/** Pair classification accuracy: among all unordered pairs, fraction agreeing on same/diff. */
function pairAccuracy(trueLab: number[], predLab: number[]): { acc: number; tp: number; tn: number; fp: number; fn: number } {
  const n = trueLab.length;
  // Use cluster-size sums to compute in O(K) instead of O(n^2)
  const trueSizes = new Map<number, number>();
  const predSizes = new Map<number, number>();
  const both = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    trueSizes.set(trueLab[i]!, (trueSizes.get(trueLab[i]!) ?? 0) + 1);
    predSizes.set(predLab[i]!, (predSizes.get(predLab[i]!) ?? 0) + 1);
    const k = `${trueLab[i]},${predLab[i]}`;
    both.set(k, (both.get(k) ?? 0) + 1);
  }
  const totalPairs = comb2(n);
  let trueSamePairs = 0;
  for (const v of trueSizes.values()) trueSamePairs += comb2(v);
  let predSamePairs = 0;
  for (const v of predSizes.values()) predSamePairs += comb2(v);
  let tp = 0; // same-true AND same-pred
  for (const v of both.values()) tp += comb2(v);
  const fn = trueSamePairs - tp;
  const fp = predSamePairs - tp;
  const tn = totalPairs - tp - fn - fp;
  return { acc: (tp + tn) / totalPairs, tp, tn, fp, fn };
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const args = parseArgs(process.argv);
  console.log(`Benchmark: ${args.targetDir}`);
  console.log(
    `  weights=${JSON.stringify(args.weights)} dist_matrix=${args.distMatrix ?? "(none)"} dmw=${args.distMatrixWeight}`,
  );

  const fnToTrue = loadTrueLabels(args.targetDir);
  console.log(`  ground truth: ${new Set(fnToTrue.values()).size} groups, ${fnToTrue.size} image refs`);

  const result = await runRustLinkage(args);

  // Build pred label per filename
  const fnToPred = new Map<string, number>();
  result.clusters.forEach((c, ci) => {
    for (const f of c.images) fnToPred.set(f, ci);
  });

  // Intersect with ground truth (the rust pipeline uses the content_hashes filename set,
  // groups file may reference deleted files)
  let common: string[] = [];
  for (const [fn] of fnToPred) if (fnToTrue.has(fn)) common.push(fn);
  if (common.length === 0) throw new Error("no overlap between predicted clusters and ground-truth groups");

  // Filter to eval subset if provided (clustering already ran on the full set;
  // we just restrict the metric computation to these images).
  if (args.evalSubset) {
    const subsetList: string[] = JSON.parse(readFileSync(args.evalSubset, "utf-8"));
    const subset = new Set(subsetList);
    const before = common.length;
    common = common.filter((f) => subset.has(f));
    console.log(`  eval subset: ${common.length}/${before} images retained from ${args.evalSubset}`);
    if (common.length === 0) throw new Error("eval subset filtered out all images");
  }

  const trueLab = common.map((f) => fnToTrue.get(f)!);
  const predLab = common.map((f) => fnToPred.get(f)!);
  const dropped = fnToPred.size - common.length;
  const ungrouped = [...fnToPred.keys()].filter((f) => !fnToTrue.has(f)).length;

  const nTrue = new Set(trueLab).size;
  const nPred = new Set(predLab).size;

  const ari = adjustedRandIndex(trueLab, predLab);
  const nmi = normalizedMutualInformation(trueLab, predLab);
  const pa = pairAccuracy(trueLab, predLab);

  console.log("");
  console.log(`Scored on ${common.length} images (${ungrouped} predicted images had no ground-truth label, dropped from scoring)`);
  console.log(`  true clusters:  ${nTrue}`);
  console.log(`  pred clusters:  ${nPred}`);
  console.log("");
  console.log(`  ARI:            ${ari.toFixed(4)}`);
  console.log(`  NMI:            ${nmi.toFixed(4)}`);
  console.log(`  pair accuracy:  ${pa.acc.toFixed(4)}   (tp=${pa.tp} tn=${pa.tn} fp=${pa.fp} fn=${pa.fn})`);
  if (dropped) console.log(`  (note: ${dropped} predicted-cluster images missing from ground truth)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
