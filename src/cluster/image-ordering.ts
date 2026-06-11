// Similarity ordering for ungrouped images on the reorder page: the same
// weighted-cosine blend as nn-query / the Rust cluster pipeline, computed
// directly over the cached embeddings (no subprocess needed — unlike group
// ordering there is no patch aggregation step), then reduced to a 1D sequence
// with the same mode algorithms as group ordering.

import type { GroupOrderMode, WeightConfig } from "../shared/types.ts";
import { activeModelsFromWeights, l2Norm, loadModelEmbedding } from "./embeddings.ts";
import { orderByDistanceMatrix } from "./group-ordering.ts";
import { rescaleLearnedProjWeight } from "./pipeline.ts";

// The matrix build is O(n²·dim) and tree mode's optimal leaf ordering is
// O(n³). Measured on an M-series Mac: matrix build ≈ 0.7s at n=1000 / 2.5s at
// n=2000 (dim 2048); chain/spectral/minimal stay ≤ ~1s at n=2000; tree ≈ 6s at
// n=1200 and ≈ 20s (and ~200MB) at n=1500, growing cubically. Progress streams
// while it runs; the caps only reject sizes that would block for minutes or
// exhaust memory.
const MAX_IMAGES = 3000;
const MAX_IMAGES_TREE = 1500;

// The matrix build runs on the event loop (no subprocess); yield between
// stages so queued SSE progress events actually reach the client.
const flushEvents = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

/**
 * Order `filenames` (the client's gallery order; the first entry anchors the
 * sequence) by blended embedding similarity. Filenames without embeddings are
 * dropped from the result — the client keeps them in place.
 */
export async function orderImagesBySimilarity(
  targetDir: string,
  filenames: string[],
  weights: WeightConfig,
  opts?: { mode?: GroupOrderMode; minimalLocality?: number },
  onProgress?: (msg: string) => void,
): Promise<string[]> {
  const mode = opts?.mode ?? "chain";

  const models = activeModelsFromWeights(rescaleLearnedProjWeight(weights));
  if (models.length === 0) {
    throw new Error("Ungrouped sort requires at least one positive model weight.");
  }

  // All models share the content-hash filename order, so one index suffices.
  const first = loadModelEmbedding(targetDir, models[0]!.key);
  const fnToIdx = new Map(first.filenames.map((f, i) => [f, i]));
  const known: string[] = [];
  const knownIdx: number[] = [];
  const seen = new Set<string>();
  for (const fn of filenames) {
    if (seen.has(fn)) continue;
    seen.add(fn);
    const idx = fnToIdx.get(fn);
    if (idx !== undefined) {
      known.push(fn);
      knownIdx.push(idx);
    }
  }
  const n = known.length;
  if (n <= 2) return known;
  if (n > MAX_IMAGES) {
    throw new Error(
      `Too many images to sort (${n}, limit ${MAX_IMAGES}) — select a subset or group some images first.`,
    );
  }
  if (mode === "tree" && n > MAX_IMAGES_TREE) {
    throw new Error(
      `Tree mode is limited to ${MAX_IMAGES_TREE} images (${n} present; it grows cubically) — use Chain, Spectral or Minimal.`,
    );
  }

  // Blended cosine distance over the subset, accumulated model by model.
  const acc = new Float64Array(n * n);
  const totalPairs = (n * (n - 1)) / 2;
  let totalWeight = 0;
  for (const { key, weight } of models) {
    onProgress?.(`Blending ${key} distances for ${n} images...`);
    await flushEvents();
    const { data, dim, normalized } = loadModelEmbedding(targetDir, key);
    // Gather (and L2-normalize where needed) the subset rows so the O(n²)
    // pass below is pure dot products over contiguous memory.
    const rows = new Float64Array(n * dim);
    for (let i = 0; i < n; i++) {
      const src = data.subarray(knownIdx[i]! * dim, (knownIdx[i]! + 1) * dim);
      const nrm = normalized ? 1 : l2Norm(src) || 1e-10;
      for (let d = 0; d < dim; d++) rows[i * dim + d] = src[d]! / nrm;
    }
    let lastPct = -1;
    for (let i = 0; i < n; i++) {
      const a = i * dim;
      for (let j = i + 1; j < n; j++) {
        const b = j * dim;
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += rows[a + d]! * rows[b + d]!;
        acc[i * n + j] = acc[i * n + j]! + weight * Math.max(0, 1 - dot);
      }
      // Early rows carry the longest j-spans, so report by pairs done, and
      // yield only on a percent change to keep the loop tight.
      if (i % 64 === 0 && onProgress) {
        const done = i * n - (i * (i + 1)) / 2;
        const pct = Math.round((done / totalPairs) * 100);
        if (pct !== lastPct) {
          lastPct = pct;
          onProgress(`Blending ${key} distances for ${n} images... ${pct}%`);
          await flushEvents();
        }
      }
    }
    totalWeight += weight;
  }

  // Rough wall-clock estimate from the measured ~20s at n=1500, cubic growth.
  const treeEta = Math.max(5, Math.round((n / 1500) ** 3 * 20));
  onProgress?.(
    mode === "tree" && n > 800
      ? `Ordering ${n} images (tree) — optimal leaf ordering is heavy, ~${treeEta}s...`
      : `Ordering ${n} images (${mode})...`,
  );
  await flushEvents();
  const dist: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const sim: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const raw = acc[i * n + j]! / totalWeight;
      const d = Number.isFinite(raw) ? raw : 2;
      dist[i]![j] = dist[j]![i] = d;
      sim[i]![j] = sim[j]![i] = Math.max(0, 1 - d);
    }
  }
  return orderByDistanceMatrix(known, dist, sim, {
    mode,
    anchorId: known[0],
    minimalLocality: opts?.minimalLocality,
  });
}
