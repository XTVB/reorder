// Batch similarity ordering driver: hands a set of ordering jobs (the ungrouped
// set, or one per selected group) to the Rust `order-tool` binary, which loads
// the cached embeddings once and orders every job in parallel (rayon). It moves
// the O(n²·dim) matrix build and the seriation off the JS event loop, so
// 4–8k-image ungrouped sorts run in seconds.
//
// The binary builds a per-model linear-weighted cosine distance matrix and
// reduces it to a 1D order via six modes (chain/tree/spectral/minimal/stable/
// gather). Weights are rescaled here the same way the cluster pipeline does
// before being passed through.

import { existsSync } from "node:fs";
import { unlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { ensureHashOrderJson, resolveHashCachePath } from "../cache-utils.ts";
import { cacheDir, contentHashesPath, HASH_ORDER_FILE } from "../fs/paths.ts";
import type { GroupOrderMode, WeightConfig } from "../shared/types.ts";
import { ORDER_BINARY } from "./binaries.ts";
import { activeModelsFromWeights } from "./embeddings.ts";
import { availableLearnedKeys, rescaleLearnedProjWeight } from "./pipeline.ts";
import { spawn } from "./subprocess.ts";

export interface OrderJob {
  id: string;
  /** Filenames in the client's current order; the first anchors the sequence. */
  filenames: string[];
}

export interface OrderJobResult {
  id: string;
  orderedIds: string[];
  skipped: number;
  clusters?: number;
  moved?: number;
}

export interface OrderBatchOpts {
  mode: GroupOrderMode;
  minimalLocality?: number;
  stableClusters?: number;
  gatherMinGain?: number;
}

/**
 * Order each job's images by blended embedding similarity via the Rust binary.
 * Jobs are independent (each anchored within itself). Returns one result per
 * input job in the same order; filenames without embeddings are dropped and
 * counted in `skipped`.
 */
export async function orderImagesBatch(
  targetDir: string,
  jobs: OrderJob[],
  weights: WeightConfig,
  opts: OrderBatchOpts,
  onProgress?: (msg: string) => void,
): Promise<OrderJobResult[]> {
  if (jobs.length === 0) return [];

  if (!existsSync(ORDER_BINARY)) {
    throw new Error(
      `order-tool binary not found at ${ORDER_BINARY}. Build with: cd rust/order-tool && cargo build --release`,
    );
  }

  const cache = cacheDir(targetDir);
  const hashCachePath = resolveHashCachePath(cache);
  if (!existsSync(hashCachePath)) {
    throw new Error(
      "Embeddings cache not found — run feature extraction (Compute in Cluster mode) first.",
    );
  }
  ensureHashOrderJson(cache); // regenerate the hash_cache_order.json sidecar if stale
  const hashOrderPath = join(cache, HASH_ORDER_FILE);
  const contentHashesP = contentHashesPath(targetDir);

  // Rescale the learned-head dials the same way the cluster pipeline does,
  // then pass the resulting per-model weights. Missing learned
  // arrays are zeroed by the rescale and skipped by the binary, so the
  // zero-shot models absorb the remainder — identical to the TS path.
  const rescaled = rescaleLearnedProjWeight(weights, availableLearnedKeys(hashCachePath));
  if (activeModelsFromWeights(rescaled).length === 0) {
    throw new Error("Similarity sort requires at least one positive model weight.");
  }

  const input = {
    mode: opts.mode,
    ...(opts.minimalLocality !== undefined && { minimalLocality: opts.minimalLocality }),
    ...(opts.stableClusters !== undefined && { stableClusters: opts.stableClusters }),
    ...(opts.gatherMinGain !== undefined && { gatherMinGain: opts.gatherMinGain }),
    jobs: jobs.map((j) => ({ id: j.id, filenames: j.filenames })),
  };

  // Unique per-call jobs file so concurrent sorts can't clobber each other.
  const jobsPath = join(cache, `.order_jobs_${process.pid}_${performance.now().toString(36)}.json`);
  await writeFile(jobsPath, JSON.stringify(input));

  try {
    const args = [
      ORDER_BINARY,
      "--hash-cache",
      hashCachePath,
      "--content-hashes",
      contentHashesP,
      "--hash-order",
      hashOrderPath,
      "--jobs",
      jobsPath,
    ];
    for (const [key, val] of Object.entries(rescaled)) {
      if (val !== undefined) args.push(`--${key.replace(/_/g, "-")}-weight`, String(val));
    }
    const { stdout } = await spawn(args, { label: "order-tool", onProgress });
    return JSON.parse(stdout) as OrderJobResult[];
  } finally {
    await unlink(jobsPath).catch(() => {});
  }
}
