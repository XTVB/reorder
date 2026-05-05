// Nearest-neighbor query (SSE — extraction may be needed on the fly).

import type { NNAggregation, NNFilter } from "../../client/types.ts";
import {
  ensurePatchDistMatrix,
  extractFeatures,
  getClusterAbortSignal,
  invalidateClusterCache,
  isClusterJobRunning,
  loadPatchDistMatrix,
  setClusterJobRunning,
} from "../../cluster/index.ts";
import { findNearestNeighbors, ModelMissingError } from "../../nn-query.ts";
import type { WeightConfig } from "../../shared/types.ts";
import { json } from "../middleware/response.ts";
import { sseResponse } from "../middleware/sse.ts";
import type { RouteHandler } from "../types.ts";

export const nnRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;
  if (path !== "/api/cluster/nn-query" || req.method !== "POST") return null;

  const body = (await req.json()) as {
    queryFilenames?: string[];
    topN?: number;
    filter?: NNFilter;
    aggregation?: NNAggregation;
    weights?: WeightConfig;
    usePatches?: boolean;
    restrictToFilenames?: string[];
    excludeQuery?: boolean;
  };
  if (!Array.isArray(body.queryFilenames) || body.queryFilenames.length === 0) {
    return json({ error: "queryFilenames must be non-empty" }, 400);
  }
  const queryFilenames = body.queryFilenames;
  const usePatches = body.usePatches ?? false;
  const opts = {
    topN: Math.max(1, Math.min(body.topN ?? 50, 500)),
    filter: body.filter ?? "any",
    aggregation: body.aggregation ?? "centroid",
    excludeQuery: body.excludeQuery !== false,
    weights: body.weights ?? {},
    usePatches,
    restrictToFilenames: body.restrictToFilenames,
  };

  return sseResponse(async (send) => {
    let mutexTaken = false;
    const onProgress = (msg: string) => send("progress", { message: msg });
    const takeMutex = (): boolean => {
      if (mutexTaken) return true;
      if (isClusterJobRunning()) {
        send("error", { error: "Clustering in progress — retry shortly" });
        return false;
      }
      setClusterJobRunning(true);
      mutexTaken = true;
      return true;
    };
    try {
      let patchDistances: Float64Array | null = null;
      if (usePatches) {
        if (!takeMutex()) return;
        await ensurePatchDistMatrix(targetDir, onProgress);
        patchDistances = loadPatchDistMatrix(targetDir).distances;
      }

      // Retry once: on ModelMissingError, extract the missing model and retry.
      for (let attempt = 0; attempt <= 1; attempt++) {
        try {
          const result = findNearestNeighbors(targetDir, queryFilenames, opts, patchDistances);
          send("result", result);
          return;
        } catch (err) {
          if (!(err instanceof ModelMissingError) || attempt === 1) throw err;
          if (!takeMutex()) return;
          send("progress", { message: `Extracting missing model: ${err.modelKey}` });
          await extractFeatures(targetDir, onProgress, {
            force: [err.modelKey],
            signal: getClusterAbortSignal(),
          });
          invalidateClusterCache();
        }
      }
    } finally {
      if (mutexTaken) setClusterJobRunning(false);
    }
  });
};
