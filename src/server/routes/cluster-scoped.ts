// Scoped clustering (subset of groups, SSE) + scoped recut.

import { runScopedFull, runScopedRecut } from "../../cluster/index.ts";
import { log } from "../../log.ts";
import type { WeightConfig } from "../../shared/types.ts";
import { runClusterJobSSE } from "../middleware/cluster-job.ts";
import { json } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

export const clusterScopedRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;

  if (path === "/api/cluster/scoped" && req.method === "POST") {
    const body = (await req.json()) as {
      groupIds?: string[];
      nClusters?: number;
      weights?: WeightConfig;
    };
    if (!Array.isArray(body.groupIds) || body.groupIds.length === 0) {
      return json({ error: "groupIds must be non-empty" }, 400);
    }
    const groupIds = body.groupIds;
    const nClusters = body.nClusters ?? 50;
    const weights = body.weights;

    log("cluster", `Scoped cluster request: groups=${groupIds.join(",")} n=${nClusters}`);

    return runClusterJobSSE(
      (_send, onProgress) => runScopedFull(targetDir, groupIds, nClusters, weights, onProgress),
      "Clustering already in progress",
    );
  }

  if (path === "/api/cluster/scoped/recut" && req.method === "POST") {
    const body = (await req.json()) as {
      scopeKey?: string;
      nClusters?: number;
      threshold?: number;
      minClusterSize?: number;
    };
    if (!body.scopeKey) return json({ error: "scopeKey required" }, 400);
    const result = await runScopedRecut(targetDir, body.scopeKey, {
      nClusters: body.nClusters,
      threshold: body.threshold,
      minClusterSize: body.minClusterSize,
    });
    return json(result);
  }

  return null;
};
