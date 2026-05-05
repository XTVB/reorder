// Core cluster routes: full cluster (SSE), status, cancel, progress (re-attach),
// recut, test linkage, cache-status, embeddings-status.

import type { ClusterData } from "../../client/types.ts";
import {
  cancelClusterJob,
  getLastProgress,
  invalidateClusterCache,
  isClusterJobRunning,
  loadImportedClusters,
  runFullCluster,
  runLinkageOnly,
  runRecut,
  runRecutAdaptive,
  runRecutByThreshold,
} from "../../cluster/index.ts";
import { contentHashesPath, linkageTreePath } from "../../fs/paths.ts";
import { log } from "../../log.ts";
import type { WeightConfig } from "../../shared/types.ts";
import { runClusterJobSSE } from "../middleware/cluster-job.ts";
import { json } from "../middleware/response.ts";
import { subscribeProgressSSE } from "../middleware/sse.ts";
import type { RouteHandler } from "../types.ts";

export const clusterRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;

  if (path === "/api/cluster" && req.method === "POST") {
    const body = (await req.json()) as {
      nClusters?: number;
      weights?: WeightConfig;
      usePatches?: boolean;
      useRerank?: boolean;
      rerankBlend?: number;
    };
    const nClusters = body.nClusters ?? 200;
    const weights = body.weights;
    const options = {
      usePatches: body.usePatches,
      useRerank: body.useRerank ?? true,
      rerankBlend: body.rerankBlend,
    };

    log("cluster", `Full cluster request (SSE): n=${nClusters} ${JSON.stringify(options)}`);

    return runClusterJobSSE(async (_send, onProgress) => {
      const result = await runFullCluster(
        targetDir,
        nClusters,
        (line) => {
          log("cluster", line);
          onProgress(line);
        },
        weights,
        options,
      );
      invalidateClusterCache();
      log("cluster", `Returned ${result.clusters.length} clusters`);
      return result;
    }, "Clustering already in progress");
  }

  if (path === "/api/cluster/status" && req.method === "GET") {
    return json({ running: isClusterJobRunning(), progress: getLastProgress() });
  }

  if (path === "/api/cluster/cancel" && req.method === "POST") {
    if (!isClusterJobRunning()) {
      return json({ ok: false, error: "No cluster job running" }, 409);
    }
    cancelClusterJob();
    return json({ ok: true });
  }

  if (path === "/api/cluster/progress" && req.method === "GET") {
    if (!isClusterJobRunning()) {
      return json({ running: false });
    }
    return subscribeProgressSSE();
  }

  if (path === "/api/cluster/cache-status" && req.method === "GET") {
    const cached = await Bun.file(linkageTreePath(targetDir)).exists();
    const imported = await loadImportedClusters(targetDir);
    return json({ cached, imported });
  }

  if (path === "/api/cluster/embeddings-status" && req.method === "GET") {
    const npzExists = await Bun.file(contentHashesPath(targetDir)).exists();
    return json({ ready: npzExists });
  }

  if (path === "/api/cluster/recut" && req.method === "POST") {
    const body = (await req.json()) as {
      nClusters?: number;
      threshold?: number;
      minClusterSize?: number;
    };
    let result: ClusterData;
    if (body.minClusterSize != null) {
      log("cluster", `Re-cut request: adaptive minClusterSize=${body.minClusterSize}`);
      result = await runRecutAdaptive(targetDir, body.minClusterSize);
    } else if (body.threshold != null) {
      log("cluster", `Re-cut request: threshold=${body.threshold}`);
      result = await runRecutByThreshold(targetDir, body.threshold);
    } else {
      log("cluster", `Re-cut request: n=${body.nClusters}`);
      result = await runRecut(targetDir, body.nClusters ?? 200);
    }
    log("cluster", `Re-cut returned ${result.clusters.length} clusters`);
    return json(result);
  }

  if (path === "/api/cluster/test" && req.method === "POST") {
    const body = (await req.json()) as {
      nClusters?: number;
      weights?: WeightConfig;
      usePatches?: boolean;
      useRerank?: boolean;
      rerankBlend?: number;
    };
    const nClusters = body.nClusters ?? 200;
    const opts = {
      usePatches: body.usePatches,
      useRerank: body.useRerank ?? true,
      rerankBlend: body.rerankBlend,
    };
    log(
      "cluster",
      `Test linkage: n=${nClusters} weights=${JSON.stringify(body.weights)} opts=${JSON.stringify(opts)}`,
    );
    const result = await runLinkageOnly(targetDir, nClusters, body.weights, opts);
    return json(result);
  }

  return null;
};
