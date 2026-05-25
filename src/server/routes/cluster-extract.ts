// Embeddings extraction (SSE), import/imported management, contact-sheet generation.

import { basename, join } from "node:path";
import {
  buildImportedResult,
  clearImportedClusters,
  extractFeatures,
  generateContactSheet,
  getClusterAbortSignal,
  invalidateClusterCache,
  loadImportedClusters,
  saveImportedClusters,
} from "../../cluster/index.ts";
import { contactSheetsDir } from "../../fs/paths.ts";
import { log } from "../../log.ts";
import type { ImportClusterInput } from "../../shared/types.ts";
import { runClusterJobSSE } from "../middleware/cluster-job.ts";
import { json, serveFileWithCache } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

export const clusterExtractRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;

  if (path === "/api/cluster/extract" && req.method === "POST") {
    const body = (await req.json().catch(() => ({}))) as { models?: string[] };
    const models = body.models;
    log(
      "cluster",
      `Extract embeddings request (SSE)${models ? ` models=${models.join(",")}` : ""}`,
    );

    return runClusterJobSSE(async (_send, onProgress) => {
      const signal = getClusterAbortSignal();
      const result = await extractFeatures(
        targetDir,
        onProgress,
        models ? { force: models, signal } : { signal },
      );
      invalidateClusterCache();
      return result;
    }, "Extraction already in progress");
  }

  if (path === "/api/cluster/import" && req.method === "POST") {
    const body = (await req.json()) as { clusters?: ImportClusterInput[] };
    if (!body.clusters || !Array.isArray(body.clusters) || body.clusters.length === 0) {
      return json({ error: "Body must include non-empty `clusters` array" }, 400);
    }
    for (const c of body.clusters) {
      if (typeof c.name !== "string" || !Array.isArray(c.images)) {
        return json({ error: "Each cluster needs `name` string and `images` string[]" }, 400);
      }
    }
    log("cluster", `Import request: ${body.clusters.length} clusters`);
    const result = await buildImportedResult(targetDir, body.clusters);
    await saveImportedClusters(targetDir, result);
    return json(result);
  }

  if (path === "/api/cluster/imported" && req.method === "GET") {
    const data = await loadImportedClusters(targetDir);
    if (!data) return json({ error: "No imported clusters" }, 404);
    return json(data);
  }

  if (path === "/api/cluster/imported" && req.method === "DELETE") {
    await clearImportedClusters(targetDir);
    log("cluster", "Cleared imported clusters cache");
    return json({ ok: true });
  }

  if (path === "/api/cluster/contact-sheet" && req.method === "POST") {
    const body = (await req.json()) as {
      filenames: string[];
      clusterName: string;
      withLabels?: boolean;
    };
    const outPath = await generateContactSheet(
      targetDir,
      body.filenames,
      body.clusterName,
      body.withLabels ?? false,
    );
    return json({ path: outPath, filename: basename(outPath) });
  }

  if (path.startsWith("/api/contact-sheet/") && req.method === "GET") {
    const name = decodeURIComponent(path.slice("/api/contact-sheet/".length));
    if (name.includes("/") || name.includes("..")) return json({ error: "Invalid name" }, 400);
    const filePath = join(contactSheetsDir(targetDir), name);
    return serveFileWithCache(req, filePath, "no-cache");
  }

  return null;
};
