// Bun.serve + dispatcher. Walks the route handlers in order, returning the
// first non-null response. The startup `recoverPendingRename` call is wrapped
// in `withRenameLock` so live operations (which also acquire the lock) wait.

import { join } from "node:path";
import { recoverPendingRename, withRenameLock } from "../fs/index.ts";
import { initLog, log, logError } from "../log.ts";
import { json, mimeType } from "./middleware/response.ts";
import { clusterRoutes } from "./routes/cluster.ts";
import { clusterExtractRoutes } from "./routes/cluster-extract.ts";
import { constraintsRoutes } from "./routes/constraints.ts";
import { czkawkaRoutes } from "./routes/czkawka.ts";
import { deleteRoutes } from "./routes/delete.ts";
import { foldersRoutes } from "./routes/folders.ts";
import { groupsRoutes } from "./routes/groups.ts";
import { imagesRoutes } from "./routes/images.ts";
import { mergeRoutes } from "./routes/merge.ts";
import { nnRoutes } from "./routes/nn.ts";
import { organizeRoutes } from "./routes/organize.ts";
import { rankRoutes } from "./routes/rank.ts";
import { remapGroups, renameRoutes } from "./routes/rename.ts";
import { treeNavRoutes } from "./routes/tree-nav.ts";
import type { RouteHandler } from "./types.ts";

const ROUTE_HANDLERS: RouteHandler[] = [
  imagesRoutes,
  renameRoutes,
  deleteRoutes,
  groupsRoutes,
  rankRoutes,
  constraintsRoutes,
  organizeRoutes,
  foldersRoutes,
  treeNavRoutes,
  clusterExtractRoutes,
  nnRoutes,
  mergeRoutes,
  clusterRoutes,
  czkawkaRoutes,
];

async function dispatchAPI(req: Request, path: string, targetDir: string): Promise<Response> {
  try {
    const ctx = { path, targetDir };
    for (const handler of ROUTE_HANDLERS) {
      const result = await handler(req, ctx);
      if (result) return result;
    }
    return json({ error: "Not found" }, 404);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return json({ error: message }, 500);
  }
}

export function createServer(targetDir: string, distDir: string, port: number) {
  initLog(targetDir).catch(() => {});

  // Complete any interrupted rename — runs inside the lock so it can't race with live operations.
  // GET /api/images and GET /api/groups also acquire the lock, so the client
  // blocks until recovery finishes — preventing stale-cleanup from wiping groups.
  withRenameLock(async () => {
    const result = await recoverPendingRename(targetDir);
    if (result.status === "none") return;
    log("recovery", result.message);
    if (result.status === "completed" && result.mappings && result.completed > 0) {
      await remapGroups(targetDir, result.mappings);
      log("recovery", "Remapped groups");
    }
  }).catch((err) => {
    logError("recovery", "Failed", err);
  });

  return Bun.serve({
    port,
    idleTimeout: 255, // max allowed — SSE streams for extraction can have long gaps between messages
    async fetch(req) {
      const url = new URL(req.url);
      const path = url.pathname;

      if (path.startsWith("/api/")) {
        return dispatchAPI(req, path, targetDir);
      }

      if (path === "/" || path === "/index.html") {
        return new Response(Bun.file(join(distDir, "index.html")), {
          headers: { "Content-Type": "text/html" },
        });
      }

      const filePath = join(distDir, path);
      const file = Bun.file(filePath);
      if (await file.exists()) {
        return new Response(file, {
          headers: { "Content-Type": mimeType(filePath) },
        });
      }

      // Client-side routing fallback — any non-API, non-asset path gets index.html
      if (!path.includes(".")) {
        return new Response(Bun.file(join(distDir, "index.html")), {
          headers: { "Content-Type": "text/html" },
        });
      }

      return new Response("Not Found", { status: 404 });
    },
  });
}
