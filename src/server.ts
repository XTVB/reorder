import type { Stats } from "node:fs";
import { stat } from "node:fs/promises";
import { basename, extname, join } from "node:path";
import type { ClusterData, ImageGroup, MergeSuggestionSimilar } from "./client/types.ts";
import {
  broadcastProgress,
  buildImportedResult,
  cancelClusterJob,
  clearImportedClusters,
  computeMergeSuggestions,
  ensureTextEmbeddings,
  extractFeatures,
  generateContactSheet,
  getClusterAbortSignal,
  getLastProgress,
  hasImportedClusters,
  type ImportClusterInput,
  invalidateClusterCache,
  isClusterJobRunning,
  loadImportedClusters,
  runFullCluster,
  runLinkageOnly,
  runRecut,
  runRecutAdaptive,
  runRecutByThreshold,
  saveImportedClusters,
  setClusterJobRunning,
  subscribeProgress,
  type WeightConfig,
} from "./cluster.ts";
import { initLog, log, logData, logError } from "./log.ts";
import {
  canUndo,
  computeOrganize,
  computeRenames,
  executeFolderSave,
  executeOrganize,
  executeRenames,
  type FolderSaveRequest,
  listFolderData,
  listImages,
  type OrganizeGroup,
  type RenameMapping,
  recoverPendingRename,
  undoRenames,
  withRenameLock,
} from "./rename.ts";
import { getThumbnail } from "./thumbnails.ts";

const GROUPS_FILE = ".reorder-groups.json";

async function readGroupsFile(targetDir: string): Promise<ImageGroup[]> {
  try {
    const file = Bun.file(join(targetDir, GROUPS_FILE));
    if (await file.exists()) {
      const data = await file.json();
      return Array.isArray(data) ? data : Array.isArray(data?.groups) ? data.groups : [];
    }
  } catch {}
  return [];
}

async function writeGroupsFile(targetDir: string, groups: ImageGroup[]) {
  const mainPath = join(targetDir, GROUPS_FILE);
  const backupPath = join(targetDir, ".reorder-groups.bak.json");
  try {
    const existing = Bun.file(mainPath);
    if (await existing.exists()) {
      await Bun.write(backupPath, existing);
    }
  } catch {}
  await Bun.write(mainPath, JSON.stringify(groups, null, 2));
}

async function remapGroups(
  targetDir: string,
  renames: RenameMapping[],
): Promise<{ before: ImageGroup[]; after: ImageGroup[] }> {
  const groups = await readGroupsFile(targetDir);
  if (groups.length === 0) return { before: [], after: [] };
  const renameMap = new Map(renames.map((r) => [r.from, r.to]));
  const remapped = groups.map((g) => ({
    ...g,
    images: g.images.map((fn) => renameMap.get(fn) ?? fn),
  }));
  await writeGroupsFile(targetDir, remapped);
  return { before: groups, after: remapped };
}

/**
 * Remap groups after a rename operation, collecting non-fatal warnings.
 * Called identically from /api/save and /api/undo.
 */
async function remapContentHashes(targetDir: string, renames: RenameMapping[]) {
  const hashesPath = join(targetDir, ".reorder-cache", "content_hashes.json");
  try {
    const hashes: Record<string, string> = await Bun.file(hashesPath).json();
    const renameMap = new Map(renames.map((r) => [r.from, r.to]));
    const updated: Record<string, string> = {};
    for (const [filename, hash] of Object.entries(hashes)) {
      updated[renameMap.get(filename) ?? filename] = hash;
    }
    await Bun.write(hashesPath, JSON.stringify(updated));
  } catch {
    // content_hashes.json doesn't exist yet — extraction hasn't run
  }
}

async function remapAfterRename(
  targetDir: string,
  renames: RenameMapping[],
  label: string,
  warnings: string[],
) {
  try {
    const { after } = await remapGroups(targetDir, renames);
    log(label, `Remapped groups: ${after.length} groups`);
    logData(
      label,
      `Groups (post-${label})`,
      after.map((g) => `  ${g.name}: [${g.images.join(", ")}]`).join("\n"),
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logError(label, "remapGroups failed", err);
    warnings.push(`Group remapping failed: ${msg}`);
  }
  try {
    await remapContentHashes(targetDir, renames);
    log(label, "Remapped content_hashes.json");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logError(label, "remapContentHashes failed", err);
    warnings.push(`Content hashes remapping failed: ${msg}`);
  }
  invalidateClusterCache();
}

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
  ".avif": "image/avif",
  ".bmp": "image/bmp",
  ".tiff": "image/tiff",
  ".tif": "image/tiff",
};

function mimeType(path: string): string {
  return MIME_TYPES[extname(path).toLowerCase()] ?? "application/octet-stream";
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function makeETag(ino: number, size: number): string {
  return `W/"${ino}-${size}"`;
}

async function serveFileWithCache(
  req: Request,
  filePath: string,
  cacheControl: string,
): Promise<Response> {
  let s: Stats;
  try {
    s = await stat(filePath);
  } catch {
    return json({ error: "File not found" }, 404);
  }
  const etag = makeETag(s.ino, s.size);
  if (req.headers.get("If-None-Match") === etag) {
    return new Response(null, { status: 304 });
  }
  return new Response(Bun.file(filePath), {
    headers: {
      "Content-Type": mimeType(filePath),
      "Cache-Control": cacheControl,
      ETag: etag,
    },
  });
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

      // API routes
      if (path.startsWith("/api/")) {
        return handleAPI(req, path, targetDir);
      }

      // Static files
      if (path === "/" || path === "/index.html") {
        return new Response(Bun.file(join(distDir, "index.html")), {
          headers: { "Content-Type": "text/html" },
        });
      }

      // Serve built assets
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

async function handleAPI(req: Request, path: string, targetDir: string): Promise<Response> {
  try {
    if (path === "/api/dir" && req.method === "GET") {
      return json({ dir: targetDir });
    }

    if (path === "/api/images" && req.method === "GET") {
      // Wait for any in-progress recovery/rename to finish before listing
      return withRenameLock(async () => {
        const images = await listImages(targetDir);
        return json({ images: images.map((filename) => ({ filename })) });
      });
    }

    if (path.startsWith("/api/images/") && req.method === "GET") {
      const filename = decodeURIComponent(path.slice("/api/images/".length));
      return serveFileWithCache(req, join(targetDir, filename), "no-cache");
    }

    if (path.startsWith("/api/thumbnails/") && req.method === "GET") {
      const filename = decodeURIComponent(path.slice("/api/thumbnails/".length));
      const { path: thumbPath } = await getThumbnail(targetDir, filename);
      return serveFileWithCache(req, thumbPath, "no-cache");
    }

    if (path === "/api/preview" && req.method === "POST") {
      const body = (await req.json()) as { order: string[] };
      const renames = computeRenames(body.order);
      return json({ renames });
    }

    if (path === "/api/save" && req.method === "POST") {
      const body = (await req.json()) as { order: string[]; groups?: ImageGroup[] };
      return withRenameLock(async () => {
        const t0 = Date.now();
        log("save", `Received save request: ${body.order.length} files in order`);
        logData("save", "Input order", body.order.join("\n"));
        const warnings: string[] = [];

        if (body.groups) {
          const groupSummary = body.groups
            .map((g) => `  ${g.name}: [${g.images.join(", ")}]`)
            .join("\n");
          log("save", `Writing ${body.groups.length} groups to disk before rename`);
          logData("save", "Groups (pre-rename)", groupSummary);
          await writeGroupsFile(targetDir, body.groups);
        }

        log("save", "Executing filesystem renames...");
        const renames = await executeRenames(targetDir, body.order);
        const effective = renames.filter((r) => r.from !== r.to);
        log(
          "save",
          `Filesystem renames complete: ${effective.length} changed, ${renames.length - effective.length} unchanged`,
        );
        logData(
          "save",
          "All rename mappings",
          renames
            .map((r) => (r.from === r.to ? `  ${r.from} (unchanged)` : `  ${r.from} → ${r.to}`))
            .join("\n"),
        );

        await remapAfterRename(targetDir, renames, "save", warnings);

        const elapsed = Date.now() - t0;
        log(
          "save",
          `Complete in ${elapsed}ms — ${effective.length} files renamed${warnings.length > 0 ? `, ${warnings.length} warning(s)` : ""}`,
        );
        return json({ success: true, renames, warnings });
      });
    }

    if (path === "/api/reorder-by-groups" && req.method === "POST") {
      return withRenameLock(async () => {
        const t0 = Date.now();
        const label = "reorder-by-groups";
        log(label, "Received reorder-by-groups request");
        const warnings: string[] = [];

        const [groups, diskImages] = await Promise.all([
          readGroupsFile(targetDir),
          listImages(targetDir),
        ]);
        const diskSet = new Set(diskImages);

        log(
          label,
          `Loaded ${groups.length} groups from ${GROUPS_FILE}, ${diskImages.length} images on disk`,
        );

        const seen = new Set<string>();
        const order: string[] = [];
        const missing: string[] = [];
        let groupedCount = 0;

        for (const g of groups) {
          for (const fn of g.images) {
            if (seen.has(fn)) continue;
            if (!diskSet.has(fn)) {
              missing.push(`${g.name}: ${fn}`);
              continue;
            }
            seen.add(fn);
            order.push(fn);
            groupedCount++;
          }
        }

        for (const fn of diskImages) {
          if (seen.has(fn)) continue;
          seen.add(fn);
          order.push(fn);
        }

        const ungroupedCount = order.length - groupedCount;

        if (missing.length > 0) {
          warnings.push(`${missing.length} group member(s) not found on disk (skipped)`);
          logData(label, "Missing group members", missing.join("\n"));
        }

        log(
          label,
          `Computed order: ${order.length} files (${groupedCount} grouped, ${ungroupedCount} ungrouped at end)`,
        );
        logData(label, "Input order", order.join("\n"));

        log(label, "Executing filesystem renames...");
        const renames = await executeRenames(targetDir, order);
        const effective = renames.filter((r) => r.from !== r.to);
        log(
          label,
          `Filesystem renames complete: ${effective.length} changed, ${renames.length - effective.length} unchanged`,
        );
        logData(
          label,
          "All rename mappings",
          renames
            .map((r) => (r.from === r.to ? `  ${r.from} (unchanged)` : `  ${r.from} → ${r.to}`))
            .join("\n"),
        );

        await remapAfterRename(targetDir, renames, label, warnings);

        const elapsed = Date.now() - t0;
        log(
          label,
          `Complete in ${elapsed}ms — ${effective.length} files renamed${warnings.length > 0 ? `, ${warnings.length} warning(s)` : ""}`,
        );
        return json({ success: true, renames, warnings });
      });
    }

    if (path === "/api/undo" && req.method === "POST") {
      return withRenameLock(async () => {
        const t0 = Date.now();
        log("undo", "Received undo request");
        const warnings: string[] = [];

        const renames = await undoRenames(targetDir);
        const effective = renames.filter((r) => r.from !== r.to);
        log("undo", `Filesystem undo complete: ${effective.length} files reversed`);
        logData(
          "undo",
          "All undo mappings",
          renames
            .map((r) => (r.from === r.to ? `  ${r.from} (unchanged)` : `  ${r.from} → ${r.to}`))
            .join("\n"),
        );

        await remapAfterRename(targetDir, renames, "undo", warnings);

        const elapsed = Date.now() - t0;
        log(
          "undo",
          `Complete in ${elapsed}ms — ${effective.length} files reversed${warnings.length > 0 ? `, ${warnings.length} warning(s)` : ""}`,
        );
        return json({ success: true, renames, warnings });
      });
    }

    if (path === "/api/can-undo" && req.method === "GET") {
      return withRenameLock(async () => {
        const available = await canUndo(targetDir);
        return json({ canUndo: available });
      });
    }

    if (path === "/api/groups" && req.method === "GET") {
      return withRenameLock(async () => {
        const groups = await readGroupsFile(targetDir);
        return json(groups);
      });
    }

    if (path === "/api/groups" && req.method === "POST") {
      const groups = (await req.json()) as ImageGroup[];
      return withRenameLock(async () => {
        log("groups", `Persisting ${groups.length} groups`);
        logData(
          "groups",
          "Groups snapshot",
          groups.map((g) => `  ${g.name}: [${g.images.join(", ")}]`).join("\n"),
        );
        await writeGroupsFile(targetDir, groups);
        return json({ success: true });
      });
    }

    if (path === "/api/organize/preview" && req.method === "POST") {
      const body = (await req.json()) as { groups: OrganizeGroup[]; order: string[] };
      const mappings = computeOrganize(body.groups, body.order);
      return json({ mappings });
    }

    if (path === "/api/organize" && req.method === "POST") {
      const body = (await req.json()) as { groups: OrganizeGroup[]; order: string[] };
      return withRenameLock(async () => {
        const mappings = await executeOrganize(targetDir, body.groups, body.order);
        return json({ success: true, mappings });
      });
    }

    // ---- Folder mode routes ----

    if (path === "/api/folders" && req.method === "GET") {
      return withRenameLock(async () => {
        const data = await listFolderData(targetDir);
        return json(data);
      });
    }

    if (path === "/api/folders/save" && req.method === "POST") {
      const body = (await req.json()) as FolderSaveRequest;
      return withRenameLock(async () => {
        const t0 = Date.now();
        const totalImages =
          body.folders.reduce((n, f) => n + f.images.length, 0) + body.rootImages.length;
        log("folders-save", `Received save: ${body.folders.length} folders, ${totalImages} images`);
        const result = await executeFolderSave(targetDir, body, log);
        const elapsed = Date.now() - t0;
        log(
          "folders-save",
          `Complete in ${elapsed}ms — ${result.moves.length} moves, ${result.foldersCreated.length} created, ${result.foldersRemoved.length} removed`,
        );
        return json({ success: true, ...result });
      });
    }

    // ---- Cluster routes ----

    if (path === "/api/cluster" && req.method === "POST") {
      const body = (await req.json()) as {
        nClusters?: number;
        weights?: WeightConfig;
        usePatches?: boolean;
      };
      const nClusters = body.nClusters ?? 200;
      const weights = body.weights;
      const usePatches = body.usePatches ?? false;

      if (isClusterJobRunning()) {
        return json({ error: "Clustering already in progress" }, 409);
      }
      setClusterJobRunning(true);
      log("cluster", `Full cluster request (SSE): n=${nClusters} usePatches=${usePatches}`);

      const stream = new ReadableStream({
        async start(controller) {
          const enc = new TextEncoder();
          let closed = false;
          const send = (event: string, data: string) => {
            if (closed) return;
            try {
              controller.enqueue(enc.encode(`event: ${event}\ndata: ${data}\n\n`));
            } catch {
              closed = true;
            }
          };
          const keepalive = setInterval(() => {
            if (closed) return;
            try {
              controller.enqueue(enc.encode(": keepalive\n\n"));
            } catch {
              closed = true;
            }
          }, 30_000);
          try {
            const result = await runFullCluster(
              targetDir,
              nClusters,
              (line) => {
                log("cluster", line);
                broadcastProgress(line);
                send("progress", JSON.stringify({ message: line }));
              },
              weights,
              usePatches,
            );
            invalidateClusterCache();
            log("cluster", `Returned ${result.clusters.length} clusters`);
            broadcastProgress("");
            send("result", JSON.stringify(result));
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            broadcastProgress("");
            send("error", JSON.stringify({ error: msg }));
          } finally {
            clearInterval(keepalive);
            setClusterJobRunning(false);
            if (!closed) controller.close();
          }
        },
      });

      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
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
      let cleanupProgressSSE: (() => void) | null = null;
      const stream = new ReadableStream({
        start(controller) {
          const enc = new TextEncoder();
          let closed = false;
          const cleanup = () => {
            if (closed) return;
            closed = true;
            unsub();
            clearInterval(keepalive);
          };
          cleanupProgressSSE = cleanup;
          const write = (s: string) => {
            if (closed) return;
            try {
              controller.enqueue(enc.encode(s));
            } catch {
              cleanup();
            }
          };
          // Send current progress immediately
          const last = getLastProgress();
          if (last) write(`event: progress\ndata: ${JSON.stringify({ message: last })}\n\n`);
          // Keepalive
          const keepalive = setInterval(() => {
            write(": keepalive\n\n");
          }, 30_000);
          const unsub = subscribeProgress((msg) => {
            if (!msg) {
              write(`event: result\ndata: {}\n\n`);
              cleanup();
              try {
                controller.close();
              } catch {}
            } else {
              write(`event: progress\ndata: ${JSON.stringify({ message: msg })}\n\n`);
            }
          });
        },
        cancel() {
          cleanupProgressSSE?.();
        },
      });
      return new Response(stream, {
        headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
      });
    }

    if (path === "/api/cluster/cache-status" && req.method === "GET") {
      const cached = await Bun.file(join(targetDir, ".reorder-cache", "linkage_tree.bin")).exists();
      const imported = hasImportedClusters(targetDir);
      return json({ cached, imported });
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

    if (path === "/api/cluster/embeddings-status" && req.method === "GET") {
      const npzExists = await Bun.file(
        join(targetDir, ".reorder-cache", "content_hashes.json"),
      ).exists();
      return json({ ready: npzExists });
    }

    if (path === "/api/cluster/extract" && req.method === "POST") {
      if (isClusterJobRunning()) {
        return json({ error: "Extraction already in progress" }, 409);
      }
      const body = (await req.json().catch(() => ({}))) as { models?: string[] };
      const models = body.models; // e.g. ["pecore_l", "pecore_g"]
      setClusterJobRunning(true);
      log(
        "cluster",
        `Extract embeddings request (SSE)${models ? ` models=${models.join(",")}` : ""}`,
      );

      const stream = new ReadableStream({
        async start(controller) {
          const enc = new TextEncoder();
          let closed = false;
          const send = (event: string, data: string) => {
            if (closed) return;
            try {
              controller.enqueue(enc.encode(`event: ${event}\ndata: ${data}\n\n`));
            } catch {
              closed = true;
            }
          };
          // SSE keepalive: send a comment every 30s to prevent idle timeout
          const keepalive = setInterval(() => {
            if (closed) return;
            try {
              controller.enqueue(enc.encode(": keepalive\n\n"));
            } catch {
              closed = true;
            }
          }, 30_000);
          try {
            const signal = getClusterAbortSignal();
            const result = await extractFeatures(
              targetDir,
              (line) => {
                broadcastProgress(line);
                send("progress", JSON.stringify({ message: line }));
              },
              models ? { force: models, signal } : { signal },
            );
            invalidateClusterCache();
            await ensureTextEmbeddings(targetDir);
            broadcastProgress("");
            send("result", JSON.stringify(result));
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            broadcastProgress("");
            send("error", JSON.stringify({ error: msg }));
          } finally {
            clearInterval(keepalive);
            setClusterJobRunning(false);
            if (!closed) controller.close();
          }
        },
      });
      return new Response(stream, {
        headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
      });
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
      };
      const nClusters = body.nClusters ?? 200;
      log(
        "cluster",
        `Test linkage: n=${nClusters} weights=${JSON.stringify(body.weights)} usePatches=${body.usePatches}`,
      );
      const result = await runLinkageOnly(targetDir, nClusters, body.weights, body.usePatches);
      return json(result);
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
      const filePath = join(targetDir, ".reorder-cache", "contact_sheets", name);
      return serveFileWithCache(req, filePath, "no-cache");
    }

    // ── Merge suggestions (DINOv3 patch matching) ─────────────────────
    if (path === "/api/merge-suggestions" && req.method === "POST") {
      const body = (await req.json()) as {
        threshold?: number;
        maxPerGroup?: number;
        fullResolution?: boolean;
        maxCombinedSize?: number;
      };
      const threshold = body.threshold ?? 0.65;
      const maxPerGroup = body.maxPerGroup ?? 8;
      const fullResolution = body.fullResolution ?? false;
      const maxCombinedSize = Math.max(0, Math.floor(body.maxCombinedSize ?? 0));

      // SSE stream for progress + result (matches /api/cluster pattern)
      const stream = new ReadableStream({
        async start(controller) {
          const enc = new TextEncoder();
          let closed = false;
          const send = (event: string, data: unknown) => {
            if (closed) return;
            try {
              controller.enqueue(enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
            } catch {
              closed = true;
            }
          };
          const keepalive = setInterval(() => {
            if (closed) return;
            try {
              controller.enqueue(enc.encode(": keepalive\n\n"));
            } catch {
              closed = true;
            }
          }, 30_000);
          try {
            const startTime = performance.now();

            const entries = await computeMergeSuggestions(targetDir, threshold, {
              fullResolution,
              maxCombinedSize,
              onProgress: (msg) => send("progress", { message: msg }),
            });

            const groups = (await readGroupsFile(targetDir)) as ImageGroup[];
            const groupMap = new Map(groups.map((g) => [g.id, g]));

            const rowMap = new Map<
              string,
              { refGroupId: string; similar: MergeSuggestionSimilar[] }
            >();

            for (const d of entries) {
              const gA = groupMap.get(d.group_a);
              const gB = groupMap.get(d.group_b);
              if (!gA || !gB) continue;

              // 1 - patch_median so lower = more similar, matching the Ward-distance semantics used by other UI.
              const displayDist = 1 - d.patch_median;

              for (const [srcId, other] of [
                [d.group_a, gB],
                [d.group_b, gA],
              ] as const) {
                let row = rowMap.get(srcId);
                if (!row) {
                  row = { refGroupId: srcId, similar: [] };
                  rowMap.set(srcId, row);
                }
                row.similar.push({
                  groupId: other.id,
                  groupName: other.name,
                  groupImages: other.images,
                  distance: displayDist,
                });
              }
            }

            const suggestions = Array.from(rowMap.values())
              .map((row) => {
                const g = groupMap.get(row.refGroupId)!;
                row.similar.sort((a, b) => a.distance - b.distance);
                row.similar = row.similar.slice(0, maxPerGroup);
                return {
                  refGroupId: row.refGroupId,
                  refGroupName: g.name,
                  refGroupImages: g.images,
                  similar: row.similar,
                };
              })
              .sort((a, b) => a.similar[0]!.distance - b.similar[0]!.distance);

            const computeTimeMs = Math.round(performance.now() - startTime);
            send("result", { suggestions, computeTimeMs });
          } catch (err) {
            send("error", { error: err instanceof Error ? err.message : "Unknown error" });
          } finally {
            clearInterval(keepalive);
            if (!closed) controller.close();
          }
        },
      });

      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }

    return json({ error: "Not found" }, 404);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return json({ error: message }, 500);
  }
}
