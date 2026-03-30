import { join, extname } from "node:path";
import { stat } from "node:fs/promises";
import {
  listImages,
  computeRenames,
  executeRenames,
  canUndo,
  undoRenames,
  computeOrganize,
  executeOrganize,
  recoverPendingRename,
  withRenameLock,
  listFolderData,
  executeFolderSave,
  type OrganizeGroup,
  type RenameMapping,
  type FolderSaveRequest,
} from "./rename.ts";
import { getThumbnail } from "./thumbnails.ts";
import { ingestTags, getAllTags, getClothingStructured, getDbStatus, remapTagsDb } from "./tags-db.ts";
import { initLog, log, logError, logData } from "./log.ts";

const GROUPS_FILE = ".reorder-groups.json";

async function readGroupsFile(targetDir: string): Promise<unknown[]> {
  try {
    const file = Bun.file(join(targetDir, GROUPS_FILE));
    if (await file.exists()) {
      const data = await file.json();
      return Array.isArray(data) ? data : Array.isArray(data?.groups) ? data.groups : [];
    }
  } catch {}
  return [];
}

async function writeGroupsFile(targetDir: string, groups: unknown[]) {
  await Bun.write(join(targetDir, GROUPS_FILE), JSON.stringify(groups, null, 2));
}

async function remapGroups(targetDir: string, renames: RenameMapping[]): Promise<{ before: unknown[]; after: unknown[] }> {
  const groups = await readGroupsFile(targetDir);
  if (groups.length === 0) return { before: [], after: [] };
  const renameMap = new Map(renames.map((r) => [r.from, r.to]));
  const remapped = groups.map((g: any) => ({
    ...g,
    images: Array.isArray(g.images)
      ? g.images.map((fn: string) => renameMap.get(fn) ?? fn)
      : g.images,
  }));
  await writeGroupsFile(targetDir, remapped);
  return { before: groups, after: remapped };
}

/**
 * Remap groups and tags DB after a rename operation, collecting non-fatal warnings.
 * Called identically from /api/save and /api/undo.
 */
async function remapAfterRename(
  targetDir: string,
  renames: RenameMapping[],
  label: string,
  warnings: string[],
) {
  try {
    const { after } = await remapGroups(targetDir, renames);
    log(label, `Remapped groups: ${after.length} groups`);
    logData(label, `Groups (post-${label})`,
      after.map((g: any) => `  ${g.name}: [${g.images?.join(", ")}]`).join("\n")
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logError(label, "remapGroups failed", err);
    warnings.push(`Group remapping failed: ${msg}`);
  }

  try {
    remapTagsDb(targetDir, renames);
    log(label, "Remapped tags DB");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logError(label, "remapTagsDb failed", err);
    warnings.push(`Tags DB remapping failed: ${msg}`);
  }
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
  cacheControl: string
): Promise<Response> {
  let s;
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
      "ETag": etag,
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
      remapTagsDb(targetDir, result.mappings);
      log("recovery", "Remapped groups and tags DB");
    }
  }).catch((err) => {
    logError("recovery", "Failed", err);
  });

  return Bun.serve({
    port,
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

      return new Response("Not Found", { status: 404 });
    },
  });
}

async function handleAPI(
  req: Request,
  path: string,
  targetDir: string
): Promise<Response> {
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
      const body = (await req.json()) as { order: string[]; groups?: unknown[] };
      return withRenameLock(async () => {
        const t0 = Date.now();
        log("save", `Received save request: ${body.order.length} files in order`);
        logData("save", "Input order", body.order.join("\n"));
        const warnings: string[] = [];

        if (body.groups) {
          const groupSummary = (body.groups as any[]).map((g: any) =>
            `  ${g.name}: [${g.images?.join(", ")}]`
          ).join("\n");
          log("save", `Writing ${(body.groups as unknown[]).length} groups to disk before rename`);
          logData("save", "Groups (pre-rename)", groupSummary);
          await writeGroupsFile(targetDir, body.groups);
        }

        log("save", "Executing filesystem renames...");
        const renames = await executeRenames(targetDir, body.order);
        const effective = renames.filter(r => r.from !== r.to);
        log("save", `Filesystem renames complete: ${effective.length} changed, ${renames.length - effective.length} unchanged`);
        logData("save", "All rename mappings",
          renames.map(r => r.from === r.to ? `  ${r.from} (unchanged)` : `  ${r.from} → ${r.to}`).join("\n")
        );

        await remapAfterRename(targetDir, renames, "save", warnings);

        const elapsed = Date.now() - t0;
        log("save", `Complete in ${elapsed}ms — ${effective.length} files renamed${warnings.length > 0 ? `, ${warnings.length} warning(s)` : ""}`);
        return json({ success: true, renames, warnings });
      });
    }

    if (path === "/api/undo" && req.method === "POST") {
      return withRenameLock(async () => {
        const t0 = Date.now();
        log("undo", "Received undo request");
        const warnings: string[] = [];

        const renames = await undoRenames(targetDir);
        const effective = renames.filter(r => r.from !== r.to);
        log("undo", `Filesystem undo complete: ${effective.length} files reversed`);
        logData("undo", "All undo mappings",
          renames.map(r => r.from === r.to ? `  ${r.from} (unchanged)` : `  ${r.from} → ${r.to}`).join("\n")
        );

        await remapAfterRename(targetDir, renames, "undo", warnings);

        const elapsed = Date.now() - t0;
        log("undo", `Complete in ${elapsed}ms — ${effective.length} files reversed${warnings.length > 0 ? `, ${warnings.length} warning(s)` : ""}`);
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
      const groups = (await req.json()) as unknown[];
      return withRenameLock(async () => {
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

    // Tag database routes
    if (path === "/api/tags/status" && req.method === "GET") {
      return json(getDbStatus(targetDir));
    }

    if (path === "/api/tags/ingest" && req.method === "POST") {
      const body = (await req.json()) as { data: unknown };
      const result = await ingestTags(targetDir, body.data);
      return json(result);
    }

    if (path === "/api/tags/all" && req.method === "GET") {
      const result = getAllTags(targetDir);
      if (!result) return json({ error: "No tags database found. Ingest a tags file first." }, 404);
      return json(result);
    }

    if (path === "/api/tags/clothing-structured" && req.method === "GET") {
      const result = getClothingStructured(targetDir);
      if (!result) return json({ error: "No tags database found" }, 404);
      return json(result);
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
        const totalImages = body.folders.reduce((n, f) => n + f.images.length, 0) + body.rootImages.length;
        log("folders-save", `Received save: ${body.folders.length} folders, ${totalImages} images`);
        const result = await executeFolderSave(targetDir, body, log);
        const elapsed = Date.now() - t0;
        log("folders-save", `Complete in ${elapsed}ms — ${result.moves.length} moves, ${result.foldersCreated.length} created, ${result.foldersRemoved.length} removed`);
        return json({ success: true, ...result });
      });
    }

    return json({ error: "Not found" }, 404);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return json({ error: message }, 500);
  }
}
