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
  type OrganizeGroup,
} from "./rename.ts";
import { getThumbnail, remapCache } from "./thumbnails.ts";

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

function makeETag(mtimeMs: number, size: number): string {
  return `W/"${Math.floor(mtimeMs)}-${size}"`;
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
  const etag = makeETag(s.mtimeMs, s.size);
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
  // Deterministic per directory — cache-busts when switching directories but
  // preserves cache across server restarts on the same directory
  const cacheNonce = new Bun.CryptoHasher("md5").update(targetDir).digest("hex").slice(0, 8);

  return Bun.serve({
    port,
    async fetch(req) {
      const url = new URL(req.url);
      const path = url.pathname;

      // API routes
      if (path.startsWith("/api/")) {
        return handleAPI(req, path, targetDir, cacheNonce);
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
  targetDir: string,
  cacheNonce: string
): Promise<Response> {
  try {
    if (path === "/api/dir" && req.method === "GET") {
      return json({ dir: targetDir, cacheNonce });
    }

    if (path === "/api/images" && req.method === "GET") {
      const images = await listImages(targetDir);
      return json(images.map((filename) => ({ filename })));
    }

    if (path.startsWith("/api/images/") && req.method === "GET") {
      const filename = decodeURIComponent(path.slice("/api/images/".length));
      return serveFileWithCache(req, join(targetDir, filename), "public, max-age=3600");
    }

    if (path.startsWith("/api/thumbnails/") && req.method === "GET") {
      const filename = decodeURIComponent(path.slice("/api/thumbnails/".length));
      const { path: thumbPath } = await getThumbnail(targetDir, filename);
      return serveFileWithCache(req, thumbPath, "public, max-age=86400");
    }

    if (path === "/api/preview" && req.method === "POST") {
      const body = (await req.json()) as { order: string[] };
      const renames = computeRenames(body.order);
      return json({ renames });
    }

    if (path === "/api/save" && req.method === "POST") {
      const body = (await req.json()) as { order: string[] };
      const renames = await executeRenames(targetDir, body.order);
      remapCache(targetDir, renames).catch(() => {});
      return json({ success: true, renames });
    }

    if (path === "/api/undo" && req.method === "POST") {
      const renames = await undoRenames(targetDir);
      remapCache(targetDir, renames).catch(() => {});
      return json({ success: true, renames });
    }

    if (path === "/api/can-undo" && req.method === "GET") {
      const available = await canUndo(targetDir);
      return json({ canUndo: available });
    }

    if (path === "/api/groups" && req.method === "GET") {
      const groups = await readGroupsFile(targetDir);
      return json(groups);
    }

    if (path === "/api/groups" && req.method === "POST") {
      const groups = (await req.json()) as unknown[];
      await writeGroupsFile(targetDir, groups);
      return json({ success: true });
    }

    if (path === "/api/organize/preview" && req.method === "POST") {
      const body = (await req.json()) as { groups: OrganizeGroup[]; order: string[] };
      const mappings = computeOrganize(body.groups, body.order);
      return json({ mappings });
    }

    if (path === "/api/organize" && req.method === "POST") {
      const body = (await req.json()) as { groups: OrganizeGroup[]; order: string[] };
      const mappings = await executeOrganize(targetDir, body.groups, body.order);
      return json({ success: true, mappings });
    }

    return json({ error: "Not found" }, 404);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return json({ error: message }, 500);
  }
}
