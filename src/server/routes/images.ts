// Routes that serve images, thumbnails, and basic directory metadata.

import { join } from "node:path";
import { listImages, withRenameLock } from "../../fs/index.ts";
import { getThumbnail } from "../../thumbnails.ts";
import { json, serveFileWithCache } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

export const imagesRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;

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

  return null;
};
