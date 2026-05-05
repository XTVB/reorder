// Folder mode: list folder data, save folder restructure.

import {
  executeFolderSave,
  type FolderSaveRequest,
  listFolderData,
  withRenameLock,
} from "../../fs/index.ts";
import { log } from "../../log.ts";
import { json } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

export const foldersRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;

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

  return null;
};
