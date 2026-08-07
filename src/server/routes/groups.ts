// GET /api/groups — load groups from disk
// POST /api/groups — persist groups to disk
// POST /api/groups/prune — drop group members whose files no longer exist

import type { ImageGroup } from "../../client/types.ts";
import { listImages, loadGroups, withRenameLock, writeGroupsFile } from "../../fs/index.ts";
import { log, logData } from "../../log.ts";
import { pruneGroupsToDisk } from "../cleanup.ts";
import { json } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

export const groupsRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;

  // Manual cleanup for members left dangling by an undoable delete, which keeps
  // membership so undo can restore it. Reads the directory under the rename
  // lock so a half-applied rename can't make live files look missing.
  if (path === "/api/groups/prune" && req.method === "POST") {
    return withRenameLock(async () => {
      const present = new Set(await listImages(targetDir));
      const result = await pruneGroupsToDisk(targetDir, present);
      return json({ success: true, ...result });
    });
  }

  if (path !== "/api/groups") return null;

  if (req.method === "GET") {
    return withRenameLock(async () => {
      return json(loadGroups(targetDir));
    });
  }

  if (req.method === "POST") {
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

  return null;
};
