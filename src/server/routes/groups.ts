// GET /api/groups — load groups from disk
// POST /api/groups — persist groups to disk

import type { ImageGroup } from "../../client/types.ts";
import { loadGroups, withRenameLock, writeGroupsFile } from "../../fs/index.ts";
import { log, logData } from "../../log.ts";
import { json } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

export const groupsRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;
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
