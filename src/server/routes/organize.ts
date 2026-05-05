// Organize-into-subfolders preview + execute.

import {
  computeOrganize,
  executeOrganize,
  type OrganizeGroup,
  withRenameLock,
} from "../../fs/index.ts";
import { json } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

export const organizeRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;

  if (path === "/api/organize/preview" && req.method === "POST") {
    const body = (await req.json()) as {
      groups: OrganizeGroup[];
      order: string[];
      numbered?: boolean;
    };
    const mappings = computeOrganize(body.groups, body.order, { numbered: body.numbered });
    return json({ mappings });
  }

  if (path === "/api/organize" && req.method === "POST") {
    const body = (await req.json()) as {
      groups: OrganizeGroup[];
      order: string[];
      numbered?: boolean;
    };
    return withRenameLock(async () => {
      const mappings = await executeOrganize(targetDir, body.groups, body.order, {
        numbered: body.numbered,
      });
      return json({ success: true, mappings });
    });
  }

  return null;
};
