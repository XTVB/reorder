// /api/delete — move files to Trash and prune groups + content_hashes.

import { executeDelete, withRenameLock } from "../../fs/index.ts";
import { log, logData } from "../../log.ts";
import { cleanupAfterPermanentDelete } from "../cleanup.ts";
import { json } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

export const deleteRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;
  if (path !== "/api/delete" || req.method !== "POST") return null;

  const body = (await req.json()) as { filenames?: string[] };
  const filenames = Array.isArray(body.filenames) ? body.filenames : [];
  if (filenames.length === 0) {
    return json({ error: "filenames must be a non-empty array" }, 400);
  }
  // Reject any compound paths — only bare filenames in targetDir are allowed
  for (const fn of filenames) {
    if (fn.includes("/") || fn.includes("..") || fn.startsWith(".")) {
      return json({ error: `Invalid filename: ${fn}` }, 400);
    }
  }
  return withRenameLock(async () => {
    const t0 = Date.now();
    log("delete", `Received delete request: ${filenames.length} files`);
    logData("delete", "Files to delete", filenames.join("\n"));
    const warnings: string[] = [];

    const { deleted, missing } = await executeDelete(targetDir, filenames);
    log("delete", `Trash complete: ${deleted.length} moved to Trash, ${missing.length} missing`);
    if (missing.length > 0) {
      warnings.push(`${missing.length} file(s) not found on disk (skipped)`);
      logData("delete", "Missing files", missing.join("\n"));
    }

    warnings.push(...(await cleanupAfterPermanentDelete(targetDir, deleted)));

    const elapsed = Date.now() - t0;
    log(
      "delete",
      `Complete in ${elapsed}ms — ${deleted.length} files moved to Trash${warnings.length > 0 ? `, ${warnings.length} warning(s)` : ""}`,
    );
    return json({ success: true, deleted, missing, warnings });
  });
};
