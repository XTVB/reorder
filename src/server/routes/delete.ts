// /api/delete — move files to Trash and prune groups + content_hashes.

import {
  invalidateClusterCache,
  removeLinkageTree,
  removeRerankDistMatrix,
} from "../../cluster/index.ts";
import { pruneContentHashes } from "../../fs/content-hashes.ts";
import { executeDelete, loadGroups, withRenameLock, writeGroupsFile } from "../../fs/index.ts";
import { log, logData, logError } from "../../log.ts";
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

    if (deleted.length > 0) {
      const deletedSet = new Set(deleted);

      const safeStep = async (step: string, fn: () => Promise<void>) => {
        try {
          await fn();
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          logError("delete", `${step} failed`, err);
          warnings.push(`${step} failed: ${msg}`);
        }
      };

      await Promise.all([
        safeStep("Group cleanup", async () => {
          const groups = loadGroups(targetDir);
          if (groups.length === 0) return;
          if (!groups.some((g) => g.images.some((fn) => deletedSet.has(fn)))) return;
          const cleaned = groups
            .map((g) => ({ ...g, images: g.images.filter((fn) => !deletedSet.has(fn)) }))
            .filter((g) => g.images.length > 0);
          await writeGroupsFile(targetDir, cleaned);
          log("delete", `Pruned groups: ${cleaned.length} remaining`);
        }),
        safeStep("Content hashes cleanup", () => pruneContentHashes(targetDir, deletedSet)),
        // linkage_tree.bin and rerank_dist_matrix.bin are indexed by image
        // position in the sorted filename list — any deletion shifts those
        // indices, so the on-disk artifacts must go. Embeddings (hash-keyed)
        // survive deletes.
        safeStep("Linkage tree invalidation", () => removeLinkageTree(targetDir)),
        safeStep("Rerank matrix invalidation", () => removeRerankDistMatrix(targetDir)),
      ]);
      invalidateClusterCache();
    }

    const elapsed = Date.now() - t0;
    log(
      "delete",
      `Complete in ${elapsed}ms — ${deleted.length} files moved to Trash${warnings.length > 0 ? `, ${warnings.length} warning(s)` : ""}`,
    );
    return json({ success: true, deleted, missing, warnings });
  });
};
