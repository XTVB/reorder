// Shared post-delete bookkeeping: prune groups + content_hashes and drop the
// position-indexed cluster artifacts (linkage tree, rerank matrix) whose row
// indices are invalidated by any file removal. Used by the reorder delete
// route and the czkawka compare actions. Caller must hold withRenameLock.

import {
  invalidateClusterCache,
  removeLinkageTree,
  removeRerankDistMatrix,
} from "../cluster/index.ts";
import { pruneContentHashes } from "../fs/content-hashes.ts";
import { loadGroups, writeGroupsFile } from "../fs/index.ts";
import { log, logError } from "../log.ts";

function makeSafeStep(warnings: string[]) {
  return async (step: string, fn: () => Promise<void>) => {
    try {
      await fn();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logError("delete", `${step} failed`, err);
      warnings.push(`${step} failed: ${msg}`);
    }
  };
}

/** Drop the position-indexed cluster artifacts and in-memory caches. Enough
 * on its own when directory contents changed without any file going away
 * (e.g. czkawka undo restoring files). */
export async function invalidateDerivedCaches(targetDir: string): Promise<string[]> {
  const warnings: string[] = [];
  const safeStep = makeSafeStep(warnings);
  await Promise.all([
    safeStep("Linkage tree invalidation", () => removeLinkageTree(targetDir)),
    safeStep("Rerank matrix invalidation", () => removeRerankDistMatrix(targetDir)),
  ]);
  invalidateClusterCache();
  return warnings;
}

/** Run the cleanup steps, collecting warnings instead of throwing — a failed
 * prune must not roll back the delete that already happened. */
export async function cleanupAfterDelete(targetDir: string, deleted: string[]): Promise<string[]> {
  if (deleted.length === 0) return [];
  const deletedSet = new Set(deleted);
  const warnings: string[] = [];
  const safeStep = makeSafeStep(warnings);

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

  return warnings;
}
