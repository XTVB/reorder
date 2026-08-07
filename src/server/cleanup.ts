// Shared post-delete bookkeeping: prune content_hashes and optionally groups,
// and drop the position-indexed cluster artifacts (linkage tree, rerank matrix)
// whose row indices are invalidated by any file removal. Used by the reorder
// delete route and the czkawka compare actions. Caller must hold withRenameLock.
// (Rank scores need no pruning — they're keyed by content hash, so a deleted
// photo's entry simply stops resolving; see fs/rank-scores.ts.)
//
// The group prune is the one lossy step: it discards each deleted image's
// group membership and its index within that group. An undoable delete must
// not prune, or undo restores the file to disk with nowhere to put it back.
//
// Leaving a dangling name in .reorder-groups.json is safe — every consumer
// already filters against what's on disk: reorder-by-groups skips members
// missing from `diskSet`, buildClustersFromLabels only looks up filenames
// sourced from the on-disk list, and constraint resolution drops entries that
// don't resolve through hashToFilename. Stale entries are cleared by the next
// Apply/Save renumber, or on demand via /api/groups/prune.

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

export interface PruneGroupsResult {
  removedImages: number;
  /** Groups dropped because the prune left them empty. */
  removedGroups: number;
}

/** Drop group members failing `keep`, plus any group left empty. Returns what
 * it removed; writes only when something changed. */
async function pruneGroupsBy(
  targetDir: string,
  keep: (filename: string) => boolean,
): Promise<PruneGroupsResult> {
  const groups = loadGroups(targetDir);
  if (groups.length === 0) return { removedImages: 0, removedGroups: 0 };

  const cleaned = groups
    .map((g) => ({ ...g, images: g.images.filter(keep) }))
    .filter((g) => g.images.length > 0);

  const before = groups.reduce((n, g) => n + g.images.length, 0);
  const after = cleaned.reduce((n, g) => n + g.images.length, 0);
  const result = {
    removedImages: before - after,
    removedGroups: groups.length - cleaned.length,
  };
  if (result.removedImages === 0 && result.removedGroups === 0) return result;

  await writeGroupsFile(targetDir, cleaned);
  log(
    "prune",
    `Pruned groups: ${cleaned.length} remaining (dropped ${result.removedImages} member(s), ${result.removedGroups} empty group(s))`,
  );
  return result;
}

/** Drop group members whose files aren't in `present` (the manual prune). */
export function pruneGroupsToDisk(
  targetDir: string,
  present: Set<string>,
): Promise<PruneGroupsResult> {
  return pruneGroupsBy(targetDir, (fn) => present.has(fn));
}

/** Run the cleanup steps, collecting warnings instead of throwing — a failed
 * step must not roll back the delete that already happened.
 *
 * Leaves reorder-group membership intact, so it is safe after an *undoable*
 * delete. One-way-door deletes want `cleanupAfterPermanentDelete` instead. */
export async function cleanupAfterDelete(targetDir: string, deleted: string[]): Promise<string[]> {
  if (deleted.length === 0) return [];
  const deletedSet = new Set(deleted);
  const warnings: string[] = [];
  const safeStep = makeSafeStep(warnings);

  await Promise.all([
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

/** `cleanupAfterDelete` plus the lossy group prune. Only for deletes that can't
 * be undone — see the header note. */
export async function cleanupAfterPermanentDelete(
  targetDir: string,
  deleted: string[],
): Promise<string[]> {
  if (deleted.length === 0) return [];
  const deletedSet = new Set(deleted);
  const warnings: string[] = [];
  await makeSafeStep(warnings)("Group cleanup", async () => {
    await pruneGroupsBy(targetDir, (fn) => !deletedSet.has(fn));
  });
  warnings.push(...(await cleanupAfterDelete(targetDir, deleted)));
  return warnings;
}
