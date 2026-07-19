// Single source of truth for filesystem path conventions.

import { join } from "node:path";

// Cache directory (inside the target image directory)
export const CACHE_DIRNAME = ".reorder-cache";

export function cacheDir(targetDir: string): string {
  return join(targetDir, CACHE_DIRNAME);
}

// State files (in the target directory)
export const GROUPS_FILE = ".reorder-groups.json";
export const GROUPS_BACKUP_FILE = ".reorder-groups.bak.json";
export const CONSTRAINTS_FILE = ".reorder-constraints.json";
export const CONSTRAINTS_BACKUP_FILE = ".reorder-constraints.bak.json";
export const PENDING_RENAME_FILE = ".reorder-pending.json";
export const PENDING_FOLDER_SAVE_FILE = ".reorder-folders-pending.json";
export const HISTORY_FILE = ".reorder-history.json";
export const HISTORY_PREV_FILE = ".reorder-history.prev.json";
export const LOG_FILE = ".reorder-log";
export const TAGS_FILE = "tags.json";
export const TEMP_PREFIX = "__reorder_tmp_";

export function groupsPath(targetDir: string): string {
  return join(targetDir, GROUPS_FILE);
}
export function groupsBackupPath(targetDir: string): string {
  return join(targetDir, GROUPS_BACKUP_FILE);
}
export function constraintsPath(targetDir: string): string {
  return join(targetDir, CONSTRAINTS_FILE);
}
export function constraintsBackupPath(targetDir: string): string {
  return join(targetDir, CONSTRAINTS_BACKUP_FILE);
}
export function pendingRenamePath(targetDir: string): string {
  return join(targetDir, PENDING_RENAME_FILE);
}
export function pendingFolderSavePath(targetDir: string): string {
  return join(targetDir, PENDING_FOLDER_SAVE_FILE);
}
export function historyPath(targetDir: string): string {
  return join(targetDir, HISTORY_FILE);
}
export function historyPrevPath(targetDir: string): string {
  return join(targetDir, HISTORY_PREV_FILE);
}
export function logPath(targetDir: string): string {
  return join(targetDir, LOG_FILE);
}
export function tagsPath(targetDir: string): string {
  return join(targetDir, TAGS_FILE);
}

// Cache files (inside .reorder-cache/)
export const HASH_CACHE_FILE = "embeddings_hash_cache.npz";
// Legacy filename retained so we can transparently migrate caches written
// before the CLIP/PE-L/DINOv2 cleanup. Bun-side readers fall back to this.
export const LEGACY_HASH_CACHE_FILE = "clip_hash_cache.npz";
export const HASH_ORDER_FILE = "hash_cache_order.json";
export const CONTENT_HASHES_FILE = "content_hashes.json";
export const CONTENT_HASHES_TMP_FILE = "content_hashes.json.tmp";
export const LINKAGE_TREE_FILE = "linkage_tree.bin";
export const PATCH_DIST_MATRIX_FILE = "patch_dist_matrix.bin";
export const RERANK_DIST_MATRIX_FILE = "rerank_dist_matrix.bin";
export const IMPORTED_CLUSTERS_FILE = "imported_clusters.json";
export const DINOV3_PATCHES_FILE = "dinov3_patches_hash_cache.npy";
export const DINOV3_PATCHES_FULL_FILE = "dinov3_patches_full_hash_cache.npy";
export const DINOV3_PATCHES_HASHES_FILE = "dinov3_patches_hashes.json";
export const CONTACT_SHEETS_DIRNAME = "contact_sheets";
export const CZKAWKA_SESSION_FILE = "czkawka_session.json";

// Constraint files (Rust-resolved, inside .reorder-cache/)
export const CANNOT_LINK_RESOLVED_FILE = ".cannot_link_resolved.json";
export const LOCKED_GROUPS_RESOLVED_FILE = ".locked_groups_resolved.json";

export function contentHashesPath(targetDir: string): string {
  return join(cacheDir(targetDir), CONTENT_HASHES_FILE);
}
export function contentHashesTmpPath(targetDir: string): string {
  return join(cacheDir(targetDir), CONTENT_HASHES_TMP_FILE);
}
export function linkageTreePath(targetDir: string): string {
  return join(cacheDir(targetDir), LINKAGE_TREE_FILE);
}
export function patchDistMatrixPath(targetDir: string): string {
  return join(cacheDir(targetDir), PATCH_DIST_MATRIX_FILE);
}
export function rerankDistMatrixPath(targetDir: string): string {
  return join(cacheDir(targetDir), RERANK_DIST_MATRIX_FILE);
}
export function importedClustersPath(targetDir: string): string {
  return join(cacheDir(targetDir), IMPORTED_CLUSTERS_FILE);
}
export function contactSheetsDir(targetDir: string): string {
  return join(cacheDir(targetDir), CONTACT_SHEETS_DIRNAME);
}
export function czkawkaSessionPath(targetDir: string): string {
  return join(cacheDir(targetDir), CZKAWKA_SESSION_FILE);
}
/** Current hash-tool pipeline version prefix (v2 = fast_image_resize
 * pre-shrink; v3 = flop-only orientation hashes + luma JPEG decode). Bump the
 * version whenever rust/hash-tool changes its hash values or output schema so
 * stale caches are ignored rather than mixed with fresh hashes; files with an
 * older prefix are pruned on the next run. */
export const CZKAWKA_HASH_CACHE_PREFIX = "czkawka_hashes_v3_";

/** Per-config perceptual-hash cache — separate file per (alg, filter, size) so
 * switching configs never evicts another config's hashes. */
export function czkawkaHashCachePath(
  targetDir: string,
  hashAlg: string,
  imageFilter: string,
  hashSize: number,
): string {
  return join(
    cacheDir(targetDir),
    `${CZKAWKA_HASH_CACHE_PREFIX}${hashAlg}_${imageFilter}_${hashSize}.json`,
  );
}
