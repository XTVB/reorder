// Mutators for `.reorder-cache/content_hashes.json`. Used by rename and delete
// routes after disk-state changes. Writes go through writeJsonAtomic so a
// concurrent cluster-job reader never sees an empty intermediate state.

import { writeJsonAtomic } from "./atomic-json.ts";
import { contentHashesPath } from "./paths.ts";

/** Apply a from→to filename rename map to content_hashes.json. No-op when the
 * file doesn't exist (extraction hasn't run yet). */
export async function remapContentHashes(
  targetDir: string,
  renameMap: Map<string, string>,
): Promise<void> {
  const path = contentHashesPath(targetDir);
  const file = Bun.file(path);
  if (!(await file.exists())) return;
  const hashes: Record<string, string> = await file.json();
  const updated: Record<string, string> = {};
  for (const [filename, hash] of Object.entries(hashes)) {
    updated[renameMap.get(filename) ?? filename] = hash;
  }
  await writeJsonAtomic(path, updated, { pretty: false, atomic: true });
}

/** Drop every entry whose filename is in `deletedSet`. No-op when the file
 * doesn't exist or no entries match. */
export async function pruneContentHashes(
  targetDir: string,
  deletedSet: Set<string>,
): Promise<void> {
  const path = contentHashesPath(targetDir);
  const file = Bun.file(path);
  if (!(await file.exists())) return;
  const hashes: Record<string, string> = await file.json();
  const filtered: Record<string, string> = {};
  let changed = false;
  for (const [k, v] of Object.entries(hashes)) {
    if (deletedSet.has(k)) changed = true;
    else filtered[k] = v;
  }
  if (changed) await writeJsonAtomic(path, filtered, { pretty: false, atomic: true });
}
