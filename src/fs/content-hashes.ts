// The rename-surviving image identity — blake2b of the first 16KB + file size —
// plus the mutators for `.reorder-cache/content_hashes.json`, the extraction
// pipeline's filename→hash map. Used by rename and delete routes after
// disk-state changes. Writes go through writeJsonAtomic so a concurrent
// cluster-job reader never sees an empty intermediate state.

import { createHash } from "node:crypto";
import { open, stat } from "node:fs/promises";
import { join } from "node:path";
import { writeJsonAtomic } from "./atomic-json.ts";
import { listImages } from "./images.ts";
import { contentHashesPath } from "./paths.ts";

const HEAD_BYTES = 16384;

/** Hash a file by its first 16KB + size — cheap, and stable across renames. */
export async function computeContentHash(
  filePath: string,
): Promise<{ hash: string; size: number }> {
  const file = await open(filePath, "r");
  try {
    const buf = Buffer.alloc(HEAD_BYTES);
    const { bytesRead } = await file.read(buf, 0, HEAD_BYTES);
    const stats = await file.stat();
    const h = createHash("blake2b256");
    h.update(buf.subarray(0, bytesRead));
    h.update(String(stats.size));
    return { hash: h.digest("hex"), size: stats.size };
  } finally {
    await file.close();
  }
}

/** Hashes memoized per inode, revalidated by (size, mtime, ctime) — ctime moves
 * on a rename and mtime on a content change, so a recycled inode can never
 * serve another file's hash. A rename costs one re-read of the 16KB head. */
const hashMemo = new Map<number, { key: string; hash: string }>();

function statKey(stats: { size: number; mtimeMs: number; ctimeMs: number }): string {
  return `${stats.size}:${stats.mtimeMs}:${stats.ctimeMs}`;
}

/** Hash every image in `dir` → `Map<filename, hash>`. Unlike content_hashes.json
 * this doesn't need the (~5min, Python) extraction to have ever run, so callers
 * that only want a stable image identity can rely on it in a fresh directory. */
export async function imageContentHashes(dir: string): Promise<Map<string, string>> {
  // Crude bound so scanning many large dirs over a long-lived server can't
  // grow the memo forever; a reset just costs one 16KB head re-read per file.
  if (hashMemo.size > 100_000) hashMemo.clear();
  const filenames = await listImages(dir);
  const out = new Map<string, string>();
  const CONCURRENCY = 16;
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(CONCURRENCY, filenames.length) }, async () => {
      while (next < filenames.length) {
        const filename = filenames[next++]!;
        const path = join(dir, filename);
        const stats = await stat(path).catch(() => null);
        if (!stats) continue; // vanished mid-scan
        const key = statKey(stats);
        const memo = hashMemo.get(stats.ino);
        if (memo?.key === key) {
          out.set(filename, memo.hash);
          continue;
        }
        const { hash } = await computeContentHash(path).catch(() => ({ hash: "" }));
        if (!hash) continue;
        hashMemo.set(stats.ino, { key, hash });
        out.set(filename, hash);
      }
    }),
  );
  return out;
}

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
