import type { Stats } from "node:fs";
import { mkdir, readdir, rm, stat, unlink } from "node:fs/promises";
import { join } from "node:path";
import sharp from "sharp";

const CACHE_DIR = ".reorder-cache";
const THUMB_WIDTH = 400;
const THUMB_QUALITY = 80;

/** Content-addressable cache key: inode + file size → deterministic, rename-proof. */
function cacheKey(ino: number, size: number): string {
  return `${ino}-${size}.webp`;
}

const _ensuredDirs = new Set<string>();

async function ensureCacheDir(targetDir: string): Promise<string> {
  const dir = join(targetDir, CACHE_DIR);
  if (_ensuredDirs.has(dir)) return dir;
  await mkdir(dir, { recursive: true });
  const noBackup = Bun.file(join(dir, ".nobackup"));
  if (!(await noBackup.exists())) await Bun.write(noBackup, "").catch(() => {});
  _ensuredDirs.add(dir);
  return dir;
}

async function generateThumbnail(sourcePath: string, cachePath: string): Promise<void> {
  await sharp(sourcePath)
    .resize({ width: THUMB_WIDTH, withoutEnlargement: true })
    .webp({ quality: THUMB_QUALITY })
    .toFile(cachePath);
}

export async function getThumbnail(targetDir: string, filename: string): Promise<{ path: string }> {
  const cacheDir = await ensureCacheDir(targetDir);
  const sourcePath = join(targetDir, filename);

  let s: Stats;
  try {
    s = await stat(sourcePath);
  } catch {
    return { path: sourcePath };
  }

  const cachePath = join(cacheDir, cacheKey(s.ino, s.size));

  // Cache hit — file exists with matching identity
  try {
    await stat(cachePath);
    return { path: cachePath };
  } catch {}

  try {
    await generateThumbnail(sourcePath, cachePath);
    return { path: cachePath };
  } catch {
    return { path: sourcePath };
  }
}

export async function clearCache(targetDir: string): Promise<void> {
  await rm(join(targetDir, CACHE_DIR), { recursive: true, force: true });
  _ensuredDirs.delete(join(targetDir, CACHE_DIR));
}

export async function preGenerateThumbnails(
  targetDir: string,
  filenames: string[],
  concurrency = 8,
): Promise<void> {
  const cacheDir = await ensureCacheDir(targetDir);
  let completed = 0;
  const total = filenames.length;
  let lastLog = Date.now();
  const validKeys = new Set<string>();

  const queue = [...filenames];

  async function processNext(): Promise<void> {
    while (queue.length > 0) {
      const filename = queue.shift()!;
      const sourcePath = join(targetDir, filename);

      try {
        const s = await stat(sourcePath);
        const key = cacheKey(s.ino, s.size);
        validKeys.add(key);
        const cachePath = join(cacheDir, key);

        // Skip if already cached (content-addressable → survives renames)
        try {
          await stat(cachePath);
        } catch {
          await generateThumbnail(sourcePath, cachePath);
        }
      } catch {
        // Skip failures silently
      }

      completed++;
      const now = Date.now();
      if (now - lastLog >= 5000 || completed === total) {
        console.log(`Thumbnails: ${completed}/${total}`);
        lastLog = now;
      }
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, total) }, () => processNext());
  await Promise.all(workers);
  console.log(`Thumbnails: ${total} done.`);

  // Remove orphaned cache entries (deleted files, old filename-keyed thumbnails, etc.)
  try {
    const entries = await readdir(cacheDir);
    const orphans = entries.filter((name) => name.endsWith(".webp") && !validKeys.has(name));
    if (orphans.length > 0) {
      await Promise.all(orphans.map((name) => unlink(join(cacheDir, name)).catch(() => {})));
      console.log(`Thumbnails: removed ${orphans.length} orphaned cache entries.`);
    }
  } catch {
    // Cache dir read failed — not critical
  }
}
