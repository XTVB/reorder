import { join, extname } from "node:path";
import { stat, mkdir } from "node:fs/promises";
import sharp from "sharp";

const CACHE_DIR = ".reorder-cache";
const THUMB_WIDTH = 400;
const THUMB_QUALITY = 80;

function thumbFilename(originalFilename: string): string {
  const name = originalFilename.slice(0, originalFilename.length - extname(originalFilename).length);
  return `${name}.webp`;
}

export function getCachePath(targetDir: string, filename: string): string {
  return join(targetDir, CACHE_DIR, thumbFilename(filename));
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

async function isFresh(sourcePath: string, cachePath: string): Promise<boolean> {
  try {
    const [src, cached] = await Promise.all([stat(sourcePath), stat(cachePath)]);
    return cached.mtimeMs >= src.mtimeMs;
  } catch {
    return false;
  }
}

export async function generateThumbnail(sourcePath: string, cachePath: string): Promise<void> {
  await sharp(sourcePath)
    .resize({ width: THUMB_WIDTH, withoutEnlargement: true })
    .webp({ quality: THUMB_QUALITY })
    .toFile(cachePath);
}

export async function getThumbnail(
  targetDir: string,
  filename: string
): Promise<{ path: string }> {
  await ensureCacheDir(targetDir);
  const sourcePath = join(targetDir, filename);
  const cachePath = getCachePath(targetDir, filename);

  if (await isFresh(sourcePath, cachePath)) {
    return { path: cachePath };
  }

  try {
    await generateThumbnail(sourcePath, cachePath);
    return { path: cachePath };
  } catch {
    return { path: sourcePath };
  }
}

export async function preGenerateThumbnails(
  targetDir: string,
  filenames: string[],
  concurrency = 8
): Promise<void> {
  await ensureCacheDir(targetDir);
  let completed = 0;
  const total = filenames.length;
  let lastLog = Date.now();

  // Simple semaphore
  let running = 0;
  const queue = [...filenames];

  async function processNext(): Promise<void> {
    while (queue.length > 0) {
      const filename = queue.shift()!;
      const sourcePath = join(targetDir, filename);
      const cachePath = getCachePath(targetDir, filename);

      try {
        if (await isFresh(sourcePath, cachePath)) {
          completed++;
          continue;
        }
        await generateThumbnail(sourcePath, cachePath);
      } catch {
        // Skip failures silently
      }
      completed++; // counts both successful generations and failures
      const now = Date.now();
      if (now - lastLog >= 5000 || completed === total) {
        console.log(`Thumbnails: ${completed}/${total} generated...`);
        lastLog = now;
      }
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, total) }, () => processNext());
  await Promise.all(workers);
  console.log(`Thumbnails: ${completed}/${total} done.`);
}
