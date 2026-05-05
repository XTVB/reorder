// Atomic JSON writers with optional .bak backup. readJsonTolerant returns a
// fallback when the file is missing or unparseable.

import { rename } from "node:fs/promises";

interface WriteOpts {
  /** Copy the existing file to `${path}.bak.json` (replacing extension) before
   * writing. Default: false. */
  backup?: boolean;
  /** Pretty-print with 2-space indent. Default: true. */
  pretty?: boolean;
  /** Use a tmp-then-rename to make the write atomic from concurrent readers'
   * perspective. Default: false (Bun.write is already a single syscall write,
   * but it's not atomic across the whole pipeline if a reader could see the
   * intermediate empty state). */
  atomic?: boolean;
}

/**
 * Compute the backup path for a given JSON file: `foo.json` → `foo.bak.json`.
 */
export function backupPath(path: string): string {
  if (path.endsWith(".json")) return `${path.slice(0, -5)}.bak.json`;
  return `${path}.bak`;
}

/**
 * Write JSON to disk. Optionally back up the existing file first, and
 * optionally write atomically via tmp+rename.
 */
export async function writeJsonAtomic(
  path: string,
  data: unknown,
  opts: WriteOpts = {},
): Promise<void> {
  const { backup = false, pretty = true, atomic = false } = opts;

  if (backup) {
    try {
      const existing = Bun.file(path);
      if (await existing.exists()) {
        await Bun.write(backupPath(path), existing);
      }
    } catch {
      // backup is best-effort
    }
  }

  const body = pretty ? JSON.stringify(data, null, 2) : JSON.stringify(data);

  if (atomic) {
    const tmpPath = `${path}.tmp`;
    await Bun.write(tmpPath, body);
    await rename(tmpPath, path);
  } else {
    await Bun.write(path, body);
  }
}

/**
 * Read and parse JSON, returning `fallback` when the file is missing,
 * unreadable, or unparseable.
 */
export async function readJsonTolerant<T>(path: string, fallback: T): Promise<T> {
  try {
    const f = Bun.file(path);
    if (!(await f.exists())) return fallback;
    return (await f.json()) as T;
  } catch {
    return fallback;
  }
}
