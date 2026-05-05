// Small filesystem helpers shared across the fs modules.

import { access, constants } from "node:fs/promises";
import { extname, join } from "node:path";

/**
 * Extract the title portion from a numbered filename like "003 - Beach Sunset.jpg".
 * Returns the " - title" part (including separator) or empty string if no title.
 */
export function extractTitle(filename: string): string {
  const name = filename.slice(0, filename.length - extname(filename).length);
  const match = name.match(/^\d+(\s*-\s*.+)$/);
  return match?.[1] ?? "";
}

/** Throw if any of the given paths (relative to dir) are missing. */
export async function assertFilesExist(
  dir: string,
  paths: string[],
  context: string,
): Promise<void> {
  const missing: string[] = [];
  await Promise.all(
    paths.map(async (p) => {
      try {
        await access(join(dir, p), constants.F_OK);
      } catch {
        missing.push(p);
      }
    }),
  );
  if (missing.length > 0) {
    throw new Error(
      `Cannot ${context}: ${missing.length} file(s) not found on disk (${missing.slice(0, 5).join(", ")}${missing.length > 5 ? ", ..." : ""}). Try refreshing first.`,
    );
  }
}

/** Throw if the directory isn't writable. */
export async function assertWritable(dir: string): Promise<void> {
  try {
    await access(dir, constants.W_OK);
  } catch {
    throw new Error(`No write permission for directory: ${dir}`);
  }
}
