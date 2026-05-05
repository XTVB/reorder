// Move files to macOS Trash via Finder (preserves "Put Back"). osascript
// invocation rather than node:fs because Finder's Trash is the user-visible
// behaviour we want; an `unlink` would delete permanently.

import { access, constants } from "node:fs/promises";
import { join } from "node:path";
import { tagsPath } from "./paths.ts";
import { readTagsJson } from "./tags.ts";

export interface DeleteResult {
  deleted: string[];
  missing: string[];
}

export async function executeDelete(dir: string, filenames: string[]): Promise<DeleteResult> {
  try {
    await access(dir, constants.W_OK);
  } catch {
    throw new Error(`No write permission for directory: ${dir}`);
  }

  const present: string[] = [];
  const missing: string[] = [];
  await Promise.all(
    filenames.map(async (fn) => {
      try {
        await access(join(dir, fn), constants.F_OK);
        present.push(fn);
      } catch {
        missing.push(fn);
      }
    }),
  );
  if (present.length === 0) return { deleted: [], missing };

  // Use Finder via osascript so files land in Trash with "Put Back" support.
  // Pass all paths in a single `delete {…}` call so Finder treats it as one
  // operation (single undo, single Trash sound, no per-file progress flicker).
  const fileList = present
    .map((fn) => {
      const abs = join(dir, fn).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
      return `POSIX file "${abs}"`;
    })
    .join(", ");
  const args: string[] = [
    "osascript",
    "-e",
    'tell application "Finder"',
    "-e",
    `delete {${fileList}}`,
    "-e",
    "end tell",
  ];

  const proc = Bun.spawn(args, { stdout: "pipe", stderr: "pipe" });
  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    const stderr = await new Response(proc.stderr).text();
    throw new Error(`Trash delete failed (osascript exit ${exitCode}): ${stderr.trim()}`);
  }

  const tags = await readTagsJson(dir);
  if (tags) {
    const deletedSet = new Set(present);
    const remaining = Object.fromEntries(
      Object.entries(tags).filter(([key]) => !deletedSet.has(key)),
    );
    await Bun.write(tagsPath(dir), JSON.stringify(remaining, null, 2));
  }

  return { deleted: present, missing };
}
