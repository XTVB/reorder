// Czkawka duplicate-compare session state: rename-based Trash moves (so undo
// can rename files back — Finder's osascript trash in trash.ts has no
// programmatic restore path) and a write-ahead session journal in
// .reorder-cache/ so groups and the undo stack survive a server restart.

import { access, mkdir, rename } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, extname, join } from "node:path";
import type { CzkawkaDirEntry, CzkawkaGroup } from "../shared/types.ts";
import { readJsonTolerant, writeJsonAtomic } from "./atomic-json.ts";
import { cacheDir, czkawkaSessionPath } from "./paths.ts";

const TRASH_DIR = join(homedir(), ".Trash");
const MAX_UNDO_ENTRIES = 50;

export interface CzkawkaTrashEntry {
  /** Original absolute path (restore destination). */
  path: string;
  /** Where the file landed in ~/.Trash. */
  trashPath: string;
}

export interface CzkawkaActionEntry {
  snapshotBefore: CzkawkaGroup[];
  /** In execution order; undo restores in reverse. */
  trashed: CzkawkaTrashEntry[];
  /** Files created by copy-replace that undo must remove before restoring. */
  copiedTargets: string[];
  deletedCount: number;
}

export interface CzkawkaSessionData {
  groups: CzkawkaGroup[];
  computeTimeMs: number;
  undoStack: CzkawkaActionEntry[];
  /** Directories in the last/current comparison; always includes targetDir. */
  dirs: CzkawkaDirEntry[];
}

export async function loadCzkawkaSession(targetDir: string): Promise<CzkawkaSessionData> {
  const data = await readJsonTolerant<Partial<CzkawkaSessionData>>(
    czkawkaSessionPath(targetDir),
    {},
  );
  const dirs = Array.isArray(data.dirs) ? data.dirs : [];
  if (!dirs.some((d) => d.path === targetDir)) {
    dirs.unshift({ path: targetDir, reference: false });
  }
  return {
    groups: Array.isArray(data.groups) ? data.groups : [],
    computeTimeMs: data.computeTimeMs ?? 0,
    undoStack: Array.isArray(data.undoStack) ? data.undoStack : [],
    dirs,
  };
}

export async function saveCzkawkaSession(
  targetDir: string,
  session: CzkawkaSessionData,
): Promise<void> {
  if (session.undoStack.length > MAX_UNDO_ENTRIES) {
    session.undoStack = session.undoStack.slice(-MAX_UNDO_ENTRIES);
  }
  await mkdir(cacheDir(targetDir), { recursive: true });
  await writeJsonAtomic(czkawkaSessionPath(targetDir), session, { pretty: false, atomic: true });
}

/** Rename one file into ~/.Trash, suffixing on name collisions. */
async function moveToTrash(absPath: string): Promise<string> {
  const filename = basename(absPath);
  const ext = extname(filename);
  const base = basename(filename, ext);
  let dest = join(TRASH_DIR, filename);
  let counter = 1;
  while (true) {
    try {
      await access(dest);
    } catch {
      break; // destination free
    }
    dest = join(TRASH_DIR, `${base}_${counter}${ext}`);
    counter++;
    if (counter > 9999) throw new Error(`Too many Trash collisions for ${filename}`);
  }
  await rename(absPath, dest);
  return dest;
}

/** Move files (absolute paths) to ~/.Trash sequentially (collision probing
 * is stateful). Missing files are skipped. Returns entries in execution
 * order. */
export async function trashFilesRestorable(paths: string[]): Promise<CzkawkaTrashEntry[]> {
  const entries: CzkawkaTrashEntry[] = [];
  for (const abs of paths) {
    try {
      await access(abs);
    } catch {
      continue;
    }
    entries.push({ path: abs, trashPath: await moveToTrash(abs) });
  }
  return entries;
}

/** Rename trashed files back to their original paths, tolerating entries that
 * were already restored or removed from the Trash. */
export async function restoreFromTrash(entries: CzkawkaTrashEntry[]): Promise<void> {
  for (const e of entries) {
    try {
      await rename(e.trashPath, e.path);
    } catch {
      // already restored or manually emptied from Trash — nothing to do
    }
  }
}
