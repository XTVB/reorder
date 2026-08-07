// Czkawka duplicate-compare session state: rename-based Trash moves (so undo
// can rename files back — Finder's osascript trash in trash.ts has no
// programmatic restore path) and a write-ahead session journal in
// .reorder-cache/ so groups and the undo stack survive a server restart.

import { access, mkdir, rename } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, extname, join } from "node:path";
import type { CzkawkaDirEntry, CzkawkaGroup, CzkawkaImage } from "../shared/types.ts";
import { readJsonTolerant, writeJsonAtomic } from "./atomic-json.ts";
import { cacheDir, czkawkaSessionPath } from "./paths.ts";

const TRASH_DIR = join(homedir(), ".Trash");

export interface CzkawkaTrashEntry {
  /** Original absolute path (restore destination). */
  path: string;
  /** Where the file landed in ~/.Trash. */
  trashPath: string;
}

/**
 * The inverse of one action's edit to `groups`.
 *
 * Actions only ever *remove* — images from a group, or whole groups (resolved,
 * or left with <2 images). So an undo step is fully described by what went away
 * and where it sat, which is O(changed) rather than O(all groups).
 */
export interface CzkawkaGroupsPatch {
  /** `index` is the group's position in the old array. */
  removedGroups: { index: number; group: CzkawkaGroup }[];
  /**
   * Images dropped from groups that survived. `index` is the group's index in
   * the *new* array; `at` is the image's position in the old group.
   */
  removedImages: { index: number; at: number; image: CzkawkaImage }[];
}

export interface CzkawkaActionEntry {
  patch?: CzkawkaGroupsPatch;
  /** Legacy: full pre-action snapshot from journals written before `patch`.
   * `groupsBeforeEntry` handles either. */
  snapshotBefore?: CzkawkaGroup[];
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
  /** Directories in the last/current comparison. Seeded with targetDir for a
   * brand-new session, but the user may remove it later. */
  dirs: CzkawkaDirEntry[];
}

/** Stable identity for an image within a comparison (paths are unique). */
export const imageKey = (i: CzkawkaImage) => join(i.dir, i.filename);

/**
 * Diff `before` -> `after` into the patch undo needs. Assumes the edit only
 * removed images/groups and preserved relative order, which is what
 * applyOperations does.
 */
export function buildGroupsPatch(
  before: CzkawkaGroup[],
  after: CzkawkaGroup[],
): CzkawkaGroupsPatch {
  const removedGroups: CzkawkaGroupsPatch["removedGroups"] = [];
  const removedImages: CzkawkaGroupsPatch["removedImages"] = [];

  // Walk both arrays in order; a surviving group keeps its images' relative
  // order, so matching on the first shared image identifies it unambiguously.
  const afterKeys = after.map((g) => new Set(g.images.map(imageKey)));
  let ai = 0;
  for (let bi = 0; bi < before.length; bi++) {
    const beforeGroup = before[bi]!;
    const survives =
      ai < after.length && beforeGroup.images.some((img) => afterKeys[ai]!.has(imageKey(img)));
    if (!survives) {
      removedGroups.push({ index: bi, group: beforeGroup });
      continue;
    }
    const keptKeys = afterKeys[ai]!;
    beforeGroup.images.forEach((img, at) => {
      if (!keptKeys.has(imageKey(img))) removedImages.push({ index: ai, at, image: img });
    });
    ai++;
  }
  return { removedGroups, removedImages };
}

export function applyGroupsPatch(after: CzkawkaGroup[], patch: CzkawkaGroupsPatch): CzkawkaGroup[] {
  // Re-insert images into their surviving groups first, so group indices still
  // refer to the post-action array.
  const groups = after.map((g) => ({ images: [...g.images] }));
  const byGroup = new Map<number, { at: number; image: CzkawkaImage }[]>();
  for (const r of patch.removedImages) {
    const list = byGroup.get(r.index);
    if (list) list.push(r);
    else byGroup.set(r.index, [r]);
  }
  for (const [index, items] of byGroup) {
    const group = groups[index];
    if (!group) continue; // defensive: stale patch
    for (const { at, image } of [...items].sort((a, b) => a.at - b.at)) {
      group.images.splice(Math.min(at, group.images.length), 0, image);
    }
  }
  // Ascending, so earlier indices stay correct as later ones fill in.
  for (const { index, group } of [...patch.removedGroups].sort((a, b) => a.index - b.index)) {
    groups.splice(Math.min(index, groups.length), 0, { images: [...group.images] });
  }
  return groups;
}

/** The pre-action groups for an undo entry, from either journal format. */
export function groupsBeforeEntry(
  current: CzkawkaGroup[],
  entry: CzkawkaActionEntry,
): CzkawkaGroup[] {
  if (entry.patch) return applyGroupsPatch(current, entry.patch);
  return entry.snapshotBefore ?? current;
}

export async function loadCzkawkaSession(targetDir: string): Promise<CzkawkaSessionData> {
  const data = await readJsonTolerant<Partial<CzkawkaSessionData>>(
    czkawkaSessionPath(targetDir),
    {},
  );
  // A brand-new session defaults to the launch dir; a saved session is
  // respected verbatim (the user may have removed the launch dir on purpose).
  const dirs: CzkawkaDirEntry[] = Array.isArray(data.dirs)
    ? data.dirs.map((d) => ({
        path: d.path,
        reference: d.reference === true,
        recursive: d.recursive === true,
      }))
    : [{ path: targetDir, reference: false, recursive: false }];
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
