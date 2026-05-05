// Folder mode: treat subdirectories as groups. The save flow performs an
// atomic two-phase rename (everything to root temps first) so partial states
// can never collide on disk.

import { mkdir, readdir, rename, rmdir, unlink } from "node:fs/promises";
import { extname, join } from "node:path";
import type { FolderData } from "../shared/types.ts";
import { assertFilesExist, extractTitle } from "./helpers.ts";
import { listImages } from "./images.ts";
import { pendingFolderSavePath, TEMP_PREFIX } from "./paths.ts";

const HIDDEN_DIRS = new Set([".reorder-cache"]);

export type { FolderData };

export async function listSubdirectories(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  return entries
    .filter((e) => e.isDirectory() && !e.name.startsWith(".") && !HIDDEN_DIRS.has(e.name))
    .map((e) => e.name)
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

export async function listFolderData(dir: string): Promise<FolderData> {
  const subdirs = await listSubdirectories(dir);
  const rootImages = await listImages(dir);
  const folders = await Promise.all(
    subdirs.map(async (name) => ({
      name,
      images: await listImages(join(dir, name)),
    })),
  );
  return { folders, rootImages };
}

/**
 * The desired folder structure sent by the client.
 * Each image is identified by its original compound path (e.g. "OldFolder/001.jpg")
 * or bare filename for root images. The server resolves these to physical files.
 */
export interface FolderSaveRequest {
  folders: { title: string; images: string[] }[]; // images are original compound paths
  rootImages: string[]; // original bare filenames
  numbered?: boolean; // when false, target folders use bare title with no NNN- prefix (default true)
}

interface FolderMove {
  from: string; // relative to targetDir, e.g. "001 - Day 1/001.jpg" or "standalone.jpg"
  to: string; // relative to targetDir, e.g. "001 - Beach/003.jpg"
}

interface FolderSaveManifest {
  batchId: string;
  moves: FolderMove[];
  foldersToCreate: string[];
  foldersToRemove: string[];
  timestamp: string;
}

/**
 * Compute what the final filesystem state should look like, then execute
 * all moves atomically.
 *
 * Strategy:
 *   1. Write manifest describing every move
 *   2. Phase 1 — rename every source file to a temp name in the root dir
 *   3. Phase 2 — create target folders, rename temps to final destinations
 *   4. Remove now-empty old folders
 *   5. Delete manifest
 *
 * Moving everything to root temps first guarantees zero filename collisions
 * at any point, and recovery is simple: if temps exist, the manifest says
 * where they go.
 */
export async function executeFolderSave(
  dir: string,
  req: FolderSaveRequest,
  logFn?: (label: string, msg: string) => void,
): Promise<{ moves: FolderMove[]; foldersCreated: string[]; foldersRemoved: string[] }> {
  const _log = logFn ?? (() => {});

  // --- Compute target folder names ---
  const numbered = req.numbered ?? true;
  const folderPadLen = Math.max(3, String(req.folders.length).length);
  const targetFolders = req.folders.map((f, i) => {
    if (!numbered) return f.title;
    const num = String(i + 1).padStart(folderPadLen, "0");
    return `${num} - ${f.title}`;
  });

  // --- Compute every file move ---
  const moves: FolderMove[] = [];

  for (let fi = 0; fi < req.folders.length; fi++) {
    const folder = req.folders[fi]!;
    const destFolder = targetFolders[fi]!;
    const imgPadLen = Math.max(3, String(folder.images.length).length);

    for (let ii = 0; ii < folder.images.length; ii++) {
      const srcPath = folder.images[ii]!;
      const num = String(ii + 1).padStart(imgPadLen, "0");
      const title = extractTitle(
        srcPath.includes("/") ? srcPath.slice(srcPath.lastIndexOf("/") + 1) : srcPath,
      );
      const ext = extname(srcPath).toLowerCase();
      const destFile = `${num}${title}${ext}`;
      moves.push({ from: srcPath, to: `${destFolder}/${destFile}` });
    }
  }

  // Root images
  const rootPadLen = Math.max(3, String(req.rootImages.length).length);
  for (let i = 0; i < req.rootImages.length; i++) {
    const srcPath = req.rootImages[i]!;
    const num = String(i + 1).padStart(rootPadLen, "0");
    const title = extractTitle(
      srcPath.includes("/") ? srcPath.slice(srcPath.lastIndexOf("/") + 1) : srcPath,
    );
    const ext = extname(srcPath).toLowerCase();
    const destFile = `${num}${title}${ext}`;
    moves.push({ from: srcPath, to: destFile });
  }

  // Skip no-op moves
  const effectiveMoves = moves.filter((m) => m.from !== m.to);
  if (effectiveMoves.length === 0) {
    _log("folders-save", "No changes to apply");
    return { moves, foldersCreated: [], foldersRemoved: [] };
  }

  // --- Determine folders to create and remove ---
  const existingFolders = await listSubdirectories(dir);
  const existingSet = new Set(existingFolders);
  const targetSet = new Set(targetFolders);
  const foldersToCreate = targetFolders.filter((f) => !existingSet.has(f));
  const foldersToRemove = existingFolders.filter((f) => !targetSet.has(f));

  await assertFilesExist(
    dir,
    effectiveMoves.map((m) => m.from),
    "save folders",
  );

  _log(
    "folders-save",
    `${effectiveMoves.length} file moves, ${foldersToCreate.length} folders to create, ${foldersToRemove.length} to remove`,
  );

  // --- Write manifest ---
  const batchId = crypto.randomUUID().slice(0, 8);
  const manifest: FolderSaveManifest = {
    batchId,
    moves: effectiveMoves,
    foldersToCreate,
    foldersToRemove,
    timestamp: new Date().toISOString(),
  };
  await Bun.write(pendingFolderSavePath(dir), JSON.stringify(manifest, null, 2));

  // --- Phase 1: move all source files to temp names in root ---
  const tempMap = new Map<string, string>();
  for (const { from } of effectiveMoves) {
    if (tempMap.has(from)) continue;
    const temp = `${TEMP_PREFIX}${batchId}_${tempMap.size}${extname(from)}`;
    await rename(join(dir, from), join(dir, temp));
    tempMap.set(from, temp);
  }
  _log("folders-save", `Phase 1 complete: ${tempMap.size} files moved to temps`);

  // --- Phase 2: create target dirs and move temps to final destinations ---
  for (const folder of foldersToCreate) {
    await mkdir(join(dir, folder), { recursive: true });
  }
  for (const folder of targetFolders) {
    if (!foldersToCreate.includes(folder)) {
      await mkdir(join(dir, folder), { recursive: true });
    }
  }

  for (const { from, to } of effectiveMoves) {
    const temp = tempMap.get(from)!;
    const slashIdx = to.lastIndexOf("/");
    if (slashIdx >= 0) {
      await mkdir(join(dir, to.slice(0, slashIdx)), { recursive: true });
    }
    await rename(join(dir, temp), join(dir, to));
  }
  _log("folders-save", `Phase 2 complete: all files at final destinations`);

  // --- Remove empty old folders ---
  for (const folder of foldersToRemove) {
    try {
      await rmdir(join(dir, folder));
      _log("folders-save", `Removed empty folder: ${folder}`);
    } catch {
      _log("folders-save", `Could not remove folder (not empty?): ${folder}`);
    }
  }

  // --- Remove manifest ---
  await unlink(pendingFolderSavePath(dir)).catch(() => {});
  _log("folders-save", `Done — batch ${batchId}`);

  return { moves, foldersCreated: foldersToCreate, foldersRemoved: foldersToRemove };
}
