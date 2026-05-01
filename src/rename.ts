import { access, constants, mkdir, readdir, rename, rmdir, unlink } from "node:fs/promises";
import { extname, join } from "node:path";

const IMAGE_EXTENSIONS = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".gif",
  ".webp",
  ".svg",
  ".avif",
  ".bmp",
  ".tiff",
  ".tif",
]);

export interface RenameMapping {
  from: string;
  to: string;
}

interface ReorderHistory {
  batchId?: string;
  timestamp: string;
  renames: RenameMapping[];
  originalTags?: Record<string, unknown>;
}

const HISTORY_FILE = ".reorder-history.json";
const HISTORY_PREV_FILE = ".reorder-history.prev.json";
const TAGS_FILE = "tags.json";
const TEMP_PREFIX = "__reorder_tmp_";
const PENDING_FILE = ".reorder-pending.json";

interface PendingManifest {
  batchId: string;
  mappings: RenameMapping[];
  timestamp: string;
}

// Serialize all filesystem-mutating operations
let _renameLock: Promise<unknown> = Promise.resolve();

export function withRenameLock<T>(fn: () => Promise<T>): Promise<T> {
  const prev = _renameLock;
  let resolve: () => void;
  _renameLock = new Promise<void>((r) => {
    resolve = r;
  });
  return prev.then(fn).finally(() => resolve!());
}

export function isImageFile(filename: string): boolean {
  if (filename.startsWith(".")) return false;
  return IMAGE_EXTENSIONS.has(extname(filename).toLowerCase());
}

export async function listImages(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  return entries
    .filter((e) => (e.isFile() || e.isSymbolicLink()) && isImageFile(e.name))
    .map((e) => e.name)
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

/**
 * Extract the title portion from a numbered filename like "003 - Beach Sunset.jpg".
 * Returns the " - title" part (including separator) or empty string if no title.
 */
function extractTitle(filename: string): string {
  const name = filename.slice(0, filename.length - extname(filename).length);
  const match = name.match(/^\d+(\s*-\s*.+)$/);
  return match?.[1] ?? "";
}

export function computeRenames(order: string[]): RenameMapping[] {
  const seen = new Set<string>();
  for (const fn of order) {
    if (seen.has(fn)) throw new Error(`Duplicate filename in order: ${fn}`);
    seen.add(fn);
  }
  const padLen = Math.max(3, String(order.length).length);
  return order.map((filename, i) => {
    const title = extractTitle(filename);
    const num = String(i + 1).padStart(padLen, "0");
    return {
      from: filename,
      to: `${num}${title}${extname(filename).toLowerCase()}`,
    };
  });
}

async function twoPhaseRename(dir: string, mappings: RenameMapping[]): Promise<string> {
  const id = crypto.randomUUID().slice(0, 8);

  // Write manifest before touching any files
  const manifest: PendingManifest = {
    batchId: id,
    mappings,
    timestamp: new Date().toISOString(),
  };
  await Bun.write(join(dir, PENDING_FILE), JSON.stringify(manifest, null, 2));

  // Step 1: rename to temp names
  const tempNames: { temp: string; final: string }[] = [];
  for (const { from, to } of mappings) {
    const temp = `${TEMP_PREFIX}${id}_${to}`;
    await rename(join(dir, from), join(dir, temp));
    tempNames.push({ temp, final: to });
  }

  // Step 2: rename temp names to final names
  for (const { temp, final } of tempNames) {
    await rename(join(dir, temp), join(dir, final));
  }

  // Clean up manifest on success
  await unlink(join(dir, PENDING_FILE)).catch(() => {});

  return id;
}

async function readTagsJson(dir: string): Promise<Record<string, unknown> | null> {
  try {
    const raw = await Bun.file(join(dir, TAGS_FILE)).text();
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

async function writeRemappedTags(
  dir: string,
  tags: Record<string, unknown>,
  mappings: RenameMapping[],
): Promise<void> {
  const renameMap = new Map(mappings.map((m) => [m.from, m.to]));
  const remapped = Object.entries(tags).map(
    ([key, value]) => [renameMap.get(key) ?? key, value] as const,
  );

  // Sort so image keys (e.g. 001.jpg) appear in order; non-image keys first
  remapped.sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }));

  await Bun.write(join(dir, TAGS_FILE), JSON.stringify(Object.fromEntries(remapped), null, 2));
}

async function assertFilesExist(dir: string, paths: string[], context: string): Promise<void> {
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

export async function executeRenames(dir: string, order: string[]): Promise<RenameMapping[]> {
  // Check write access
  try {
    await access(dir, constants.W_OK);
  } catch {
    throw new Error(`No write permission for directory: ${dir}`);
  }

  const mappings = computeRenames(order);

  // Skip files that wouldn't change
  const effectiveMappings = mappings.filter((m) => m.from !== m.to);

  if (effectiveMappings.length === 0) {
    return mappings;
  }

  await assertFilesExist(
    dir,
    effectiveMappings.map((m) => m.from),
    "rename",
  );

  // Snapshot tags.json before renaming (for undo)
  const originalTags = await readTagsJson(dir);

  const batchId = await twoPhaseRename(dir, effectiveMappings);

  // Update tags.json keys to match new filenames
  if (originalTags) {
    await writeRemappedTags(dir, originalTags, mappings);
  }

  // Back up existing history before overwriting
  const historyPath = join(dir, HISTORY_FILE);
  const historyPrevPath = join(dir, HISTORY_PREV_FILE);
  try {
    await access(historyPath, constants.F_OK);
    await rename(historyPath, historyPrevPath);
  } catch {
    // No existing history to back up
  }

  // Write history manifest for undo
  const history: ReorderHistory = {
    batchId,
    timestamp: new Date().toISOString(),
    renames: mappings,
    ...(originalTags && { originalTags }),
  };
  await Bun.write(historyPath, JSON.stringify(history, null, 2));

  return mappings;
}

export async function canUndo(dir: string): Promise<boolean> {
  try {
    await access(join(dir, HISTORY_FILE), constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

export async function undoRenames(dir: string): Promise<RenameMapping[]> {
  const historyPath = join(dir, HISTORY_FILE);
  let history: ReorderHistory;

  try {
    const raw = await Bun.file(historyPath).text();
    history = JSON.parse(raw);
  } catch {
    throw new Error("No undo history found");
  }

  // Reverse mapping: to → from
  const reverseMappings = history.renames.map((m) => ({
    from: m.to,
    to: m.from,
  }));

  // Only rename files that actually changed
  const effectiveReverse = reverseMappings.filter((m) => m.from !== m.to);

  if (effectiveReverse.length > 0) {
    await twoPhaseRename(dir, effectiveReverse);
  }

  // Restore original tags.json if it was saved
  if (history.originalTags) {
    await Bun.write(join(dir, TAGS_FILE), JSON.stringify(history.originalTags, null, 2));
  }

  // Remove history file and backup
  await unlink(historyPath);
  await unlink(join(dir, HISTORY_PREV_FILE)).catch(() => {});

  return reverseMappings;
}

/* ------------------------------------------------------------------ */
/*  Organize groups into subfolders                                    */
/* ------------------------------------------------------------------ */

export interface OrganizeGroup {
  name: string;
  images: string[];
}

export interface OrganizeMapping {
  folder: string;
  files: RenameMapping[];
}

export interface OrganizeOptions {
  numbered?: boolean;
}

export function computeOrganize(
  groups: OrganizeGroup[],
  imageOrder: string[],
  opts: OrganizeOptions = {},
): OrganizeMapping[] {
  const numbered = opts.numbered ?? true;
  // Determine group order based on first appearance in imageOrder
  const posMap = new Map(imageOrder.map((fn, i) => [fn, i]));
  const sorted = [...groups].sort((a, b) => {
    const aMin = Math.min(...a.images.map((fn) => posMap.get(fn) ?? Infinity));
    const bMin = Math.min(...b.images.map((fn) => posMap.get(fn) ?? Infinity));
    return aMin - bMin;
  });

  const folderPadLen = Math.max(3, String(sorted.length).length);
  return sorted.map((g, i) => {
    const folderNum = String(i + 1).padStart(folderPadLen, "0");
    const filePadLen = Math.max(3, String(g.images.length).length);
    const files = g.images.map((filename, j) => {
      const title = extractTitle(filename);
      const num = String(j + 1).padStart(filePadLen, "0");
      return {
        from: filename,
        to: `${num}${title}${extname(filename).toLowerCase()}`,
      };
    });
    return {
      folder: numbered ? `${folderNum} - ${g.name}` : g.name,
      files,
    };
  });
}

export async function executeOrganize(
  dir: string,
  groups: OrganizeGroup[],
  imageOrder: string[],
  opts: OrganizeOptions = {},
): Promise<OrganizeMapping[]> {
  try {
    await access(dir, constants.W_OK);
  } catch {
    throw new Error(`No write permission for directory: ${dir}`);
  }

  const mappings = computeOrganize(groups, imageOrder, opts);

  await Promise.all(
    mappings.map(async ({ folder, files }) => {
      const subdir = join(dir, folder);
      await mkdir(subdir, { recursive: true });
      await Promise.all(files.map(({ from, to }) => rename(join(dir, from), join(subdir, to))));
    }),
  );

  return mappings;
}

/* ------------------------------------------------------------------ */
/*  Move files to macOS Trash via Finder (preserves "Put Back")        */
/* ------------------------------------------------------------------ */

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
    await Bun.write(join(dir, TAGS_FILE), JSON.stringify(remaining, null, 2));
  }

  return { deleted: present, missing };
}

/* ------------------------------------------------------------------ */
/*  Recovery: complete stalled two-phase renames using manifest         */
/* ------------------------------------------------------------------ */

export interface RecoveryResult {
  status: "completed" | "orphaned" | "none";
  completed: number;
  message: string;
  /** Mappings applied during recovery — allows caller to remap dependent data (e.g. groups) */
  mappings?: RenameMapping[];
}

/**
 * Check for and resolve interrupted two-phase renames on startup.
 *
 * Handles all crash points:
 * - Step 1 incomplete (some temps, some originals) → complete step 1, then step 2
 * - Step 1 complete, step 2 incomplete (all temps) → complete step 2
 * - Step 2 complete (stale manifest) → remove manifest
 * - No manifest + temps → orphaned, warn
 */
export async function recoverPendingRename(dir: string): Promise<RecoveryResult> {
  // Check for manifest
  let manifest: PendingManifest | null = null;
  try {
    const raw = await Bun.file(join(dir, PENDING_FILE)).text();
    manifest = JSON.parse(raw);
  } catch {
    // No manifest
  }

  // Check for any temp files on disk
  const entries = await readdir(dir, { withFileTypes: true });
  const tempFiles = entries
    .filter((e) => e.isFile() && e.name.startsWith(TEMP_PREFIX))
    .map((e) => e.name);

  if (!manifest && tempFiles.length === 0) {
    return { status: "none", completed: 0, message: "" };
  }

  if (!manifest && tempFiles.length > 0) {
    return {
      status: "orphaned",
      completed: 0,
      message: `Found ${tempFiles.length} orphaned temp files with no manifest — manual recovery needed`,
    };
  }

  // We have a manifest
  const { batchId, mappings } = manifest!;
  const batchPrefix = `${TEMP_PREFIX}${batchId}_`;
  const batchTemps = new Set(tempFiles.filter((f) => f.startsWith(batchPrefix)));

  if (batchTemps.size === 0) {
    // No matching temps — check if step 2 already completed (all target files exist)
    const existResults = await Promise.all(
      mappings.map(({ to }) => Bun.file(join(dir, to)).exists()),
    );
    const missing = mappings.filter((_, i) => !existResults[i]).map((m) => m.to);

    if (missing.length === 0) {
      console.log(
        `[recovery] Stale manifest for batch ${batchId} — all targets exist. Removing manifest.`,
      );
      await unlink(join(dir, PENDING_FILE)).catch(() => {});
      return {
        status: "completed",
        completed: 0,
        message: `Batch ${batchId} already completed — removed stale manifest`,
      };
    }

    return {
      status: "orphaned",
      completed: 0,
      message: `Manifest for batch ${batchId} exists but no temps and ${missing.length} target files missing (${missing.slice(0, 5).join(", ")}${missing.length > 5 ? "..." : ""}) — manual recovery needed`,
    };
  }

  // Some temps exist — complete both steps for any incomplete mappings
  let step1Completed = 0;
  let step2Completed = 0;

  // Step 1: rename any remaining originals to their temp names
  for (const { from, to } of mappings) {
    const temp = `${batchPrefix}${to}`;
    if (!batchTemps.has(temp)) {
      // This mapping's temp doesn't exist — try to complete step 1
      try {
        await rename(join(dir, from), join(dir, temp));
        step1Completed++;
      } catch {
        // Original already gone (step 2 may have already placed its final name)
      }
    }
  }

  // Step 2: rename all temps to final names
  for (const { to } of mappings) {
    const temp = `${batchPrefix}${to}`;
    try {
      await rename(join(dir, temp), join(dir, to));
      step2Completed++;
    } catch {
      // Already at final name
    }
  }

  await unlink(join(dir, PENDING_FILE)).catch(() => {});

  // Check for orphaned temps from OTHER batches
  const otherTemps = tempFiles.filter((f) => !f.startsWith(batchPrefix));
  const otherWarning =
    otherTemps.length > 0
      ? ` (${otherTemps.length} orphaned temp files from other batches remain)`
      : "";

  const total = step1Completed + step2Completed;
  return {
    status: "completed",
    completed: total,
    message: `Recovered batch ${batchId}: ${step1Completed} step-1 + ${step2Completed} step-2 renames${otherWarning}`,
    mappings,
  };
}

/* ------------------------------------------------------------------ */
/*  Folder mode: treat subdirectories as groups                        */
/* ------------------------------------------------------------------ */

export interface FolderGroup {
  name: string; // subdirectory name on disk
  images: string[]; // bare filenames within that subdir
}

export interface FolderData {
  folders: FolderGroup[];
  rootImages: string[];
}

const HIDDEN_DIRS = new Set([".reorder-cache"]);

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

/* ------------------------------------------------------------------ */
/*  Folder-mode save: atomic application of desired folder structure   */
/* ------------------------------------------------------------------ */

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

const FOLDER_PENDING_FILE = ".reorder-folders-pending.json";

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
      const srcPath = folder.images[ii]!; // compound path like "OldFolder/001 - Title.jpg"
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
  await Bun.write(join(dir, FOLDER_PENDING_FILE), JSON.stringify(manifest, null, 2));

  // --- Phase 1: move all source files to temp names in root ---
  const tempMap = new Map<string, string>(); // from → tempName
  for (const { from } of effectiveMoves) {
    if (tempMap.has(from)) continue; // same source in multiple moves shouldn't happen, but guard
    const temp = `${TEMP_PREFIX}${batchId}_${tempMap.size}${extname(from)}`;
    await rename(join(dir, from), join(dir, temp));
    tempMap.set(from, temp);
  }
  _log("folders-save", `Phase 1 complete: ${tempMap.size} files moved to temps`);

  // --- Phase 2: create target dirs and move temps to final destinations ---
  for (const folder of foldersToCreate) {
    await mkdir(join(dir, folder), { recursive: true });
  }
  // Also ensure existing target folders exist (they might have been renamed)
  for (const folder of targetFolders) {
    if (!foldersToCreate.includes(folder)) {
      await mkdir(join(dir, folder), { recursive: true });
    }
  }

  for (const { from, to } of effectiveMoves) {
    const temp = tempMap.get(from)!;
    // Ensure parent dir exists (for files going into folders)
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
      // Not empty (has non-image files) — leave it
      _log("folders-save", `Could not remove folder (not empty?): ${folder}`);
    }
  }

  // --- Remove manifest ---
  await unlink(join(dir, FOLDER_PENDING_FILE)).catch(() => {});
  _log("folders-save", `Done — batch ${batchId}`);

  return { moves, foldersCreated: foldersToCreate, foldersRemoved: foldersToRemove };
}
