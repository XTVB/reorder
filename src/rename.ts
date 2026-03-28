import { readdir, rename, stat, access, constants, unlink, mkdir } from "node:fs/promises";
import { join, extname } from "node:path";

const IMAGE_EXTENSIONS = new Set([
  ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".avif", ".bmp", ".tiff", ".tif",
]);

export interface RenameMapping {
  from: string;
  to: string;
}

interface ReorderHistory {
  timestamp: string;
  renames: RenameMapping[];
  originalTags?: Record<string, unknown>;
}

const HISTORY_FILE = ".reorder-history.json";
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
  _renameLock = new Promise<void>((r) => { resolve = r; });
  return prev.then(fn).finally(() => resolve!());
}

export function isImageFile(filename: string): boolean {
  if (filename.startsWith(".")) return false;
  return IMAGE_EXTENSIONS.has(extname(filename).toLowerCase());
}

export async function listImages(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  return entries
    .filter((e) => e.isFile() && isImageFile(e.name))
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

async function twoPhaseRename(
  dir: string,
  mappings: RenameMapping[]
): Promise<void> {
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
}

async function readTagsJson(
  dir: string
): Promise<Record<string, unknown> | null> {
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
  mappings: RenameMapping[]
): Promise<void> {
  const renameMap = new Map(mappings.map((m) => [m.from, m.to]));
  const remapped = Object.entries(tags).map(
    ([key, value]) => [renameMap.get(key) ?? key, value] as const
  );

  // Sort so image keys (e.g. 001.jpg) appear in order; non-image keys first
  remapped.sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }));

  await Bun.write(join(dir, TAGS_FILE), JSON.stringify(Object.fromEntries(remapped), null, 2));
}

export async function executeRenames(
  dir: string,
  order: string[]
): Promise<RenameMapping[]> {
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

  // Snapshot tags.json before renaming (for undo)
  const originalTags = await readTagsJson(dir);

  await twoPhaseRename(dir, effectiveMappings);

  // Update tags.json keys to match new filenames
  if (originalTags) {
    await writeRemappedTags(dir, originalTags, mappings);
  }

  // Write history manifest for undo
  const history: ReorderHistory = {
    timestamp: new Date().toISOString(),
    renames: mappings,
    ...(originalTags && { originalTags }),
  };
  await Bun.write(join(dir, HISTORY_FILE), JSON.stringify(history, null, 2));

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
    await Bun.write(
      join(dir, TAGS_FILE),
      JSON.stringify(history.originalTags, null, 2)
    );
  }

  // Remove history file
  await unlink(historyPath);

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

export function computeOrganize(
  groups: OrganizeGroup[],
  imageOrder: string[]
): OrganizeMapping[] {
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
      folder: `${folderNum} - ${g.name}`,
      files,
    };
  });
}

export async function executeOrganize(
  dir: string,
  groups: OrganizeGroup[],
  imageOrder: string[]
): Promise<OrganizeMapping[]> {
  try {
    await access(dir, constants.W_OK);
  } catch {
    throw new Error(`No write permission for directory: ${dir}`);
  }

  const mappings = computeOrganize(groups, imageOrder);

  await Promise.all(mappings.map(async ({ folder, files }) => {
    const subdir = join(dir, folder);
    await mkdir(subdir, { recursive: true });
    await Promise.all(files.map(({ from, to }) =>
      rename(join(dir, from), join(subdir, to))
    ));
  }));

  return mappings;
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
      mappings.map(({ to }) => Bun.file(join(dir, to)).exists())
    );
    const missing = mappings.filter((_, i) => !existResults[i]).map((m) => m.to);

    if (missing.length === 0) {
      console.log(`[recovery] Stale manifest for batch ${batchId} — all targets exist. Removing manifest.`);
      await unlink(join(dir, PENDING_FILE)).catch(() => {});
      return { status: "completed", completed: 0, message: `Batch ${batchId} already completed — removed stale manifest` };
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
  const otherWarning = otherTemps.length > 0
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
