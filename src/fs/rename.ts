// Two-phase rename + history + undo. Every mutating call MUST be wrapped in
// withRenameLock by the caller (server middleware).

import { access, constants, rename, unlink } from "node:fs/promises";
import { extname, join } from "node:path";
import type { RenameMapping } from "../shared/types.ts";
import { assertFilesExist, assertWritable, extractTitle } from "./helpers.ts";
import {
  HISTORY_FILE,
  HISTORY_PREV_FILE,
  historyPath,
  historyPrevPath,
  PENDING_RENAME_FILE,
  pendingRenamePath,
  TEMP_PREFIX,
} from "./paths.ts";
import { readTagsJson, writeRemappedTags } from "./tags.ts";

interface ReorderHistory {
  batchId?: string;
  timestamp: string;
  renames: RenameMapping[];
  originalTags?: Record<string, unknown>;
}

export interface PendingManifest {
  batchId: string;
  mappings: RenameMapping[];
  timestamp: string;
}

export type { RenameMapping };

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

export async function twoPhaseRename(dir: string, mappings: RenameMapping[]): Promise<string> {
  const id = crypto.randomUUID().slice(0, 8);

  // Write manifest before touching any files
  const manifest: PendingManifest = {
    batchId: id,
    mappings,
    timestamp: new Date().toISOString(),
  };
  await Bun.write(pendingRenamePath(dir), JSON.stringify(manifest, null, 2));

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
  await unlink(pendingRenamePath(dir)).catch(() => {});

  return id;
}

export async function executeRenames(dir: string, order: string[]): Promise<RenameMapping[]> {
  await assertWritable(dir);

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
  const histPath = historyPath(dir);
  const histPrev = historyPrevPath(dir);
  try {
    await access(histPath, constants.F_OK);
    await rename(histPath, histPrev);
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
  await Bun.write(histPath, JSON.stringify(history, null, 2));

  return mappings;
}

export async function canUndo(dir: string): Promise<boolean> {
  try {
    await access(historyPath(dir), constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

export async function undoRenames(dir: string): Promise<RenameMapping[]> {
  const histPath = historyPath(dir);
  let history: ReorderHistory;

  try {
    const raw = await Bun.file(histPath).text();
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
    const { tagsPath } = await import("./paths.ts");
    await Bun.write(tagsPath(dir), JSON.stringify(history.originalTags, null, 2));
  }

  // Remove history file and backup
  await unlink(histPath);
  await unlink(historyPrevPath(dir)).catch(() => {});

  return reverseMappings;
}

export type { ReorderHistory };
// Re-export internal constants used by recovery.ts
export { HISTORY_FILE, HISTORY_PREV_FILE, PENDING_RENAME_FILE, TEMP_PREFIX };
