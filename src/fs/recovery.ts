// Crash recovery for two-phase renames. Runs at server startup inside
// withRenameLock so client requests block until disk state is consistent.

import { readdir, rename, unlink } from "node:fs/promises";
import { join } from "node:path";
import { log } from "../log.ts";
import type { RenameMapping } from "../shared/types.ts";
import { pendingRenamePath, TEMP_PREFIX } from "./paths.ts";
import type { PendingManifest } from "./rename.ts";

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
  let manifest: PendingManifest | null = null;
  try {
    const raw = await Bun.file(pendingRenamePath(dir)).text();
    manifest = JSON.parse(raw);
  } catch {
    // No manifest
  }

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
      log(
        "recovery",
        `Stale manifest for batch ${batchId} — all targets exist. Removing manifest.`,
      );
      await unlink(pendingRenamePath(dir)).catch(() => {});
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

  await unlink(pendingRenamePath(dir)).catch(() => {});

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
