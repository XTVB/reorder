// Rename-related routes: preview, save, reorder-by-groups, undo, can-undo.
// Includes the post-rename remap helpers since they're only used here.

import type { ImageGroup, RenameMapping } from "../../client/types.ts";
import { invalidateClusterCache, pruneDanglingConstraints } from "../../cluster/index.ts";
import { remapContentHashes } from "../../fs/content-hashes.ts";
import {
  canUndo,
  computeRenames,
  executeRenames,
  listImages,
  loadGroups,
  undoRenames,
  withRenameLock,
  writeGroupsFile,
} from "../../fs/index.ts";
import { GROUPS_FILE } from "../../fs/paths.ts";
import { log, logData, logError } from "../../log.ts";
import { json } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

async function remapGroups(
  targetDir: string,
  renames: RenameMapping[],
): Promise<{ before: ImageGroup[]; after: ImageGroup[] }> {
  const groups = loadGroups(targetDir);
  if (groups.length === 0) return { before: [], after: [] };
  const renameMap = new Map(renames.map((r) => [r.from, r.to]));
  const remapped = groups.map((g) => ({
    ...g,
    images: g.images.map((fn) => renameMap.get(fn) ?? fn),
  }));
  await writeGroupsFile(targetDir, remapped);
  return { before: groups, after: remapped };
}

async function safeStep(
  label: string,
  step: string,
  fn: () => Promise<void>,
  warnings: string[],
): Promise<boolean> {
  try {
    await fn();
    return true;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logError(label, `${step} failed`, err);
    warnings.push(`${step} failed: ${msg}`);
    return false;
  }
}

async function remapAfterRename(
  targetDir: string,
  renames: RenameMapping[],
  label: string,
  warnings: string[],
) {
  const renameMap = new Map(renames.map((r) => [r.from, r.to]));
  await safeStep(
    label,
    "Group remapping",
    async () => {
      const { after } = await remapGroups(targetDir, renames);
      log(label, `Remapped groups: ${after.length} groups`);
      logData(
        label,
        `Groups (post-${label})`,
        after.map((g) => `  ${g.name}: [${g.images.join(", ")}]`).join("\n"),
      );
    },
    warnings,
  );
  await safeStep(
    label,
    "Content hashes remapping",
    async () => {
      await remapContentHashes(targetDir, renameMap);
      log(label, "Remapped content_hashes.json");
    },
    warnings,
  );
  await safeStep(label, "Constraint pruning", () => pruneDanglingConstraints(targetDir), warnings);
  invalidateClusterCache();
}

export { remapGroups };

export const renameRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;

  if (path === "/api/preview" && req.method === "POST") {
    const body = (await req.json()) as { order: string[] };
    const renames = computeRenames(body.order);
    return json({ renames });
  }

  if (path === "/api/save" && req.method === "POST") {
    const body = (await req.json()) as { order: string[]; groups?: ImageGroup[] };
    return withRenameLock(async () => {
      const t0 = Date.now();
      log("save", `Received save request: ${body.order.length} files in order`);
      logData("save", "Input order", body.order.join("\n"));
      const warnings: string[] = [];

      if (body.groups) {
        const groupSummary = body.groups
          .map((g) => `  ${g.name}: [${g.images.join(", ")}]`)
          .join("\n");
        log("save", `Writing ${body.groups.length} groups to disk before rename`);
        logData("save", "Groups (pre-rename)", groupSummary);
        await writeGroupsFile(targetDir, body.groups);
      }

      log("save", "Executing filesystem renames...");
      const renames = await executeRenames(targetDir, body.order);
      const effective = renames.filter((r) => r.from !== r.to);
      log(
        "save",
        `Filesystem renames complete: ${effective.length} changed, ${renames.length - effective.length} unchanged`,
      );
      logData(
        "save",
        "All rename mappings",
        renames
          .map((r) => (r.from === r.to ? `  ${r.from} (unchanged)` : `  ${r.from} → ${r.to}`))
          .join("\n"),
      );

      await remapAfterRename(targetDir, renames, "save", warnings);

      const elapsed = Date.now() - t0;
      log(
        "save",
        `Complete in ${elapsed}ms — ${effective.length} files renamed${warnings.length > 0 ? `, ${warnings.length} warning(s)` : ""}`,
      );
      return json({ success: true, renames, warnings });
    });
  }

  if (path === "/api/reorder-by-groups" && req.method === "POST") {
    return withRenameLock(async () => {
      const t0 = Date.now();
      const label = "reorder-by-groups";
      log(label, "Received reorder-by-groups request");
      const warnings: string[] = [];

      const groups = loadGroups(targetDir);
      const diskImages = await listImages(targetDir);
      const diskSet = new Set(diskImages);

      log(
        label,
        `Loaded ${groups.length} groups from ${GROUPS_FILE}, ${diskImages.length} images on disk`,
      );

      const seen = new Set<string>();
      const order: string[] = [];
      const missing: string[] = [];
      let groupedCount = 0;

      for (const g of groups) {
        for (const fn of g.images) {
          if (seen.has(fn)) continue;
          if (!diskSet.has(fn)) {
            missing.push(`${g.name}: ${fn}`);
            continue;
          }
          seen.add(fn);
          order.push(fn);
          groupedCount++;
        }
      }

      for (const fn of diskImages) {
        if (seen.has(fn)) continue;
        seen.add(fn);
        order.push(fn);
      }

      const ungroupedCount = order.length - groupedCount;

      if (missing.length > 0) {
        warnings.push(`${missing.length} group member(s) not found on disk (skipped)`);
        logData(label, "Missing group members", missing.join("\n"));
      }

      log(
        label,
        `Computed order: ${order.length} files (${groupedCount} grouped, ${ungroupedCount} ungrouped at end)`,
      );
      logData(label, "Input order", order.join("\n"));

      log(label, "Executing filesystem renames...");
      const renames = await executeRenames(targetDir, order);
      const effective = renames.filter((r) => r.from !== r.to);
      log(
        label,
        `Filesystem renames complete: ${effective.length} changed, ${renames.length - effective.length} unchanged`,
      );
      logData(
        label,
        "All rename mappings",
        renames
          .map((r) => (r.from === r.to ? `  ${r.from} (unchanged)` : `  ${r.from} → ${r.to}`))
          .join("\n"),
      );

      await remapAfterRename(targetDir, renames, label, warnings);

      const elapsed = Date.now() - t0;
      log(
        label,
        `Complete in ${elapsed}ms — ${effective.length} files renamed${warnings.length > 0 ? `, ${warnings.length} warning(s)` : ""}`,
      );
      return json({ success: true, renames, warnings });
    });
  }

  if (path === "/api/undo" && req.method === "POST") {
    return withRenameLock(async () => {
      const t0 = Date.now();
      log("undo", "Received undo request");
      const warnings: string[] = [];

      const renames = await undoRenames(targetDir);
      const effective = renames.filter((r) => r.from !== r.to);
      log("undo", `Filesystem undo complete: ${effective.length} files reversed`);
      logData(
        "undo",
        "All undo mappings",
        renames
          .map((r) => (r.from === r.to ? `  ${r.from} (unchanged)` : `  ${r.from} → ${r.to}`))
          .join("\n"),
      );

      await remapAfterRename(targetDir, renames, "undo", warnings);

      const elapsed = Date.now() - t0;
      log(
        "undo",
        `Complete in ${elapsed}ms — ${effective.length} files reversed${warnings.length > 0 ? `, ${warnings.length} warning(s)` : ""}`,
      );
      return json({ success: true, renames, warnings });
    });
  }

  if (path === "/api/can-undo" && req.method === "GET") {
    return withRenameLock(async () => {
      const available = await canUndo(targetDir);
      return json({ canUndo: available });
    });
  }

  return null;
};
