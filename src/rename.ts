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
  const tempNames: { temp: string; final: string }[] = [];
  const id = crypto.randomUUID().slice(0, 8);

  // Step 1: rename to temp names
  for (const { from, to } of mappings) {
    const temp = `${TEMP_PREFIX}${id}_${to}`;
    await rename(join(dir, from), join(dir, temp));
    tempNames.push({ temp, final: to });
  }

  // Pre-index for O(1) rollback lookups
  const finalToOriginal = new Map(mappings.map((m) => [m.to, m.from]));

  // Step 2: rename temp names to final names
  for (const { temp, final } of tempNames) {
    try {
      await rename(join(dir, temp), join(dir, final));
    } catch (err) {
      // Attempt rollback of remaining temps
      console.error(`Failed renaming ${temp} → ${final}, attempting rollback...`);
      for (const t of tempNames) {
        try {
          await access(join(dir, t.temp), constants.F_OK);
          const orig = finalToOriginal.get(t.final);
          if (orig) {
            await rename(join(dir, t.temp), join(dir, orig));
          }
        } catch {
          // temp file already renamed or doesn't exist
        }
      }
      throw err;
    }
  }
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
  files: string[];
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

  const padLen = Math.max(3, String(sorted.length).length);
  return sorted.map((g, i) => {
    const num = String(i + 1).padStart(padLen, "0");
    return {
      folder: `${num} - ${g.name}`,
      files: g.images,
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
    await Promise.all(files.map((filename) =>
      rename(join(dir, filename), join(subdir, filename))
    ));
  }));

  return mappings;
}
