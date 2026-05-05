// Read/write `.reorder-groups.json`. Tolerates legacy formats and writes
// atomically with a `.bak.json` backup.

import { existsSync, readFileSync } from "node:fs";
import type { ImageGroup } from "../shared/types.ts";
import { writeJsonAtomic } from "./atomic-json.ts";
import { groupsPath } from "./paths.ts";

/**
 * Read .reorder-groups.json, tolerating both array and `{groups: [...]}` shapes.
 * Returns an empty array if the file is missing or unparseable.
 */
export function loadGroups(targetDir: string): ImageGroup[] {
  const path = groupsPath(targetDir);
  if (!existsSync(path)) return [];
  try {
    const raw = JSON.parse(readFileSync(path, "utf-8"));
    if (Array.isArray(raw)) return raw;
    if (Array.isArray(raw?.groups)) return raw.groups;
  } catch {}
  return [];
}

/** Write groups to disk, backing up the previous file to `.reorder-groups.bak.json`. */
export async function writeGroupsFile(targetDir: string, groups: ImageGroup[]): Promise<void> {
  await writeJsonAtomic(groupsPath(targetDir), groups, { backup: true });
}
