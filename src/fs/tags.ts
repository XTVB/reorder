// tags.json read/write — used by rename (snapshot+remap on rename) and
// trash (purge entries for deleted files).

import type { RenameMapping } from "../shared/types.ts";
import { tagsPath } from "./paths.ts";

export async function readTagsJson(dir: string): Promise<Record<string, unknown> | null> {
  try {
    const raw = await Bun.file(tagsPath(dir)).text();
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export async function writeRemappedTags(
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

  await Bun.write(tagsPath(dir), JSON.stringify(Object.fromEntries(remapped), null, 2));
}
