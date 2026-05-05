// Organize groups into subfolders. The group order on screen determines
// folder numbering (1, 2, 3 …); within each folder, files are renumbered
// from 1 too.

import { mkdir, rename } from "node:fs/promises";
import { extname, join } from "node:path";
import type { RenameMapping } from "../shared/types.ts";
import { assertWritable, extractTitle } from "./helpers.ts";

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
  await assertWritable(dir);

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
