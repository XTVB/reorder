import type { ImageInfo, ImageGroup } from "../types.ts";
import { toGroupSortId } from "./helpers.ts";

/** Reorder items by splicing selected items to the drop target position. */
export function multiDragReorder(
  items: string[],
  selected: Set<string>,
  activeId: string,
  overId: string
): string[] {
  const sel = items.filter((id) => selected.has(id));
  const rest = items.filter((id) => !selected.has(id));
  const overInRest = rest.indexOf(overId);
  const activeOrig = items.indexOf(activeId);
  const overOrig = items.indexOf(overId);
  const insertAt = activeOrig < overOrig ? overInRest + 1 : overInRest;
  const out = [...rest];
  out.splice(insertAt, 0, ...sel);
  return out;
}

/** After a save, filenames change. Remap group members to new filenames. */
export function remapGroupsAfterSave(
  groups: ImageGroup[],
  oldOrder: string[],
  newOrder: string[]
): ImageGroup[] {
  const renameMap = new Map<string, string>();
  for (let i = 0; i < oldOrder.length; i++) {
    if (oldOrder[i] !== newOrder[i]) {
      renameMap.set(oldOrder[i]!, newOrder[i]!);
    }
  }
  if (renameMap.size === 0) return groups;
  return groups.map((g) => ({
    ...g,
    images: g.images.map((fn) => renameMap.get(fn) ?? fn),
  }));
}

export function flattenOrder(
  ids: string[],
  groups: ImageGroup[],
  images: ImageInfo[]
): ImageInfo[] {
  const imap = new Map(images.map((i) => [i.filename, i]));
  const gmap = new Map(groups.map((g) => [toGroupSortId(g.id), g]));
  const seen = new Set<string>();
  const out: ImageInfo[] = [];
  for (const id of ids) {
    const g = gmap.get(id);
    if (g) {
      for (const fn of g.images) {
        if (seen.has(fn)) continue;
        seen.add(fn);
        const img = imap.get(fn);
        if (img) out.push(img);
      }
    } else {
      if (seen.has(id)) continue;
      seen.add(id);
      const img = imap.get(id);
      if (img) out.push(img);
    }
  }
  return out;
}

export function consolidateBlock(images: ImageInfo[], filenames: Set<string>): ImageInfo[] {
  const selected = images.filter((i) => filenames.has(i.filename));
  const out: ImageInfo[] = [];
  let inserted = false;
  for (const img of images) {
    if (filenames.has(img.filename)) {
      if (!inserted) {
        out.push(...selected);
        inserted = true;
      }
    } else {
      out.push(img);
    }
  }
  return out;
}

export function repositionBlock(
  imgs: ImageInfo[],
  orderedFilenames: string[]
): ImageInfo[] {
  const set = new Set(orderedFilenames);
  const imgMap = new Map(imgs.map((i) => [i.filename, i]));
  const reordered = orderedFilenames.map((fn) => imgMap.get(fn)!).filter(Boolean);
  const rest = imgs.filter((i) => !set.has(i.filename));
  const firstOrigIdx = imgs.findIndex((i) => set.has(i.filename));
  const insertAt = imgs.slice(0, firstOrigIdx).filter((i) => !set.has(i.filename)).length;
  const out = [...rest];
  out.splice(insertAt, 0, ...reordered);
  return out;
}
