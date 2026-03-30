import type { ImageInfo, ImageGroup, FolderGroup, GridItem } from "../types.ts";
import { toGroupSortId, toFolderSortId } from "./helpers.ts";

type GridMode =
  | { mode: "folders"; folders: FolderGroup[]; expandedFolderName: string | null }
  | { mode: "groups"; groups: ImageGroup[]; enabled: boolean; expandedGroupId: string | null };

export function computeGridItems(images: ImageInfo[], opts: GridMode): GridItem[] {
  if (opts.mode === "folders") {
    const folderImageSet = new Set<string>();
    const out: GridItem[] = [];
    for (const folder of opts.folders) {
      out.push({ type: "folder", folderName: folder.name });
      for (const fn of folder.images) folderImageSet.add(fn);
      if (folder.name === opts.expandedFolderName) {
        for (const fn of folder.images) {
          out.push({ type: "folder-image", folderName: folder.name, filename: fn });
        }
      }
    }
    for (const img of images) {
      if (!folderImageSet.has(img.filename)) {
        out.push({ type: "image", filename: img.filename });
      }
    }
    return out;
  }

  // Group mode
  const { groups, enabled, expandedGroupId } = opts;
  if (!enabled || groups.length === 0)
    return images.map((i) => ({ type: "image" as const, filename: i.filename }));

  const fnToGroup = new Map<string, string>();
  for (const g of groups) for (const fn of g.images) fnToGroup.set(fn, g.id);

  const groupById = new Map(groups.map((g) => [g.id, g]));
  const seen = new Set<string>();
  const out: GridItem[] = [];
  for (const img of images) {
    const gid = fnToGroup.get(img.filename);
    if (gid) {
      if (!seen.has(gid)) {
        seen.add(gid);
        out.push({ type: "group", groupId: gid });
        if (gid === expandedGroupId) {
          const group = groupById.get(gid)!;
          for (const fn of group.images) {
            out.push({ type: "group-image", groupId: gid, filename: fn });
          }
        }
      }
    } else {
      out.push({ type: "image", filename: img.filename });
    }
  }
  return out;
}

export function gridItemId(item: GridItem): string {
  if (item.type === "group") return toGroupSortId(item.groupId);
  if (item.type === "folder") return toFolderSortId(item.folderName);
  return item.filename;
}
