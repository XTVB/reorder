import type { ImageInfo, ImageGroup, GridItem } from "../types.ts";
import { toGroupSortId } from "./helpers.ts";

export function computeGridItems(
  images: ImageInfo[],
  groups: ImageGroup[],
  enabled: boolean,
  expandedGroupId: string | null
): GridItem[] {
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
  return item.filename;
}
