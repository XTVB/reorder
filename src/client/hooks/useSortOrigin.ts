import { useMemo } from "react";
import { useSortHistoryStore } from "../stores/sortHistoryStore.ts";
import type { ImageGroup, ImageInfo } from "../types.ts";
import { computeGridItems, gridItemId, isTopLevelItem } from "../utils/gridItems.ts";

export interface SortOrigin {
  row: number;
  column: number;
  /** False when the sort relocated this card; true when neighbours pushed it. */
  displaced: boolean;
}

interface Params {
  currentIds: string[];
  columnCount: number;
  groupsEnabled: boolean;
  enabled: boolean;
}

export interface SortOriginMap {
  /** Only cards whose slot changed appear. */
  origins: Map<string, SortOrigin>;
  available: boolean;
}

function topLevelCardIds(
  images: ImageInfo[],
  groups: ImageGroup[],
  groupsEnabled: boolean,
): string[] {
  return computeGridItems(images, {
    mode: "groups",
    groups,
    enabled: groupsEnabled,
    expandedGroupId: null,
  })
    .filter(isTopLevelItem)
    .map(gridItemId);
}

function findRelocated(previousIndices: number[]): boolean[] {
  const n = previousIndices.length;
  const relocated = new Array<boolean>(n).fill(true);
  if (n === 0) return relocated;

  const tails: number[] = [];
  const parent = new Array<number>(n).fill(-1);

  for (let i = 0; i < n; i++) {
    const value = previousIndices[i]!;
    let lo = 0;
    let hi = tails.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (previousIndices[tails[mid]!]! < value) lo = mid + 1;
      else hi = mid;
    }
    if (lo > 0) parent[i] = tails[lo - 1]!;
    tails[lo] = i;
  }

  for (let i = tails[tails.length - 1] ?? -1; i !== -1; i = parent[i]!) {
    relocated[i] = false;
  }
  return relocated;
}

export function useSortOrigin({
  currentIds,
  columnCount,
  groupsEnabled,
  enabled,
}: Params): SortOriginMap {
  const previousOrder = useSortHistoryStore((s) => s.previousOrder);

  return useMemo(() => {
    if (!enabled || !previousOrder || columnCount < 1) {
      return { origins: new Map(), available: false };
    }

    const previousIndexById = new Map<string, number>();
    for (const [i, id] of topLevelCardIds(
      previousOrder.images,
      previousOrder.groups,
      groupsEnabled,
    ).entries()) {
      previousIndexById.set(id, i);
    }

    // Cards missing from the snapshot were created or deleted since the sort
    // and have no previous slot. The rest carry the permutation, and all of it
    // feeds the LIS — restricting to cards that moved would skew the result.
    const carried: string[] = [];
    const previousIndices: number[] = [];
    const currentIndices: number[] = [];
    for (const [currentIndex, id] of currentIds.entries()) {
      const index = previousIndexById.get(id);
      if (index === undefined) continue;
      carried.push(id);
      previousIndices.push(index);
      currentIndices.push(currentIndex);
    }

    const relocated = findRelocated(previousIndices);

    const origins = new Map<string, SortOrigin>();
    for (let i = 0; i < carried.length; i++) {
      const index = previousIndices[i]!;
      if (index === currentIndices[i]) continue;
      origins.set(carried[i]!, {
        row: Math.floor(index / columnCount),
        column: index % columnCount,
        displaced: !relocated[i],
      });
    }

    return { origins, available: true };
  }, [previousOrder, currentIds, columnCount, groupsEnabled, enabled]);
}
