// Single source of truth for group-membership mutations. Every flow that
// assigns files to a group goes through one of these helpers so the same
// filename can never end up in two groups — that would crash Save with
// "Duplicate filename in order" (computeRenames in src/fs/rename.ts).
//
// `appendNewGroup`/`appendNewGroups`/`dedupeGroupMemberships` return the
// same array reference when nothing changes. `addFilenamesToGroup` always
// allocates when the target group is found (the inputs are repinned to the
// tail, which is observable even when the resulting set is unchanged).

import type { ImageGroup, ImageInfo } from "../types.ts";

/** Every filename that lives in some group (one pass; for ungrouped math). */
export function groupedFilenameSet(groups: ImageGroup[]): Set<string> {
  const set = new Set<string>();
  for (const g of groups) for (const fn of g.images) set.add(fn);
  return set;
}

function stripFromOthers(
  groups: ImageGroup[],
  filenames: Set<string>,
  keepInGroupId?: string,
): ImageGroup[] {
  if (filenames.size === 0) return groups;
  let changed = false;
  const out = groups.map((g) => {
    if (g.id === keepInGroupId) return g;
    if (!g.images.some((fn) => filenames.has(fn))) return g;
    changed = true;
    return { ...g, images: g.images.filter((fn) => !filenames.has(fn)) };
  });
  return changed ? out : groups;
}

/**
 * Append `filenames` to `targetGroupId`, removing them from every other group.
 *
 * Files in the input that are already present in the target are repinned to
 * the end (matches the prior `addImagesToGroup` behavior). Pass only files
 * the caller actually wants to add — the helper does not pre-filter.
 */
export function addFilenamesToGroup(
  groups: ImageGroup[],
  targetGroupId: string,
  filenames: string[],
): ImageGroup[] {
  if (filenames.length === 0) return groups;
  if (!groups.some((g) => g.id === targetGroupId)) return groups;
  const set = new Set(filenames);
  const stripped = stripFromOthers(groups, set, targetGroupId);
  return stripped.map((g) => {
    if (g.id !== targetGroupId) return g;
    const base = g.images.some((fn) => set.has(fn))
      ? g.images.filter((fn) => !set.has(fn))
      : g.images;
    return { ...g, images: [...base, ...filenames] };
  });
}

/**
 * Append a new group, stripping its filenames from any pre-existing group so
 * each file has exactly one home.
 */
export function appendNewGroup(groups: ImageGroup[], newGroup: ImageGroup): ImageGroup[] {
  return [...stripFromOthers(groups, new Set(newGroup.images)), newGroup];
}

/**
 * Strip `filenames` from every group. Returns the same array reference if no
 * group contained any of them.
 */
export function removeFilenamesFromGroups(
  groups: ImageGroup[],
  filenames: Set<string>,
): ImageGroup[] {
  return stripFromOthers(groups, filenames);
}

/**
 * Append multiple new groups in one pass — strips all their filenames from
 * existing groups in a single iteration. Assumes the new groups are mutually
 * disjoint (true for cluster results by construction).
 */
export function appendNewGroups(groups: ImageGroup[], newGroups: ImageGroup[]): ImageGroup[] {
  if (newGroups.length === 0) return groups;
  const set = new Set<string>();
  for (const g of newGroups) for (const fn of g.images) set.add(fn);
  return [...stripFromOthers(groups, set), ...newGroups];
}

/**
 * Groups sorted by where they appear in the gallery (earliest member image
 * wins). This is the user-visible group order, as opposed to the JSON order.
 */
export function groupsInGalleryOrder(groups: ImageGroup[], images: ImageInfo[]): ImageGroup[] {
  const imageIndex = new Map(images.map((img, i) => [img.filename, i]));
  const firstIdx = (g: ImageGroup) =>
    g.images.reduce((min, fn) => Math.min(min, imageIndex.get(fn) ?? Infinity), Infinity);
  return [...groups].sort((a, b) => firstIdx(a) - firstIdx(b));
}

/**
 * Re-pin locked groups to their pre-sort slots: groups marked `locked` keep
 * the position (among groups) they hold in `currentOrder`, and the remaining
 * slots are filled with the unlocked groups in `proposed` order. Defensive
 * about set mismatches — a locked group absent from `proposed` is simply not
 * pinned, and proposed groups absent from `currentOrder` fill slots normally.
 */
export function withLockedGroupsInPlace(
  currentOrder: ImageGroup[],
  proposed: ImageGroup[],
): ImageGroup[] {
  const proposedById = new Map(proposed.map((g) => [g.id, g]));
  // Slot index (among groups in the current order) -> locked group id.
  const lockedAt = new Map<number, string>();
  currentOrder.forEach((g, i) => {
    if (g.locked && proposedById.has(g.id)) lockedAt.set(i, g.id);
  });
  if (lockedAt.size === 0) return proposed;

  const lockedIds = new Set(lockedAt.values());
  const rest = proposed.filter((g) => !lockedIds.has(g.id));
  const out: ImageGroup[] = [];
  let u = 0;
  for (let i = 0; i < currentOrder.length && out.length < proposed.length; i++) {
    const lockedId = lockedAt.get(i);
    if (lockedId) out.push(proposedById.get(lockedId)!);
    else if (u < rest.length) out.push(rest[u++]!);
    // else: more current slots than proposed groups — nothing to fill here.
  }
  while (u < rest.length) out.push(rest[u++]!);
  return out;
}

/**
 * Recovery helper: enforce single-group membership by dropping later
 * occurrences of any filename that appears in multiple groups (first listing
 * wins). Used on group load so legacy `.reorder-groups.json` files written
 * before this fix don't poison subsequent operations.
 */
export function dedupeGroupMemberships(groups: ImageGroup[]): ImageGroup[] {
  const seen = new Set<string>();
  let changed = false;
  const out = groups.map((g) => {
    let next: string[] | null = null;
    for (let i = 0; i < g.images.length; i++) {
      const fn = g.images[i]!;
      if (seen.has(fn)) {
        if (!next) next = g.images.slice(0, i);
        changed = true;
      } else {
        seen.add(fn);
        if (next) next.push(fn);
      }
    }
    return next ? { ...g, images: next } : g;
  });
  return changed ? out : groups;
}
