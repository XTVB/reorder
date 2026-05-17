// Single source of truth for group-membership mutations. Every flow that
// assigns files to a group goes through one of these helpers so the same
// filename can never end up in two groups — that would crash Save with
// "Duplicate filename in order" (computeRenames in src/fs/rename.ts).
//
// `appendNewGroup`/`appendNewGroups`/`dedupeGroupMemberships` return the
// same array reference when nothing changes. `addFilenamesToGroup` always
// allocates when the target group is found (the inputs are repinned to the
// tail, which is observable even when the resulting set is unchanged).

import type { ImageGroup } from "../types.ts";

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
