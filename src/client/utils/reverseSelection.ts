import { useSelectionStore } from "../stores/core/selectionStore.ts";
import { useFolderStore } from "../stores/folderStore.ts";
import { useGroupStore } from "../stores/groupStore.ts";
import { useImageStore } from "../stores/imageStore.ts";
import { computeGridItems, gridItemId } from "./gridItems.ts";
import { fromFolderSortId, isFolderSortId, isGroupSortId } from "./helpers.ts";
import { flattenOrder, reverseSelectedInPlace } from "./reorder.ts";

/**
 * Reverse the order of the current reorder-mode selection, in place: every
 * selected slot is rewritten in reverse order while unselected items stay put.
 *
 * Reversal happens independently within each container the selection touches, so
 * groups/folders stay atomic at the top level while images selected *inside* an
 * expanded group/folder reverse within that container:
 *   - top level — ungrouped images + group/folder cards (cards reverse as a unit)
 *   - each expanded group — its selected member images
 *   - each expanded folder — its selected member images
 *   - root images (folder mode)
 *
 * No-op unless at least one container has >= 2 selected items. Selection is left
 * intact, so the action is its own inverse (press again to undo).
 */
export function reverseSelection(): void {
  const sel = useSelectionStore.getState().contexts.reorder;
  if (sel.size < 2) return;

  if (useFolderStore.getState().folderModeEnabled) {
    reverseFolderMode(sel);
  } else {
    reverseGroupMode(sel);
  }
}

function reverseGroupMode(sel: Set<string>): void {
  const { groups, groupsEnabled, groupMap, reorderGroup } = useGroupStore.getState();

  const fnToGroup = new Map<string, string>();
  if (groupsEnabled) {
    for (const g of groups) for (const fn of g.images) fnToGroup.set(fn, g.id);
  }

  const topSel = new Set<string>();
  const perGroup = new Map<string, Set<string>>();
  for (const id of sel) {
    if (isGroupSortId(id)) {
      topSel.add(id);
      continue;
    }
    const gid = fnToGroup.get(id);
    if (gid) {
      let members = perGroup.get(gid);
      if (!members) {
        members = new Set();
        perGroup.set(gid, members);
      }
      members.add(id);
    } else {
      topSel.add(id);
    }
  }

  // Within-group reversals first; each updates the group + the flat image array.
  for (const [gid, members] of perGroup) {
    if (members.size < 2) continue;
    const g = groupMap.get(gid);
    if (!g) continue;
    reorderGroup(gid, reverseSelectedInPlace(g.images, members));
  }

  // Top-level reversal (group cards atomic). Re-read state in case a within-group
  // reversal above already mutated the image array.
  if (topSel.size >= 2) {
    const { images, setImages } = useImageStore.getState();
    const curGroups = useGroupStore.getState().groups;
    const topItems = computeGridItems(images, {
      mode: "groups",
      groups: curGroups,
      enabled: groupsEnabled,
      expandedGroupId: null,
    });
    const reversed = reverseSelectedInPlace(topItems.map(gridItemId), topSel);
    setImages(flattenOrder(reversed, curGroups, images));
  }
}

function reverseFolderMode(sel: Set<string>): void {
  const { folders, rootImages, folderMap, reorderFolders, reorderWithinFolder, reorderRootImages } =
    useFolderStore.getState();

  const fnToFolder = new Map<string, string>();
  for (const f of folders) for (const fn of f.images) fnToFolder.set(fn, f.name);
  const rootSet = new Set(rootImages);

  const folderCardSel = new Set<string>(); // bare folder names
  const rootSel = new Set<string>();
  const perFolder = new Map<string, Set<string>>();
  for (const id of sel) {
    if (isFolderSortId(id)) {
      folderCardSel.add(fromFolderSortId(id));
      continue;
    }
    const fname = fnToFolder.get(id);
    if (fname) {
      let members = perFolder.get(fname);
      if (!members) {
        members = new Set();
        perFolder.set(fname, members);
      }
      members.add(id);
    } else if (rootSet.has(id)) {
      rootSel.add(id);
    }
  }

  for (const [fname, members] of perFolder) {
    if (members.size < 2) continue;
    const folder = folderMap.get(fname);
    if (!folder) continue;
    reorderWithinFolder(fname, reverseSelectedInPlace(folder.images, members));
  }

  if (folderCardSel.size >= 2) {
    reorderFolders(
      reverseSelectedInPlace(
        folders.map((f) => f.name),
        folderCardSel,
      ),
    );
  }

  if (rootSel.size >= 2) {
    reorderRootImages(reverseSelectedInPlace(rootImages, rootSel));
  }
}
