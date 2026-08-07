import { create } from "zustand";
import { getJson, postJson } from "../api/client.ts";
import type { ImageGroup, RenameMapping, SaveResponse } from "../types.ts";
import { addFilenamesToGroup, appendNewGroup, dedupeGroupMemberships } from "../utils/groups.ts";
import { getErrorMessage } from "../utils/helpers.ts";
import { consolidateBlock, repositionBlock } from "../utils/reorder.ts";
import { useModalStore } from "./core/modalStore.ts";
import { useSelectionStore } from "./core/selectionStore.ts";
import { useSessionStore } from "./core/sessionStore.ts";
import { useToastStore } from "./core/toastStore.ts";
import { useImageStore } from "./imageStore.ts";
import { useLockedImagesStore } from "./lockedImagesStore.ts";
import { noteOrderWrite, useSortHistoryStore } from "./sortHistoryStore.ts";
import { useTrashStore } from "./trashStore.ts";

const GROUPS_ENABLED_KEY = "reorder-groups-enabled";

let _persistTimer: ReturnType<typeof setTimeout> | null = null;

function persistGroupsToServer(groups: ImageGroup[]) {
  if (_persistTimer) clearTimeout(_persistTimer);
  _persistTimer = setTimeout(() => {
    _persistTimer = null;
    postJson<unknown>("/api/groups", groups).catch(() => {});
  }, 300);
}

interface GroupState {
  groups: ImageGroup[];
  groupsEnabled: boolean;
  expandedGroupId: string | null;
  groupMap: Map<string, ImageGroup>;
  groupsLoaded: boolean;

  updateGroups: (fn: (prev: ImageGroup[]) => ImageGroup[]) => void;
  fetchGroups: () => Promise<void>;
  toggleGroupsEnabled: () => void;
  expandGroup: (id: string | null) => void;
  collapseGroup: () => void;

  /** Cancel any pending debounced group persist and flush synchronously. */
  flushPending: () => Promise<void>;

  // Hoisted business actions (previously closures in useGroupOperations / App.tsx)
  createGroupFromSelection: () => void;
  createGroupFromSelectionAutoNamed: () => void;
  addImagesToGroup: (groupId: string, filenames: string[]) => void;
  reorderGroup: (groupId: string, newOrder: string[]) => void;
  removeFromGroup: (groupId: string, filename: string) => void;
  renameGroupPrompt: (groupId: string) => void;
  deleteGroup: (groupId: string) => void;
  /**
   * Merge the given groups into the one that appears first in the current
   * frontend order (earliest member in the gallery). The survivor keeps its
   * id and name; the others' images are appended to it and they are removed.
   */
  mergeGroups: (groupIds: string[]) => void;
  /**
   * Toggle the sort lock on the given groups: locks all of them unless every
   * one is already locked, in which case all are unlocked. Locked groups keep
   * their relative order when a sort is applied (unlocked groups may interleave
   * between them).
   */
  toggleGroupsLocked: (groupIds: string[]) => void;

  saveRenames: () => Promise<void>;
  applyOrganize: () => Promise<void>;
}

function deriveGroupMap(groups: ImageGroup[]) {
  return new Map(groups.map((g) => [g.id, g]));
}

// Shared body for the create-group actions: consolidates the selection into a
// contiguous block, appends a new group with `name`, and clears the selection.
function createGroupWithName(get: () => GroupState, name: string) {
  const sel = useSelectionStore.getState();
  const selectedIds = sel.contexts.reorder;
  if (selectedIds.size === 0) return;

  const { images, setImages } = useImageStore.getState();
  const id = crypto.randomUUID();
  const selectedInOrder = images.filter((i) => selectedIds.has(i.filename)).map((i) => i.filename);

  setImages(consolidateBlock(images, selectedIds));
  get().updateGroups((prev) => appendNewGroup(prev, { id, name, images: selectedInOrder }));
  sel.clear("reorder");
}

// Next "Cluster N" name based on existing auto-named groups (max + 1, starting at 1).
function nextClusterName(groups: ImageGroup[]) {
  let max = 0;
  for (const g of groups) {
    const m = /^Cluster (\d+)$/.exec(g.name.trim());
    if (m) max = Math.max(max, Number(m[1]));
  }
  return `Cluster ${max + 1}`;
}

export const useGroupStore = create<GroupState>((set, get) => ({
  groups: [],
  groupsEnabled: localStorage.getItem(GROUPS_ENABLED_KEY) !== "false",
  expandedGroupId: null,
  groupMap: new Map(),
  groupsLoaded: false,

  // Applies fn, prunes empty groups, skips persist if unchanged
  updateGroups: (fn) => {
    const { groups, groupsLoaded } = get();
    if (!groupsLoaded) {
      console.warn("updateGroups called before groups were loaded — ignoring to prevent data loss");
      return;
    }
    let next = fn(groups);
    if (next === groups) return;
    if (next.some((g) => g.images.length === 0)) {
      next = next.filter((g) => g.images.length > 0);
    }
    // Only membership/order reshapes the cards; rename/tag/lock edits leave the
    // sort snapshot valid.
    const cardsChanged =
      next.length !== groups.length ||
      next.some((g, i) => {
        const prev = groups[i];
        return (
          !prev ||
          g.id !== prev.id ||
          g.images.length !== prev.images.length ||
          g.images.some((fn, j) => fn !== prev.images[j])
        );
      });
    if (cardsChanged) noteOrderWrite();
    persistGroupsToServer(next);
    set({ groups: next, groupMap: deriveGroupMap(next) });
  },

  fetchGroups: async () => {
    try {
      const raw = await getJson<ImageGroup[]>("/api/groups");
      // Heal legacy overlapping memberships on read; see dedupeGroupMemberships.
      const groups = dedupeGroupMemberships(raw).filter((g) => g.images.length > 0);
      set({ groups, groupMap: deriveGroupMap(groups), groupsLoaded: true });
    } catch {
      set({ groupsLoaded: true });
    }
  },

  toggleGroupsEnabled: () => {
    const next = !get().groupsEnabled;
    localStorage.setItem(GROUPS_ENABLED_KEY, String(next));
    set({ groupsEnabled: next, expandedGroupId: null });
  },

  expandGroup: (id) => set({ expandedGroupId: id }),
  collapseGroup: () => set({ expandedGroupId: null }),

  flushPending: () => {
    if (_persistTimer) {
      clearTimeout(_persistTimer);
      _persistTimer = null;
      return postJson<unknown>("/api/groups", get().groups).then(() => {});
    }
    return Promise.resolve();
  },

  createGroupFromSelection: () => {
    const name = prompt("Enter group name:");
    if (!name?.trim()) return;
    createGroupWithName(get, name.trim());
  },

  createGroupFromSelectionAutoNamed: () => {
    createGroupWithName(get, nextClusterName(get().groups));
  },

  addImagesToGroup: (groupId, filenames) => {
    const { groups, updateGroups } = get();
    const { images, setImages } = useImageStore.getState();
    const sel = useSelectionStore.getState();

    const fileSet = new Set(filenames);
    const newGroups = addFilenamesToGroup(groups, groupId, filenames);
    updateGroups(() => newGroups);
    const targetGroup = newGroups.find((g) => g.id === groupId);
    if (!targetGroup) return;

    const allGroupImages = new Set(targetGroup.images);
    const toMove = images.filter((i) => fileSet.has(i.filename));
    const rest = images.filter((i) => !fileSet.has(i.filename));
    let lastIdx = -1;
    for (let i = 0; i < rest.length; i++) {
      if (allGroupImages.has(rest[i]!.filename)) lastIdx = i;
    }
    if (lastIdx !== -1) {
      const out = [...rest];
      out.splice(lastIdx + 1, 0, ...toMove);
      setImages(out);
    }

    sel.remove("reorder", filenames);
  },

  reorderGroup: (groupId, newOrder) => {
    const { updateGroups } = get();
    const { images, setImages } = useImageStore.getState();
    updateGroups((prev) => prev.map((g) => (g.id === groupId ? { ...g, images: newOrder } : g)));
    setImages(repositionBlock(images, newOrder));
  },

  removeFromGroup: (groupId, filename) => {
    const { groupMap, collapseGroup, updateGroups } = get();
    const group = groupMap.get(groupId);
    if (group && group.images.length <= 1) collapseGroup();
    updateGroups((prev) =>
      prev.map((g) =>
        g.id === groupId ? { ...g, images: g.images.filter((fn) => fn !== filename) } : g,
      ),
    );
  },

  renameGroupPrompt: (groupId) => {
    const { groupMap, updateGroups } = get();
    const group = groupMap.get(groupId);
    if (!group) return;
    const name = prompt("New group name:", group.name);
    if (!name?.trim()) return;
    updateGroups((prev) => prev.map((g) => (g.id === groupId ? { ...g, name: name.trim() } : g)));
  },

  deleteGroup: (groupId) => {
    const { expandedGroupId, collapseGroup, updateGroups } = get();
    updateGroups((prev) => prev.filter((g) => g.id !== groupId));
    if (expandedGroupId === groupId) collapseGroup();
  },

  mergeGroups: (groupIds) => {
    const { groupMap, expandedGroupId, collapseGroup, updateGroups } = get();
    const { images, setImages } = useImageStore.getState();
    const sel = useSelectionStore.getState();

    const targets = groupIds.filter((id) => groupMap.has(id));
    if (targets.length < 2) return;

    // Order the groups by their position in the gallery: the survivor is the
    // group whose earliest member appears first in the current image order.
    const indexOf = new Map(images.map((i, idx) => [i.filename, idx]));
    const firstSlot = (gid: string) => {
      let min = Infinity;
      for (const fn of groupMap.get(gid)!.images) {
        const i = indexOf.get(fn);
        if (i !== undefined && i < min) min = i;
      }
      return min;
    };
    const ordered = [...targets].sort((a, b) => firstSlot(a) - firstSlot(b));

    const survivorId = ordered[0]!;
    const removed = new Set(ordered.slice(1));
    const mergedImages: string[] = [];
    const seen = new Set<string>();
    for (const gid of ordered) {
      for (const fn of groupMap.get(gid)!.images) {
        if (seen.has(fn)) continue;
        seen.add(fn);
        mergedImages.push(fn);
      }
    }

    updateGroups((prev) =>
      prev
        .map((g) => (g.id === survivorId ? { ...g, images: mergedImages } : g))
        .filter((g) => !removed.has(g.id)),
    );
    setImages(repositionBlock(images, mergedImages));
    if (expandedGroupId && removed.has(expandedGroupId)) collapseGroup();

    sel.clear("reorder");
    useToastStore.getState().showToast(`Merged ${ordered.length} groups`, "success");
  },

  toggleGroupsLocked: (groupIds) => {
    const { groupMap, updateGroups } = get();
    const targets = groupIds.filter((id) => groupMap.has(id));
    if (targets.length === 0) return;
    const lock = !targets.every((id) => groupMap.get(id)!.locked);
    const idSet = new Set(targets);
    updateGroups((prev) =>
      prev.map((g) => {
        if (!idSet.has(g.id) || Boolean(g.locked) === lock) return g;
        // `undefined` rather than `false` keeps the persisted JSON clean.
        return { ...g, locked: lock ? true : undefined };
      }),
    );
    useToastStore
      .getState()
      .showToast(
        lock
          ? `Locked ${targets.length} group${targets.length === 1 ? "" : "s"} — relative order kept when sorting`
          : `Unlocked ${targets.length} group${targets.length === 1 ? "" : "s"}`,
        "success",
      );
  },

  saveRenames: async () => {
    const session = useSessionStore.getState();
    const modal = useModalStore.getState();
    const { showToast } = useToastStore.getState();
    const { images, fetchImages } = useImageStore.getState();

    session.setSaving(true);
    modal.closeModal("preview");
    try {
      // Cancel any pending debounced group persist to prevent it from
      // racing with the save and overwriting remapped groups on disk.
      await get().flushPending();
      const oldFilenames = images.map((i) => i.filename);
      const currentGroups = get().groups;
      const data = await postJson<SaveResponse>("/api/save", {
        order: oldFilenames,
        groups: currentGroups,
      });
      if (!data.success) throw new Error("Rename failed");
      const renames = (data.renames ?? []) as RenameMapping[];
      useTrashStore.getState().remap(renames);
      useLockedImagesStore.getState().remap(renames);
      // Filenames just changed on disk — the pre-sort ⌥-peek snapshot is stale.
      useSortHistoryStore.getState().clearPreviousOrder();
      if (data.warnings && data.warnings.length > 0) {
        showToast(`Files renamed (${data.warnings.length} warning(s))`, "warning");
      } else {
        showToast("Files renamed successfully", "success");
      }
      await fetchImages();
      await Promise.all([get().fetchGroups(), session.checkUndo()]);
    } catch (err) {
      showToast(getErrorMessage(err, "Rename failed"), "error");
    } finally {
      session.setSaving(false);
    }
  },

  applyOrganize: async () => {
    const session = useSessionStore.getState();
    const modal = useModalStore.getState();
    const { showToast } = useToastStore.getState();
    const { images, fetchImages } = useImageStore.getState();
    const { groups } = get();

    session.setSaving(true);
    modal.closeModal("organize");
    try {
      await postJson<{ success: boolean }>("/api/organize", {
        groups: groups.map((g) => ({ name: g.name, images: g.images })),
        order: images.map((i) => i.filename),
        numbered: useSessionStore.getState().numberedFolderPrefix,
      });
      get().updateGroups(() => []);
      get().collapseGroup();
      useSortHistoryStore.getState().clearPreviousOrder();
      showToast("Files organized into folders", "success");
      await Promise.all([fetchImages(), session.checkUndo()]);
    } catch (err) {
      showToast(getErrorMessage(err, "Organize failed"), "error");
    } finally {
      session.setSaving(false);
    }
  },
}));
