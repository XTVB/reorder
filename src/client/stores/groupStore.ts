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
  addImagesToGroup: (groupId: string, filenames: string[]) => void;
  reorderGroup: (groupId: string, newOrder: string[]) => void;
  removeFromGroup: (groupId: string, filename: string) => void;
  renameGroupPrompt: (groupId: string) => void;
  deleteGroup: (groupId: string) => void;

  saveRenames: () => Promise<void>;
  applyOrganize: () => Promise<void>;
}

function deriveGroupMap(groups: ImageGroup[]) {
  return new Map(groups.map((g) => [g.id, g]));
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
    const sel = useSelectionStore.getState();
    const selectedIds = sel.contexts.reorder;
    if (selectedIds.size === 0) return;
    const name = prompt("Enter group name:");
    if (!name?.trim()) return;

    const { images, setImages } = useImageStore.getState();
    const id = crypto.randomUUID();
    const selectedInOrder = images
      .filter((i) => selectedIds.has(i.filename))
      .map((i) => i.filename);

    setImages(consolidateBlock(images, selectedIds));
    get().updateGroups((prev) =>
      appendNewGroup(prev, { id, name: name.trim(), images: selectedInOrder }),
    );
    sel.clear("reorder");
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
      showToast("Files organized into folders", "success");
      await Promise.all([fetchImages(), session.checkUndo()]);
    } catch (err) {
      showToast(getErrorMessage(err, "Organize failed"), "error");
    } finally {
      session.setSaving(false);
    }
  },
}));
