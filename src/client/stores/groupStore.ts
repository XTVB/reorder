import { create } from "zustand";
import type { ImageGroup } from "../types.ts";
import { postJson } from "../utils/helpers.ts";

const GROUPS_ENABLED_KEY = "reorder-groups-enabled";

let _persistTimer: ReturnType<typeof setTimeout> | null = null;

function persistGroupsToServer(groups: ImageGroup[]) {
  if (_persistTimer) clearTimeout(_persistTimer);
  _persistTimer = setTimeout(() => {
    _persistTimer = null;
    postJson("/api/groups", groups).catch(() => {});
  }, 300);
}

/** Cancel any pending debounced group persist and flush synchronously. */
export function flushGroupPersist(): Promise<void> {
  if (_persistTimer) {
    clearTimeout(_persistTimer);
    _persistTimer = null;
    return postJson("/api/groups", useGroupStore.getState().groups).then(() => {});
  }
  return Promise.resolve();
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
    const { groups } = get();
    let next = fn(groups);
    if (next === groups) return;
    next = next.filter((g) => g.images.length > 0);
    persistGroupsToServer(next);
    set({ groups: next, groupMap: deriveGroupMap(next) });
  },

  fetchGroups: async () => {
    try {
      const res = await fetch("/api/groups");
      const groups: ImageGroup[] = await res.json();
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
}));

