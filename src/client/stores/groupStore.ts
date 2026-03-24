import { create } from "zustand";
import type { ImageGroup } from "../types.ts";
import { postJson } from "../utils/helpers.ts";

const GROUPS_ENABLED_KEY = "reorder-groups-enabled";

let _persistTimer: ReturnType<typeof setTimeout> | null = null;

function persistGroupsToServer(groups: ImageGroup[]) {
  if (_persistTimer) clearTimeout(_persistTimer);
  _persistTimer = setTimeout(() => {
    postJson("/api/groups", groups).catch(() => {});
  }, 300);
}

interface GroupState {
  groups: ImageGroup[];
  groupsEnabled: boolean;
  expandedGroupId: string | null;
  groupMap: Map<string, ImageGroup>;
  groupsLoaded: boolean;

  setGroups: (groups: ImageGroup[]) => void;
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

  setGroups: (groups) => {
    set({ groups, groupMap: deriveGroupMap(groups) });
  },

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
      const serverGroups: unknown = await res.json();
      let groups: ImageGroup[] = Array.isArray(serverGroups)
        ? serverGroups
        : Array.isArray((serverGroups as any)?.groups)
          ? (serverGroups as any).groups
          : [];

      // One-time migration from localStorage
      if (groups.length === 0) {
        try {
          const lsGroups = JSON.parse(localStorage.getItem("reorder-groups") || "[]");
          if (Array.isArray(lsGroups) && lsGroups.length > 0) {
            groups = lsGroups;
            persistGroupsToServer(groups);
            localStorage.removeItem("reorder-groups");
          }
        } catch {}
      }

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
