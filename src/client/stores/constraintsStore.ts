import { create } from "zustand";
import { postJson } from "../utils/helpers.ts";
import { useClusterStore } from "./clusterStore.ts";

interface ResolvedCannotLink {
  imageHash: string;
  groupId: string;
  currentFilename: string | null;
}

interface ServerResponse {
  version: 1;
  imageGroupCannotLink: { imageHash: string; groupId: string }[];
  lockedGroupIds: string[];
  imageGroupCannotLinkResolved?: ResolvedCannotLink[];
  treeStale?: boolean;
}

interface ConstraintsState {
  lockedGroupIds: Set<string>;
  loaded: boolean;

  // O(1) lookup: filename → set of group ids that have rejected it.
  index: Map<string, Set<string>>;

  loadConstraints: () => Promise<void>;
  addImageGroupCannotLink: (filename: string, groupId: string) => Promise<void>;
  removeImageGroupCannotLink: (filename: string, groupId: string) => Promise<void>;
  toggleGroupLock: (groupId: string) => Promise<void>;
  isCannotLinked: (filename: string, groupId: string) => boolean;
  isGroupLocked: (groupId: string) => boolean;
}

function buildIndex(resolved: ResolvedCannotLink[]): Map<string, Set<string>> {
  const idx = new Map<string, Set<string>>();
  for (const r of resolved) {
    if (!r.currentFilename) continue;
    let s = idx.get(r.currentFilename);
    if (!s) {
      s = new Set();
      idx.set(r.currentFilename, s);
    }
    s.add(r.groupId);
  }
  return idx;
}

function applyServer(payload: ServerResponse) {
  return {
    lockedGroupIds: new Set(payload.lockedGroupIds),
    index: buildIndex(payload.imageGroupCannotLinkResolved ?? []),
    loaded: true,
  };
}

async function applyMutationResponse(res: Response) {
  if (!res.ok) return;
  const payload = (await res.json()) as ServerResponse;
  useConstraintsStore.setState(applyServer(payload));
  if (payload.treeStale) useClusterStore.getState().markTreeStale();
}

export const useConstraintsStore = create<ConstraintsState>((set, get) => ({
  lockedGroupIds: new Set(),
  loaded: false,
  index: new Map(),

  loadConstraints: async () => {
    try {
      const res = await fetch("/api/constraints");
      const payload = (await res.json()) as ServerResponse;
      set(applyServer(payload));
    } catch {
      set({ loaded: true });
    }
  },

  addImageGroupCannotLink: async (filename, groupId) => {
    if (get().isCannotLinked(filename, groupId)) return;
    await applyMutationResponse(
      await postJson("/api/constraints/cannot-link", {
        imageFilename: filename,
        groupId,
        action: "add",
      }),
    );
  },

  removeImageGroupCannotLink: async (filename, groupId) => {
    if (!get().isCannotLinked(filename, groupId)) return;
    await applyMutationResponse(
      await postJson("/api/constraints/cannot-link", {
        imageFilename: filename,
        groupId,
        action: "remove",
      }),
    );
  },

  toggleGroupLock: async (groupId) => {
    const wasLocked = get().lockedGroupIds.has(groupId);
    await applyMutationResponse(
      await postJson("/api/constraints/group-lock", { groupId, locked: !wasLocked }),
    );
  },

  isCannotLinked: (filename, groupId) => {
    const s = get().index.get(filename);
    return s ? s.has(groupId) : false;
  },

  isGroupLocked: (groupId) => get().lockedGroupIds.has(groupId),
}));
