import { create } from "zustand";
import { getJson, postJson } from "../api/client.ts";
import { useListStore } from "./modes/cluster/listStore.ts";

interface ResolvedCannotLink {
  imageHash: string;
  groupId: string;
  currentFilename: string | null;
}

export interface RejectedMergePair {
  groupA: string;
  groupB: string;
}

interface ServerResponse {
  version: 1;
  imageGroupCannotLink: { imageHash: string; groupId: string }[];
  lockedGroupIds: string[];
  rejectedMergePairs?: RejectedMergePair[];
  imageGroupCannotLinkResolved?: ResolvedCannotLink[];
  treeStale?: boolean;
}

interface ConstraintsState {
  lockedGroupIds: Set<string>;
  loaded: boolean;

  // O(1) lookup: filename → set of group ids that have rejected it.
  index: Map<string, Set<string>>;

  // Normalized "groupA|groupB" keys (groupA < groupB lexicographically).
  rejectedMerges: Set<string>;

  loadConstraints: () => Promise<void>;
  addImageGroupCannotLink: (pairs: CannotLinkPair[]) => Promise<void>;
  removeImageGroupCannotLink: (pairs: CannotLinkPair[]) => Promise<void>;
  toggleGroupLock: (groupId: string) => Promise<void>;
  addRejectedMerges: (pairs: RejectedMergePair[]) => Promise<void>;
  isCannotLinked: (filename: string, groupId: string) => boolean;
  isGroupLocked: (groupId: string) => boolean;
  isMergeRejected: (groupA: string, groupB: string) => boolean;
}

export interface CannotLinkPair {
  filename: string;
  groupId: string;
}

function mergePairKey(a: string, b: string): string {
  return a <= b ? `${a}|${b}` : `${b}|${a}`;
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

function setsEqual<T>(a: Set<T>, b: Set<T>): boolean {
  if (a.size !== b.size) return false;
  for (const x of a) if (!b.has(x)) return false;
  return true;
}

function buildRejectedSet(pairs: RejectedMergePair[] | undefined): Set<string> {
  const s = new Set<string>();
  if (!pairs) return s;
  for (const p of pairs) s.add(mergePairKey(p.groupA, p.groupB));
  return s;
}

function reconcileLocked(prev: Set<string>, next: Set<string>): Set<string> {
  return setsEqual(prev, next) ? prev : next;
}

function reconcileIndex(
  prev: Map<string, Set<string>>,
  next: Map<string, Set<string>>,
): Map<string, Set<string>> {
  if (prev.size !== next.size) return next;
  for (const [k, prevSet] of prev) {
    const nextSet = next.get(k);
    if (!nextSet || !setsEqual(prevSet, nextSet)) return next;
  }
  return prev;
}

function applyServer(payload: ServerResponse, prev: ConstraintsState) {
  const nextLocked = new Set(payload.lockedGroupIds);
  const nextIndex = buildIndex(payload.imageGroupCannotLinkResolved ?? []);
  const nextRejected = buildRejectedSet(payload.rejectedMergePairs);
  return {
    lockedGroupIds: reconcileLocked(prev.lockedGroupIds, nextLocked),
    index: reconcileIndex(prev.index, nextIndex),
    rejectedMerges: setsEqual(prev.rejectedMerges, nextRejected)
      ? prev.rejectedMerges
      : nextRejected,
    loaded: true,
  };
}

function applyMutationPayload(payload: ServerResponse) {
  useConstraintsStore.setState((prev) => applyServer(payload, prev));
  if (payload.treeStale) useListStore.getState().markTreeStale();
}

export const useConstraintsStore = create<ConstraintsState>((set, get) => ({
  lockedGroupIds: new Set(),
  loaded: false,
  index: new Map(),
  rejectedMerges: new Set(),

  loadConstraints: async () => {
    try {
      const payload = await getJson<ServerResponse>("/api/constraints");
      set((prev) => applyServer(payload, prev));
    } catch {
      set({ loaded: true });
    }
  },

  addImageGroupCannotLink: async (pairs) => {
    const novel = pairs.filter((p) => !get().isCannotLinked(p.filename, p.groupId));
    if (novel.length === 0) return;
    try {
      const payload = await postJson<ServerResponse>("/api/constraints/cannot-link", {
        pairs: novel.map((p) => ({ imageFilename: p.filename, groupId: p.groupId })),
        action: "add",
      });
      applyMutationPayload(payload);
    } catch {}
  },

  removeImageGroupCannotLink: async (pairs) => {
    const present = pairs.filter((p) => get().isCannotLinked(p.filename, p.groupId));
    if (present.length === 0) return;
    try {
      const payload = await postJson<ServerResponse>("/api/constraints/cannot-link", {
        pairs: present.map((p) => ({ imageFilename: p.filename, groupId: p.groupId })),
        action: "remove",
      });
      applyMutationPayload(payload);
    } catch {}
  },

  toggleGroupLock: async (groupId) => {
    const wasLocked = get().lockedGroupIds.has(groupId);
    try {
      const payload = await postJson<ServerResponse>("/api/constraints/group-lock", {
        groupId,
        locked: !wasLocked,
      });
      applyMutationPayload(payload);
    } catch {}
  },

  addRejectedMerges: async (pairs) => {
    const isRejected = get().isMergeRejected;
    const novel = pairs.filter((p) => p.groupA !== p.groupB && !isRejected(p.groupA, p.groupB));
    if (novel.length === 0) return;
    try {
      const payload = await postJson<ServerResponse>("/api/constraints/rejected-merge", {
        pairs: novel,
        action: "add",
      });
      applyMutationPayload(payload);
    } catch {}
  },

  isCannotLinked: (filename, groupId) => {
    const s = get().index.get(filename);
    return s ? s.has(groupId) : false;
  },

  isGroupLocked: (groupId) => get().lockedGroupIds.has(groupId),

  isMergeRejected: (groupA, groupB) => get().rejectedMerges.has(mergePairKey(groupA, groupB)),
}));
