import { create } from "zustand";
import type {
  ImageGroup,
  MergeSuggestionRow,
  MergeSuggestionsResponse,
} from "../types.ts";
import { postJson } from "../utils/helpers.ts";
import { useGroupStore } from "./groupStore.ts";

const MAX_PER_GROUP = 8;

interface MergeSuggestionsState {
  suggestions: MergeSuggestionRow[] | null;
  loading: boolean;
  error: string | null;
  computeTimeMs: number | null;

  threshold: number;

  collapsedRows: Set<string>;
  pendingMerges: Map<string, Set<string>>; // refGroupId → candidate groupIds
  undoStack: ImageGroup[][];

  setThreshold: (t: number) => void;
  fetchSuggestions: () => Promise<void>;
  toggleRowCollapse: (groupId: string) => void;
  collapseAllRows: () => void;
  expandAllRows: () => void;
  toggleMergeCandidate: (refId: string, candidateId: string) => void;
  selectAllInRow: (refId: string) => void;
  deselectAllInRow: (refId: string) => void;
  clearPendingMerges: () => void;
  applyMerges: () => Promise<void>;
  undo: () => Promise<void>;
  pendingMergeCount: () => number;
}

export const useMergeSuggestionsStore = create<MergeSuggestionsState>(
  (set, get) => ({
    suggestions: null,
    loading: false,
    error: null,
    computeTimeMs: null,

    threshold: 0.65,

    collapsedRows: new Set(),
    pendingMerges: new Map(),
    undoStack: [],

    setThreshold: (t) => set({ threshold: t }),

    fetchSuggestions: async () => {
      const { threshold } = get();
      set({ loading: true, error: null });
      try {
        const resp = await postJson("/api/merge-suggestions", {
          threshold,
          maxPerGroup: MAX_PER_GROUP,
        });
        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.error || "Failed to fetch suggestions");
        }
        const data: MergeSuggestionsResponse = await resp.json();
        set({
          suggestions: data.suggestions,
          computeTimeMs: data.computeTimeMs,
          loading: false,
          pendingMerges: new Map(),
        });
      } catch (err) {
        set({
          loading: false,
          error: err instanceof Error ? err.message : "Unknown error",
        });
      }
    },

    toggleRowCollapse: (groupId) => {
      set((s) => {
        const next = new Set(s.collapsedRows);
        if (next.has(groupId)) next.delete(groupId);
        else next.add(groupId);
        return { collapsedRows: next };
      });
    },

    collapseAllRows: () => {
      const { suggestions } = get();
      if (!suggestions) return;
      set({ collapsedRows: new Set(suggestions.map((s) => s.refGroupId)) });
    },

    expandAllRows: () => set({ collapsedRows: new Set() }),

    toggleMergeCandidate: (refId, candidateId) => {
      set((s) => {
        const next = new Map(s.pendingMerges);
        const existing = next.get(refId) ?? new Set();
        const updated = new Set(existing);
        if (updated.has(candidateId)) updated.delete(candidateId);
        else updated.add(candidateId);
        if (updated.size === 0) next.delete(refId);
        else next.set(refId, updated);
        return { pendingMerges: next };
      });
    },

    selectAllInRow: (refId) => {
      const { suggestions } = get();
      if (!suggestions) return;
      const row = suggestions.find((s) => s.refGroupId === refId);
      if (!row) return;
      set((s) => {
        const next = new Map(s.pendingMerges);
        next.set(refId, new Set(row.similar.map((c) => c.groupId)));
        return { pendingMerges: next };
      });
    },

    deselectAllInRow: (refId) => {
      set((s) => {
        const next = new Map(s.pendingMerges);
        next.delete(refId);
        return { pendingMerges: next };
      });
    },

    clearPendingMerges: () => set({ pendingMerges: new Map() }),

    applyMerges: async () => {
      const { pendingMerges, undoStack } = get();
      if (pendingMerges.size === 0) return;

      const groupStore = useGroupStore.getState();
      const currentGroups = [...groupStore.groups.map((g) => ({ ...g, images: [...g.images] }))];

      const newUndoStack = [...undoStack, currentGroups].slice(-10);

      groupStore.updateGroups((prev) => {
        const groups = prev.map((g) => ({ ...g, images: [...g.images] }));
        const byId = new Map(groups.map((g) => [g.id, g]));
        const toRemove = new Set<string>();

        for (const [refId, candidateIds] of pendingMerges) {
          const ref = byId.get(refId);
          if (!ref) continue;
          const existing = new Set(ref.images);
          for (const candId of candidateIds) {
            const cand = byId.get(candId);
            if (!cand) continue;
            for (const img of cand.images) {
              if (!existing.has(img)) {
                ref.images.push(img);
                existing.add(img);
              }
            }
            toRemove.add(candId);
          }
        }

        return groups.filter((g) => !toRemove.has(g.id));
      });

      set({ undoStack: newUndoStack, pendingMerges: new Map() });
      await get().fetchSuggestions();
    },

    undo: async () => {
      const { undoStack } = get();
      if (undoStack.length === 0) return;

      const previousGroups = undoStack[undoStack.length - 1]!;
      const newStack = undoStack.slice(0, -1);

      const groupStore = useGroupStore.getState();
      groupStore.updateGroups(() => previousGroups);

      set({ undoStack: newStack, pendingMerges: new Map() });
      await get().fetchSuggestions();
    },

    pendingMergeCount: () => {
      const { pendingMerges } = get();
      let count = 0;
      for (const candidates of pendingMerges.values()) count += candidates.size;
      return count;
    },
  }),
);
