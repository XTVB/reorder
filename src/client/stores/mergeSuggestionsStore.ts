import { create } from "zustand";

import type { ImageGroup, MergeSuggestionRow, MergeSuggestionsResponse } from "../types.ts";
import { postJson } from "../utils/helpers.ts";
import { consumeSSE } from "../utils/sse.ts";
import { flushGroupPersist, useGroupStore } from "./groupStore.ts";

const MAX_PER_GROUP = 8;

interface MergeSuggestionsState {
  suggestions: MergeSuggestionRow[] | null;
  loading: boolean;
  error: string | null;
  computeTimeMs: number | null;
  progress: string | null;

  threshold: number;
  fullResolution: boolean;

  collapsedRows: Set<string>;
  pendingMerges: Map<string, Set<string>>; // refGroupId → candidate groupIds
  selectionAnchors: Map<string, string>; // refGroupId → last-toggled candidateId (range-select anchor)
  undoStack: ImageGroup[][];

  setThreshold: (t: number) => void;
  setFullResolution: (v: boolean) => void;
  fetchSuggestions: () => Promise<void>;
  toggleRowCollapse: (groupId: string) => void;
  collapseAllRows: () => void;
  expandAllRows: () => void;
  toggleMergeCandidate: (refId: string, candidateId: string) => void;
  rangeSelectInRow: (refId: string, candidateId: string) => void;
  selectAllInRow: (refId: string) => void;
  deselectAllInRow: (refId: string) => void;
  clearPendingMerges: () => void;
  applyMerges: () => Promise<void>;
  undo: () => Promise<void>;
  pendingMergeCount: () => number;
}

export const useMergeSuggestionsStore = create<MergeSuggestionsState>((set, get) => ({
  suggestions: null,
  loading: false,
  error: null,
  computeTimeMs: null,
  progress: null,

  threshold: 0.65,
  fullResolution: false,

  collapsedRows: new Set(),
  pendingMerges: new Map(),
  selectionAnchors: new Map(),
  undoStack: [],

  setThreshold: (t) => set({ threshold: t }),
  setFullResolution: (v) => set({ fullResolution: v }),

  fetchSuggestions: async () => {
    const { threshold, fullResolution } = get();
    set({ loading: true, error: null, progress: "Starting..." });
    try {
      const resp = await postJson("/api/merge-suggestions", {
        threshold,
        maxPerGroup: MAX_PER_GROUP,
        fullResolution,
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.error || "Failed to fetch suggestions");
      }
      await consumeSSE(resp, {
        onProgress: (message) => set({ progress: message }),
        onResult: (data) => {
          const result = data as MergeSuggestionsResponse;
          set({
            suggestions: result.suggestions,
            computeTimeMs: result.computeTimeMs,
            loading: false,
            progress: null,
            pendingMerges: new Map(),
          });
        },
        onError: (error) => {
          set({ loading: false, error, progress: null });
        },
      });
      if (get().loading) {
        set({ loading: false, progress: null, error: "Stream ended unexpectedly" });
      }
    } catch (err) {
      set({
        loading: false,
        progress: null,
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
      const anchors = new Map(s.selectionAnchors);
      anchors.set(refId, candidateId);
      return { pendingMerges: next, selectionAnchors: anchors };
    });
  },

  rangeSelectInRow: (refId, candidateId) => {
    const { suggestions, selectionAnchors } = get();
    const row = suggestions?.find((s) => s.refGroupId === refId);
    if (!row) return;
    const anchor = selectionAnchors.get(refId);
    const endIdx = row.similar.findIndex((c) => c.groupId === candidateId);
    if (endIdx === -1) return;
    const anchorIdx = anchor ? row.similar.findIndex((c) => c.groupId === anchor) : -1;
    const [lo, hi] =
      anchorIdx === -1
        ? [endIdx, endIdx]
        : [Math.min(anchorIdx, endIdx), Math.max(anchorIdx, endIdx)];
    set((s) => {
      const next = new Map(s.pendingMerges);
      const updated = new Set(next.get(refId) ?? new Set());
      for (let i = lo; i <= hi; i++) updated.add(row.similar[i]!.groupId);
      next.set(refId, updated);
      const anchors = new Map(s.selectionAnchors);
      anchors.set(refId, candidateId);
      return { pendingMerges: next, selectionAnchors: anchors };
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
    await flushGroupPersist();
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
    await flushGroupPersist();
    await get().fetchSuggestions();
  },

  pendingMergeCount: () => {
    const { pendingMerges } = get();
    let count = 0;
    for (const candidates of pendingMerges.values()) count += candidates.size;
    return count;
  },
}));
