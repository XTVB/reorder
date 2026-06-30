import { create } from "zustand";
import { consumeSSE, startSSE } from "../api/sse.ts";
import type { ImageGroup, MergeSuggestionRow, MergeSuggestionsResponse } from "../types.ts";
import { useSelectionStore } from "./core/selectionStore.ts";
import { useGroupStore } from "./groupStore.ts";
import { useListStore } from "./modes/cluster/listStore.ts";

const MAX_PER_GROUP = 8;

export type MergeSortMode = "topMatch" | "groupOrder";
/** "patches" = DINOv3 patch matching; "embeddings" = weighted CLS-embedding blend. */
export type MergeMethod = "patches" | "embeddings";

/**
 * Per-method "Min similarity" slider bounds + default. Embedding-cosine scores
 * sit higher and span a wider band than patch-match scores (medians observed
 * ~0.40–0.83), so the embeddings range is wider and shifted up.
 */
export const METHOD_THRESHOLDS: Record<
  MergeMethod,
  { min: number; max: number; step: number; default: number }
> = {
  patches: { min: 0.5, max: 0.8, step: 0.01, default: 0.65 },
  embeddings: { min: 0.6, max: 0.95, step: 0.01, default: 0.8 },
};

const clampThreshold = (t: number, method: MergeMethod) => {
  const { min, max } = METHOD_THRESHOLDS[method];
  return Math.min(max, Math.max(min, t));
};

interface MergeSuggestionsState {
  suggestions: MergeSuggestionRow[] | null;
  loading: boolean;
  error: string | null;
  computeTimeMs: number | null;
  progress: string | null;

  threshold: number;
  method: MergeMethod;
  fullResolution: boolean;
  maxCombinedSize: number;
  sortMode: MergeSortMode;

  collapsedRows: Set<string>;
  undoStack: ImageGroup[][];

  setThreshold: (t: number) => void;
  setMethod: (m: MergeMethod) => void;
  setFullResolution: (v: boolean) => void;
  setMaxCombinedSize: (n: number) => void;
  setSortMode: (m: MergeSortMode) => void;
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

  threshold: METHOD_THRESHOLDS.embeddings.default,
  method: "embeddings",
  fullResolution: false,
  maxCombinedSize: 0,
  sortMode: "topMatch",

  collapsedRows: new Set(),
  undoStack: [],

  setThreshold: (t) => set({ threshold: t }),
  // Clamp the threshold into the new method's range so the slider stays valid.
  setMethod: (m) => set((s) => ({ method: m, threshold: clampThreshold(s.threshold, m) })),
  setFullResolution: (v) => set({ fullResolution: v }),
  setMaxCombinedSize: (n) => set({ maxCombinedSize: Math.max(0, Math.floor(n)) }),
  setSortMode: (m) => set({ sortMode: m }),

  fetchSuggestions: async () => {
    const { threshold, method, fullResolution, maxCombinedSize } = get();
    set({ loading: true, error: null, progress: "Starting..." });
    try {
      // Embeddings mode reuses the cluster pipeline's per-model weights.
      const weights = useListStore.getState().weights;
      const start = await startSSE("/api/merge-suggestions", {
        threshold,
        maxPerGroup: MAX_PER_GROUP,
        method,
        fullResolution,
        maxCombinedSize,
        weights,
      });
      if (start.kind === "conflict") {
        set({ loading: false, progress: null, error: start.message });
        return;
      }
      await consumeSSE(start.response, {
        onProgress: (message) => set({ progress: message }),
        onResult: (data) => {
          const result = data as MergeSuggestionsResponse;
          useSelectionStore.getState().clearRowContext("merge-suggestions");
          set({
            suggestions: result.suggestions,
            computeTimeMs: result.computeTimeMs,
            loading: false,
            progress: null,
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
    useSelectionStore.getState().toggleInRow("merge-suggestions", refId, candidateId);
  },

  rangeSelectInRow: (refId, candidateId) => {
    const row = get().suggestions?.find((s) => s.refGroupId === refId);
    if (!row) return;
    const allIds = row.similar.map((c) => c.groupId);
    useSelectionStore.getState().rangeSelectInRow("merge-suggestions", refId, allIds, candidateId);
  },

  selectAllInRow: (refId) => {
    const row = get().suggestions?.find((s) => s.refGroupId === refId);
    if (!row) return;
    useSelectionStore.getState().selectAllInRow(
      "merge-suggestions",
      refId,
      row.similar.map((c) => c.groupId),
    );
  },

  deselectAllInRow: (refId) => {
    useSelectionStore.getState().deselectAllInRow("merge-suggestions", refId);
  },

  clearPendingMerges: () => {
    useSelectionStore.getState().clearRowContext("merge-suggestions");
  },

  applyMerges: async () => {
    const pendingMerges = useSelectionStore.getState().rowSelections["merge-suggestions"];
    const { undoStack } = get();
    if (pendingMerges.size === 0) return;

    const groupStore = useGroupStore.getState();
    const currentGroups = [...groupStore.groups.map((g) => ({ ...g, images: [...g.images] }))];

    const newUndoStack = [...undoStack, currentGroups].slice(-10);

    const removedIds = new Set<string>();
    for (const candidateIds of pendingMerges.values()) {
      for (const candId of candidateIds) removedIds.add(candId);
    }

    groupStore.updateGroups((prev) => {
      const groups = prev.map((g) => ({ ...g, images: [...g.images] }));
      const byId = new Map(groups.map((g) => [g.id, g]));

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
        }
      }

      return groups.filter((g) => !removedIds.has(g.id));
    });

    // No recompute on Apply — user iterates merge-by-merge and runs Compute when ready.
    const groupById = new Map(useGroupStore.getState().groups.map((g) => [g.id, g]));
    set((s) => {
      if (!s.suggestions) return {};
      const nextSuggestions: typeof s.suggestions = [];
      for (const row of s.suggestions) {
        if (removedIds.has(row.refGroupId)) continue;
        const filteredSimilar = row.similar.filter((c) => !removedIds.has(c.groupId));
        if (filteredSimilar.length === 0) continue;
        const refGroup = groupById.get(row.refGroupId);
        nextSuggestions.push({
          ...row,
          refGroupImages: refGroup ? refGroup.images : row.refGroupImages,
          similar: filteredSimilar.map((c) => {
            const g = groupById.get(c.groupId);
            return g ? { ...c, groupImages: g.images } : c;
          }),
        });
      }
      return { suggestions: nextSuggestions };
    });

    useSelectionStore.getState().clearRowContext("merge-suggestions");
    set({ undoStack: newUndoStack });
    await useGroupStore.getState().flushPending();
  },

  undo: async () => {
    const { undoStack } = get();
    if (undoStack.length === 0) return;

    const previousGroups = undoStack[undoStack.length - 1]!;
    const newStack = undoStack.slice(0, -1);

    const groupStore = useGroupStore.getState();
    groupStore.updateGroups(() => previousGroups);

    useSelectionStore.getState().clearRowContext("merge-suggestions");
    set({ undoStack: newStack });
    await useGroupStore.getState().flushPending();
    await get().fetchSuggestions();
  },

  pendingMergeCount: () => {
    const pendingMerges = useSelectionStore.getState().rowSelections["merge-suggestions"];
    let count = 0;
    for (const candidates of pendingMerges.values()) count += candidates.size;
    return count;
  },
}));
