// Cross-mode selection — one store, one API, multiple named contexts.
//
// Flat contexts hold a `Set<string>` of selected ids; the per-row context
// "merge-suggestions" holds `Map<rowKey, Set<id>>` because each row carries
// its own selection. Range-select anchors are tracked per context.

import { create } from "zustand";

export type SelectionContext =
  | "reorder" // grid filenames
  | "cluster:images" // composite "clusterId:filename"
  | "cluster:merge" // cluster IDs in merge bar
  | "expand" // filenames in ExpandModal
  | "nn" // result filenames in NNResultsModal
  | "trash"; // filenames marked for trash

export type RowSelectionContext = "merge-suggestions";

interface SelectionState {
  contexts: Record<SelectionContext, Set<string>>;
  anchors: Record<SelectionContext, string | null>;
  rowSelections: Record<RowSelectionContext, Map<string, Set<string>>>;
  rowAnchors: Record<RowSelectionContext, Map<string, string>>;

  get: (ctx: SelectionContext) => Set<string>;
  isSelected: (ctx: SelectionContext, id: string) => boolean;
  toggle: (ctx: SelectionContext, id: string, anchor?: string) => void;
  add: (ctx: SelectionContext, ids: Iterable<string>) => void;
  remove: (ctx: SelectionContext, ids: Iterable<string>) => void;
  setSelection: (ctx: SelectionContext, ids: Iterable<string>) => void;
  rangeSelect: (ctx: SelectionContext, allIds: string[], to: string) => void;
  clear: (ctx: SelectionContext) => void;
  /** Clear every context whose name starts with the prefix. */
  clearMode: (prefix: string) => void;
  /** Drop ids that aren't in `valid` from the given context. */
  pruneToValid: (ctx: SelectionContext, valid: Set<string>) => void;
  /** Apply a rename map to the given context's ids. */
  remap: (ctx: SelectionContext, renameMap: Map<string, string>) => void;

  // Row-context (merge-suggestions): per-row selections.
  getRow: (ctx: RowSelectionContext, rowKey: string) => Set<string>;
  toggleInRow: (ctx: RowSelectionContext, rowKey: string, id: string) => void;
  rangeSelectInRow: (
    ctx: RowSelectionContext,
    rowKey: string,
    allIds: string[],
    to: string,
  ) => void;
  selectAllInRow: (ctx: RowSelectionContext, rowKey: string, ids: Iterable<string>) => void;
  deselectAllInRow: (ctx: RowSelectionContext, rowKey: string) => void;
  clearRowContext: (ctx: RowSelectionContext) => void;
}

const EMPTY_SET: Set<string> = new Set();

const INITIAL_CONTEXTS: Record<SelectionContext, Set<string>> = {
  reorder: new Set(),
  "cluster:images": new Set(),
  "cluster:merge": new Set(),
  expand: new Set(),
  nn: new Set(),
  trash: new Set(),
};

const INITIAL_ANCHORS: Record<SelectionContext, string | null> = {
  reorder: null,
  "cluster:images": null,
  "cluster:merge": null,
  expand: null,
  nn: null,
  trash: null,
};

export const useSelectionStore = create<SelectionState>((set, get) => ({
  contexts: { ...INITIAL_CONTEXTS },
  anchors: { ...INITIAL_ANCHORS },
  rowSelections: { "merge-suggestions": new Map() },
  rowAnchors: { "merge-suggestions": new Map() },

  get: (ctx) => get().contexts[ctx] ?? EMPTY_SET,
  isSelected: (ctx, id) => (get().contexts[ctx] ?? EMPTY_SET).has(id),

  toggle: (ctx, id, anchor) => {
    const cur = get().contexts[ctx];
    const next = new Set(cur);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    set({
      contexts: { ...get().contexts, [ctx]: next },
      anchors: { ...get().anchors, [ctx]: anchor ?? id },
    });
  },

  add: (ctx, ids) => {
    const cur = get().contexts[ctx];
    const next = new Set(cur);
    let changed = false;
    for (const id of ids) {
      if (!next.has(id)) {
        next.add(id);
        changed = true;
      }
    }
    if (!changed) return;
    set({ contexts: { ...get().contexts, [ctx]: next } });
  },

  remove: (ctx, ids) => {
    const cur = get().contexts[ctx];
    if (cur.size === 0) return;
    const next = new Set(cur);
    let changed = false;
    for (const id of ids) {
      if (next.delete(id)) changed = true;
    }
    if (!changed) return;
    set({ contexts: { ...get().contexts, [ctx]: next } });
  },

  setSelection: (ctx, ids) => {
    set({ contexts: { ...get().contexts, [ctx]: new Set(ids) } });
  },

  rangeSelect: (ctx, allIds, to) => {
    const anchor = get().anchors[ctx];
    const cur = get().contexts[ctx];
    const toIdx = allIds.indexOf(to);
    if (toIdx === -1) return;
    if (anchor === null) {
      set({
        contexts: { ...get().contexts, [ctx]: new Set([to]) },
        anchors: { ...get().anchors, [ctx]: to },
      });
      return;
    }
    const fromIdx = allIds.indexOf(anchor);
    if (fromIdx === -1) {
      set({
        contexts: { ...get().contexts, [ctx]: new Set([to]) },
        anchors: { ...get().anchors, [ctx]: to },
      });
      return;
    }
    const lo = Math.min(fromIdx, toIdx);
    const hi = Math.max(fromIdx, toIdx);
    const next = new Set(cur);
    for (let i = lo; i <= hi; i++) {
      const id = allIds[i];
      if (id) next.add(id);
    }
    set({ contexts: { ...get().contexts, [ctx]: next } });
  },

  clear: (ctx) => {
    if (get().contexts[ctx].size === 0 && get().anchors[ctx] === null) return;
    set({
      contexts: { ...get().contexts, [ctx]: new Set() },
      anchors: { ...get().anchors, [ctx]: null },
    });
  },

  clearMode: (prefix) => {
    const contexts = { ...get().contexts };
    const anchors = { ...get().anchors };
    let changed = false;
    for (const key of Object.keys(contexts) as SelectionContext[]) {
      if (key.startsWith(prefix)) {
        if (contexts[key].size > 0) {
          contexts[key] = new Set();
          changed = true;
        }
        if (anchors[key] !== null) {
          anchors[key] = null;
          changed = true;
        }
      }
    }
    if (changed) set({ contexts, anchors });
  },

  pruneToValid: (ctx, valid) => {
    const cur = get().contexts[ctx];
    if (cur.size === 0) return;
    const next = new Set<string>();
    for (const id of cur) if (valid.has(id)) next.add(id);
    if (next.size !== cur.size) {
      set({ contexts: { ...get().contexts, [ctx]: next } });
    }
  },

  remap: (ctx, renameMap) => {
    const cur = get().contexts[ctx];
    if (cur.size === 0 || renameMap.size === 0) return;
    const next = new Set<string>();
    let changed = false;
    for (const id of cur) {
      const mapped = renameMap.get(id);
      if (mapped) {
        next.add(mapped);
        changed = true;
      } else {
        next.add(id);
      }
    }
    if (changed) set({ contexts: { ...get().contexts, [ctx]: next } });
  },

  getRow: (ctx, rowKey) => get().rowSelections[ctx].get(rowKey) ?? EMPTY_SET,

  toggleInRow: (ctx, rowKey, id) => {
    const map = new Map(get().rowSelections[ctx]);
    const cur = map.get(rowKey) ?? new Set<string>();
    const next = new Set(cur);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    if (next.size === 0) map.delete(rowKey);
    else map.set(rowKey, next);
    const anchorMap = new Map(get().rowAnchors[ctx]);
    anchorMap.set(rowKey, id);
    set({
      rowSelections: { ...get().rowSelections, [ctx]: map },
      rowAnchors: { ...get().rowAnchors, [ctx]: anchorMap },
    });
  },

  rangeSelectInRow: (ctx, rowKey, allIds, to) => {
    const anchor = get().rowAnchors[ctx].get(rowKey);
    const toIdx = allIds.indexOf(to);
    if (toIdx === -1) return;
    const anchorIdx = anchor ? allIds.indexOf(anchor) : -1;
    const [lo, hi] =
      anchorIdx === -1 ? [toIdx, toIdx] : [Math.min(anchorIdx, toIdx), Math.max(anchorIdx, toIdx)];
    const map = new Map(get().rowSelections[ctx]);
    const cur = map.get(rowKey) ?? new Set<string>();
    const next = new Set(cur);
    for (let i = lo; i <= hi; i++) {
      const id = allIds[i];
      if (id) next.add(id);
    }
    map.set(rowKey, next);
    const anchorMap = new Map(get().rowAnchors[ctx]);
    anchorMap.set(rowKey, to);
    set({
      rowSelections: { ...get().rowSelections, [ctx]: map },
      rowAnchors: { ...get().rowAnchors, [ctx]: anchorMap },
    });
  },

  selectAllInRow: (ctx, rowKey, ids) => {
    const map = new Map(get().rowSelections[ctx]);
    map.set(rowKey, new Set(ids));
    set({ rowSelections: { ...get().rowSelections, [ctx]: map } });
  },

  deselectAllInRow: (ctx, rowKey) => {
    const map = new Map(get().rowSelections[ctx]);
    if (!map.has(rowKey)) return;
    map.delete(rowKey);
    set({ rowSelections: { ...get().rowSelections, [ctx]: map } });
  },

  clearRowContext: (ctx) => {
    if (get().rowSelections[ctx].size === 0) return;
    set({
      rowSelections: { ...get().rowSelections, [ctx]: new Map() },
      rowAnchors: { ...get().rowAnchors, [ctx]: new Map() },
    });
  },
}));
