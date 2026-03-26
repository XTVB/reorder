import { create } from "zustand";

interface SelectionState {
  selectedIds: Set<string>;
  lastClickedIndex: number | null;

  toggleSelect: (id: string, index: number) => void;
  rangeSelect: (index: number, allIds: { id: string; index: number }[]) => void;
  clearSelection: () => void;
  selectAll: (ids: string[]) => void;
  removeFromSelection: (ids: string[]) => void;
}

export const useSelectionStore = create<SelectionState>((set, get) => ({
  selectedIds: new Set(),
  lastClickedIndex: null,

  toggleSelect: (id, index) => {
    const { selectedIds } = get();
    const next = new Set(selectedIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    set({ selectedIds: next, lastClickedIndex: index });
  },

  rangeSelect: (index, allIds) => {
    const { selectedIds, lastClickedIndex } = get();
    if (lastClickedIndex !== null) {
      const start = Math.min(lastClickedIndex, index);
      const end = Math.max(lastClickedIndex, index);
      const next = new Set(selectedIds);
      for (const item of allIds) {
        if (item.index >= start && item.index <= end) {
          next.add(item.id);
        }
      }
      set({ selectedIds: next });
    } else {
      const item = allIds.find((i) => i.index === index);
      if (item) set({ selectedIds: new Set([item.id]), lastClickedIndex: index });
    }
  },

  clearSelection: () => {
    if (get().selectedIds.size === 0) return;
    set({ selectedIds: new Set() });
  },

  selectAll: (ids) => set({ selectedIds: new Set(ids) }),

  removeFromSelection: (ids) => {
    const { selectedIds } = get();
    if (!ids.some((id) => selectedIds.has(id))) return;
    const next = new Set(selectedIds);
    for (const id of ids) next.delete(id);
    set({ selectedIds: next });
  },
}));
