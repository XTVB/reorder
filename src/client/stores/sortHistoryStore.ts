import { create } from "zustand";
import type { ImageGroup, ImageInfo } from "../types.ts";

/** Gallery order as it was just before the last applied sort. */
export interface OrderSnapshot {
  images: ImageInfo[];
  groups: ImageGroup[];
}

interface SortHistoryState {
  /** Pre-sort order kept for the ⌥ compare peek; null until a sort is applied. */
  previousOrder: OrderSnapshot | null;
  /** True while ⌥ is held and the grid is showing previousOrder. */
  peeking: boolean;

  setPreviousOrder: (snapshot: OrderSnapshot) => void;
  /** Drop the snapshot when it goes stale (e.g. filenames changed on save). */
  clearPreviousOrder: () => void;
  setPeeking: (peeking: boolean) => void;
}

export const useSortHistoryStore = create<SortHistoryState>((set, get) => ({
  previousOrder: null,
  peeking: false,

  setPreviousOrder: (previousOrder) => set({ previousOrder }),
  clearPreviousOrder: () => {
    if (get().previousOrder !== null || get().peeking) {
      set({ previousOrder: null, peeking: false });
    }
  },
  setPeeking: (peeking) => {
    if (get().peeking !== peeking) set({ peeking });
  },
}));
