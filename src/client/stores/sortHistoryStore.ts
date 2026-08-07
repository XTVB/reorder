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

// A sort applies its order over several writes (setImages, then updateGroups to
// mirror it into group contents), all synchronous within one handler — so the
// window closes on the next microtask rather than after a fixed write count.
let sortWindowOpen = false;

export function openSortWriteWindow(): void {
  if (sortWindowOpen) return;
  sortWindowOpen = true;
  queueMicrotask(() => {
    sortWindowOpen = false;
  });
}

/** Drops the snapshot unless this write belongs to a sort in progress. */
export function noteOrderWrite(): void {
  if (sortWindowOpen) return;
  useSortHistoryStore.getState().clearPreviousOrder();
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
