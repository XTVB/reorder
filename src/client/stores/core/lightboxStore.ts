// Single source of truth for the lightbox modal across all modes and modals.
// One global `open`/`source` pair lets each consumer gate `source === "x"` and
// render its own <Lightbox /> from the snapshotted `filenames`. Modal-scoped
// consumers should call `useCloseLightboxOnUnmount(source)` so state doesn't
// leak between sessions.

import { useEffect } from "react";
import { create } from "zustand";

export type LightboxSource =
  | "reorder"
  | "cluster"
  | "compare"
  | "merge"
  | "review"
  | "trash"
  | "expand"
  | "nn";

interface LightboxState {
  open: boolean;
  filenames: string[];
  index: number;
  source: LightboxSource | null;

  openLightbox: (filenames: string[], startIndex: number, source: LightboxSource) => void;
  close: () => void;
}

export const useLightboxStore = create<LightboxState>((set) => ({
  open: false,
  filenames: [],
  index: 0,
  source: null,

  openLightbox: (filenames, startIndex, source) => {
    if (filenames.length === 0) return;
    const index = Math.min(Math.max(0, startIndex), filenames.length - 1);
    set({ open: true, filenames, index, source });
  },
  close: () => set({ open: false, filenames: [], index: 0, source: null }),
}));

export function useCloseLightboxOnUnmount(source: LightboxSource) {
  useEffect(() => {
    return () => {
      const s = useLightboxStore.getState();
      if (s.source === source) s.close();
    };
  }, [source]);
}
