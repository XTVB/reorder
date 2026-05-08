// Single source of truth for the lightbox modal. One global open/filenames/index
// triple drives the single <Lightbox /> rendered at the app shell level.

import { create } from "zustand";

interface LightboxState {
  open: boolean;
  filenames: string[];
  index: number;

  openLightbox: (filenames: string[], startIndex: number) => void;
  close: () => void;
}

export const useLightboxStore = create<LightboxState>((set) => ({
  open: false,
  filenames: [],
  index: 0,

  openLightbox: (filenames, startIndex) => {
    if (filenames.length === 0) return;
    const index = Math.min(Math.max(0, startIndex), filenames.length - 1);
    set({ open: true, filenames, index });
  },
  close: () => set({ open: false, filenames: [], index: 0 }),
}));
