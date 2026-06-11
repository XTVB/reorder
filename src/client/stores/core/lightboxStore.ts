// Single source of truth for the lightbox modal. One global open/filenames/index
// triple drives the single <Lightbox /> rendered at the app shell level.

import { create } from "zustand";

interface LightboxState {
  open: boolean;
  filenames: string[];
  index: number;
  // Bumped on every openLightbox so the view can remount on a fresh session —
  // needed because the inner view seeds its index once and reads filenames
  // reactively, so re-opening while already open would otherwise desync them.
  seq: number;

  openLightbox: (filenames: string[], startIndex: number) => void;
  close: () => void;
}

export const useLightboxStore = create<LightboxState>((set) => ({
  open: false,
  filenames: [],
  index: 0,
  seq: 0,

  openLightbox: (filenames, startIndex) => {
    if (filenames.length === 0) return;
    const index = Math.min(Math.max(0, startIndex), filenames.length - 1);
    set((s) => ({ open: true, filenames, index, seq: s.seq + 1 }));
  },
  close: () => set({ open: false, filenames: [], index: 0 }),
}));
