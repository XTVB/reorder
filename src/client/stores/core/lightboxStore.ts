// Single source of truth for the lightbox modal. One global open/filenames/index
// triple drives the single <Lightbox /> rendered at the app shell level.

import { create } from "zustand";

interface LightboxOptions {
  /** Show the mark-for-deletion control (reorder workflow). Default true;
   * czkawka mode passes false — its deletes are immediate, not marked. */
  trashMark?: boolean;
  /** Explicit image URLs, parallel to filenames. When omitted the viewer
   * falls back to the target-dir image route; czkawka passes these so files
   * from other comparison directories render too. */
  urls?: string[];
}

interface LightboxState {
  open: boolean;
  filenames: string[];
  urls: string[] | null;
  index: number;
  trashMark: boolean;
  // Bumped on every openLightbox so the view can remount on a fresh session —
  // needed because the inner view seeds its index once and reads filenames
  // reactively, so re-opening while already open would otherwise desync them.
  seq: number;

  openLightbox: (filenames: string[], startIndex: number, options?: LightboxOptions) => void;
  /** Replace the image set in place — no remount, so zoom/pan survive. Used
   * when an action mutates the current group while the lightbox is open.
   * Closes when the new set is empty. */
  updateFilenames: (filenames: string[], urls?: string[]) => void;
  close: () => void;
}

export const useLightboxStore = create<LightboxState>((set) => ({
  open: false,
  filenames: [],
  urls: null,
  index: 0,
  trashMark: true,
  seq: 0,

  openLightbox: (filenames, startIndex, options) => {
    if (filenames.length === 0) return;
    const index = Math.min(Math.max(0, startIndex), filenames.length - 1);
    set((s) => ({
      open: true,
      filenames,
      urls: options?.urls ?? null,
      index,
      trashMark: options?.trashMark ?? true,
      seq: s.seq + 1,
    }));
  },

  updateFilenames: (filenames, urls) => {
    set((s) => {
      if (!s.open) return {};
      if (filenames.length === 0) return { open: false, filenames: [], urls: null, index: 0 };
      return {
        filenames,
        urls: urls ?? s.urls,
        index: Math.min(s.index, filenames.length - 1),
      };
    });
  },

  close: () => set({ open: false, filenames: [], urls: null, index: 0 }),
}));
