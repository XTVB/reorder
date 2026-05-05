// Single source of truth for the lightbox modal. Consolidates four prior
// states: `uiStore.lightboxIndex` (Reorder), `clusterStore.lightbox` (Cluster),
// and the local useState lightboxes in MergeSuggestions.tsx and
// ClusterCompare.tsx.

import { create } from "zustand";

export type LightboxSource =
  | "reorder"
  | "cluster"
  | "compare"
  | "merge"
  | "review"
  | "trash"
  | "nn";

interface LightboxState {
  open: boolean;
  filenames: string[];
  index: number;
  source: LightboxSource | null;
  /** Optional metadata for cluster mode: which cluster the images came from. */
  clusterId: string | null;

  openLightbox: (
    filenames: string[],
    startIndex: number,
    source: LightboxSource,
    clusterId?: string,
  ) => void;
  close: () => void;
  setIndex: (index: number) => void;
  next: () => void;
  prev: () => void;
}

export const useLightboxStore = create<LightboxState>((set, get) => ({
  open: false,
  filenames: [],
  index: 0,
  source: null,
  clusterId: null,

  openLightbox: (filenames, startIndex, source, clusterId) => {
    if (filenames.length === 0) return;
    const index = Math.min(Math.max(0, startIndex), filenames.length - 1);
    set({ open: true, filenames, index, source, clusterId: clusterId ?? null });
  },
  close: () => set({ open: false, filenames: [], index: 0, source: null, clusterId: null }),
  setIndex: (index) => {
    const { filenames } = get();
    if (filenames.length === 0) return;
    const clamped = Math.min(Math.max(0, index), filenames.length - 1);
    if (clamped !== get().index) set({ index: clamped });
  },
  next: () => {
    const { filenames, index } = get();
    if (filenames.length === 0) return;
    const nextIdx = (index + 1) % filenames.length;
    set({ index: nextIdx });
  },
  prev: () => {
    const { filenames, index } = get();
    if (filenames.length === 0) return;
    const prevIdx = (index - 1 + filenames.length) % filenames.length;
    set({ index: prevIdx });
  },
}));
