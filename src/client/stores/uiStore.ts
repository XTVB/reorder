import { create } from "zustand";
import type {
  CanUndoResponse,
  DirResponse,
  OrganizeMapping,
  RenameMapping,
  Toast,
} from "../types.ts";

let _toastTimer: ReturnType<typeof setTimeout> | undefined;

const SLIDESHOW_INTERVAL_KEY = "reorder-slideshow-interval";
const SLIDESHOW_SHUFFLE_KEY = "reorder-slideshow-shuffle";
const SLIDESHOW_TRANSITION_KEY = "reorder-slideshow-transition";
const SLIDESHOW_INTERVAL_DEFAULT = 3000;
const SLIDESHOW_INTERVAL_MIN = 250;
const SLIDESHOW_INTERVAL_MAX = 60000;

export type SlideshowTransition = "fade" | "none";
const SLIDESHOW_TRANSITIONS: readonly SlideshowTransition[] = ["fade", "none"];

function readStoredTransition(): SlideshowTransition {
  const raw = localStorage.getItem(SLIDESHOW_TRANSITION_KEY);
  return SLIDESHOW_TRANSITIONS.includes(raw as SlideshowTransition)
    ? (raw as SlideshowTransition)
    : "fade";
}

function clampInterval(ms: number): number {
  if (!Number.isFinite(ms)) return SLIDESHOW_INTERVAL_DEFAULT;
  return Math.min(SLIDESHOW_INTERVAL_MAX, Math.max(SLIDESHOW_INTERVAL_MIN, Math.round(ms)));
}

function readStoredInterval(): number {
  const raw = localStorage.getItem(SLIDESHOW_INTERVAL_KEY);
  if (!raw) return SLIDESHOW_INTERVAL_DEFAULT;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? clampInterval(parsed) : SLIDESHOW_INTERVAL_DEFAULT;
}

interface SlideshowState {
  open: boolean;
  startIndex: number;
  playing: boolean;
  intervalMs: number;
  shuffle: boolean;
  transition: SlideshowTransition;
}

interface UIState {
  lightboxIndex: number | null;
  saving: boolean;
  error: string | null;
  showPreview: boolean;
  showOrganize: boolean;
  showPaths: boolean;
  showReview: boolean;
  showGroupPicker: boolean;
  slideshow: SlideshowState;
  toast: Toast | null;
  canUndo: boolean;
  targetDir: string;
  previewRenames: RenameMapping[];
  organizeMappings: OrganizeMapping[];
  headerSubtitle: string;

  openLightbox: (index: number) => void;
  closeLightbox: () => void;
  setSaving: (saving: boolean) => void;
  setError: (error: string | null) => void;
  setShowPreview: (show: boolean) => void;
  setShowOrganize: (show: boolean) => void;
  setShowPaths: (show: boolean) => void;
  setShowReview: (show: boolean) => void;
  setShowGroupPicker: (show: boolean) => void;
  openSlideshow: (startIndex: number) => void;
  closeSlideshow: () => void;
  setSlideshowPlaying: (playing: boolean) => void;
  setSlideshowInterval: (ms: number) => void;
  setSlideshowShuffle: (shuffle: boolean) => void;
  setSlideshowTransition: (transition: SlideshowTransition) => void;
  showToast: (message: string, type: Toast["type"]) => void;
  setPreviewRenames: (renames: RenameMapping[]) => void;
  setOrganizeMappings: (mappings: OrganizeMapping[]) => void;
  checkUndo: () => Promise<void>;
  fetchTargetDir: () => Promise<void>;
  setHeaderSubtitle: (s: string) => void;
}

export const useUIStore = create<UIState>((set, get) => ({
  lightboxIndex: null,
  saving: false,
  error: null,
  showPreview: false,
  showOrganize: false,
  showPaths: false,
  showReview: false,
  showGroupPicker: false,
  slideshow: {
    open: false,
    startIndex: 0,
    playing: false,
    intervalMs: readStoredInterval(),
    shuffle: localStorage.getItem(SLIDESHOW_SHUFFLE_KEY) === "true",
    transition: readStoredTransition(),
  },
  toast: null,
  canUndo: false,
  targetDir: "",
  previewRenames: [],
  organizeMappings: [],
  headerSubtitle: "",

  openLightbox: (index) => set({ lightboxIndex: index }),
  closeLightbox: () => set({ lightboxIndex: null }),
  setSaving: (saving) => set({ saving }),
  setError: (error) => set({ error }),
  setShowPreview: (show) => set({ showPreview: show }),
  setShowOrganize: (show) => set({ showOrganize: show }),
  setShowPaths: (show) => set({ showPaths: show }),
  setShowReview: (show) => set({ showReview: show }),
  setShowGroupPicker: (show) => set({ showGroupPicker: show }),

  openSlideshow: (startIndex) =>
    set((s) => ({
      slideshow: { ...s.slideshow, open: true, startIndex, playing: false },
    })),
  closeSlideshow: () =>
    set((s) => ({ slideshow: { ...s.slideshow, open: false, playing: false } })),
  setSlideshowPlaying: (playing) => set((s) => ({ slideshow: { ...s.slideshow, playing } })),
  setSlideshowInterval: (ms) => {
    const clamped = clampInterval(ms);
    localStorage.setItem(SLIDESHOW_INTERVAL_KEY, String(clamped));
    set((s) => ({ slideshow: { ...s.slideshow, intervalMs: clamped } }));
  },
  setSlideshowShuffle: (shuffle) => {
    localStorage.setItem(SLIDESHOW_SHUFFLE_KEY, String(shuffle));
    set((s) => ({ slideshow: { ...s.slideshow, shuffle } }));
  },
  setSlideshowTransition: (transition) => {
    localStorage.setItem(SLIDESHOW_TRANSITION_KEY, transition);
    set((s) => ({ slideshow: { ...s.slideshow, transition } }));
  },

  showToast: (message, type) => {
    if (_toastTimer) clearTimeout(_toastTimer);
    set({ toast: { message, type } });
    _toastTimer = setTimeout(() => set({ toast: null }), 3000);
  },

  setPreviewRenames: (renames) => set({ previewRenames: renames }),
  setOrganizeMappings: (mappings) => set({ organizeMappings: mappings }),

  checkUndo: async () => {
    try {
      const res = await fetch("/api/can-undo");
      const { canUndo }: CanUndoResponse = await res.json();
      set({ canUndo });
    } catch {
      set({ canUndo: false });
    }
  },

  setHeaderSubtitle: (s) => {
    if (s !== get().headerSubtitle) set({ headerSubtitle: s });
  },

  fetchTargetDir: async () => {
    try {
      const res = await fetch("/api/dir");
      const { dir }: DirResponse = await res.json();
      set({ targetDir: dir });
    } catch {}
  },
}));
