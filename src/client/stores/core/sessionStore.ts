// Session-wide UI bag. Holds everything that's neither selection, modal,
// lightbox, nor toast: targetDir, undo availability, header subtitle,
// preview/organize transient state, slideshow prefs, saving spinner, error
// banner. Persists slideshow + numbered-folder prefs to localStorage.

import { create } from "zustand";
import { getJson } from "../../api/client.ts";
import type { CanUndoResponse, DirResponse, OrganizeMapping, RenameMapping } from "../../types.ts";

const SLIDESHOW_INTERVAL_KEY = "reorder-slideshow-interval";
const SLIDESHOW_SHUFFLE_KEY = "reorder-slideshow-shuffle";
const SLIDESHOW_TRANSITION_KEY = "reorder-slideshow-transition";
const NUMBERED_FOLDER_PREFIX_KEY = "reorder-numbered-folder-prefix";
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

export interface SlideshowState {
  open: boolean;
  startIndex: number;
  playing: boolean;
  intervalMs: number;
  shuffle: boolean;
  transition: SlideshowTransition;
}

interface SessionState {
  saving: boolean;
  error: string | null;
  canUndo: boolean;
  targetDir: string;
  previewRenames: RenameMapping[];
  organizeMappings: OrganizeMapping[];
  headerSubtitle: string;
  numberedFolderPrefix: boolean;
  slideshow: SlideshowState;

  setSaving: (saving: boolean) => void;
  setError: (error: string | null) => void;
  setPreviewRenames: (renames: RenameMapping[]) => void;
  setOrganizeMappings: (mappings: OrganizeMapping[]) => void;
  checkUndo: () => Promise<void>;
  fetchTargetDir: () => Promise<void>;
  setHeaderSubtitle: (s: string) => void;
  setNumberedFolderPrefix: (v: boolean) => void;

  openSlideshow: (startIndex: number) => void;
  closeSlideshow: () => void;
  setSlideshowPlaying: (playing: boolean) => void;
  setSlideshowInterval: (ms: number) => void;
  setSlideshowShuffle: (shuffle: boolean) => void;
  setSlideshowTransition: (transition: SlideshowTransition) => void;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  saving: false,
  error: null,
  canUndo: false,
  targetDir: "",
  previewRenames: [],
  organizeMappings: [],
  headerSubtitle: "",
  numberedFolderPrefix: localStorage.getItem(NUMBERED_FOLDER_PREFIX_KEY) !== "false",
  slideshow: {
    open: false,
    startIndex: 0,
    playing: false,
    intervalMs: readStoredInterval(),
    shuffle: localStorage.getItem(SLIDESHOW_SHUFFLE_KEY) === "true",
    transition: readStoredTransition(),
  },

  setSaving: (saving) => set({ saving }),
  setError: (error) => set({ error }),
  setPreviewRenames: (renames) => set({ previewRenames: renames }),
  setOrganizeMappings: (mappings) => set({ organizeMappings: mappings }),

  checkUndo: async () => {
    try {
      const { canUndo } = await getJson<CanUndoResponse>("/api/can-undo");
      set({ canUndo });
    } catch {
      set({ canUndo: false });
    }
  },

  fetchTargetDir: async () => {
    try {
      const { dir } = await getJson<DirResponse>("/api/dir");
      set({ targetDir: dir });
    } catch {}
  },

  setHeaderSubtitle: (s) => {
    if (s !== get().headerSubtitle) set({ headerSubtitle: s });
  },

  setNumberedFolderPrefix: (v) => {
    localStorage.setItem(NUMBERED_FOLDER_PREFIX_KEY, String(v));
    set({ numberedFolderPrefix: v });
  },

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
}));
