import { create } from "zustand";
import type { Toast, RenameMapping, OrganizeMapping, DirResponse, CanUndoResponse, AppMode } from "../types.ts";

let _toastTimer: ReturnType<typeof setTimeout> | undefined;

interface UIState {
  lightboxIndex: number | null;
  saving: boolean;
  error: string | null;
  showPreview: boolean;
  showOrganize: boolean;
  showPaths: boolean;
  toast: Toast | null;
  canUndo: boolean;
  targetDir: string;
  previewRenames: RenameMapping[];
  organizeMappings: OrganizeMapping[];
  appMode: AppMode;

  openLightbox: (index: number) => void;
  closeLightbox: () => void;
  setSaving: (saving: boolean) => void;
  setError: (error: string | null) => void;
  setShowPreview: (show: boolean) => void;
  setShowOrganize: (show: boolean) => void;
  setShowPaths: (show: boolean) => void;
  showToast: (message: string, type: Toast["type"]) => void;
  setPreviewRenames: (renames: RenameMapping[]) => void;
  setOrganizeMappings: (mappings: OrganizeMapping[]) => void;
  checkUndo: () => Promise<void>;
  fetchTargetDir: () => Promise<void>;
  setAppMode: (mode: AppMode) => void;
}

export const useUIStore = create<UIState>((set) => ({
  lightboxIndex: null,
  saving: false,
  error: null,
  showPreview: false,
  showOrganize: false,
  showPaths: false,
  toast: null,
  canUndo: false,
  targetDir: "",
  previewRenames: [],
  organizeMappings: [],
  appMode: "reorder",

  openLightbox: (index) => set({ lightboxIndex: index }),
  closeLightbox: () => set({ lightboxIndex: null }),
  setSaving: (saving) => set({ saving }),
  setError: (error) => set({ error }),
  setShowPreview: (show) => set({ showPreview: show }),
  setShowOrganize: (show) => set({ showOrganize: show }),
  setShowPaths: (show) => set({ showPaths: show }),

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

  setAppMode: (mode) => set({ appMode: mode }),

  fetchTargetDir: async () => {
    try {
      const res = await fetch("/api/dir");
      const { dir }: DirResponse = await res.json();
      set({ targetDir: dir });
    } catch {}
  },
}));
