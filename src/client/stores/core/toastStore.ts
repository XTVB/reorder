// Toast notifications. Extracted from uiStore. Single-toast queue with a
// 3-second auto-clear timer; calling `showToast` resets the timer so the
// most recent message stays visible.

import { create } from "zustand";
import type { Toast } from "../../types.ts";

interface ToastState {
  toast: Toast | null;
  showToast: (message: string, type: Toast["type"]) => void;
  clearToast: () => void;
}

let _toastTimer: ReturnType<typeof setTimeout> | undefined;

export const useToastStore = create<ToastState>((set) => ({
  toast: null,

  showToast: (message, type) => {
    if (_toastTimer) clearTimeout(_toastTimer);
    set({ toast: { message, type } });
    _toastTimer = setTimeout(() => set({ toast: null }), 3000);
  },

  clearToast: () => {
    if (_toastTimer) clearTimeout(_toastTimer);
    set({ toast: null });
  },
}));
