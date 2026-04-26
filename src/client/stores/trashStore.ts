import { create } from "zustand";
import { postJson } from "../utils/helpers.ts";
import { useUIStore } from "./uiStore.ts";

interface DeleteResponse {
  success: boolean;
  deleted: string[];
  missing: string[];
  warnings?: string[];
  error?: string;
}

interface TrashState {
  markedIds: Set<string>;

  mark: (filenames: string[]) => void;
  unmark: (filenames: string[]) => void;
  toggle: (filename: string) => void;
  clear: () => void;
  pruneToValid: (validFilenames: Iterable<string>) => void;
  confirmDelete: () => Promise<DeleteResponse>;
}

export const useTrashStore = create<TrashState>((set, get) => ({
  markedIds: new Set(),

  mark: (filenames) => {
    if (filenames.length === 0) return;
    const current = get().markedIds;
    const next = new Set(current);
    let changed = false;
    for (const fn of filenames) {
      if (!next.has(fn)) {
        next.add(fn);
        changed = true;
      }
    }
    if (changed) set({ markedIds: next });
  },

  unmark: (filenames) => {
    if (filenames.length === 0) return;
    const next = new Set(get().markedIds);
    let changed = false;
    for (const fn of filenames) {
      if (next.delete(fn)) changed = true;
    }
    if (changed) set({ markedIds: next });
  },

  toggle: (filename) => {
    const next = new Set(get().markedIds);
    if (next.has(filename)) next.delete(filename);
    else next.add(filename);
    set({ markedIds: next });
  },

  clear: () => {
    if (get().markedIds.size === 0) return;
    set({ markedIds: new Set() });
  },

  pruneToValid: (validFilenames) => {
    const { markedIds } = get();
    if (markedIds.size === 0) return;
    const valid = validFilenames instanceof Set ? validFilenames : new Set(validFilenames);
    const next = new Set<string>();
    for (const fn of markedIds) if (valid.has(fn)) next.add(fn);
    if (next.size !== markedIds.size) set({ markedIds: next });
  },

  confirmDelete: async () => {
    const filenames = [...get().markedIds];
    if (filenames.length === 0) {
      return { success: true, deleted: [], missing: [] };
    }
    const res = await postJson("/api/delete", { filenames });
    const data = (await res.json()) as DeleteResponse;
    if (!res.ok || !data.success) {
      throw new Error(data.error ?? "Delete failed");
    }
    set({ markedIds: new Set() });
    useUIStore.getState().setShowTrashModal(false);
    return data;
  },
}));
