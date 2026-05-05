// Trash actions facade. Selected filenames live in selectionStore "trash"
// context; this store provides the action surface (mark/unmark/clear/remap/
// confirmDelete).

import { create } from "zustand";
import { postJson } from "../api/client.ts";
import type { RenameMapping } from "../types.ts";
import { useModalStore } from "./core/modalStore.ts";
import { useSelectionStore } from "./core/selectionStore.ts";

interface DeleteResponse {
  success: boolean;
  deleted: string[];
  missing: string[];
  warnings?: string[];
  error?: string;
}

interface TrashState {
  mark: (filenames: string[]) => void;
  unmark: (filenames: string[]) => void;
  toggle: (filename: string) => void;
  clear: () => void;
  pruneToValid: (validFilenames: Iterable<string>) => void;
  remap: (renames: RenameMapping[]) => void;
  confirmDelete: () => Promise<DeleteResponse>;
}

export const useTrashStore = create<TrashState>(() => ({
  mark: (filenames) => {
    if (filenames.length === 0) return;
    useSelectionStore.getState().add("trash", filenames);
  },

  unmark: (filenames) => {
    if (filenames.length === 0) return;
    useSelectionStore.getState().remove("trash", filenames);
  },

  toggle: (filename) => {
    useSelectionStore.getState().toggle("trash", filename);
  },

  clear: () => {
    useSelectionStore.getState().clear("trash");
  },

  pruneToValid: (validFilenames) => {
    const valid = validFilenames instanceof Set ? validFilenames : new Set(validFilenames);
    useSelectionStore.getState().pruneToValid("trash", valid as Set<string>);
  },

  remap: (renames) => {
    if (renames.length === 0) return;
    const map = new Map<string, string>();
    for (const r of renames) if (r.from !== r.to) map.set(r.from, r.to);
    if (map.size === 0) return;
    useSelectionStore.getState().remap("trash", map);
  },

  confirmDelete: async () => {
    const filenames = [...useSelectionStore.getState().contexts.trash];
    if (filenames.length === 0) {
      return { success: true, deleted: [], missing: [] };
    }
    const data = await postJson<DeleteResponse>("/api/delete", { filenames });
    if (!data.success) {
      throw new Error(data.error ?? "Delete failed");
    }
    useSelectionStore.getState().clear("trash");
    useModalStore.getState().closeModal("trash");
    return data;
  },
}));
