// Locked-image actions facade. Locked ungrouped filenames live in the
// selectionStore "lock" context; this store is the action surface (toggle/
// clear/prune/remap). Locking is a session concept — like the unsaved reorder
// arrangement it applies to, it is not persisted to disk.
//
// Semantics: a locked ungrouped image keeps its RELATIVE order among the other
// locked images when "Sort Similar" runs on the ungrouped set (or a selection).
// Unlocked images may still be reordered around and between locked ones — only
// the locked subsequence is pinned. See reorderSubsetWithinSlots in
// utils/reorder.ts for how the pin is applied to the server's proposed order.

import { create } from "zustand";
import type { RenameMapping } from "../types.ts";
import { useSelectionStore } from "./core/selectionStore.ts";
import { useToastStore } from "./core/toastStore.ts";

interface LockedImagesState {
  /**
   * Toggle the lock on the given filenames: lock all of them unless every one
   * is already locked, in which case unlock all. Mirrors the group-lock toggle.
   */
  toggleLocked: (filenames: string[]) => void;
  clear: () => void;
  pruneToValid: (validFilenames: Iterable<string>) => void;
  remap: (renames: RenameMapping[]) => void;
}

export const useLockedImagesStore = create<LockedImagesState>(() => ({
  toggleLocked: (filenames) => {
    if (filenames.length === 0) return;
    const sel = useSelectionStore.getState();
    const lockSet = sel.contexts.lock;
    const lock = !filenames.every((fn) => lockSet.has(fn));
    if (lock) sel.add("lock", filenames);
    else sel.remove("lock", filenames);
    useToastStore
      .getState()
      .showToast(
        lock
          ? `Locked ${filenames.length} image${filenames.length === 1 ? "" : "s"} — relative order kept when sorting`
          : `Unlocked ${filenames.length} image${filenames.length === 1 ? "" : "s"}`,
        "success",
      );
  },

  clear: () => {
    useSelectionStore.getState().clear("lock");
  },

  pruneToValid: (validFilenames) => {
    const valid = validFilenames instanceof Set ? validFilenames : new Set(validFilenames);
    useSelectionStore.getState().pruneToValid("lock", valid as Set<string>);
  },

  remap: (renames) => {
    if (renames.length === 0) return;
    const map = new Map<string, string>();
    for (const r of renames) if (r.from !== r.to) map.set(r.from, r.to);
    if (map.size === 0) return;
    useSelectionStore.getState().remap("lock", map);
  },
}));
