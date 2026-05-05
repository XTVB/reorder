import { useEffect, useRef } from "react";
import { useModalStore } from "../stores/core/modalStore.ts";
import { useSelectionStore } from "../stores/core/selectionStore.ts";
import { useGroupStore } from "../stores/groupStore.ts";
import { useTrashStore } from "../stores/trashStore.ts";
import { selectedImageFilenames } from "../utils/helpers.ts";

interface KeyboardShortcutsDeps {
  isLightboxOpen: boolean;
  isSlideshowOpen: boolean;
  searchState: { isOpen: boolean; close: () => void };
  onCreateGroup: () => void;
}

export function useKeyboardShortcuts({
  isLightboxOpen,
  isSlideshowOpen,
  searchState,
  onCreateGroup,
}: KeyboardShortcutsDeps) {
  const collapseGroup = useGroupStore((s) => s.collapseGroup);
  const clear = useSelectionStore((s) => s.clear);

  const lightboxOpenRef = useRef(false);
  lightboxOpenRef.current = isLightboxOpen;
  const slideshowOpenRef = useRef(false);
  slideshowOpenRef.current = isSlideshowOpen;
  const createGroupRef = useRef(onCreateGroup);
  createGroupRef.current = onCreateGroup;

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (slideshowOpenRef.current) return;
      if (e.key === "Escape") {
        if (lightboxOpenRef.current) return;
        if (searchState.isOpen) {
          searchState.close();
          return;
        }
        const expId = useGroupStore.getState().expandedGroupId;
        if (expId) collapseGroup();
        else if (useSelectionStore.getState().contexts.reorder.size > 0) clear("reorder");
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "g") {
        const { groupsEnabled } = useGroupStore.getState();
        const selectedIds = useSelectionStore.getState().contexts.reorder;
        if (groupsEnabled && selectedIds.size > 0) createGroupRef.current();
      } else if (e.key === "h") {
        const { groupsEnabled, groups } = useGroupStore.getState();
        const selectedIds = useSelectionStore.getState().contexts.reorder;
        if (groupsEnabled && selectedIds.size > 0 && groups.length > 0) {
          useModalStore.getState().openModal("groupPicker");
        }
      } else if (e.key === "d" || e.key === "D") {
        const fns = selectedImageFilenames(useSelectionStore.getState().contexts.reorder);
        if (fns.length === 0) return;
        const trash = useTrashStore.getState();
        const trashSet = useSelectionStore.getState().contexts.trash;
        const allMarked = fns.every((fn) => trashSet.has(fn));
        if (allMarked) trash.unmark(fns);
        else trash.mark(fns);
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [collapseGroup, clear, searchState]);
}
