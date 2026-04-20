import { useEffect, useRef } from "react";
import { useGroupStore } from "../stores/groupStore.ts";
import { useSelectionStore } from "../stores/selectionStore.ts";
import { useUIStore } from "../stores/uiStore.ts";

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
  const clearSelection = useSelectionStore((s) => s.clearSelection);

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
        else if (useSelectionStore.getState().selectedIds.size > 0) clearSelection();
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "g") {
        const { groupsEnabled } = useGroupStore.getState();
        const { selectedIds } = useSelectionStore.getState();
        if (groupsEnabled && selectedIds.size > 0) createGroupRef.current();
      } else if (e.key === "h") {
        const { groupsEnabled, groups } = useGroupStore.getState();
        const { selectedIds } = useSelectionStore.getState();
        if (groupsEnabled && selectedIds.size > 0 && groups.length > 0) {
          useUIStore.getState().setShowGroupPicker(true);
        }
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [collapseGroup, clearSelection, searchState]);
}
