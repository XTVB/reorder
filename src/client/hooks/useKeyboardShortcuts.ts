import { useEffect, useRef } from "react";
import { useGroupStore } from "../stores/groupStore.ts";
import { useSelectionStore } from "../stores/selectionStore.ts";

interface KeyboardShortcutsDeps {
  isLightboxOpen: boolean;
  searchState: { isOpen: boolean; close: () => void };
  onCreateGroup: () => void;
}

export function useKeyboardShortcuts({
  isLightboxOpen,
  searchState,
  onCreateGroup,
}: KeyboardShortcutsDeps) {
  const collapseGroup = useGroupStore((s) => s.collapseGroup);
  const clearSelection = useSelectionStore((s) => s.clearSelection);

  const lightboxOpenRef = useRef(false);
  lightboxOpenRef.current = isLightboxOpen;
  const createGroupRef = useRef(onCreateGroup);
  createGroupRef.current = onCreateGroup;

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
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
      if (e.key === "g" && !e.metaKey && !e.ctrlKey && !(e.target instanceof HTMLInputElement)) {
        const { groupsEnabled } = useGroupStore.getState();
        const { selectedIds } = useSelectionStore.getState();
        if (groupsEnabled && selectedIds.size > 0) createGroupRef.current();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [collapseGroup, clearSelection, searchState]);
}
