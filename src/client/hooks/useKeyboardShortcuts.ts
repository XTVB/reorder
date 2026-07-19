import { useEffect, useRef } from "react";
import { useModalStore } from "../stores/core/modalStore.ts";
import { useSelectionStore } from "../stores/core/selectionStore.ts";
import { useFolderStore } from "../stores/folderStore.ts";
import { useGroupStore } from "../stores/groupStore.ts";
import { useLockedImagesStore } from "../stores/lockedImagesStore.ts";
import { useTrashStore } from "../stores/trashStore.ts";
import {
  fromFolderSortId,
  fromGroupSortId,
  isFolderSortId,
  isGroupSortId,
  selectedImageFilenames,
} from "../utils/helpers.ts";
import { reverseSelection } from "../utils/reverseSelection.ts";

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
        if (useSelectionStore.getState().contexts.reorder.size > 0) clear("reorder");
        else if (useGroupStore.getState().expandedGroupId) collapseGroup();
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "g" || e.key === "G") {
        const { groupsEnabled, createGroupFromSelectionAutoNamed } = useGroupStore.getState();
        const selectedIds = useSelectionStore.getState().contexts.reorder;
        if (groupsEnabled && selectedIds.size > 0) {
          if (e.shiftKey) createGroupFromSelectionAutoNamed();
          else createGroupRef.current();
        }
      } else if (e.key === "h") {
        const { groupsEnabled, groups } = useGroupStore.getState();
        const selectedIds = useSelectionStore.getState().contexts.reorder;
        if (groupsEnabled && selectedIds.size > 0 && groups.length > 0) {
          useModalStore.getState().openModal("groupPicker");
        }
      } else if (e.key === "m" || e.key === "M") {
        const { groupsEnabled, mergeGroups } = useGroupStore.getState();
        if (!groupsEnabled) return;
        const groupIds = [...useSelectionStore.getState().contexts.reorder]
          .filter(isGroupSortId)
          .map(fromGroupSortId);
        if (groupIds.length >= 2) mergeGroups(groupIds);
      } else if (e.key === "r" || e.key === "R") {
        if (useSelectionStore.getState().contexts.reorder.size >= 2) reverseSelection();
      } else if (e.key === "l" || e.key === "L") {
        // Toggle the sort lock on the selection. Groups keep their gallery
        // slot; ungrouped images keep their relative order (see
        // lockedImagesStore). Guard against the lightbox and modals — the
        // grouping-sort modal uses l/h for navigation.
        if (lightboxOpenRef.current) return;
        if (Object.values(useModalStore.getState().open).some(Boolean)) return;
        const selection = [...useSelectionStore.getState().contexts.reorder];
        const groupIds = selection.filter(isGroupSortId).map(fromGroupSortId);
        if (groupIds.length > 0) useGroupStore.getState().toggleGroupsLocked(groupIds);
        // Locking individual images only makes sense in the normal grid, not
        // folder mode (where plain ids are a folder's contents).
        if (!useFolderStore.getState().folderModeEnabled) {
          const imageFilenames = selectedImageFilenames(selection);
          if (imageFilenames.length > 0)
            useLockedImagesStore.getState().toggleLocked(imageFilenames);
        }
      } else if (e.key === "d" || e.key === "D") {
        const selection = useSelectionStore.getState().contexts.reorder;
        if (selection.size === 0) return;
        const groupMap = useGroupStore.getState().groupMap;
        const folderMap = useFolderStore.getState().folderMap;
        const fns = new Set<string>();
        for (const id of selection) {
          if (isGroupSortId(id)) {
            const group = groupMap.get(fromGroupSortId(id));
            if (group) for (const fn of group.images) fns.add(fn);
          } else if (isFolderSortId(id)) {
            const folder = folderMap.get(fromFolderSortId(id));
            if (folder) for (const fn of folder.images) fns.add(fn);
          } else {
            fns.add(id);
          }
        }
        useTrashStore.getState().toggleMany([...fns]);
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [collapseGroup, clear, searchState]);
}
