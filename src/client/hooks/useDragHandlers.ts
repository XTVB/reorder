import type { DragEndEvent, DragStartEvent } from "@dnd-kit/core";
import { arrayMove } from "@dnd-kit/sortable";
import { useCallback, useEffect, useRef } from "react";
import { useDndStore } from "../stores/dndStore.ts";
import { useFolderStore } from "../stores/folderStore.ts";
import { useGroupStore } from "../stores/groupStore.ts";
import { useImageStore } from "../stores/imageStore.ts";
import { useSelectionStore } from "../stores/selectionStore.ts";
import { computeGridItems, gridItemId } from "../utils/gridItems.ts";
import {
  fromFolderSortId,
  isFolderSortId,
  isGroupSortId,
  setDragEndTime,
  toFolderSortId,
} from "../utils/helpers.ts";
import { flattenOrder, multiDragReorder, repositionBlock } from "../utils/reorder.ts";

interface DragHandlersDeps {
  addImagesToGroup: (groupId: string, filenames: string[]) => void;
  handleGroupReorder: (groupId: string, newOrder: string[]) => void;
}

export function useDragHandlers({ addImagesToGroup, handleGroupReorder }: DragHandlersDeps) {
  // Subscribe only to what effects need as dependencies
  const activeId = useDndStore((s) => s.activeId);
  const setFrozenGroup = useDndStore((s) => s.setFrozenGroupId);
  const setDragOverGroupId = useDndStore((s) => s.setDragOverGroupId);

  const expandedGroupIdRef = useRef<string | null>(null);
  expandedGroupIdRef.current = useGroupStore.getState().expandedGroupId;
  const expandedFolderNameRef = useRef<string | null>(null);
  expandedFolderNameRef.current = useFolderStore.getState().expandedFolderName;

  // Pointer tracking for dwell-based group/folder drop
  useEffect(() => {
    if (!activeId || isGroupSortId(activeId) || isFolderSortId(activeId)) {
      setFrozenGroup(null);
      setDragOverGroupId(null);
      return;
    }

    let dwellTimer: ReturnType<typeof setTimeout> | undefined;
    let currentHoverGroupId: string | null = null;
    let rafId: number | null = null;

    function findGroupAtPoint(x: number, y: number): string | null {
      const els = document.elementsFromPoint(x, y);
      for (const el of els) {
        const popoverEl = el.closest("[data-group-popover]") as HTMLElement | null;
        if (popoverEl?.dataset.groupPopover) return popoverEl.dataset.groupPopover;
        const folderPopoverEl = el.closest("[data-folder-popover]") as HTMLElement | null;
        if (folderPopoverEl?.dataset.folderPopover)
          return toFolderSortId(folderPopoverEl.dataset.folderPopover);
        // Check folder cards
        const folderEl = el.closest("[data-folder-name]") as HTMLElement | null;
        if (folderEl?.dataset.folderName) {
          if (folderEl.dataset.folderName !== expandedFolderNameRef.current) {
            return toFolderSortId(folderEl.dataset.folderName);
          }
        }
        // Check group cards
        const groupEl = el.closest("[data-group-id]") as HTMLElement | null;
        if (groupEl?.dataset.groupId && groupEl.dataset.groupId !== expandedGroupIdRef.current) {
          return groupEl.dataset.groupId;
        }
      }
      return null;
    }

    function onPointerMove(e: PointerEvent) {
      if (rafId !== null) return;
      rafId = requestAnimationFrame(() => {
        rafId = null;
        const groupId = findGroupAtPoint(e.clientX, e.clientY);
        if (groupId === currentHoverGroupId) return;
        clearTimeout(dwellTimer);
        if (currentHoverGroupId) {
          setFrozenGroup(null);
          setDragOverGroupId(null);
        }
        currentHoverGroupId = groupId;
        if (groupId) {
          setFrozenGroup(groupId);
          dwellTimer = setTimeout(() => setDragOverGroupId(groupId), 300);
        }
      });
    }

    window.addEventListener("pointermove", onPointerMove);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      clearTimeout(dwellTimer);
      if (rafId !== null) cancelAnimationFrame(rafId);
      setFrozenGroup(null);
      setDragOverGroupId(null);
    };
  }, [activeId, setFrozenGroup, setDragOverGroupId]);

  // Drag start
  const handleDragStartImpl = (event: DragStartEvent) => {
    const { setActiveId } = useDndStore.getState();
    const { selectedIds, clearSelection } = useSelectionStore.getState();
    const id = event.active.id as string;
    setActiveId(id);
    if (selectedIds.size > 0 && !selectedIds.has(id)) {
      clearSelection();
    }
  };
  const dragStartRef = useRef(handleDragStartImpl);
  dragStartRef.current = handleDragStartImpl;
  const handleDragStart = useCallback((event: DragStartEvent) => dragStartRef.current(event), []);

  // Drag end
  const handleDragEndImpl = (event: DragEndEvent) => {
    const { dragOverGroupId, clearDrag } = useDndStore.getState();
    const { selectedIds, removeFromSelection } = useSelectionStore.getState();
    const { images, setImages } = useImageStore.getState();
    const { groups, groupsEnabled, expandedGroupId, groupMap, updateGroups, collapseGroup } =
      useGroupStore.getState();
    const {
      folderModeEnabled,
      folders,
      expandedFolderName,
      reorderFolders,
      moveImages,
      reorderWithinFolder,
    } = useFolderStore.getState();

    const dropGroupId = dragOverGroupId;
    clearDrag();
    setDragEndTime();
    const { active, over } = event;
    if (!active) return;

    const aid = active.id as string;

    // ---- Folder mode drag end ----
    if (folderModeEnabled) {
      const isAidFolder = isFolderSortId(aid);

      // Handle image dropped onto a folder card (via dwell)
      if (!isAidFolder && dropGroupId && isFolderSortId(dropGroupId)) {
        const targetFolderName = fromFolderSortId(dropGroupId);
        moveImages([aid], targetFolderName);
        return;
      }

      if (!over || active.id === over.id) return;
      const oid = over.id as string;

      // Folder-to-folder reorder
      if (isAidFolder && isFolderSortId(oid)) {
        const folderNames = folders.map((f) => f.name);
        const aidName = fromFolderSortId(aid);
        const oidName = fromFolderSortId(oid);
        const oldIdx = folderNames.indexOf(aidName);
        const newIdx = folderNames.indexOf(oidName);
        if (oldIdx >= 0 && newIdx >= 0 && oldIdx !== newIdx) {
          reorderFolders(arrayMove([...folderNames], oldIdx, newIdx));
        }
        return;
      }

      // Reorder within expanded folder (image-to-image in same folder)
      if (!isAidFolder && !isFolderSortId(oid) && expandedFolderName) {
        const folder = folders.find((f) => f.name === expandedFolderName);
        if (folder) {
          const compoundList = folder.images.map((fn) => `${folder.name}/${fn}`);
          const expandedCompound = new Set(compoundList);
          if (expandedCompound.has(aid) && expandedCompound.has(oid)) {
            const oldIdx = compoundList.indexOf(aid);
            const newIdx = compoundList.indexOf(oid);
            if (oldIdx >= 0 && newIdx >= 0) {
              reorderWithinFolder(folder.name, arrayMove([...compoundList], oldIdx, newIdx));
            }
            return;
          }
        }
      }

      return;
    }

    // ---- Regular (non-folder) mode drag end ----
    const isAidGroup = isGroupSortId(aid);
    const toAdd =
      !isAidGroup && selectedIds.size > 0 && selectedIds.has(aid) ? [...selectedIds] : [aid];

    if (dropGroupId && dropGroupId !== expandedGroupId && !isAidGroup && groupsEnabled) {
      addImagesToGroup(dropGroupId, toAdd);
      return;
    }

    if (dropGroupId && dropGroupId === expandedGroupId && (!over || active.id === over.id)) {
      addImagesToGroup(dropGroupId, toAdd);
      return;
    }

    if (!over || active.id === over.id) return;
    const oid = over.id as string;

    const gridItems = computeGridItems(images, {
      mode: "groups",
      groups,
      enabled: groupsEnabled,
      expandedGroupId,
    });
    const gridIds = gridItems.map(gridItemId);

    const expandedGroup = expandedGroupId ? (groupMap.get(expandedGroupId) ?? null) : null;
    const expandedSet = expandedGroup ? new Set(expandedGroup.images) : null;
    const activeInExpanded = expandedSet?.has(aid) ?? false;
    const overInExpanded = expandedSet?.has(oid) ?? false;

    // Reorder within expanded group
    if (activeInExpanded && overInExpanded && expandedGroup) {
      const selectedInGroup =
        selectedIds.size > 0 && selectedIds.has(aid)
          ? new Set([...selectedIds].filter((fn) => expandedSet!.has(fn)))
          : null;
      const newOrder =
        selectedInGroup && selectedInGroup.size > 1
          ? selectedInGroup.has(oid)
            ? null
            : multiDragReorder(expandedGroup.images, selectedInGroup, aid, oid)
          : arrayMove(
              [...expandedGroup.images],
              expandedGroup.images.indexOf(aid),
              expandedGroup.images.indexOf(oid),
            );
      if (!newOrder) return;
      handleGroupReorder(expandedGroup.id, newOrder);
      return;
    }

    // Dragged out of expanded group
    if (activeInExpanded && !overInExpanded && expandedGroup) {
      const toRemove =
        selectedIds.size > 0 && selectedIds.has(aid)
          ? new Set([...selectedIds].filter((fn) => expandedSet!.has(fn)))
          : new Set([aid]);
      const updatedGroups = groups
        .map((g) =>
          g.id === expandedGroup.id
            ? { ...g, images: g.images.filter((fn) => !toRemove.has(fn)) }
            : g,
        )
        .filter((g) => g.images.length > 0);
      updateGroups(() => updatedGroups);
      if (!updatedGroups.some((g) => g.id === expandedGroupId)) collapseGroup();
      const newIds =
        toRemove.size > 1
          ? multiDragReorder(gridIds, toRemove, aid, oid)
          : arrayMove([...gridIds], gridIds.indexOf(aid), gridIds.indexOf(oid));
      setImages(flattenOrder(newIds, updatedGroups, images));
      return;
    }

    // Outside image dragged into expanded group
    if (!activeInExpanded && overInExpanded && expandedGroup) {
      const insertIdx = expandedGroup.images.indexOf(oid);
      const aidPos = gridIds.indexOf(aid);
      const oidPos = gridIds.indexOf(oid);
      const newGroupImages = [...expandedGroup.images];
      newGroupImages.splice(aidPos < oidPos ? insertIdx + 1 : insertIdx, 0, aid);
      const updatedGroups = groups.map((g) => {
        if (g.id === expandedGroup.id) return { ...g, images: newGroupImages };
        return { ...g, images: g.images.filter((fn) => fn !== aid) };
      });
      updateGroups(() => updatedGroups);
      setImages(repositionBlock(images, newGroupImages));
      removeFromSelection([aid]);
      return;
    }

    // Default: normal grid reorder
    const isMultiDrag = selectedIds.size > 1 && selectedIds.has(aid);
    const newIds = isMultiDrag
      ? selectedIds.has(oid)
        ? null
        : multiDragReorder(gridIds, selectedIds, aid, oid)
      : arrayMove([...gridIds], gridIds.indexOf(aid), gridIds.indexOf(oid));
    if (!newIds) return;
    setImages(flattenOrder(newIds, groups, images));
  };
  const dragEndRef = useRef(handleDragEndImpl);
  dragEndRef.current = handleDragEndImpl;
  const handleDragEnd = useCallback((event: DragEndEvent) => dragEndRef.current(event), []);

  return { handleDragStart, handleDragEnd };
}
