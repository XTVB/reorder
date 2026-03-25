import React, { useEffect, useCallback, useRef, useMemo } from "react";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragOverlay,
  type DragStartEvent,
  type DragEndEvent,
  type CollisionDetection,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  rectSortingStrategy,
} from "@dnd-kit/sortable";
import { useVirtualizer } from "@tanstack/react-virtual";

import type { ImageInfo, RenameMapping } from "./types.ts";
import { useImageStore } from "./stores/imageStore.ts";
import { useSelectionStore } from "./stores/selectionStore.ts";
import { useDndStore } from "./stores/dndStore.ts";
import { useGroupStore } from "./stores/groupStore.ts";
import { useUIStore } from "./stores/uiStore.ts";
import {
  isGroupSortId,
  fromGroupSortId,
  toGroupSortId,
  getErrorMessage,
  imageUrl,
  setDragEndTime,
  postJson,
} from "./utils/helpers.ts";
import { computeGridItems, gridItemId } from "./utils/gridItems.ts";
import {
  multiDragReorder,
  flattenOrder,
  consolidateBlock,
  repositionBlock,
  remapGroupsAfterSave,
} from "./utils/reorder.ts";

import { Toolbar } from "./components/Toolbar.tsx";
import { SortableCard } from "./components/SortableCard.tsx";
import { SortableGroupCard } from "./components/SortableGroupCard.tsx";
import { ExpandedGroupItem } from "./components/ExpandedGroupItem.tsx";
import { GroupThumbGrid } from "./components/GroupThumbGrid.tsx";
import { Lightbox } from "./components/Lightbox.tsx";
import { Modal } from "./components/Modal.tsx";
import { SearchBar, SearchContext, useSearchState } from "./components/SearchBar.tsx";

/* ------------------------------------------------------------------ */
/*  Grid layout hook                                                   */
/* ------------------------------------------------------------------ */

const INFO_HEIGHT = 40; // card-info bar: 8px padding + 24px badge + 8px padding

/**
 * Reads column count and row height from actual CSS grid layout.
 * An empty hidden div with the same auto-fill grid rule is rendered in the
 * flow; CSS decides columns and we read them via getComputedStyle.
 * Row height is derived from the resolved track width (thumbnail is
 * aspect-ratio:1 so height = width) + the fixed info bar + gap.
 */
function useGridLayout() {
  const [layout, setLayout] = React.useState({ columnCount: 6, rowHeight: 220 });
  const observerRef = useRef<ResizeObserver | null>(null);

  const measureRowRef = useCallback((node: HTMLDivElement | null) => {
    if (observerRef.current) {
      observerRef.current.disconnect();
      observerRef.current = null;
    }

    if (!node) return;

    function measure() {
      if (!node) return;

      // CSS resolved grid-template-columns is e.g. "200px 200px 200px ..."
      const tracks = getComputedStyle(node).gridTemplateColumns;
      const trackValues = tracks.split(" ");
      const cols = Math.max(1, trackValues.length);

      // Card width from the first track; thumbnail is aspect-ratio:1, so height = width
      const trackWidth = parseFloat(trackValues[0]!) || 160;
      const gap = parseFloat(getComputedStyle(node).gap) || 16;
      const rowHeight = Math.ceil(trackWidth + INFO_HEIGHT + gap);

      setLayout((prev) =>
        prev.columnCount === cols && prev.rowHeight === rowHeight
          ? prev
          : { columnCount: cols, rowHeight }
      );
    }

    const observer = new ResizeObserver(() => measure());
    observer.observe(node);
    observerRef.current = observer;
    requestAnimationFrame(measure);
  }, []);

  return { ...layout, measureRowRef };
}

/* ------------------------------------------------------------------ */
/*  App                                                                */
/* ------------------------------------------------------------------ */

export function App() {
  // ---- Store subscriptions ----
  const images = useImageStore((s) => s.images);
  const imageMap = useImageStore((s) => s.imageMap);
  const loading = useImageStore((s) => s.loading);
  const setImages = useImageStore((s) => s.setImages);
  const fetchImages = useImageStore((s) => s.fetchImages);

  const selectedIds = useSelectionStore((s) => s.selectedIds);
  const clearSelection = useSelectionStore((s) => s.clearSelection);
  const toggleSelect = useSelectionStore((s) => s.toggleSelect);
  const rangeSelect = useSelectionStore((s) => s.rangeSelect);
  const removeFromSelection = useSelectionStore((s) => s.removeFromSelection);

  const activeId = useDndStore((s) => s.activeId);
  const dragOverGroupId = useDndStore((s) => s.dragOverGroupId);
  const frozenGroupId = useDndStore((s) => s.frozenGroupId);
  const setActiveId = useDndStore((s) => s.setActiveId);
  const setDragOverGroupId = useDndStore((s) => s.setDragOverGroupId);
  const setFrozenGroup = useDndStore((s) => s.setFrozenGroupId);
  const clearDrag = useDndStore((s) => s.clearDrag);

  const groups = useGroupStore((s) => s.groups);
  const groupsEnabled = useGroupStore((s) => s.groupsEnabled);
  const expandedGroupId = useGroupStore((s) => s.expandedGroupId);
  const groupMap = useGroupStore((s) => s.groupMap);
  const groupsLoaded = useGroupStore((s) => s.groupsLoaded);
  const updateGroups = useGroupStore((s) => s.updateGroups);
  const expandGroup = useGroupStore((s) => s.expandGroup);
  const collapseGroup = useGroupStore((s) => s.collapseGroup);
  const fetchGroups = useGroupStore((s) => s.fetchGroups);

  const lightboxIndex = useUIStore((s) => s.lightboxIndex);
  const saving = useUIStore((s) => s.saving);
  const error = useUIStore((s) => s.error);
  const showPreview = useUIStore((s) => s.showPreview);
  const showOrganize = useUIStore((s) => s.showOrganize);
  const showPaths = useUIStore((s) => s.showPaths);
  const toast = useUIStore((s) => s.toast);
  const targetDir = useUIStore((s) => s.targetDir);
  const previewRenames = useUIStore((s) => s.previewRenames);
  const organizeMappings = useUIStore((s) => s.organizeMappings);
  const openLightbox = useUIStore((s) => s.openLightbox);
  const closeLightbox = useUIStore((s) => s.closeLightbox);
  const showToast = useUIStore((s) => s.showToast);
  const setSaving = useUIStore((s) => s.setSaving);
  const setError = useUIStore((s) => s.setError);
  const setShowPreview = useUIStore((s) => s.setShowPreview);
  const setShowOrganize = useUIStore((s) => s.setShowOrganize);
  const setShowPaths = useUIStore((s) => s.setShowPaths);
  const checkUndo = useUIStore((s) => s.checkUndo);
  const bumpCacheNonce = useUIStore((s) => s.bumpCacheNonce);
  const fetchTargetDir = useUIStore((s) => s.fetchTargetDir);

  // ---- DnD sensors ----
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );

  // ---- Computed grid ----
  const gridItems = useMemo(
    () => computeGridItems(images, groups, groupsEnabled, expandedGroupId),
    [images, groups, groupsEnabled, expandedGroupId]
  );
  const gridIds = useMemo(() => gridItems.map(gridItemId), [gridItems]);

  // Visible items (excludes group-image which render inside popovers)
  const visibleItems = useMemo(
    () => gridItems.filter((item) => item.type !== "group-image"),
    [gridItems]
  );

  const isMultiDragging =
    activeId !== null && selectedIds.size > 1 && selectedIds.has(activeId);

  // ---- Virtualization ----
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const { columnCount, rowHeight, measureRowRef } = useGridLayout();

  const rows = useMemo(() => {
    const result: typeof visibleItems[] = [];
    for (let i = 0; i < visibleItems.length; i += columnCount) {
      result.push(visibleItems.slice(i, i + columnCount));
    }
    return result;
  }, [visibleItems, columnCount]);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollContainerRef.current,
    estimateSize: () => rowHeight,
    overscan: 5,
  });

  // ---- Search ----
  const searchState = useSearchState();
  const scrollToRow = useCallback((rowIndex: number) => {
    virtualizer.scrollToIndex(rowIndex, { align: "center" });
  }, [virtualizer]);

  // ---- Stable refs for callbacks ----
  const activeIdRef = useRef<string | null>(null);
  activeIdRef.current = activeId;
  const expandedGroupIdRef = useRef<string | null>(null);
  expandedGroupIdRef.current = expandedGroupId;

  // ---- Collision detection (stable) ----
  const frozenGroupRef = useRef<string | null>(null);
  frozenGroupRef.current = frozenGroupId;

  const stableCollision = useCallback<CollisionDetection>((args) => {
    const results = closestCenter(args);
    const aid = activeIdRef.current;
    const frozen = frozenGroupRef.current;
    if (aid && !isGroupSortId(aid) && frozen) {
      const excludeId = toGroupSortId(frozen);
      return results.filter((c) => String(c.id) !== excludeId);
    }
    return results;
  }, []);

  // ---- Initial data fetch ----
  // fetchTargetDir must resolve first so cacheNonce is set before imageUrl() is called
  useEffect(() => {
    fetchTargetDir().then(() => {
      fetchImages().catch((err: unknown) => {
        setError(getErrorMessage(err, "Failed to load images"));
      });
    });
    checkUndo();
    fetchGroups();
  }, [fetchImages, checkUndo, fetchTargetDir, fetchGroups, setError]);

  // Clean stale group entries when images change
  useEffect(() => {
    if (images.length === 0 || !groupsLoaded) return;
    const existing = new Set(images.map((i) => i.filename));
    updateGroups((prev) => {
      let changed = false;
      const cleaned = prev.reduce<typeof prev>((acc, g) => {
        const filtered = g.images.filter((fn) => existing.has(fn));
        if (filtered.length !== g.images.length) changed = true;
        if (filtered.length > 0) acc.push({ ...g, images: filtered });
        else changed = true;
        return acc;
      }, []);
      return changed ? cleaned : prev;
    });
  }, [images, groupsLoaded, updateGroups]);

  // ---- Keyboard shortcuts ----
  const createGroupRef = useRef<() => void>(() => {});
  const lightboxOpenRef = useRef(false);
  lightboxOpenRef.current = lightboxIndex !== null;

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        if (lightboxOpenRef.current) return;
        if (searchState.isOpen) { searchState.close(); return; }
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
  }, [expandGroup, clearSelection, searchState]);

  // ---- Pointer tracking for dwell-based group drop ----
  useEffect(() => {
    if (!activeId || isGroupSortId(activeId)) {
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
        const groupEl = el.closest("[data-group-id]") as HTMLElement | null;
        if (groupEl?.dataset.groupId && groupEl.dataset.groupId !== expandedGroupIdRef.current) {
          return groupEl.dataset.groupId;
        }
      }
      return null;
    }

    function onPointerMove(e: PointerEvent) {
      // Throttle with rAF to avoid calling elementsFromPoint on every pointermove
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

  // ---- Group operations ----

  function handleCreateGroup() {
    if (selectedIds.size === 0) return;
    const name = prompt("Enter group name:");
    if (!name?.trim()) return;

    const id = crypto.randomUUID();
    const selectedInOrder = images
      .filter((i) => selectedIds.has(i.filename))
      .map((i) => i.filename);

    setImages(consolidateBlock(images, selectedIds));
    updateGroups((prev) => [
      ...prev,
      { id, name: name.trim(), images: selectedInOrder },
    ]);
    clearSelection();
  }
  createGroupRef.current = handleCreateGroup;

  function addImagesToGroup(groupId: string, filenames: string[]) {
    const fileSet = new Set(filenames);
    const newGroups = groups.map((g) => {
      const cleaned = g.images.filter((fn) => !fileSet.has(fn));
      if (g.id === groupId) return { ...g, images: [...cleaned, ...filenames] };
      return { ...g, images: cleaned };
    });

    updateGroups(() => newGroups);
    const targetGroup = newGroups.find((g) => g.id === groupId);
    if (!targetGroup) return;

    setImages((() => {
      const allGroupImages = new Set(targetGroup.images);
      const toMove = images.filter((i) => fileSet.has(i.filename));
      const rest = images.filter((i) => !fileSet.has(i.filename));
      let lastIdx = -1;
      for (let i = 0; i < rest.length; i++) {
        if (allGroupImages.has(rest[i]!.filename)) lastIdx = i;
      }
      if (lastIdx === -1) return images;
      const out = [...rest];
      out.splice(lastIdx + 1, 0, ...toMove);
      return out;
    })());

    removeFromSelection(filenames);
  }

  function handleGroupReorder(groupId: string, newOrder: string[]) {
    updateGroups((prev) =>
      prev.map((g) => (g.id === groupId ? { ...g, images: newOrder } : g))
    );
    setImages(repositionBlock(images, newOrder));
  }

  function handleRemoveFromGroup(groupId: string, filename: string) {
    const group = groupMap.get(groupId);
    if (group && group.images.length <= 1) collapseGroup();
    updateGroups((prev) =>
      prev.map((g) =>
        g.id === groupId
          ? { ...g, images: g.images.filter((fn) => fn !== filename) }
          : g
      )
    );
  }

  function handleRenameGroup(groupId: string) {
    const group = groupMap.get(groupId);
    if (!group) return;
    const name = prompt("New group name:", group.name);
    if (!name?.trim()) return;
    updateGroups((prev) =>
      prev.map((g) => g.id === groupId ? { ...g, name: name.trim() } : g)
    );
  }

  function handleDeleteGroup(groupId: string) {
    updateGroups((prev) => prev.filter((g) => g.id !== groupId));
    if (expandedGroupId === groupId) collapseGroup();
  }

  // ---- Card click ----

  function handleGridItemClickImpl(id: string, e: React.MouseEvent) {
    const gridIdx = gridItems.findIndex((item) => gridItemId(item) === id);

    if (e.metaKey || e.ctrlKey) {
      toggleSelect(id, gridIdx);
    } else if (e.shiftKey) {
      const allIds = gridItems.map((item, i) => ({ id: gridItemId(item), index: i }));
      rangeSelect(gridIdx, allIds);
    } else if (isGroupSortId(id)) {
      const gid = fromGroupSortId(id);
      if (selectedIds.size > 0) {
        const hasOnlyImages = [...selectedIds].every((s) => !isGroupSortId(s));
        if (hasOnlyImages && groupsEnabled) {
          addImagesToGroup(gid, [...selectedIds]);
        } else {
          clearSelection();
        }
      } else {
        expandGroup(expandedGroupId === gid ? null : gid);
      }
    } else {
      if (selectedIds.size > 0) {
        clearSelection();
      } else {
        const imgIdx = images.findIndex((i) => i.filename === id);
        if (imgIdx !== -1) openLightbox(imgIdx);
      }
    }
  }

  const gridItemClickRef = useRef(handleGridItemClickImpl);
  gridItemClickRef.current = handleGridItemClickImpl;
  const handleGridItemClick = useCallback(
    (id: string, e: React.MouseEvent) => gridItemClickRef.current(id, e),
    []
  );

  // ---- Drag handlers ----

  function handleDragStart(event: DragStartEvent) {
    const id = event.active.id as string;
    setActiveId(id);
    if (selectedIds.size > 0 && !selectedIds.has(id)) {
      clearSelection();
    }
  }

  function handleDragEnd(event: DragEndEvent) {
    const dropGroupId = dragOverGroupId;
    clearDrag();
    setDragEndTime();
    const { active, over } = event;
    if (!active) return;

    const aid = active.id as string;
    const isAidGroup = isGroupSortId(aid);
    const toAdd = !isAidGroup && selectedIds.size > 0 && selectedIds.has(aid)
      ? [...selectedIds]
      : [aid];

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

    const expandedGroup = expandedGroupId ? groupMap.get(expandedGroupId) ?? null : null;
    const expandedSet = expandedGroup ? new Set(expandedGroup.images) : null;
    const activeInExpanded = expandedSet?.has(aid) ?? false;
    const overInExpanded = expandedSet?.has(oid) ?? false;

    // Reorder within expanded group
    if (activeInExpanded && overInExpanded && expandedGroup) {
      const selectedInGroup = selectedIds.size > 0 && selectedIds.has(aid)
        ? new Set([...selectedIds].filter((fn) => expandedSet!.has(fn)))
        : null;
      const newOrder = selectedInGroup && selectedInGroup.size > 1
        ? (selectedInGroup.has(oid) ? null : multiDragReorder(expandedGroup.images, selectedInGroup, aid, oid))
        : arrayMove([...expandedGroup.images], expandedGroup.images.indexOf(aid), expandedGroup.images.indexOf(oid));
      if (!newOrder) return;
      handleGroupReorder(expandedGroup.id, newOrder);
      return;
    }

    // Dragged out of expanded group
    if (activeInExpanded && !overInExpanded && expandedGroup) {
      const toRemove = selectedIds.size > 0 && selectedIds.has(aid)
        ? new Set([...selectedIds].filter((fn) => expandedSet!.has(fn)))
        : new Set([aid]);
      const updatedGroups = groups
        .map((g) =>
          g.id === expandedGroup.id
            ? { ...g, images: g.images.filter((fn) => !toRemove.has(fn)) }
            : g
        )
        .filter((g) => g.images.length > 0);
      updateGroups(() => updatedGroups);
      if (!updatedGroups.some((g) => g.id === expandedGroupId)) collapseGroup();
      const newIds = toRemove.size > 1
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
      ? (selectedIds.has(oid) ? null : multiDragReorder(gridIds, selectedIds, aid, oid))
      : arrayMove([...gridIds], gridIds.indexOf(aid), gridIds.indexOf(oid));
    if (!newIds) return;
    setImages(flattenOrder(newIds, groups, images));
  }

  // ---- Save / Confirm ----

  async function handleConfirmSave() {
    setSaving(true);
    setShowPreview(false);
    try {
      const oldFilenames = images.map((i) => i.filename);
      const res = await postJson("/api/save", { order: oldFilenames });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      const newFilenames = (data.renames as RenameMapping[]).map((r) => r.to);
      const remapped = remapGroupsAfterSave(groups, oldFilenames, newFilenames);
      updateGroups(() => remapped);
      bumpCacheNonce();
      showToast("Files renamed successfully", "success");
      await Promise.all([fetchImages(), checkUndo()]);
    } catch (err) {
      showToast(getErrorMessage(err, "Rename failed"), "error");
    } finally {
      setSaving(false);
    }
  }

  async function handleConfirmOrganize() {
    setSaving(true);
    setShowOrganize(false);
    try {
      const res = await postJson("/api/organize", {
        groups: groups.map((g) => ({ name: g.name, images: g.images })),
        order: images.map((i) => i.filename),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      updateGroups(() => []);
      collapseGroup();
      showToast("Files organized into folders", "success");
      await Promise.all([fetchImages(), checkUndo()]);
    } catch (err) {
      showToast(getErrorMessage(err, "Organize failed"), "error");
    } finally {
      setSaving(false);
    }
  }

  // ---- Drag overlay helpers ----

  const activeImage = activeId && !isGroupSortId(activeId) ? imageMap.get(activeId) ?? null : null;
  const activeGroup = activeId && isGroupSortId(activeId) ? groupMap.get(fromGroupSortId(activeId)) ?? null : null;

  const activeGridIndex = activeImage
    ? gridItems.findIndex((i) => (i.type === "image" || i.type === "group-image") && i.filename === activeImage.filename)
    : activeGroup
      ? gridItems.findIndex((i) => i.type === "group" && i.groupId === activeGroup.id)
      : -1;

  // ---- Render ----

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner" />
        Loading images...
      </div>
    );
  }

  const virtualRows = virtualizer.getVirtualItems();
  const totalHeight = virtualizer.getTotalSize();

  return (
    <SearchContext.Provider value={searchState}>
      <Toolbar onCreateGroup={handleCreateGroup} />

      {error && <div className="error-banner">{error}</div>}

      {images.length === 0 && !error ? (
        <div className="empty">
          <div className="empty-icon">📁</div>
          <div>No images found in this directory</div>
        </div>
      ) : (
        <DndContext
          sensors={sensors}
          collisionDetection={stableCollision}
          onDragStart={handleDragStart}
          onDragEnd={handleDragEnd}
        >
          <SortableContext items={gridIds} strategy={rectSortingStrategy}>
            <div style={{ position: "relative" }}>
              <SearchBar
                gridItems={gridItems}
                onScrollToRow={scrollToRow}
                columnCount={columnCount}
              />
              <div
                ref={scrollContainerRef}
                className="grid-scroll-container"
              >
                {/* Empty grid div for CSS column measurement — auto-fill decides tracks */}
                <div ref={measureRowRef} className="grid-measure-row" aria-hidden />
                <div
                  style={{
                    height: totalHeight,
                    width: "100%",
                    position: "relative",
                  }}
                >
                  {virtualRows.map((virtualRow) => {
                    const row = rows[virtualRow.index]!;
                    const hasExpandedGroup = expandedGroupId != null && row.some(
                      (item) => item.type === "group" && item.groupId === expandedGroupId
                    );
                    return (
                      <div
                        key={virtualRow.key}
                        className={hasExpandedGroup ? "grid-row grid-row-expanded" : "grid-row"}
                        style={{
                          position: "absolute",
                          top: 0,
                          left: 0,
                          width: "100%",
                          height: virtualRow.size,
                          transform: `translateY(${virtualRow.start}px)`,
                        }}
                      >
                        {row.map((item, colIdx) => {
                          // Find the true grid index for numbering
                          const visibleIdx = virtualRow.index * columnCount + colIdx;

                          if (item.type === "group") {
                            const group = groupMap.get(item.groupId);
                            if (!group) return null;
                            const gid = item.groupId;
                            const isExp = expandedGroupId === gid;
                            const sortId = toGroupSortId(gid);
                            return (
                              <SortableGroupCard
                                key={sortId}
                                group={group}
                                gridIndex={visibleIdx}
                                isDropTarget={dragOverGroupId === gid}
                                isExpanded={isExp}
                                isFrozen={frozenGroupId === gid}
                                isSelected={selectedIds.has(sortId)}
                                isGhost={isMultiDragging && selectedIds.has(sortId) && sortId !== activeId}
                                isSearchMatch={searchState.matchIds.has(sortId)}
                                isCurrentSearchMatch={searchState.currentMatchId === sortId}
                                onClick={(e: React.MouseEvent) => handleGridItemClick(sortId, e)}
                                popover={isExp ? (
                                  <div className="group-popover" data-group-popover={gid} onClick={(e) => e.stopPropagation()}>
                                    <div className="group-popover-header">
                                      <span className="group-popover-name">{group.name}</span>
                                      <span className="group-popover-count">{group.images.length} image{group.images.length !== 1 ? "s" : ""}</span>
                                      <div className="group-popover-actions">
                                        <button className="btn btn-small btn-secondary" onClick={() => handleRenameGroup(gid)}>Rename</button>
                                        <button className="btn btn-small btn-danger" onClick={() => handleDeleteGroup(gid)}>Dissolve</button>
                                        <button className="btn btn-small btn-secondary" onClick={() => collapseGroup()}>Close</button>
                                      </div>
                                    </div>
                                    <div className="group-popover-grid">
                                      {group.images.map((fn) => {
                                        const img = imageMap.get(fn);
                                        if (!img) return null;
                                        return (
                                          <ExpandedGroupItem
                                            key={fn}
                                            image={img}
                                            isSelected={selectedIds.has(fn)}
                                            isGhost={isMultiDragging && selectedIds.has(fn) && fn !== activeId}
                                            onRemove={() => handleRemoveFromGroup(gid, fn)}
                                            onCardClick={handleGridItemClick}
                                          />
                                        );
                                      })}
                                    </div>
                                  </div>
                                ) : undefined}
                              />
                            );
                          }

                          // Regular image
                          const img = imageMap.get(item.filename);
                          if (!img) return null;
                          return (
                            <SortableCard
                              key={item.filename}
                              image={img}
                              gridIndex={visibleIdx}
                              isSelected={selectedIds.has(item.filename)}
                              isGhost={
                                isMultiDragging &&
                                selectedIds.has(item.filename) &&
                                item.filename !== activeId
                              }
                              isSearchMatch={searchState.matchIds.has(item.filename)}
                              isCurrentSearchMatch={searchState.currentMatchId === item.filename}
                              onCardClick={handleGridItemClick}
                            />
                          );
                        })}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </SortableContext>
          <DragOverlay dropAnimation={null}>
            {activeImage ? (
              <div className={isMultiDragging ? "drag-overlay-multi" : undefined}>
                <div className="card card-dragging">
                  <img
                    className="card-thumb"
                    src={imageUrl(activeImage.filename)}
                    alt={activeImage.filename}
                    draggable={false}
                  />
                  <div className="card-info">
                    <span className="card-badge">{activeGridIndex + 1}</span>
                    <span className="card-name">{activeImage.filename}</span>
                  </div>
                </div>
                {isMultiDragging && (
                  <div className="drag-count">{selectedIds.size}</div>
                )}
              </div>
            ) : activeGroup ? (
              <div className={isMultiDragging ? "drag-overlay-multi" : undefined}>
                <div className="card group-card card-dragging">
                  <GroupThumbGrid images={activeGroup.images} />
                  <div className="card-info">
                    <span className="card-badge">{activeGridIndex + 1}</span>
                    <span className="card-name">{activeGroup.name}</span>
                  </div>
                </div>
                {isMultiDragging && (
                  <div className="drag-count">{selectedIds.size}</div>
                )}
              </div>
            ) : null}
          </DragOverlay>
        </DndContext>
      )}

      {lightboxIndex !== null && (
        <Lightbox
          images={images}
          initialIndex={lightboxIndex}
          onClose={closeLightbox}
        />
      )}

      {showPreview && (
        <Modal title="Preview Renames" onClose={() => setShowPreview(false)} footer={
          <>
            <button className="btn btn-secondary" onClick={() => setShowPreview(false)}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={handleConfirmSave}>
              Confirm Rename
            </button>
          </>
        }>
          <table className="rename-table">
            <thead>
              <tr>
                <th>Current Name</th>
                <th className="rename-arrow"></th>
                <th>New Name</th>
              </tr>
            </thead>
            <tbody>
              {previewRenames.map((r) => (
                <tr key={r.from}>
                  <td>{r.from}</td>
                  <td className="rename-arrow">→</td>
                  <td>{r.to}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Modal>
      )}

      {showPaths && targetDir && (() => {
        const selectedPaths = images
          .filter((i) => selectedIds.has(i.filename))
          .map((i) => `@"${targetDir}/${i.filename}"`);
        const pathsText = selectedPaths.join("\n");
        return (
          <Modal title="Copy Paths" onClose={() => setShowPaths(false)} footer={
            <>
              <button className="btn btn-secondary" onClick={() => setShowPaths(false)}>
                Close
              </button>
              <button
                className="btn btn-primary"
                onClick={() => {
                  navigator.clipboard.writeText(pathsText);
                  setShowPaths(false);
                  showToast("Paths copied to clipboard", "success");
                }}
              >
                Copy
              </button>
            </>
          }>
            <pre className="paths-list">{pathsText}</pre>
          </Modal>
        );
      })()}

      {showOrganize && (
        <Modal title="Organize into Folders" onClose={() => setShowOrganize(false)} footer={
          <>
            <button className="btn btn-secondary" onClick={() => setShowOrganize(false)}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={handleConfirmOrganize}>
              Confirm
            </button>
          </>
        }>
          <p className="organize-description">
            The following groups will be moved into subfolders:
          </p>
          {organizeMappings.map((m) => (
            <div key={m.folder} className="organize-group">
              <div className="organize-folder">{m.folder}/</div>
              <div className="organize-files">
                {m.files.map((f) => (
                  <div key={f.from} className="organize-file">
                    {f.from === f.to ? f.to : <>{f.from} <span className="rename-arrow">→</span> {f.to}</>}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </Modal>
      )}

      {toast && (
        <div className={`toast toast-${toast.type}`}>{toast.message}</div>
      )}
    </SearchContext.Provider>
  );
}
