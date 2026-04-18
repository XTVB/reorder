import {
  type CollisionDetection,
  closestCenter,
  DndContext,
  DragOverlay,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import {
  rectSortingStrategy,
  SortableContext,
  sortableKeyboardCoordinates,
} from "@dnd-kit/sortable";
import { useVirtualizer } from "@tanstack/react-virtual";
import type React from "react";
import { useCallback, useEffect, useMemo, useRef } from "react";
import { FolderPopover } from "./components/FolderPopover.tsx";
import { GroupPopover } from "./components/GroupPopover.tsx";
import { GroupThumbGrid } from "./components/GroupThumbGrid.tsx";
import { Lightbox } from "./components/Lightbox.tsx";
import { OrganizeModal } from "./components/OrganizeModal.tsx";
import { PathsModal } from "./components/PathsModal.tsx";
import { PreviewModal } from "./components/PreviewModal.tsx";
import { ReviewModal } from "./components/ReviewModal.tsx";
import { SearchBar, SearchContext, useSearchState } from "./components/SearchBar.tsx";
import { SortableCard } from "./components/SortableCard.tsx";
import { SortableFolderCard } from "./components/SortableFolderCard.tsx";
import { SortableGroupCard } from "./components/SortableGroupCard.tsx";
import { useDragHandlers } from "./hooks/useDragHandlers.ts";
import { useGridLayout } from "./hooks/useGridLayout.ts";
import { useGroupOperations } from "./hooks/useGroupOperations.ts";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts.ts";
import { useDndStore } from "./stores/dndStore.ts";
import { useFolderStore } from "./stores/folderStore.ts";
import { flushGroupPersist, useGroupStore } from "./stores/groupStore.ts";
import { useImageStore } from "./stores/imageStore.ts";
import { useSelectionStore } from "./stores/selectionStore.ts";
import { useUIStore } from "./stores/uiStore.ts";
import { computeGridItems, gridItemId } from "./utils/gridItems.ts";
import {
  fromFolderSortId,
  fromGroupSortId,
  getErrorMessage,
  imageUrl,
  isFolderSortId,
  isGroupSortId,
  postJson,
  stripFolderNumber,
  toFolderSortId,
  toGroupSortId,
} from "./utils/helpers.ts";

export function App() {
  // ---- Store subscriptions ----
  const images = useImageStore((s) => s.images);
  const imageMap = useImageStore((s) => s.imageMap);
  const loading = useImageStore((s) => s.loading);
  const fetchImages = useImageStore((s) => s.fetchImages);

  const selectedIds = useSelectionStore((s) => s.selectedIds);
  const toggleSelect = useSelectionStore((s) => s.toggleSelect);
  const rangeSelect = useSelectionStore((s) => s.rangeSelect);
  const clearSelection = useSelectionStore((s) => s.clearSelection);

  const activeId = useDndStore((s) => s.activeId);
  const dragOverGroupId = useDndStore((s) => s.dragOverGroupId);
  const frozenGroupId = useDndStore((s) => s.frozenGroupId);

  const groups = useGroupStore((s) => s.groups);
  const groupsEnabled = useGroupStore((s) => s.groupsEnabled);
  const expandedGroupId = useGroupStore((s) => s.expandedGroupId);
  const groupMap = useGroupStore((s) => s.groupMap);
  const groupsLoaded = useGroupStore((s) => s.groupsLoaded);
  const updateGroups = useGroupStore((s) => s.updateGroups);
  const expandGroup = useGroupStore((s) => s.expandGroup);
  const collapseGroup = useGroupStore((s) => s.collapseGroup);
  const fetchGroups = useGroupStore((s) => s.fetchGroups);

  const folderModeEnabled = useFolderStore((s) => s.folderModeEnabled);
  const folders = useFolderStore((s) => s.folders);
  const folderMap = useFolderStore((s) => s.folderMap);
  const expandedFolderName = useFolderStore((s) => s.expandedFolderName);
  const expandFolder = useFolderStore((s) => s.expandFolder);
  const collapseFolder = useFolderStore((s) => s.collapseFolder);
  const fetchFolders = useFolderStore((s) => s.fetchFolders);

  const lightboxIndex = useUIStore((s) => s.lightboxIndex);
  const saving = useUIStore((s) => s.saving);
  const error = useUIStore((s) => s.error);
  const showPreview = useUIStore((s) => s.showPreview);
  const showOrganize = useUIStore((s) => s.showOrganize);
  const showPaths = useUIStore((s) => s.showPaths);
  const showReview = useUIStore((s) => s.showReview);
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
  const setShowReview = useUIStore((s) => s.setShowReview);
  const checkUndo = useUIStore((s) => s.checkUndo);
  const fetchTargetDir = useUIStore((s) => s.fetchTargetDir);

  // ---- DnD sensors ----
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  // ---- Computed grid ----
  const gridItems = useMemo(
    () =>
      computeGridItems(
        images,
        folderModeEnabled
          ? { mode: "folders", folders, expandedFolderName }
          : { mode: "groups", groups, enabled: groupsEnabled, expandedGroupId },
      ),
    [
      images,
      groups,
      groupsEnabled,
      expandedGroupId,
      folders,
      folderModeEnabled,
      expandedFolderName,
    ],
  );
  const gridIds = useMemo(() => gridItems.map(gridItemId), [gridItems]);

  const visibleItems = useMemo(
    () => gridItems.filter((item) => item.type !== "group-image" && item.type !== "folder-image"),
    [gridItems],
  );

  const isMultiDragging = activeId !== null && selectedIds.size > 1 && selectedIds.has(activeId);

  // ---- Virtualization ----
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const { columnCount, rowHeight, measureRowRef } = useGridLayout();

  const rows = useMemo(() => {
    const result: (typeof visibleItems)[] = [];
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
  const scrollToRow = useCallback(
    (rowIndex: number) => {
      virtualizer.scrollToIndex(rowIndex, { align: "center" });
    },
    [virtualizer],
  );

  // ---- Collision detection (stable) ----
  const activeIdRef = useRef<string | null>(null);
  activeIdRef.current = activeId;
  const frozenGroupRef = useRef<string | null>(null);
  frozenGroupRef.current = frozenGroupId;

  const stableCollision = useCallback<CollisionDetection>((args) => {
    const results = closestCenter(args);
    const aid = activeIdRef.current;
    const frozen = frozenGroupRef.current;
    if (aid && !isGroupSortId(aid) && !isFolderSortId(aid) && frozen) {
      const excludeId = isFolderSortId(frozen) ? frozen : toGroupSortId(frozen);
      return results.filter((c) => String(c.id) !== excludeId);
    }
    return results;
  }, []);

  // ---- Hooks ----
  const groupOps = useGroupOperations();
  const { handleDragStart, handleDragEnd } = useDragHandlers(groupOps);

  useKeyboardShortcuts({
    isLightboxOpen: lightboxIndex !== null,
    searchState,
    onCreateGroup: groupOps.handleCreateGroup,
  });

  // ---- Initial data fetch ----
  // biome-ignore lint/correctness/useExhaustiveDependencies: mount-only effect — all called functions are stable Zustand selectors
  useEffect(() => {
    fetchTargetDir();
    // Skip re-fetching if images are already loaded (e.g. switching back from cluster mode).
    // Re-fetching would bump imageVersion, busting the browser's in-memory thumbnail cache
    // and causing a black flash while thumbnails reload.
    const alreadyLoaded = useImageStore.getState().images.length > 0;
    if (folderModeEnabled) {
      fetchFolders().catch((err: unknown) => {
        setError(getErrorMessage(err, "Failed to load folders"));
      });
    } else if (!alreadyLoaded) {
      fetchImages().catch((err: unknown) => {
        setError(getErrorMessage(err, "Failed to load images"));
      });
    }
    if (!folderModeEnabled) {
      // Always refresh groups — they may have changed in cluster mode (accept/merge)
      fetchGroups();
    }
    checkUndo();
  }, []);

  // Clean stale group entries when images change
  useEffect(() => {
    if (folderModeEnabled || images.length === 0 || !groupsLoaded || saving) return;
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
  }, [folderModeEnabled, images, groupsLoaded, saving, updateGroups]);

  // ---- Card click ----
  const handleGridItemClickImpl = (id: string, e: React.MouseEvent) => {
    const gridIdx = gridItems.findIndex((item) => gridItemId(item) === id);

    if (e.metaKey || e.ctrlKey) {
      toggleSelect(id, gridIdx);
    } else if (e.shiftKey) {
      const allIds = gridItems.map((item, i) => ({ id: gridItemId(item), index: i }));
      rangeSelect(gridIdx, allIds);
    } else if (isFolderSortId(id)) {
      const fname = fromFolderSortId(id);
      if (selectedIds.size > 0) {
        const { moveImages } = useFolderStore.getState();
        const toMove = [...selectedIds].filter((s) => !isFolderSortId(s));
        if (toMove.length > 0) {
          moveImages(toMove, fname);
          clearSelection();
        }
      } else {
        expandFolder(expandedFolderName === fname ? null : fname);
      }
    } else if (isGroupSortId(id)) {
      const gid = fromGroupSortId(id);
      if (selectedIds.size > 0) {
        const hasOnlyImages = [...selectedIds].every((s) => !isGroupSortId(s));
        if (hasOnlyImages && groupsEnabled) {
          groupOps.addImagesToGroup(gid, [...selectedIds]);
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
  };
  const gridItemClickRef = useRef(handleGridItemClickImpl);
  gridItemClickRef.current = handleGridItemClickImpl;
  const handleGridItemClick = useCallback(
    (id: string, e: React.MouseEvent) => gridItemClickRef.current(id, e),
    [],
  );

  // ---- Save / Confirm ----
  const handleConfirmSaveImpl = async () => {
    setSaving(true);
    setShowPreview(false);
    try {
      // Cancel any pending debounced group persist to prevent it from
      // racing with the save and overwriting remapped groups on disk
      await flushGroupPersist();
      const oldFilenames = images.map((i) => i.filename);
      const currentGroups = useGroupStore.getState().groups;
      const res = await postJson("/api/save", { order: oldFilenames, groups: currentGroups });
      const data = await res.json();
      if (!data.success) throw new Error(data.error ?? "Rename failed");
      if (data.warnings?.length > 0) {
        showToast(`Files renamed (${data.warnings.length} warning(s))`, "warning");
      } else {
        showToast("Files renamed successfully", "success");
      }
      await fetchImages();
      await Promise.all([fetchGroups(), checkUndo()]);
    } catch (err) {
      showToast(getErrorMessage(err, "Rename failed"), "error");
    } finally {
      setSaving(false);
    }
  };
  const confirmSaveRef = useRef(handleConfirmSaveImpl);
  confirmSaveRef.current = handleConfirmSaveImpl;
  const handleConfirmSave = useCallback(async () => confirmSaveRef.current(), []);

  const handleConfirmOrganizeImpl = async () => {
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
  };
  const confirmOrganizeRef = useRef(handleConfirmOrganizeImpl);
  confirmOrganizeRef.current = handleConfirmOrganizeImpl;
  const handleConfirmOrganize = useCallback(async () => confirmOrganizeRef.current(), []);

  // ---- Folder operations (all local state, nothing hits disk until Save) ----
  const handleRenameFolder = useCallback((folderName: string) => {
    const title = stripFolderNumber(folderName) || folderName;
    const newTitle = prompt("Rename folder:", title);
    if (!newTitle?.trim() || newTitle.trim() === title) return;
    useFolderStore.getState().renameFolder(folderName, newTitle.trim());
  }, []);

  const handleDissolveFolder = useCallback((folderName: string) => {
    if (
      !confirm(
        `Dissolve "${stripFolderNumber(folderName) || folderName}"? Images will be moved to root on save.`,
      )
    )
      return;
    useFolderStore.getState().dissolveFolder(folderName);
  }, []);

  const handleRemoveFromFolder = useCallback((_folderName: string, compoundFn: string) => {
    useFolderStore.getState().moveImages([compoundFn], "");
  }, []);

  // ---- Drag overlay helpers ----
  const activeImage =
    activeId && !isGroupSortId(activeId) && !isFolderSortId(activeId)
      ? (imageMap.get(activeId) ?? null)
      : null;
  const activeGroup =
    activeId && isGroupSortId(activeId) ? (groupMap.get(fromGroupSortId(activeId)) ?? null) : null;
  const activeFolder =
    activeId && isFolderSortId(activeId)
      ? (folderMap.get(fromFolderSortId(activeId)) ?? null)
      : null;

  const activeGridIndex = activeImage
    ? gridItems.findIndex(
        (i) =>
          (i.type === "image" || i.type === "group-image" || i.type === "folder-image") &&
          i.filename === activeImage.filename,
      )
    : activeGroup
      ? gridItems.findIndex((i) => i.type === "group" && i.groupId === activeGroup.id)
      : activeFolder
        ? gridItems.findIndex((i) => i.type === "folder" && i.folderName === activeFolder.name)
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
              <div ref={scrollContainerRef} className="grid-scroll-container">
                <div ref={measureRowRef} className="grid-measure-row" aria-hidden />
                <div style={{ height: totalHeight, width: "100%", position: "relative" }}>
                  {virtualRows.map((virtualRow) => {
                    const row = rows[virtualRow.index]!;
                    const hasExpandedGroup =
                      (expandedGroupId != null &&
                        row.some(
                          (item) => item.type === "group" && item.groupId === expandedGroupId,
                        )) ||
                      (expandedFolderName != null &&
                        row.some(
                          (item) =>
                            item.type === "folder" && item.folderName === expandedFolderName,
                        ));
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
                          const visibleIdx = virtualRow.index * columnCount + colIdx;

                          if (item.type === "folder") {
                            const folder = folderMap.get(item.folderName);
                            if (!folder) return null;
                            const isExp = expandedFolderName === folder.name;
                            const sortId = toFolderSortId(folder.name);
                            return (
                              <SortableFolderCard
                                key={sortId}
                                folder={folder}
                                gridIndex={visibleIdx}
                                isDropTarget={dragOverGroupId === sortId}
                                isExpanded={isExp}
                                isFrozen={frozenGroupId === sortId}
                                isSelected={selectedIds.has(sortId)}
                                isGhost={
                                  isMultiDragging && selectedIds.has(sortId) && sortId !== activeId
                                }
                                isSearchMatch={searchState.matchIds.has(sortId)}
                                isCurrentSearchMatch={searchState.currentMatchId === sortId}
                                onClick={(e: React.MouseEvent) => handleGridItemClick(sortId, e)}
                                popover={
                                  isExp ? (
                                    <FolderPopover
                                      folder={folder}
                                      imageMap={imageMap}
                                      selectedIds={selectedIds}
                                      isMultiDragging={isMultiDragging}
                                      activeId={activeId}
                                      onRename={handleRenameFolder}
                                      onDissolve={handleDissolveFolder}
                                      onCollapse={collapseFolder}
                                      onRemoveFromFolder={handleRemoveFromFolder}
                                      onCardClick={handleGridItemClick}
                                    />
                                  ) : undefined
                                }
                              />
                            );
                          }

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
                                isGhost={
                                  isMultiDragging && selectedIds.has(sortId) && sortId !== activeId
                                }
                                isSearchMatch={searchState.matchIds.has(sortId)}
                                isCurrentSearchMatch={searchState.currentMatchId === sortId}
                                onClick={(e: React.MouseEvent) => handleGridItemClick(sortId, e)}
                                popover={
                                  isExp ? (
                                    <GroupPopover
                                      group={group}
                                      imageMap={imageMap}
                                      selectedIds={selectedIds}
                                      isMultiDragging={isMultiDragging}
                                      activeId={activeId}
                                      onRename={groupOps.handleRenameGroup}
                                      onDelete={groupOps.handleDeleteGroup}
                                      onCollapse={collapseGroup}
                                      onRemoveFromGroup={groupOps.handleRemoveFromGroup}
                                      onCardClick={handleGridItemClick}
                                    />
                                  ) : undefined
                                }
                              />
                            );
                          }

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
                {isMultiDragging && <div className="drag-count">{selectedIds.size}</div>}
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
                {isMultiDragging && <div className="drag-count">{selectedIds.size}</div>}
              </div>
            ) : activeFolder ? (
              <div className="card group-card card-dragging">
                <GroupThumbGrid
                  images={activeFolder.images.map((fn) => `${activeFolder.name}/${fn}`)}
                />
                <div className="card-info">
                  <span className="card-badge">{activeGridIndex + 1}</span>
                  <span className="card-name">
                    {stripFolderNumber(activeFolder.name) || activeFolder.name}
                  </span>
                </div>
              </div>
            ) : null}
          </DragOverlay>
        </DndContext>
      )}

      {lightboxIndex !== null && (
        <Lightbox images={images} initialIndex={lightboxIndex} onClose={closeLightbox} />
      )}

      {showPreview && (
        <PreviewModal
          renames={previewRenames}
          onClose={() => setShowPreview(false)}
          onConfirm={handleConfirmSave}
        />
      )}

      {showPaths &&
        targetDir &&
        (() => {
          const selectedFilenames = images
            .filter((i) => selectedIds.has(i.filename))
            .map((i) => i.filename);
          return (
            <PathsModal
              filenames={selectedFilenames}
              targetDir={targetDir}
              onClose={() => setShowPaths(false)}
            />
          );
        })()}

      {showOrganize && (
        <OrganizeModal
          mappings={organizeMappings}
          onClose={() => setShowOrganize(false)}
          onConfirm={handleConfirmOrganize}
        />
      )}

      {showReview && <ReviewModal onClose={() => setShowReview(false)} />}
    </SearchContext.Provider>
  );
}
