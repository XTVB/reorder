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
import { useDragHandlers } from "../../hooks/useDragHandlers.ts";
import { useGridLayout } from "../../hooks/useGridLayout.ts";
import { useKeyboardShortcuts } from "../../hooks/useKeyboardShortcuts.ts";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { useModalStore } from "../../stores/core/modalStore.ts";
import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useSessionStore } from "../../stores/core/sessionStore.ts";
import { useDndStore } from "../../stores/dndStore.ts";
import { useFolderStore } from "../../stores/folderStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import { useTrashStore } from "../../stores/trashStore.ts";
import { computeGridItems, gridItemId } from "../../utils/gridItems.ts";
import {
  fromFolderSortId,
  fromGroupSortId,
  getErrorMessage,
  imageUrl,
  isFolderSortId,
  isGroupSortId,
  stripFolderNumber,
  toFolderSortId,
  toGroupSortId,
} from "../../utils/helpers.ts";
import { CreateGroupsModal } from "../shared/CreateGroupsModal.tsx";
import { FolderPopover } from "../shared/FolderPopover.tsx";
import { GroupPopover } from "../shared/GroupPopover.tsx";
import { GroupThumbGrid } from "../shared/GroupThumbGrid.tsx";
import { OrganizeModal } from "../shared/OrganizeModal.tsx";
import { PathsModal } from "../shared/PathsModal.tsx";
import { PreviewModal } from "../shared/PreviewModal.tsx";
import { ReviewModal } from "../shared/ReviewModal.tsx";
import { SearchBar, SearchContext, useSearchState } from "../shared/SearchBar.tsx";
import { Slideshow } from "../shared/Slideshow.tsx";
import { TrashModal } from "../shared/TrashModal.tsx";
import { SortableCard } from "./SortableCard.tsx";
import { SortableContainerCard } from "./SortableContainerCard.tsx";

export function ReorderView() {
  // ---- Store subscriptions ----
  const images = useImageStore((s) => s.images);
  const imageMap = useImageStore((s) => s.imageMap);
  const loading = useImageStore((s) => s.loading);
  const fetchImages = useImageStore((s) => s.fetchImages);

  const selectedIds = useSelectionStore((s) => s.contexts.reorder);
  const toggle = useSelectionStore((s) => s.toggle);
  const rangeSelect = useSelectionStore((s) => s.rangeSelect);
  const clear = useSelectionStore((s) => s.clear);

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
  const createGroupFromSelection = useGroupStore((s) => s.createGroupFromSelection);
  const addImagesToGroupAction = useGroupStore((s) => s.addImagesToGroup);
  const renameGroupAction = useGroupStore((s) => s.renameGroupPrompt);
  const deleteGroupAction = useGroupStore((s) => s.deleteGroup);
  const removeFromGroupAction = useGroupStore((s) => s.removeFromGroup);
  const reorderGroupAction = useGroupStore((s) => s.reorderGroup);
  const saveRenames = useGroupStore((s) => s.saveRenames);
  const applyOrganize = useGroupStore((s) => s.applyOrganize);

  const folderModeEnabled = useFolderStore((s) => s.folderModeEnabled);
  const folders = useFolderStore((s) => s.folders);
  const folderMap = useFolderStore((s) => s.folderMap);
  const expandedFolderName = useFolderStore((s) => s.expandedFolderName);
  const expandFolder = useFolderStore((s) => s.expandFolder);
  const collapseFolder = useFolderStore((s) => s.collapseFolder);
  const fetchFolders = useFolderStore((s) => s.fetchFolders);

  const markedTrashIds = useSelectionStore((s) => s.contexts.trash);
  const pruneTrashToValid = useTrashStore((s) => s.pruneToValid);

  const handleToggleGroupMarkAll = useCallback((groupId: string) => {
    const group = useGroupStore.getState().groups.find((g) => g.id === groupId);
    if (!group) return;
    useTrashStore.getState().toggleMany(group.images);
  }, []);

  const allFilenames = useMemo(() => images.map((i) => i.filename), [images]);

  const lightboxOpen = useLightboxStore((s) => s.open);
  const saving = useSessionStore((s) => s.saving);
  const error = useSessionStore((s) => s.error);
  const showPreview = useModalStore((s) => s.open.preview);
  const showOrganize = useModalStore((s) => s.open.organize);
  const showPaths = useModalStore((s) => s.open.paths);
  const showReview = useModalStore((s) => s.open.review);
  const showCreateGroups = useModalStore((s) => s.open.createGroups);
  const showTrashModal = useModalStore((s) => s.open.trash);
  const closeModal = useModalStore((s) => s.closeModal);
  const slideshow = useSessionStore((s) => s.slideshow);
  const closeSlideshow = useSessionStore((s) => s.closeSlideshow);
  const targetDir = useSessionStore((s) => s.targetDir);
  const previewRenames = useSessionStore((s) => s.previewRenames);
  const organizeMappings = useSessionStore((s) => s.organizeMappings);
  const setError = useSessionStore((s) => s.setError);
  const checkUndo = useSessionStore((s) => s.checkUndo);
  const fetchTargetDir = useSessionStore((s) => s.fetchTargetDir);

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

  const handleImageSelect = useCallback(
    (filename: string) => {
      toggle("reorder", filename);
    },
    [toggle],
  );

  const handleImageRangeSelect = useCallback(
    (filename: string) => {
      rangeSelect("reorder", gridIds, filename);
    },
    [rangeSelect, gridIds],
  );

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
  const { handleDragStart, handleDragEnd } = useDragHandlers({
    addImagesToGroup: addImagesToGroupAction,
    handleGroupReorder: reorderGroupAction,
  });

  useKeyboardShortcuts({
    isLightboxOpen: lightboxOpen,
    isSlideshowOpen: slideshow.open,
    searchState,
    onCreateGroup: createGroupFromSelection,
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

  // biome-ignore lint/correctness/useExhaustiveDependencies: markedTrashIds gates the work; including it would force the prune even when it's already empty
  useEffect(() => {
    if (folderModeEnabled || images.length === 0 || markedTrashIds.size === 0) return;
    pruneTrashToValid(images.map((i) => i.filename));
  }, [folderModeEnabled, images, pruneTrashToValid]);

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

  const handleGridItemClickImpl = (id: string, e: React.MouseEvent) => {
    if (e.metaKey || e.ctrlKey) {
      toggle("reorder", id);
      return;
    }
    if (e.shiftKey) {
      rangeSelect("reorder", gridIds, id);
      return;
    }
    if (isFolderSortId(id)) {
      const fname = fromFolderSortId(id);
      if (selectedIds.size > 0) {
        const { moveImages } = useFolderStore.getState();
        const toMove = [...selectedIds].filter((s) => !isFolderSortId(s));
        if (toMove.length > 0) {
          moveImages(toMove, fname);
          clear("reorder");
        }
      } else {
        expandFolder(expandedFolderName === fname ? null : fname);
      }
      return;
    }
    if (isGroupSortId(id)) {
      const gid = fromGroupSortId(id);
      if (selectedIds.size > 0) {
        const hasOnlyImages = [...selectedIds].every((s) => !isGroupSortId(s));
        if (hasOnlyImages && groupsEnabled) {
          addImagesToGroupAction(gid, [...selectedIds]);
        } else {
          clear("reorder");
        }
      } else {
        expandGroup(expandedGroupId === gid ? null : gid);
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
  const handleConfirmSave = useCallback(async () => saveRenames(), [saveRenames]);
  const handleConfirmOrganize = useCallback(async () => applyOrganize(), [applyOrganize]);

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
                              <SortableContainerCard
                                key={sortId}
                                variant="folder"
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
                                      lightboxImages={allFilenames}
                                      onRename={handleRenameFolder}
                                      onDissolve={handleDissolveFolder}
                                      onCollapse={collapseFolder}
                                      onRemoveFromFolder={handleRemoveFromFolder}
                                      onSelect={handleImageSelect}
                                      onRangeSelect={handleImageRangeSelect}
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
                            const groupAllMarked =
                              group.images.length > 0 &&
                              group.images.every((fn) => markedTrashIds.has(fn));
                            return (
                              <SortableContainerCard
                                key={sortId}
                                variant="group"
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
                                isMarkedForTrash={groupAllMarked}
                                onClick={(e: React.MouseEvent) => handleGridItemClick(sortId, e)}
                                popover={
                                  isExp ? (
                                    <GroupPopover
                                      group={group}
                                      imageMap={imageMap}
                                      selectedIds={selectedIds}
                                      markedTrashIds={markedTrashIds}
                                      isMultiDragging={isMultiDragging}
                                      activeId={activeId}
                                      lightboxImages={allFilenames}
                                      onRename={renameGroupAction}
                                      onDelete={deleteGroupAction}
                                      onCollapse={collapseGroup}
                                      onRemoveFromGroup={removeFromGroupAction}
                                      onSelect={handleImageSelect}
                                      onRangeSelect={handleImageRangeSelect}
                                      onToggleMarkAll={handleToggleGroupMarkAll}
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
                              isMarkedForTrash={markedTrashIds.has(item.filename)}
                              lightboxImages={allFilenames}
                              onSelect={handleImageSelect}
                              onRangeSelect={handleImageRangeSelect}
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
                <div className="image-thumb image-thumb-has-footer image-thumb-dragging">
                  <img
                    className="image-thumb-image"
                    src={imageUrl(activeImage.filename)}
                    alt={activeImage.filename}
                    draggable={false}
                  />
                  <div className="image-thumb-footer">
                    <span className="image-thumb-badge">{activeGridIndex + 1}</span>
                    <span className="image-thumb-name">{activeImage.filename}</span>
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

      {slideshow.open && (
        <Slideshow images={images} initialIndex={slideshow.startIndex} onClose={closeSlideshow} />
      )}

      {showPreview && (
        <PreviewModal
          renames={previewRenames}
          onClose={() => closeModal("preview")}
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
              onClose={() => closeModal("paths")}
            />
          );
        })()}

      {showOrganize && (
        <OrganizeModal
          mappings={organizeMappings}
          onClose={() => closeModal("organize")}
          onConfirm={handleConfirmOrganize}
        />
      )}

      {showReview && <ReviewModal onClose={() => closeModal("review")} />}

      {showCreateGroups && <CreateGroupsModal onClose={() => closeModal("createGroups")} />}

      {showTrashModal && <TrashModal onClose={() => closeModal("trash")} />}
    </SearchContext.Provider>
  );
}
