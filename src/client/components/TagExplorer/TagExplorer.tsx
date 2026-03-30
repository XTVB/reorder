import React, { useEffect, useMemo, useCallback, useState } from "react";
import { Toolbar } from "../Toolbar.tsx";
import { Toast } from "../Toast.tsx";
import { Lightbox } from "../Lightbox.tsx";
import { useTagStore } from "../../stores/tagStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useSelectionStore } from "../../stores/selectionStore.ts";
import { resolveFilterMode } from "../../utils/helpers.ts";
import { consolidateBlock } from "../../utils/reorder.ts";
import { useTagViewInit } from "../../hooks/useTagViewInit.ts";
import { useGroupOperations } from "../../hooks/useGroupOperations.ts";
import { FilterBar } from "./FilterBar.tsx";
import { LeftPanel } from "./LeftPanel.tsx";
import { CenterPanel } from "./CenterPanel.tsx";
import { RightPanel } from "./RightPanel.tsx";
import { IngestPrompt } from "./IngestPrompt.tsx";

export function TagExplorer() {
  useTagViewInit();

  const hasDb = useTagStore((s) => s.hasDb);
  const loading = useTagStore((s) => s.loading);
  const indexReady = useTagStore((s) => s.indexReady);

  const filters = useTagStore((s) => s.filters);
  const filteredFilenames = useTagStore((s) => s.filteredFilenames);
  const addFilter = useTagStore((s) => s.addFilter);
  const removeFilter = useTagStore((s) => s.removeFilter);
  const setFilterMode = useTagStore((s) => s.setFilterMode);
  const clearFilters = useTagStore((s) => s.clearFilters);

  const invertedIndex = useTagStore((s) => s.invertedIndex);
  const tagData = useTagStore((s) => s.tagData);
  const clothingOptions = useTagStore((s) => s.clothingOptions);
  const detailFilename = useTagStore((s) => s.detailFilename);
  const setDetailFilename = useTagStore((s) => s.setDetailFilename);
  const sidebarOpen = useTagStore((s) => s.sidebarOpen);
  const toggleSidebar = useTagStore((s) => s.toggleSidebar);
  const focusedGroupId = useTagStore((s) => s.focusedGroupId);
  const setFocusedGroup = useTagStore((s) => s.setFocusedGroup);
  const recomputeFiltered = useTagStore((s) => s.recomputeFiltered);

  const groups = useGroupStore((s) => s.groups);
  const updateGroups = useGroupStore((s) => s.updateGroups);
  const images = useImageStore((s) => s.images);
  const selectedIds = useSelectionStore((s) => s.selectedIds);
  const clearSelection = useSelectionStore((s) => s.clearSelection);

  // Recompute filtered results when images or groups change
  useEffect(() => {
    if (indexReady) recomputeFiltered();
  }, [images, groups, indexReady, recomputeFiltered]);

  const filteredSet = useMemo(() => new Set(filteredFilenames), [filteredFilenames]);

  const detailData = detailFilename
    ? tagData.get(detailFilename) ?? { filename: detailFilename, tags: {}, clothing: [] }
    : null;

  const handleTagClick = useCallback((category: string, value: string, e: React.MouseEvent) => {
    const existing = filters.find((f) => f.category === category && f.value === value);
    if (existing) {
      removeFilter(category, value);
    } else {
      addFilter(category, value, resolveFilterMode(e));
    }
  }, [filters, addFilter, removeFilter]);

  const handleClothingAdd = useCallback((piece: string, color: string) => {
    const value = color ? `${piece}|${color}` : piece;
    addFilter("__clothing_structured", value, "AND");
  }, [addFilter]);

  const handleClothingRemove = useCallback((value: string) => {
    removeFilter("__clothing_structured", value);
  }, [removeFilter]);

  const handleImageClick = useCallback((filename: string, _e: React.MouseEvent) => {
    setDetailFilename(detailFilename === filename ? null : filename);
  }, [detailFilename, setDetailFilename]);

  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  const lightboxImages = useMemo(
    () => filteredFilenames.map((fn) => ({ filename: fn })),
    [filteredFilenames],
  );

  const filteredIndexMap = useMemo(
    () => new Map(filteredFilenames.map((fn, i) => [fn, i])),
    [filteredFilenames],
  );

  const handleOpenLightbox = useCallback((filename: string) => {
    const idx = filteredIndexMap.get(filename);
    if (idx !== undefined) setLightboxIndex(idx);
  }, [filteredIndexMap]);

  const setImages = useImageStore((s) => s.setImages);

  const handleCreateGroup = useCallback(() => {
    const sel = useSelectionStore.getState().selectedIds;
    if (sel.size === 0) return;
    const name = prompt("Enter group name:");
    if (!name?.trim()) return;
    const selectedInOrder = images
      .filter((i) => sel.has(i.filename))
      .map((i) => i.filename);
    // Consolidate images together in the main order so the group
    // appears cleanly in the Reorder view
    setImages(consolidateBlock(images, sel));
    updateGroups((prev) => [
      ...prev,
      { id: crypto.randomUUID(), name: name.trim(), images: selectedInOrder },
    ]);
    clearSelection();
  }, [images, setImages, updateGroups, clearSelection]);

  // G key to create group from selection
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "g" && !e.metaKey && !e.ctrlKey && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement)) {
        if (useSelectionStore.getState().selectedIds.size > 0) {
          handleCreateGroup();
        }
      }
      if (e.key === "Escape") {
        if (useSelectionStore.getState().selectedIds.size > 0) clearSelection();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleCreateGroup, clearSelection]);

  const groupOps = useGroupOperations();

  const handleAddToGroup = useCallback((groupId: string) => {
    groupOps.addImagesToGroup(groupId, [...useSelectionStore.getState().selectedIds]);
  }, [groupOps]);

  return (
    <>
      <Toolbar onCreateGroup={handleCreateGroup} />

      {loading ? (
        <div className="loading">
          <div className="spinner" />
          Loading tags...
        </div>
      ) : !hasDb ? (
        <div className="tag-explorer">
          <IngestPrompt />
        </div>
      ) : indexReady ? (
        <div className="tag-explorer">
          <div className="tag-filter-bar">
            <FilterBar
              filters={filters}
              onRemove={removeFilter}
              onModeChange={setFilterMode}
              onClear={clearFilters}
            />
          </div>
          <div className="tag-explorer-body">
            <LeftPanel
              groups={groups}
              focusedGroupId={focusedGroupId}
              onFocusGroup={setFocusedGroup}
              filteredSet={filteredSet}
              hasFilters={filters.length > 0}
              onAddToGroup={handleAddToGroup}
              onImageClick={(fn) => setDetailFilename(detailFilename === fn ? null : fn)}
              selectedCount={selectedIds.size}
            />
            <CenterPanel onImageClick={handleImageClick} onImageDoubleClick={handleOpenLightbox} onAddToGroup={handleAddToGroup} onCreateGroup={handleCreateGroup} />
            <RightPanel
              isOpen={sidebarOpen}
              onToggle={toggleSidebar}
              detailData={detailData}
              onCloseDetail={() => setDetailFilename(null)}
              invertedIndex={invertedIndex}
              filteredFilenames={filteredFilenames}
              filters={filters}
              onTagClick={handleTagClick}
              clothingOptions={clothingOptions}
              onClothingAdd={handleClothingAdd}
              onClothingRemove={handleClothingRemove}
              onOpenLightbox={handleOpenLightbox}
            />
          </div>
        </div>
      ) : null}

      {lightboxIndex !== null && (
        <Lightbox
          images={lightboxImages}
          initialIndex={lightboxIndex}
          onClose={() => setLightboxIndex(null)}
        />
      )}
      <Toast />
    </>
  );
}
