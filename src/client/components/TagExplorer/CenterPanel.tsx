import React, { useRef, useMemo, useCallback } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { cn, imageUrl } from "../../utils/helpers.ts";
import { ScopeToggle } from "./ScopeToggle.tsx";
import { useTagStore } from "../../stores/tagStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useSelectionStore } from "../../stores/selectionStore.ts";
import { useGridLayout } from "../../hooks/useGridLayout.ts";

interface CenterPanelProps {
  onImageClick: (filename: string, e: React.MouseEvent) => void;
  onImageDoubleClick: (filename: string) => void;
  onAddToGroup: (groupId: string) => void;
  onCreateGroup: () => void;
}

export function CenterPanel({ onImageClick, onImageDoubleClick, onAddToGroup, onCreateGroup }: CenterPanelProps) {
  const filteredFilenames = useTagStore((s) => s.filteredFilenames);
  const scope = useTagStore((s) => s.scope);
  const setScope = useTagStore((s) => s.setScope);
  const detailFilename = useTagStore((s) => s.detailFilename);
  const filters = useTagStore((s) => s.filters);

  const groups = useGroupStore((s) => s.groups);
  const selectedIds = useSelectionStore((s) => s.selectedIds);
  const toggleSelect = useSelectionStore((s) => s.toggleSelect);
  const rangeSelect = useSelectionStore((s) => s.rangeSelect);
  const clearSelection = useSelectionStore((s) => s.clearSelection);

  const resultCount = filteredFilenames.length;

  // Grid layout
  const { columnCount, rowHeight, measureRowRef } = useGridLayout();
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const rows = useMemo(() => {
    const result: string[][] = [];
    for (let i = 0; i < filteredFilenames.length; i += columnCount) {
      result.push(filteredFilenames.slice(i, i + columnCount));
    }
    return result;
  }, [filteredFilenames, columnCount]);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollContainerRef.current,
    estimateSize: () => rowHeight,
    overscan: 5,
  });

  const handleCardClick = useCallback((filename: string, e: React.MouseEvent, idx: number) => {
    if (e.metaKey || e.ctrlKey) {
      toggleSelect(filename, idx);
    } else if (e.shiftKey) {
      const allIds = useTagStore.getState().filteredFilenames.map((fn, i) => ({ id: fn, index: i }));
      rangeSelect(idx, allIds);
    } else {
      if (useSelectionStore.getState().selectedIds.size > 0) {
        clearSelection();
      } else {
        onImageClick(filename, e);
      }
    }
  }, [toggleSelect, rangeSelect, clearSelection, onImageClick]);

  const virtualRows = virtualizer.getVirtualItems();
  const totalHeight = virtualizer.getTotalSize();

  return (
    <div className="tag-center-panel">
      <div className="tag-results-header">
        <span className="tag-results-count">
          {filteredFilenames.length} result{filteredFilenames.length !== 1 ? "s" : ""}
          {filters.length > 0 && ` matching ${filters.length} filter${filters.length !== 1 ? "s" : ""}`}
        </span>
        <ScopeToggle
          scope={scope}
          onChange={setScope}
          resultCount={resultCount}
        />
      </div>
      {selectedIds.size > 0 && (
        <div className="tag-selection-bar">
          <span>{selectedIds.size} selected</span>
          <button className="btn btn-primary" style={{ padding: "4px 10px", fontSize: "12px" }} onClick={onCreateGroup}>
            New Group
          </button>
          {groups.length > 0 && (
            <select
              className="clothing-filter-select"
              defaultValue=""
              onChange={(e) => {
                if (e.target.value) {
                  onAddToGroup(e.target.value);
                  e.target.value = "";
                }
              }}
            >
              <option value="">Add to group...</option>
              {groups.map((g) => (
                <option key={g.id} value={g.id}>{g.name}</option>
              ))}
            </select>
          )}
          <button className="btn btn-secondary" style={{ padding: "4px 10px", fontSize: "12px" }} onClick={clearSelection}>
            Clear
          </button>
        </div>
      )}
      <div ref={scrollContainerRef} style={{ flex: 1, overflow: "auto" }}>
        <div ref={measureRowRef} className="grid-measure-row" aria-hidden />
        <div style={{ height: totalHeight, width: "100%", position: "relative" }}>
          {virtualRows.map((virtualRow) => {
            const row = rows[virtualRow.index]!;
            return (
              <div
                key={virtualRow.key}
                className="grid-row"
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: virtualRow.size,
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              >
                {row.map((filename, colIdx) => {
                  const globalIdx = virtualRow.index * columnCount + colIdx;
                  const isSelected = selectedIds.has(filename);
                  const isDetail = detailFilename === filename;
                  return (
                    <div
                      key={filename}
                      className={cn(
                        "tag-results-card",
                        isSelected && "tag-results-card-selected",
                        isDetail && "tag-results-card-detail",
                      )}
                      onClick={(e) => handleCardClick(filename, e, globalIdx)}
                      onDoubleClick={() => onImageDoubleClick(filename)}
                    >
                      <img src={imageUrl(filename)} alt={filename} loading="lazy" draggable={false} />
                      <div className="tag-results-card-info">{filename}</div>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
