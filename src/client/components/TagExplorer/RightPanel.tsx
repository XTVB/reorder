import React from "react";
import type { ActiveFilter, ClothingOption, ImageTagData } from "../../types.ts";
import type { InvertedIndex } from "../../utils/tagIndex.ts";
import { ImageDetailPanel } from "./ImageDetailPanel.tsx";
import { TagBrowserPanel } from "./TagBrowserPanel.tsx";
import { ClothingFilterSection } from "./ClothingFilterSection.tsx";

interface RightPanelProps {
  isOpen: boolean;
  onToggle: () => void;
  detailData: ImageTagData | null;
  onCloseDetail: () => void;
  invertedIndex: InvertedIndex;
  filteredFilenames: string[];
  filters: ActiveFilter[];
  onTagClick: (category: string, value: string, e: React.MouseEvent) => void;
  clothingOptions: ClothingOption[];
  onClothingAdd: (piece: string, color: string) => void;
  onClothingRemove: (value: string) => void;
  onOpenLightbox: (filename: string) => void;
}

export function RightPanel({
  isOpen,
  onToggle,
  detailData,
  onCloseDetail,
  invertedIndex,
  filteredFilenames,
  filters,
  onTagClick,
  clothingOptions,
  onClothingAdd,
  onClothingRemove,
  onOpenLightbox,
}: RightPanelProps) {
  return (
    <>
      <button
        className="sidebar-toggle"
        onClick={onToggle}
        style={{ position: "fixed", right: isOpen ? 328 : 8, top: 74, zIndex: 200 }}
        title={isOpen ? "Hide tag panel" : "Show tag panel"}
      >
        {isOpen ? "»" : "«"}
      </button>
      {isOpen && (
        <div className="tag-right-panel">
          {detailData && (
            <ImageDetailPanel
              tagData={detailData}
              onTagClick={onTagClick}
              onClothingClick={onClothingAdd}
              onClose={onCloseDetail}
              onOpenLightbox={onOpenLightbox}
            />
          )}
          <ClothingFilterSection
            clothingOptions={clothingOptions}
            filters={filters}
            onAdd={onClothingAdd}
            onRemove={onClothingRemove}
          />
          <TagBrowserPanel
            invertedIndex={invertedIndex}
            filteredFilenames={filteredFilenames}
            filters={filters}
            onTagClick={onTagClick}
          />
        </div>
      )}
    </>
  );
}
