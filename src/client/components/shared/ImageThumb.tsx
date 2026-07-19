import type { DraggableAttributes, DraggableSyntheticListeners } from "@dnd-kit/core";
import type React from "react";
import { memo } from "react";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { cn, imageUrl, wasJustDragged } from "../../utils/helpers.ts";
import { TrashBadge } from "./TrashBadge.tsx";

interface ImageThumbProps {
  filename: string;

  isSelected?: boolean;
  isMarkedForTrash?: boolean;
  isLocked?: boolean;
  isSearchMatch?: boolean;
  isCurrentSearchMatch?: boolean;
  isGhost?: boolean;
  isDragging?: boolean;
  variant?: "confirmed" | "suggested";

  onSelect?: (filename: string, e: React.MouseEvent) => void;
  onRangeSelect?: (filename: string, e: React.MouseEvent) => void;
  /** Plain click opens the lightbox at `lightboxIndex` (or indexOf(filename) if omitted). */
  lightboxImages?: string[];
  lightboxIndex?: number;
  showSelectButton?: boolean;

  /** Tags the root for the sort FLIP animation (see utils/sortFlip.ts). */
  flipId?: string;

  footer?: React.ReactNode;
  topRight?: React.ReactNode;
  bottomLeft?: React.ReactNode;
  bottomRight?: React.ReactNode;

  dnd?: {
    setNodeRef: (node: HTMLElement | null) => void;
    style?: React.CSSProperties;
    attributes: DraggableAttributes;
    listeners: DraggableSyntheticListeners;
  };
}

export const ImageThumb = memo(function ImageThumb({
  filename,
  isSelected = false,
  isMarkedForTrash = false,
  isLocked = false,
  isSearchMatch = false,
  isCurrentSearchMatch = false,
  isGhost = false,
  isDragging = false,
  variant,
  onSelect,
  onRangeSelect,
  lightboxImages,
  lightboxIndex,
  showSelectButton = false,
  flipId,
  footer,
  topRight,
  bottomLeft,
  bottomRight,
  dnd,
}: ImageThumbProps) {
  function handleRootClick(e: React.MouseEvent) {
    if (wasJustDragged()) return;
    e.stopPropagation();
    if (e.shiftKey && onRangeSelect) {
      onRangeSelect(filename, e);
    } else if ((e.metaKey || e.ctrlKey) && onSelect) {
      onSelect(filename, e);
    } else if (lightboxImages) {
      const idx = lightboxIndex ?? lightboxImages.indexOf(filename);
      if (idx >= 0) useLightboxStore.getState().openLightbox(lightboxImages, idx);
    }
  }

  function handleSelectButtonClick(e: React.MouseEvent) {
    e.stopPropagation();
    if (e.shiftKey && onRangeSelect) onRangeSelect(filename, e);
    else if (onSelect) onSelect(filename, e);
  }

  return (
    <div
      ref={dnd?.setNodeRef}
      style={dnd?.style}
      data-flip-id={flipId}
      className={cn(
        "image-thumb",
        Boolean(footer) && "image-thumb-has-footer",
        isDragging && "image-thumb-dragging",
        isGhost && "image-thumb-ghost",
        isSelected && "image-thumb-selected",
        isSearchMatch && "image-thumb-search-match",
        isCurrentSearchMatch && "image-thumb-search-current",
        isMarkedForTrash && "image-thumb-marked-trash",
        isLocked && "image-thumb-locked",
        variant === "confirmed" && "image-thumb-confirmed",
        variant === "suggested" && "image-thumb-suggested",
      )}
      onClick={handleRootClick}
      {...dnd?.attributes}
      {...dnd?.listeners}
    >
      <img
        className="image-thumb-image"
        src={imageUrl(filename)}
        alt={filename}
        loading="lazy"
        decoding="async"
        draggable={false}
      />
      {showSelectButton && (
        <button
          type="button"
          className="image-thumb-select"
          onClick={handleSelectButtonClick}
          aria-label={isSelected ? "Deselect" : "Select"}
          title={isSelected ? "Deselect" : "Select"}
        >
          <span className="image-thumb-check" aria-hidden>
            {isSelected ? "✓" : ""}
          </span>
        </button>
      )}
      {isLocked && (
        <span
          className="image-thumb-lock-badge"
          title="Locked — keeps its relative order when sorting (L to unlock)"
          aria-label="Locked"
        >
          🔒
        </span>
      )}
      {(isMarkedForTrash || topRight) && (
        <div className="image-thumb-top-right">
          {isMarkedForTrash && <TrashBadge />}
          {topRight}
        </div>
      )}
      {bottomLeft && <div className="image-thumb-bottom-left">{bottomLeft}</div>}
      {bottomRight && <div className="image-thumb-bottom-right">{bottomRight}</div>}
      {footer && <div className="image-thumb-footer">{footer}</div>}
    </div>
  );
});
