import React from "react";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import type { ImageInfo } from "../types.ts";
import { cn, imageUrl, wasJustDragged } from "../utils/helpers.ts";

export const SortableCard = React.memo(function SortableCard({
  image,
  gridIndex,
  isSelected,
  isGhost,
  isSearchMatch,
  onCardClick,
}: {
  image: ImageInfo;
  gridIndex: number;
  isSelected: boolean;
  isGhost: boolean;
  isSearchMatch?: boolean;
  onCardClick: (filename: string, e: React.MouseEvent) => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: image.filename });

  const style = { transform: CSS.Transform.toString(transform), transition };

  function handleClick(e: React.MouseEvent) {
    if (wasJustDragged()) return;
    onCardClick(image.filename, e);
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={cn(
        "card",
        isDragging && "card-overlay",
        isSelected && "card-selected",
        isGhost && "card-ghost",
        isSearchMatch && "card-search-match",
      )}
      onClick={handleClick}
      {...attributes}
      {...listeners}
    >
      <img
        className="card-thumb"
        src={imageUrl(image.filename)}
        alt={image.filename}
        loading="lazy"
        draggable={false}
      />
      <div className="card-info">
        <span className="card-badge">{gridIndex + 1}</span>
        <span className="card-name" title={image.filename}>
          {image.filename}
        </span>
      </div>
    </div>
  );
});
