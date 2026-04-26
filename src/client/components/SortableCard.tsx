import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import React from "react";
import type { ImageInfo } from "../types.ts";
import { cn, imageUrl, wasJustDragged } from "../utils/helpers.ts";
import { TrashIcon } from "./TrashIcon.tsx";

export const SortableCard = React.memo(function SortableCard({
  image,
  gridIndex,
  isSelected,
  isGhost,
  isSearchMatch,
  isCurrentSearchMatch,
  isMarkedForTrash,
  onCardClick,
}: {
  image: ImageInfo;
  gridIndex: number;
  isSelected: boolean;
  isGhost: boolean;
  isSearchMatch?: boolean;
  isCurrentSearchMatch?: boolean;
  isMarkedForTrash?: boolean;
  onCardClick: (filename: string, e: React.MouseEvent) => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: image.filename,
  });

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
        isCurrentSearchMatch && "card-search-current",
        isMarkedForTrash && "card-marked-trash",
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
      {isMarkedForTrash && (
        <span
          className="card-trash-badge"
          aria-label="Marked for deletion"
          title="Marked for deletion"
        >
          <TrashIcon size={14} />
        </span>
      )}
      <div className="card-info">
        <span className="card-badge">{gridIndex + 1}</span>
        <span className="card-name" title={image.filename}>
          {image.filename}
        </span>
      </div>
    </div>
  );
});
