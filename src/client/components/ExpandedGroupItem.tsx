import React, { memo } from "react";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import type { ImageInfo } from "../types.ts";
import { cn, imageUrl, wasJustDragged } from "../utils/helpers.ts";

export const ExpandedGroupItem = memo(function ExpandedGroupItem({
  image,
  isSelected,
  isGhost,
  onRemove,
  onCardClick,
}: {
  image: ImageInfo;
  isSelected: boolean;
  isGhost: boolean;
  onRemove: () => void;
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
        "group-image-card",
        isDragging && "card-overlay",
        isSelected && "card-selected",
        isGhost && "card-ghost",
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
        <span className="card-name" title={image.filename}>
          {image.filename}
        </span>
        <button
          className="group-remove-btn"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          title="Remove from group"
        >
          Remove
        </button>
      </div>
    </div>
  );
});
