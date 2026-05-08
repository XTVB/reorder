import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import type React from "react";
import { memo } from "react";
import type { ImageInfo } from "../../types.ts";
import { ImageThumb } from "./ImageThumb.tsx";

export const ExpandedGroupItem = memo(function ExpandedGroupItem({
  image,
  isSelected,
  isGhost,
  isMarkedForTrash,
  lightboxImages,
  onSelect,
  onRangeSelect,
  onRemove,
}: {
  image: ImageInfo;
  isSelected: boolean;
  isGhost: boolean;
  isMarkedForTrash?: boolean;
  lightboxImages: string[];
  onSelect: (filename: string, e: React.MouseEvent) => void;
  onRangeSelect: (filename: string, e: React.MouseEvent) => void;
  onRemove: () => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: image.filename,
  });

  const style = { transform: CSS.Transform.toString(transform), transition };

  return (
    <ImageThumb
      filename={image.filename}
      isSelected={isSelected}
      isGhost={isGhost}
      isDragging={isDragging}
      isMarkedForTrash={isMarkedForTrash}
      onSelect={onSelect}
      onRangeSelect={onRangeSelect}
      lightboxImages={lightboxImages}
      footer={
        <>
          <span className="image-thumb-name" title={image.filename}>
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
        </>
      }
      dnd={{ setNodeRef, style, attributes, listeners }}
    />
  );
});
