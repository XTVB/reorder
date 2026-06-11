import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import React from "react";
import type { ImageInfo } from "../../types.ts";
import { ImageThumb } from "../shared/ImageThumb.tsx";

export const SortableCard = React.memo(function SortableCard({
  image,
  gridIndex,
  isSelected,
  isGhost,
  isSearchMatch,
  isCurrentSearchMatch,
  isMarkedForTrash,
  lightboxImages,
  onSelect,
  onRangeSelect,
}: {
  image: ImageInfo;
  gridIndex: number;
  isSelected: boolean;
  isGhost: boolean;
  isSearchMatch?: boolean;
  isCurrentSearchMatch?: boolean;
  isMarkedForTrash?: boolean;
  lightboxImages: string[];
  onSelect: (filename: string, e: React.MouseEvent) => void;
  onRangeSelect: (filename: string, e: React.MouseEvent) => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: image.filename,
  });

  const style = { transform: CSS.Transform.toString(transform), transition };

  return (
    <ImageThumb
      filename={image.filename}
      flipId={image.filename}
      isSelected={isSelected}
      isGhost={isGhost}
      isDragging={isDragging}
      isSearchMatch={isSearchMatch}
      isCurrentSearchMatch={isCurrentSearchMatch}
      isMarkedForTrash={isMarkedForTrash}
      onSelect={onSelect}
      onRangeSelect={onRangeSelect}
      lightboxImages={lightboxImages}
      footer={
        <>
          <span className="image-thumb-badge">{gridIndex + 1}</span>
          <span className="image-thumb-name" title={image.filename}>
            {image.filename}
          </span>
        </>
      }
      dnd={{ setNodeRef, style, attributes, listeners }}
    />
  );
});
