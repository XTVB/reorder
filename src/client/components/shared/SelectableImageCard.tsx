import type { MouseEvent, ReactNode } from "react";
import { cn, imageUrl } from "../../utils/helpers.ts";

export function SelectableImageCard({
  filename,
  selected,
  onToggleSelect,
  onRangeSelect,
  onOpen,
  topRight,
  bottomLeft,
  bottomRight,
}: {
  filename: string;
  selected: boolean;
  onToggleSelect: () => void;
  onRangeSelect: () => void;
  onOpen: () => void;
  topRight?: ReactNode;
  bottomLeft?: ReactNode;
  bottomRight?: ReactNode;
}) {
  function handleImageClick(e: MouseEvent) {
    e.stopPropagation();
    if (e.shiftKey) onRangeSelect();
    else if (e.metaKey || e.ctrlKey) onToggleSelect();
    else onOpen();
  }
  return (
    <div className={cn("image-card", selected && "image-card-selected")}>
      <button
        type="button"
        className="image-card-image"
        onClick={handleImageClick}
        title={`Open ${filename} (⌘ click to select, ⇧ click for range)`}
        aria-label={`Open ${filename}`}
      >
        <img src={imageUrl(filename)} alt="" loading="lazy" decoding="async" draggable={false} />
      </button>
      <button
        type="button"
        className="image-card-select"
        onClick={(e) => {
          e.stopPropagation();
          if (e.shiftKey) onRangeSelect();
          else onToggleSelect();
        }}
        aria-label={selected ? "Deselect" : "Select"}
        title={selected ? "Deselect" : "Select"}
      >
        <span className="image-card-check" aria-hidden>
          {selected ? "✓" : ""}
        </span>
      </button>
      {topRight && <div className="image-card-top-right">{topRight}</div>}
      {bottomLeft && <div className="image-card-bottom-left">{bottomLeft}</div>}
      {bottomRight && <div className="image-card-bottom-right">{bottomRight}</div>}
    </div>
  );
}
