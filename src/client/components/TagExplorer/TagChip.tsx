import React from "react";
import type { FilterMode } from "../../types.ts";
import { cn } from "../../utils/helpers.ts";

interface TagChipProps {
  category: string;
  value: string;
  count?: number;
  activeMode?: FilterMode | null;
  onClick: (e: React.MouseEvent) => void;
  onRemove?: () => void;
  onModeChange?: (mode: FilterMode) => void;
}

const MODE_CYCLE: FilterMode[] = ["AND", "OR", "NOT"];

export function TagChip({
  value,
  count,
  activeMode,
  onClick,
  onRemove,
  onModeChange,
}: TagChipProps) {
  const chipClass = cn(
    "tag-chip",
    activeMode === "AND" && "tag-chip-active-and",
    activeMode === "OR" && "tag-chip-active-or",
    activeMode === "NOT" && "tag-chip-active-not",
  );

  function handleClick(e: React.MouseEvent) {
    e.preventDefault();
    onClick(e);
  }

  function handleContextMenu(e: React.MouseEvent) {
    e.preventDefault();
    onClick(e);
  }

  function handleModeClick(e: React.MouseEvent) {
    e.stopPropagation();
    if (!onModeChange || !activeMode) return;
    const idx = MODE_CYCLE.indexOf(activeMode);
    onModeChange(MODE_CYCLE[(idx + 1) % MODE_CYCLE.length]);
  }

  return (
    <span
      className={chipClass}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
    >
      {activeMode && onModeChange && (
        <span className="tag-chip-mode" onClick={handleModeClick}>
          {activeMode}
        </span>
      )}
      {value.replace(/_/g, " ")}
      {count != null && <span className="tag-chip-count">{count}</span>}
      {onRemove && (
        <button
          className="tag-chip-remove"
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
        >
          x
        </button>
      )}
    </span>
  );
}
