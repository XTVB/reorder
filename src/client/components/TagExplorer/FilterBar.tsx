import React from "react";
import type { ActiveFilter, FilterMode } from "../../types.ts";
import { formatCategoryName, formatClothingValue } from "../../utils/helpers.ts";
import { TagChip } from "./TagChip.tsx";

interface FilterBarProps {
  filters: ActiveFilter[];
  onRemove: (category: string, value: string) => void;
  onModeChange: (category: string, value: string, mode: FilterMode) => void;
  onClear: () => void;
}

function formatFilterLabel(f: ActiveFilter): string {
  if (f.category === "__clothing_structured") return formatClothingValue(f.value);
  return `${formatCategoryName(f.category)}: ${f.value.replace(/_/g, " ")}`;
}

export function FilterBar({ filters, onRemove, onModeChange, onClear }: FilterBarProps) {
  return (
    <>
      {filters.length === 0 ? (
        <span className="tag-filter-bar-empty">No active filters — click tags to filter</span>
      ) : (
        <>
          {filters.map((f) => (
            <TagChip
              key={`${f.category}:${f.value}`}
              category={f.category}
              value={formatFilterLabel(f)}
              activeMode={f.mode}
              onClick={() => {}}
              onRemove={() => onRemove(f.category, f.value)}
              onModeChange={(mode) => onModeChange(f.category, f.value, mode)}
            />
          ))}
          <button className="btn btn-secondary" style={{ padding: "4px 10px", fontSize: "12px" }} onClick={onClear}>
            Clear all
          </button>
        </>
      )}
    </>
  );
}
