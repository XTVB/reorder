import React, { useState, useMemo } from "react";
import type { ActiveFilter, FilterMode } from "../../types.ts";
import { formatCategoryName } from "../../utils/helpers.ts";
import { TagChip } from "./TagChip.tsx";

interface CategorySectionProps {
  category: string;
  values: Map<string, number>;
  filters: ActiveFilter[];
  onTagClick: (category: string, value: string, e: React.MouseEvent) => void;
  defaultCollapsed?: boolean;
}

export function CategorySection({
  category,
  values,
  filters,
  onTagClick,
  defaultCollapsed = false,
}: CategorySectionProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  const sorted = useMemo(() => [...values.entries()].sort((a, b) => b[1] - a[1]), [values]);

  function getActiveMode(value: string): FilterMode | null {
    const f = filters.find((f) => f.category === category && f.value === value);
    return f?.mode ?? null;
  }

  return (
    <div className="tag-category">
      <div className="tag-category-header" onClick={() => setCollapsed(!collapsed)}>
        <span>{formatCategoryName(category)}</span>
        <span className={`tag-category-arrow ${collapsed ? "collapsed" : ""}`}>▼</span>
      </div>
      {!collapsed && (
        <div className="tag-category-values">
          {sorted.map(([value, count]) => (
            <TagChip
              key={value}
              category={category}
              value={value}
              count={count}
              activeMode={getActiveMode(value)}
              onClick={(e) => onTagClick(category, value, e)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
