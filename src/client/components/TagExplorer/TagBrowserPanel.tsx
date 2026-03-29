import React, { useMemo } from "react";
import type { ActiveFilter } from "../../types.ts";
import { computeTagCounts, categoryOrderIndex, type InvertedIndex } from "../../utils/tagIndex.ts";
import { CategorySection } from "./CategorySection.tsx";

interface TagBrowserPanelProps {
  invertedIndex: InvertedIndex;
  filteredFilenames: string[];
  filters: ActiveFilter[];
  onTagClick: (category: string, value: string, e: React.MouseEvent) => void;
}

function allIndexedFilenames(index: InvertedIndex): Set<string> {
  const all = new Set<string>();
  for (const valueMap of index.values()) {
    for (const filenames of valueMap.values()) {
      for (const fn of filenames) all.add(fn);
    }
  }
  return all;
}

export function TagBrowserPanel({
  invertedIndex,
  filteredFilenames,
  filters,
  onTagClick,
}: TagBrowserPanelProps) {
  // When no filters active, show counts from ALL indexed images so the user
  // can see every available tag. When filters are active, show counts within
  // the filtered result so the user can see what's left to narrow down.
  const resultSet = useMemo(() => {
    if (filters.length === 0) return allIndexedFilenames(invertedIndex);
    return new Set(filteredFilenames);
  }, [filteredFilenames, filters.length, invertedIndex]);

  const tagCounts = useMemo(
    () => computeTagCounts(invertedIndex, resultSet),
    [invertedIndex, resultSet],
  );

  const sortedCategories = useMemo(() => {
    const entries = [...tagCounts.entries()];
    entries.sort((a, b) => categoryOrderIndex(a[0]) - categoryOrderIndex(b[0]));
    return entries;
  }, [tagCounts]);

  if (sortedCategories.length === 0) {
    return (
      <div className="tag-browser" style={{ padding: 16, color: "var(--text-muted)", fontSize: 13 }}>
        No tags available
      </div>
    );
  }

  return (
    <div className="tag-browser">
      {sortedCategories.map(([category, values]) => (
        <CategorySection
          key={category}
          category={category}
          values={values}
          filters={filters}
          onTagClick={onTagClick}
          defaultCollapsed={false}
        />
      ))}
    </div>
  );
}
