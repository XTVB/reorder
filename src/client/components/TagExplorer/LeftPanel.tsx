import React, { useMemo } from "react";
import type { ImageGroup } from "../../types.ts";
import { cn, imageUrl } from "../../utils/helpers.ts";

interface LeftPanelProps {
  groups: ImageGroup[];
  focusedGroupId: string | null;
  onFocusGroup: (groupId: string | null) => void;
  filteredSet: Set<string>;
  hasFilters: boolean;
  onAddToGroup: (groupId: string) => void;
  onImageClick: (filename: string) => void;
  selectedCount: number;
}

export function LeftPanel({
  groups,
  focusedGroupId,
  onFocusGroup,
  filteredSet,
  hasFilters,
  onAddToGroup,
  onImageClick,
  selectedCount,
}: LeftPanelProps) {
  const groupStats = useMemo(() => {
    return groups.map((g) => {
      const matchCount = g.images.filter((fn) => filteredSet.has(fn)).length;
      return { group: g, matchCount };
    });
  }, [groups, filteredSet]);

  const focusedGroup = focusedGroupId
    ? groups.find((g) => g.id === focusedGroupId) ?? null
    : null;

  return (
    <div className="tag-left-panel">
      <div className="group-list-header">
        <span>Groups ({groups.length})</span>
      </div>
      <div className="group-list">
        {groupStats.map(({ group, matchCount }) => {
          const isFocused = focusedGroupId === group.id;
          const isDimmed = hasFilters && matchCount === 0;

          return (
            <div key={group.id}>
              <div
                className={cn(
                  "group-list-item",
                  isFocused && "group-list-item-focused",
                  isDimmed && "group-list-item-dimmed",
                )}
                onClick={() => onFocusGroup(isFocused ? null : group.id)}
              >
                <span className="group-list-item-name">{group.name}</span>
                <span className="group-list-item-badge">
                  {hasFilters ? (
                    <span className={matchCount > 0 ? "group-list-item-match" : ""}>
                      {matchCount}/{group.images.length}
                    </span>
                  ) : (
                    group.images.length
                  )}
                </span>
              </div>

              {isFocused && focusedGroup && (
                <div>
                  {selectedCount > 0 && (
                    <div style={{ padding: "6px 10px" }}>
                      <button
                        className="btn btn-primary"
                        style={{ width: "100%", padding: "6px", fontSize: "12px" }}
                        onClick={(e) => { e.stopPropagation(); onAddToGroup(group.id); }}
                      >
                        Add {selectedCount} selected to {group.name}
                      </button>
                    </div>
                  )}
                  <div className="focused-group-grid">
                    {focusedGroup.images.map((fn) => (
                      <img
                        key={fn}
                        src={imageUrl(fn)}
                        alt={fn}
                        loading="lazy"
                        onClick={(e) => { e.stopPropagation(); onImageClick(fn); }}
                        style={{ cursor: "pointer" }}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
        {groups.length === 0 && (
          <div style={{ padding: "16px", color: "var(--text-muted)", fontSize: "13px", textAlign: "center" }}>
            No groups yet
          </div>
        )}
      </div>
    </div>
  );
}
