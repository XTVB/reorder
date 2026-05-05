import React from "react";
import { useMergeSuggestionsStore } from "../../stores/mergeSuggestionsStore.ts";
import type { MergeSuggestionSimilar } from "../../types.ts";
import { cn } from "../../utils/helpers.ts";
import { MergeSuggestionCard, type OpenCardHandler } from "./MergeSuggestionCard.tsx";

interface Props {
  refGroupId: string;
  refGroupName: string;
  refGroupImages: string[];
  similar: MergeSuggestionSimilar[];
  collapsed: boolean;
  pendingCandidates: Set<string>;
  refCardExpanded: boolean;
  expandedCandidateId: string | null;
  isCurrentSearchMatch?: boolean;
  onOpenCard: OpenCardHandler;
}

export const MergeSuggestionRow = React.memo(function MergeSuggestionRow({
  refGroupId,
  refGroupName,
  refGroupImages,
  similar,
  collapsed,
  pendingCandidates,
  refCardExpanded,
  expandedCandidateId,
  isCurrentSearchMatch,
  onOpenCard,
}: Props) {
  const toggleRowCollapse = useMergeSuggestionsStore((s) => s.toggleRowCollapse);
  const toggleMergeCandidate = useMergeSuggestionsStore((s) => s.toggleMergeCandidate);
  const rangeSelectInRow = useMergeSuggestionsStore((s) => s.rangeSelectInRow);
  const selectAllInRow = useMergeSuggestionsStore((s) => s.selectAllInRow);
  const deselectAllInRow = useMergeSuggestionsStore((s) => s.deselectAllInRow);

  const bestDistance = similar[0]?.distance;
  const allSelected = similar.length > 0 && similar.every((s) => pendingCandidates.has(s.groupId));
  const someSelected = similar.some((s) => pendingCandidates.has(s.groupId));

  return (
    <div className={cn("merge-row", isCurrentSearchMatch && "merge-row-search-current")}>
      <div className="merge-row-header" onClick={() => toggleRowCollapse(refGroupId)}>
        <span className={`merge-row-chevron ${collapsed ? "" : "merge-row-chevron-open"}`}>
          {"\u25B6"}
        </span>
        <span className="merge-row-name">{refGroupName}</span>
        <span className="merge-row-count">{refGroupImages.length} imgs</span>
        <span className="merge-row-suggestions">{similar.length} suggestions</span>
        {bestDistance != null && (
          <span className="merge-row-best-distance">best: {bestDistance.toFixed(3)}</span>
        )}
        {someSelected && (
          <span className="merge-row-pending">{pendingCandidates.size} selected</span>
        )}
        <button
          className="btn btn-small merge-row-actions"
          onClick={(e) => {
            e.stopPropagation();
            if (allSelected) deselectAllInRow(refGroupId);
            else selectAllInRow(refGroupId);
          }}
        >
          {allSelected ? "Deselect All" : "Select All"}
        </button>
      </div>
      {!collapsed && (
        <div className="merge-row-content">
          <MergeSuggestionCard
            groupName={refGroupName}
            images={refGroupImages}
            imageCount={refGroupImages.length}
            isRef
            refGroupId={refGroupId}
            isExpanded={refCardExpanded}
            onOpenCard={onOpenCard}
          />
          <div className="merge-row-arrow">{"\u2192"}</div>
          <div className="merge-row-candidates">
            {similar.map((s) => (
              <MergeSuggestionCard
                key={s.groupId}
                groupName={s.groupName}
                images={s.groupImages}
                imageCount={s.groupImages.length}
                distance={s.distance}
                isSelected={pendingCandidates.has(s.groupId)}
                isExpanded={expandedCandidateId === s.groupId}
                refGroupId={refGroupId}
                candidateId={s.groupId}
                onToggleSelect={toggleMergeCandidate}
                onRangeSelect={rangeSelectInRow}
                onOpenCard={onOpenCard}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
});
