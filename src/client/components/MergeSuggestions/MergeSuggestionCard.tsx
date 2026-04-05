import React, { useMemo } from "react";
import { imageUrl, pickThumbSamples } from "../../utils/helpers.ts";

export interface OpenCardArgs {
  anchorEl: HTMLElement;
  displayName: string;
  images: string[];
  refGroupId: string;
  /** null when opening the ref card itself */
  candidateId: string | null;
}
export type OpenCardHandler = (args: OpenCardArgs) => void;

interface Props {
  groupName: string;
  images: string[];
  imageCount: number;
  distance?: number;
  isRef?: boolean;
  isSelected?: boolean;
  isExpanded?: boolean;
  refGroupId: string;
  candidateId?: string;
  onToggleSelect?: (refGroupId: string, candidateId: string) => void;
  onRangeSelect?: (refGroupId: string, candidateId: string) => void;
  onOpenCard: OpenCardHandler;
}

export const MergeSuggestionCard = React.memo(function MergeSuggestionCard({
  groupName,
  images,
  imageCount,
  distance,
  isRef,
  isSelected,
  isExpanded,
  refGroupId,
  candidateId,
  onToggleSelect,
  onRangeSelect,
  onOpenCard,
}: Props) {
  const thumbs = useMemo(() => pickThumbSamples(images), [images]);

  const distanceColor =
    distance != null
      ? distance < 0.15
        ? "merge-dist-low"
        : distance < 0.3
          ? "merge-dist-mid"
          : "merge-dist-high"
      : "";

  return (
    <div
      className={`merge-card ${isRef ? "merge-card-ref" : "merge-card-candidate"} ${isSelected ? "merge-card-selected" : ""} ${isExpanded ? "merge-card-active" : ""}`}
      onClick={(e) => {
        if (!isRef && candidateId) {
          if (e.shiftKey) {
            onRangeSelect?.(refGroupId, candidateId);
            return;
          }
          if (e.metaKey || e.ctrlKey) {
            onToggleSelect?.(refGroupId, candidateId);
            return;
          }
        }
        onOpenCard({
          anchorEl: e.currentTarget as HTMLElement,
          displayName: groupName,
          images,
          refGroupId,
          candidateId: isRef ? null : (candidateId ?? null),
        });
      }}
    >
      <div className="merge-card-thumbs">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="merge-card-thumb-slot">
            {thumbs[i] ? (
              <img src={imageUrl(thumbs[i])} alt="" loading="lazy" draggable={false} />
            ) : (
              <div className="merge-card-thumb-empty" />
            )}
          </div>
        ))}
      </div>
      <div className="merge-card-info">
        <span className="merge-card-name" title={groupName}>
          {groupName}
        </span>
        <span className="merge-card-count">{imageCount} imgs</span>
        {distance != null && (
          <span
            className={`merge-card-distance ${distanceColor}`}
            title={`Ward distance: ${distance.toFixed(4)}`}
          >
            {distance.toFixed(3)}
          </span>
        )}
      </div>
      {!isRef && onToggleSelect && candidateId && (
        <button
          type="button"
          className={`merge-card-checkbox ${isSelected ? "merge-card-checkbox-checked" : ""}`}
          aria-label={isSelected ? "Deselect candidate" : "Select candidate for merge"}
          onClick={(e) => {
            e.stopPropagation();
            onToggleSelect(refGroupId, candidateId);
          }}
        >
          {isSelected ? "\u2713" : ""}
        </button>
      )}
    </div>
  );
});
