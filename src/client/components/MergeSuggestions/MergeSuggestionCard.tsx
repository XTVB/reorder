import React, { useMemo, useState } from "react";
import { imageUrl, pickThumbSamples } from "../../utils/helpers.ts";

interface Props {
  groupName: string;
  images: string[];
  imageCount: number;
  distance?: number;
  isRef?: boolean;
  isSelected?: boolean;
  refGroupId?: string;
  candidateId?: string;
  onToggleSelect?: (refGroupId: string, candidateId: string) => void;
}

export const MergeSuggestionCard = React.memo(function MergeSuggestionCard({
  groupName,
  images,
  imageCount,
  distance,
  isRef,
  isSelected,
  refGroupId,
  candidateId,
  onToggleSelect,
}: Props) {
  const [expanded, setExpanded] = useState(false);

  const thumbs = useMemo(() => pickThumbSamples(images), [images]);

  const distanceColor = distance != null
    ? distance < 0.15 ? "merge-dist-low" : distance < 0.3 ? "merge-dist-mid" : "merge-dist-high"
    : "";

  return (
    <div
      className={`merge-card ${isRef ? "merge-card-ref" : "merge-card-candidate"} ${isSelected ? "merge-card-selected" : ""}`}
      onClick={() => {
        if (isRef) {
          setExpanded(!expanded);
        } else if (onToggleSelect && refGroupId && candidateId) {
          onToggleSelect(refGroupId, candidateId);
        }
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
        <span className="merge-card-name" title={groupName}>{groupName}</span>
        <span className="merge-card-count">{imageCount} imgs</span>
        {distance != null && (
          <span className={`merge-card-distance ${distanceColor}`} title={`Ward distance: ${distance.toFixed(4)}`}>
            {distance.toFixed(3)}
          </span>
        )}
      </div>
      {!isRef && (
        <div className={`merge-card-checkbox ${isSelected ? "merge-card-checkbox-checked" : ""}`}>
          {isSelected ? "\u2713" : ""}
        </div>
      )}
      {expanded && (
        <div className="merge-card-expanded">
          {images.map((img) => (
            <div key={img} className="merge-card-expanded-thumb">
              <img src={imageUrl(img)} alt={img} loading="lazy" draggable={false} />
              <span className="merge-card-expanded-name">{img}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});
