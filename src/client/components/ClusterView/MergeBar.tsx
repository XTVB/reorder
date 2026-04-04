import React, { useMemo } from "react";
import type { ClusterResultData } from "../../types.ts";

interface Props {
  selection: Set<string>;
  clusters: ClusterResultData[];
  onMerge: () => void;
  onCancel: () => void;
  onRemove: (id: string) => void;
}

export function MergeBar({ selection, clusters, onMerge, onCancel, onRemove }: Props) {
  const selected = useMemo(
    () => clusters.filter((c) => selection.has(c.id)),
    [clusters, selection],
  );

  return (
    <div className="cluster-merge-bar">
      <span className="merge-count">{selection.size} clusters selected</span>
      <div className="merge-tags">
        {selected.map((c) => (
          <span key={c.id} className="merge-tag" onClick={() => onRemove(c.id)}>
            {c.autoName || c.id} ({c.images.length}) ×
          </span>
        ))}
      </div>
      <button className="btn btn-merge" onClick={onMerge} disabled={selection.size < 2}>
        Merge
      </button>
      <button className="btn btn-small" onClick={onCancel}>
        Cancel
      </button>
    </div>
  );
}
