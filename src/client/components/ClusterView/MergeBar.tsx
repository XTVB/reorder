import { useMemo } from "react";
import { useClusterStore } from "../../stores/clusterStore.ts";
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

  const inScope = useClusterStore((s) => !!s.clusterData?.scope);
  const runScopedCluster = useClusterStore((s) => s.runScopedCluster);

  // Scoped re-cluster is available only when every selected cluster has a confirmed group
  // AND we're not already inside a scoped view.
  const scopeGroupIds = useMemo(() => {
    if (inScope) return null;
    if (selected.length < 2) return null;
    const ids: string[] = [];
    for (const c of selected) {
      if (!c.confirmedGroup) return null;
      ids.push(c.confirmedGroup.id);
    }
    return [...new Set(ids)];
  }, [selected, inScope]);

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
      {scopeGroupIds && scopeGroupIds.length >= 2 && (
        <button
          className="btn btn-secondary"
          onClick={() => runScopedCluster(scopeGroupIds)}
          title="Re-cluster over only the images in these confirmed groups"
        >
          Re-cluster these groups
        </button>
      )}
      <button className="btn btn-small" onClick={onCancel}>
        Cancel
      </button>
    </div>
  );
}
