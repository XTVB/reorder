import { useClusterStore } from "../../stores/clusterStore.ts";

export function ScopeBanner() {
  const scope = useClusterStore((s) => s.clusterData?.scope);
  const exitScope = useClusterStore((s) => s.exitScope);

  if (!scope) return null;

  const label =
    scope.groupNames.length <= 3
      ? scope.groupNames.join(", ")
      : `${scope.groupNames.slice(0, 3).join(", ")} +${scope.groupNames.length - 3}`;

  return (
    <div className="cluster-scope-banner">
      <span className="cluster-scope-dot" aria-hidden />
      <span className="cluster-scope-label">
        Scoped view · {scope.groupIds.length} groups · {scope.nImages} images
      </span>
      <span className="cluster-scope-detail" title={scope.groupNames.join(", ")}>
        {label}
      </span>
      <button type="button" className="btn btn-small" onClick={exitScope}>
        Exit scope
      </button>
    </div>
  );
}
