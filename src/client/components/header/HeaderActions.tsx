import { useInteractionsStore, useListStore } from "../../stores/modes/cluster/index.ts";
import type { AppMode } from "../../types.ts";
import { ClusterToolbar } from "../cluster/ClusterToolbar.tsx";
import { CzkawkaToolbar } from "../czkawka/CzkawkaToolbar.tsx";
import { ReorderToolbar } from "../reorder/ReorderToolbar.tsx";

export function HeaderActions({ mode }: { mode: AppMode }) {
  if (mode === "cluster") return <ClusterActions />;
  if (mode === "merge-suggestions") return null;
  if (mode === "czkawka") return <CzkawkaToolbar />;
  return <ReorderToolbar />;
}

function ClusterActions() {
  const clusterData = useListStore((s) => s.clusterData);
  const loading = useListStore((s) => s.loading);
  const progress = useListStore((s) => s.progress);
  const weights = useListStore((s) => s.weights);
  const usePatches = useListStore((s) => s.usePatches);
  const useRerank = useListStore((s) => s.useRerank);
  const rerankBlend = useListStore((s) => s.rerankBlend);
  const linkage = useListStore((s) => s.linkage);
  const setWeights = useListStore((s) => s.setWeights);
  const setUsePatches = useListStore((s) => s.setUsePatches);
  const setUseRerank = useListStore((s) => s.setUseRerank);
  const setRerankBlend = useListStore((s) => s.setRerankBlend);
  const setLinkage = useListStore((s) => s.setLinkage);
  const fetchClusters = useListStore((s) => s.fetchClusters);
  const recut = useListStore((s) => s.recut);
  const expandAll = useListStore((s) => s.expandAll);
  const collapseAll = useListStore((s) => s.collapseAll);
  const acceptAllClusters = useInteractionsStore((s) => s.acceptAllClusters);
  const importClusters = useListStore((s) => s.importClusters);
  const clearImportedClusters = useListStore((s) => s.clearImportedClusters);
  const visibleCount = clusterData?.clusters.length ?? 0;
  const hasError = progress.startsWith("Error:");

  return (
    <ClusterToolbar
      loading={loading}
      progress={progress}
      nClusters={clusterData?.nClusters ?? 200}
      totalClusters={visibleCount}
      hasError={hasError}
      distanceProfile={clusterData?.distanceProfile ?? null}
      weights={weights}
      usePatches={usePatches}
      useRerank={useRerank}
      rerankBlend={rerankBlend}
      linkage={linkage}
      onRun={fetchClusters}
      onRecut={(n) => recut({ nClusters: n })}
      onRecutAdaptive={(minClusterSize) => recut({ minClusterSize })}
      onWeightsChange={setWeights}
      onUsePatchesChange={setUsePatches}
      onUseRerankChange={setUseRerank}
      onRerankBlendChange={setRerankBlend}
      onLinkageChange={setLinkage}
      onExpandAll={expandAll}
      onCollapseAll={collapseAll}
      onAcceptAll={acceptAllClusters}
      onImportClusters={importClusters}
      onClearImported={clearImportedClusters}
    />
  );
}
