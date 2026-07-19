import { useInteractionsStore, useListStore } from "../../stores/modes/cluster/index.ts";
import type { AppMode } from "../../types.ts";
import { ClusterToolbar } from "../cluster/ClusterToolbar.tsx";
import { CzkawkaPrimary, CzkawkaTools } from "../czkawka/CzkawkaToolbar.tsx";
import {
  ReorderPrimary,
  ReorderSelectionActions,
  ReorderTools,
  ReorderToolsEnd,
} from "../reorder/ReorderToolbar.tsx";
import { ReorderToolbarOverflow } from "../reorder/ReorderToolbarOverflow.tsx";

/** Top row, right: the mode's primary action + global icons. */
export function HeaderPrimary({ mode }: { mode: AppMode }) {
  if (mode === "cluster") return <ClusterRunButton />;
  if (mode === "merge-suggestions") return null;
  if (mode === "czkawka") return <CzkawkaPrimary />;
  return <ReorderPrimary />;
}

/** Top row, middle: contextual selection actions. */
export function HeaderSelection({ mode }: { mode: AppMode }) {
  if (mode === "reorder") return <ReorderSelectionActions />;
  return null;
}

/** Bottom row: the mode's tool strip. */
export function HeaderTools({ mode }: { mode: AppMode }) {
  if (mode === "cluster") return <ClusterActions />;
  if (mode === "merge-suggestions") return null;
  if (mode === "czkawka") return <CzkawkaTools />;
  return (
    <>
      <ReorderTools />
      <ReorderToolsEnd>
        <ReorderToolbarOverflow />
      </ReorderToolsEnd>
    </>
  );
}

function ClusterRunButton() {
  const loading = useListStore((s) => s.loading);
  const fetchClusters = useListStore((s) => s.fetchClusters);
  const desiredN = useListStore((s) => s.desiredN);
  return (
    <button className="btn btn-primary" onClick={() => fetchClusters(desiredN)} disabled={loading}>
      {loading ? "Clustering…" : "Run Clustering"}
    </button>
  );
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
      totalClusters={visibleCount}
      hasError={hasError}
      distanceProfile={clusterData?.distanceProfile ?? null}
      weights={weights}
      usePatches={usePatches}
      useRerank={useRerank}
      rerankBlend={rerankBlend}
      linkage={linkage}
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
