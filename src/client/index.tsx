import { useEffect } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.tsx";
import { AppShellHeader, DEFAULT_MODE, MODES, modeFromPath } from "./components/AppShellHeader.tsx";
import { ClusterCompare } from "./components/ClusterCompare/ClusterCompare.tsx";
import { ClusterToolbar } from "./components/ClusterView/ClusterToolbar.tsx";
import { ClusterView } from "./components/ClusterView/ClusterView.tsx";
import { MergeSuggestions } from "./components/MergeSuggestions/MergeSuggestions.tsx";
import { Toast } from "./components/Toast.tsx";
import { Toolbar } from "./components/Toolbar.tsx";
import { useRouter } from "./hooks/useRouter.ts";
import { useClusterStore } from "./stores/clusterStore.ts";
import { useDndStore } from "./stores/dndStore.ts";
import { useGroupStore } from "./stores/groupStore.ts";
import { useImageStore } from "./stores/imageStore.ts";
import { useMergeSuggestionsStore } from "./stores/mergeSuggestionsStore.ts";
import { useSelectionStore } from "./stores/selectionStore.ts";
import { useUIStore } from "./stores/uiStore.ts";

// Expose all stores on window for console access / debugging
(window as unknown as Record<string, unknown>).__stores = {
  images: useImageStore,
  groups: useGroupStore,
  selection: useSelectionStore,
  dnd: useDndStore,
  ui: useUIStore,
  cluster: useClusterStore,
  mergeSuggestions: useMergeSuggestionsStore,
};

function AppShell() {
  const { pathname, navigate } = useRouter();
  const mode = modeFromPath(pathname);

  // biome-ignore lint/correctness/useExhaustiveDependencies: navigate is a stable ref from useRouter
  useEffect(() => {
    if (!MODES.some((m) => m.path === pathname)) {
      navigate(MODES.find((m) => m.key === DEFAULT_MODE)!.path);
    }
  }, [pathname]);

  return (
    <>
      <AppShellHeader mode={mode} navigate={navigate}>
        {mode === "cluster" ? (
          <ClusterActions />
        ) : mode === "cluster-compare" || mode === "merge-suggestions" ? null : (
          <Toolbar />
        )}
      </AppShellHeader>
      {mode === "cluster" ? (
        <ClusterView />
      ) : mode === "cluster-compare" ? (
        <ClusterCompare />
      ) : mode === "merge-suggestions" ? (
        <MergeSuggestions />
      ) : (
        <App />
      )}
      <Toast />
    </>
  );
}

function ClusterActions() {
  const clusterData = useClusterStore((s) => s.clusterData);
  const loading = useClusterStore((s) => s.loading);
  const progress = useClusterStore((s) => s.progress);
  const weights = useClusterStore((s) => s.weights);
  const usePatches = useClusterStore((s) => s.usePatches);
  const setWeights = useClusterStore((s) => s.setWeights);
  const setUsePatches = useClusterStore((s) => s.setUsePatches);
  const fetchClusters = useClusterStore((s) => s.fetchClusters);
  const recutClusters = useClusterStore((s) => s.recutClusters);
  const recutByThreshold = useClusterStore((s) => s.recutByThreshold);
  const recutAdaptive = useClusterStore((s) => s.recutAdaptive);
  const expandAll = useClusterStore((s) => s.expandAll);
  const collapseAll = useClusterStore((s) => s.collapseAll);
  const acceptAllClusters = useClusterStore((s) => s.acceptAllClusters);
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
      onRun={fetchClusters}
      onRecut={recutClusters}
      onRecutByThreshold={recutByThreshold}
      onRecutAdaptive={recutAdaptive}
      onWeightsChange={setWeights}
      onUsePatchesChange={setUsePatches}
      onExpandAll={expandAll}
      onCollapseAll={collapseAll}
      onAcceptAll={acceptAllClusters}
    />
  );
}

createRoot(document.getElementById("root")!).render(<AppShell />);
