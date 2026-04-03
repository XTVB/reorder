import { createRoot } from "react-dom/client";
import { useEffect } from "react";
import { App } from "./App.tsx";
import { ClusterView } from "./components/ClusterView/ClusterView.tsx";
import { Toast } from "./components/Toast.tsx";
import { AppShellHeader } from "./components/AppShellHeader.tsx";
import { Toolbar } from "./components/Toolbar.tsx";
import { ClusterToolbar } from "./components/ClusterView/ClusterToolbar.tsx";
import { useRouter } from "./hooks/useRouter.ts";
import { useImageStore } from "./stores/imageStore.ts";
import { useGroupStore } from "./stores/groupStore.ts";
import { useSelectionStore } from "./stores/selectionStore.ts";
import { useDndStore } from "./stores/dndStore.ts";
import { useUIStore } from "./stores/uiStore.ts";
import { useClusterStore } from "./stores/clusterStore.ts";

// Expose all stores on window for console access / debugging
(window as any).__stores = {
  images: useImageStore,
  groups: useGroupStore,
  selection: useSelectionStore,
  dnd: useDndStore,
  ui: useUIStore,
  cluster: useClusterStore,
};

function AppShell() {
  const { pathname, navigate } = useRouter();
  const mode = pathname === "/cluster" ? "cluster" : "reorder";

  useEffect(() => {
    if (pathname !== "/reorder" && pathname !== "/cluster") {
      navigate("/reorder");
    }
  }, [pathname]);

  return (
    <>
      <AppShellHeader mode={mode} navigate={navigate}>
        {mode === "cluster" ? <ClusterActions /> : <Toolbar />}
      </AppShellHeader>
      {mode === "cluster" ? <ClusterView /> : <App />}
      <Toast />
    </>
  );
}

function ClusterActions() {
  const clusterData = useClusterStore((s) => s.clusterData);
  const loading = useClusterStore((s) => s.loading);
  const progress = useClusterStore((s) => s.progress);
  const fetchClusters = useClusterStore((s) => s.fetchClusters);
  const recutClusters = useClusterStore((s) => s.recutClusters);
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
      suggestedCounts={clusterData?.suggestedCounts ?? []}
      totalClusters={visibleCount}
      hasError={hasError}
      onRun={fetchClusters}
      onRecut={recutClusters}
      onExpandAll={expandAll}
      onCollapseAll={collapseAll}
      onAcceptAll={acceptAllClusters}
    />
  );
}

createRoot(document.getElementById("root")!).render(<AppShell />);
