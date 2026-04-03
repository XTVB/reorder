import { createRoot } from "react-dom/client";
import { App } from "./App.tsx";
import { ClusterView } from "./components/ClusterView/ClusterView.tsx";
import { Toast } from "./components/Toast.tsx";
import { useUIStore } from "./stores/uiStore.ts";
import { useImageStore } from "./stores/imageStore.ts";
import { useGroupStore } from "./stores/groupStore.ts";
import { useSelectionStore } from "./stores/selectionStore.ts";
import { useDndStore } from "./stores/dndStore.ts";
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
  const appMode = useUIStore((s) => s.appMode);

  return (
    <>
      {appMode === "cluster" ? <ClusterView /> : <App />}
      <Toast />
    </>
  );
}

createRoot(document.getElementById("root")!).render(<AppShell />);
