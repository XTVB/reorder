import { createRoot } from "react-dom/client";
import { App } from "./App.tsx";
import { TagExplorer } from "./components/TagExplorer/TagExplorer.tsx";
import { MergeView } from "./components/MergeView/MergeView.tsx";
import { useUIStore } from "./stores/uiStore.ts";
import { useImageStore } from "./stores/imageStore.ts";
import { useGroupStore } from "./stores/groupStore.ts";
import { useSelectionStore } from "./stores/selectionStore.ts";
import { useDndStore } from "./stores/dndStore.ts";
import { useTagStore } from "./stores/tagStore.ts";

// Expose all stores on window for console access / debugging
(window as any).__stores = {
  images: useImageStore,
  groups: useGroupStore,
  selection: useSelectionStore,
  dnd: useDndStore,
  ui: useUIStore,
  tags: useTagStore,
};

function AppShell() {
  const appMode = useUIStore((s) => s.appMode);

  if (appMode === "tags") return <TagExplorer />;
  if (appMode === "merge") return <MergeView />;
  return <App />;
}

createRoot(document.getElementById("root")!).render(<AppShell />);
