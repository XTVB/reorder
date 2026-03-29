import { createRoot } from "react-dom/client";
import { App } from "./App.tsx";
import { TagExplorer } from "./components/TagExplorer/TagExplorer.tsx";
import { MergeView } from "./components/MergeView/MergeView.tsx";
import { useUIStore } from "./stores/uiStore.ts";

function AppShell() {
  const appMode = useUIStore((s) => s.appMode);

  if (appMode === "tags") return <TagExplorer />;
  if (appMode === "merge") return <MergeView />;
  return <App />;
}

createRoot(document.getElementById("root")!).render(<AppShell />);
