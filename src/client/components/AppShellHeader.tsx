import type React from "react";
import { useClusterStore } from "../stores/clusterStore.ts";
import { useSelectionStore } from "../stores/selectionStore.ts";
import { useUIStore } from "../stores/uiStore.ts";
import type { AppMode } from "../types.ts";

export const MODES: { key: AppMode; label: string; title: string; path: string }[] = [
  { key: "reorder", label: "Reorder", title: "Reorder Images", path: "/reorder" },
  { key: "cluster", label: "Cluster", title: "Cluster", path: "/cluster" },
  { key: "cluster-compare", label: "Compare", title: "Cluster", path: "/cluster-compare" },
  {
    key: "merge-suggestions",
    label: "Merge",
    title: "Merge Suggestions",
    path: "/merge-suggestions",
  },
];

export const DEFAULT_MODE: AppMode = "reorder";

export function modeFromPath(pathname: string): AppMode {
  return MODES.find((m) => m.path === pathname)?.key ?? DEFAULT_MODE;
}

export function AppShellHeader({
  mode,
  navigate,
  leftSlot,
  children,
}: {
  mode: AppMode;
  navigate: (path: string) => void;
  leftSlot?: React.ReactNode;
  children?: React.ReactNode;
}) {
  const headerSubtitle = useUIStore((s) => s.headerSubtitle);
  const clearSelection = useSelectionStore((s) => s.clearSelection);
  const clearImageSelection = useClusterStore((s) => s.clearImageSelection);

  return (
    <header className="app-header">
      <div className="app-header-left">
        <div className="mode-toggle">
          {MODES.map((m) => (
            <button
              key={m.key}
              className={`mode-toggle-btn ${mode === m.key ? "mode-toggle-active" : ""}`}
              onClick={() => {
                if (mode !== m.key) {
                  clearSelection();
                  clearImageSelection();
                  navigate(m.path);
                }
              }}
            >
              {m.label}
            </button>
          ))}
        </div>
        {leftSlot}
        {headerSubtitle && <div className="header-subtitle">{headerSubtitle}</div>}
      </div>
      <div className="app-header-right">{children}</div>
    </header>
  );
}
