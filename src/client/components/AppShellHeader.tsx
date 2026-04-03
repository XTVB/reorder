import React from "react";
import { useUIStore } from "../stores/uiStore.ts";
import { useSelectionStore } from "../stores/selectionStore.ts";
import { useClusterStore } from "../stores/clusterStore.ts";

type AppMode = "reorder" | "cluster";

const MODES: { key: AppMode; label: string; path: string }[] = [
  { key: "reorder", label: "Reorder", path: "/reorder" },
  { key: "cluster", label: "Cluster", path: "/cluster" },
];

export function AppShellHeader({ mode, navigate, children }: { mode: AppMode; navigate: (path: string) => void; children?: React.ReactNode }) {
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
      </div>
      <div className="app-header-center">
        <div className="header-title">
          {mode === "reorder" ? "Reorder Images" : "Cluster"}
        </div>
        {headerSubtitle && (
          <div className="header-subtitle">{headerSubtitle}</div>
        )}
      </div>
      <div className="app-header-right">
        {children}
      </div>
    </header>
  );
}
