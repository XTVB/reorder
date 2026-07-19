import type React from "react";
import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useSessionStore } from "../../stores/core/sessionStore.ts";
import type { AppMode } from "../../types.ts";

export const MODES: { key: AppMode; label: string; title: string; path: string }[] = [
  { key: "reorder", label: "Reorder", title: "Reorder Images", path: "/reorder" },
  { key: "cluster", label: "Cluster", title: "Cluster", path: "/cluster" },
  {
    key: "merge-suggestions",
    label: "Merge",
    title: "Merge Suggestions",
    path: "/merge-suggestions",
  },
  { key: "czkawka", label: "Dupes", title: "Czkawka Compare", path: "/czkawka" },
];

export const DEFAULT_MODE: AppMode = "reorder";

export function modeFromPath(pathname: string): AppMode {
  return MODES.find((m) => m.path === pathname)?.key ?? DEFAULT_MODE;
}

/**
 * Two-row header. The top row never changes shape: tabs on the left, status +
 * selection actions in the flexible middle, the mode's primary action pinned
 * on the right. The bottom row holds the mode's tools and wraps on its own
 * without ever colliding with the tabs.
 */
export function AppShellHeader({
  mode,
  navigate,
  selectionSlot,
  primarySlot,
  children,
}: {
  mode: AppMode;
  navigate: (path: string) => void;
  /** Contextual actions for the current selection (top row, middle). */
  selectionSlot?: React.ReactNode;
  /** The mode's primary action + global icons (top row, right). */
  primarySlot?: React.ReactNode;
  /** The mode's tool strip. */
  children?: React.ReactNode;
}) {
  const headerSubtitle = useSessionStore((s) => s.headerSubtitle);
  const clear = useSelectionStore((s) => s.clear);
  const clearMode = useSelectionStore((s) => s.clearMode);

  // Reorder's tools + selection actions need a full second row (and that row
  // is what keeps the layout from shifting as the selection changes). The
  // other modes have compact tool sets and no header selection actions, so
  // their tools sit inline in the top row's middle — one slim header.
  const inlineTools = mode !== "reorder";

  return (
    <header className="app-header">
      <div className="app-header-top">
        <div className="app-header-lead">
          <div className="mode-toggle">
            {MODES.map((m) => (
              <button
                key={m.key}
                className={`mode-toggle-btn ${mode === m.key ? "mode-toggle-active" : ""}`}
                onClick={() => {
                  if (mode !== m.key) {
                    clear("reorder");
                    clearMode("cluster");
                    navigate(m.path);
                  }
                }}
              >
                {m.label}
              </button>
            ))}
          </div>
          {headerSubtitle && <div className="header-subtitle">{headerSubtitle}</div>}
        </div>
        <div className="app-header-mid">
          {selectionSlot && <div className="app-header-selection">{selectionSlot}</div>}
          {inlineTools && children ? (
            <div className="app-header-inline-tools">{children}</div>
          ) : null}
        </div>
        <div className="app-header-primary">{primarySlot}</div>
      </div>
      {!inlineTools && children && <div className="app-header-tools">{children}</div>}
    </header>
  );
}
