import React, { useState, useEffect } from "react";

const DEFAULT_N_CLUSTERS = 200;

interface Props {
  loading: boolean;
  progress: string;
  nClusters: number;
  suggestedCounts: number[];
  totalClusters: number;
  hasError: boolean;
  onRun: (n?: number) => void;
  onRecut: (n: number) => void;
  onExpandAll: () => void;
  onCollapseAll: () => void;
  onAcceptAll: (minSize: number) => void;
}

export function ClusterToolbar({
  loading, progress, nClusters, suggestedCounts, totalClusters, hasError,
  onRun, onRecut, onExpandAll, onCollapseAll, onAcceptAll,
}: Props) {
  const [customN, setCustomN] = useState(String(nClusters));

  useEffect(() => { setCustomN(String(nClusters)); }, [nClusters]);

  return (
    <div className="cluster-toolbar">
      <div className="cluster-toolbar-left">
        <button
          className="btn btn-primary"
          onClick={() => onRun(parseInt(customN) || DEFAULT_N_CLUSTERS)}
          disabled={loading}
        >
          {loading ? "Clustering..." : "Run Clustering"}
        </button>

        <label className="cluster-n-selector">
          N=
          <select
            value={customN}
            onChange={(e) => {
              setCustomN(e.target.value);
              onRecut(parseInt(e.target.value));
            }}
            disabled={loading || !totalClusters}
          >
            {suggestedCounts.map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
          <input
            type="number"
            value={customN}
            onChange={(e) => setCustomN(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") onRecut(parseInt(customN) || DEFAULT_N_CLUSTERS);
            }}
            className="cluster-n-input"
            min={2}
            disabled={loading}
          />
          <button
            className="btn btn-small"
            onClick={() => onRecut(parseInt(customN) || DEFAULT_N_CLUSTERS)}
            disabled={loading || !totalClusters}
          >
            Re-cut
          </button>
        </label>
      </div>

      <div className="cluster-toolbar-right">
        {(loading || hasError) && (
          <span className={`cluster-progress ${hasError ? "cluster-error" : ""}`}>{progress}</span>
        )}
        {totalClusters > 0 && <span className="cluster-count">{totalClusters} clusters</span>}
        <button className="btn btn-small" onClick={onExpandAll} disabled={!totalClusters}>Expand All</button>
        <button className="btn btn-small" onClick={onCollapseAll} disabled={!totalClusters}>Collapse All</button>
        <button
          className="btn btn-accept-all"
          onClick={() => onAcceptAll(3)}
          disabled={!totalClusters}
        >
          Accept All
        </button>
      </div>
    </div>
  );
}
