import React from "react";

interface Props {
  threshold: number;
  loading: boolean;
  computeTimeMs: number | null;
  suggestionCount: number;
  pendingCount: number;
  canUndo: boolean;
  onThresholdChange: (t: number) => void;
  onCompute: () => void;
  onApply: () => void;
  onUndo: () => void;
  onClear: () => void;
  onExpandAll: () => void;
  onCollapseAll: () => void;
}

export function MergeSuggestionsToolbar({
  threshold,
  loading,
  computeTimeMs,
  suggestionCount,
  pendingCount,
  canUndo,
  onThresholdChange,
  onCompute,
  onApply,
  onUndo,
  onClear,
  onExpandAll,
  onCollapseAll,
}: Props) {
  return (
    <div className="merge-toolbar">
      <div className="merge-toolbar-row">
        <button className="btn btn-primary" onClick={onCompute} disabled={loading}>
          {loading ? "Computing..." : "Compute"}
        </button>

        <label className="merge-threshold-control">
          <span className="merge-threshold-label">Min similarity</span>
          <input
            type="range"
            min="0.50"
            max="0.80"
            step="0.01"
            value={threshold}
            onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
            disabled={loading}
          />
          <span className="merge-threshold-value">{threshold.toFixed(2)}</span>
        </label>

        {computeTimeMs != null && (
          <span className="merge-compute-time">{(computeTimeMs / 1000).toFixed(1)}s</span>
        )}

        {suggestionCount > 0 && (
          <span className="merge-suggestion-count">{suggestionCount} rows</span>
        )}

        <button className="btn btn-small" onClick={onExpandAll} disabled={!suggestionCount}>
          Expand All
        </button>
        <button className="btn btn-small" onClick={onCollapseAll} disabled={!suggestionCount}>
          Collapse All
        </button>
      </div>

      {pendingCount > 0 && (
        <div className="merge-toolbar-actions">
          <button className="btn btn-success" onClick={onApply}>
            Apply {pendingCount} merge{pendingCount > 1 ? "s" : ""}
          </button>
          <button className="btn btn-small" onClick={onClear}>
            Clear
          </button>
          {canUndo && (
            <button className="btn btn-small" onClick={onUndo}>
              Undo
            </button>
          )}
        </div>
      )}
      {pendingCount === 0 && canUndo && (
        <div className="merge-toolbar-actions">
          <button className="btn btn-small" onClick={onUndo}>
            Undo Last Apply
          </button>
        </div>
      )}
    </div>
  );
}
