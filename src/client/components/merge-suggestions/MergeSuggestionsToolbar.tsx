interface Props {
  threshold: number;
  loading: boolean;
  computeTimeMs: number | null;
  progress: string | null;
  suggestionCount: number;
  pendingCount: number;
  canUndo: boolean;
  fullResolution: boolean;
  maxCombinedSize: number;
  onThresholdChange: (t: number) => void;
  onFullResolutionChange: (v: boolean) => void;
  onMaxCombinedSizeChange: (n: number) => void;
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
  progress,
  suggestionCount,
  pendingCount,
  canUndo,
  fullResolution,
  maxCombinedSize,
  onThresholdChange,
  onFullResolutionChange,
  onMaxCombinedSizeChange,
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

        <div className="merge-method-toggle">
          <button
            className={`btn btn-small ${!fullResolution ? "btn-active" : ""}`}
            onClick={() => onFullResolutionChange(false)}
            disabled={loading}
            title="7x7 averaged patches — fast (~20s)"
          >
            Fast
          </button>
          <button
            className={`btn btn-small ${fullResolution ? "btn-active" : ""}`}
            onClick={() => onFullResolutionChange(true)}
            disabled={loading}
            title="14x14 full-resolution patches — slower (~4 min) but slightly more accurate"
          >
            Full-res
          </button>
        </div>

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

        <label
          className="merge-threshold-control"
          title="Skip pairs whose combined image count would exceed this value. 0 = no limit. Smaller limits make compute much faster."
        >
          <span className="merge-threshold-label">Max combined size</span>
          <input
            type="number"
            className="merge-max-size-input"
            min="0"
            step="1"
            value={maxCombinedSize}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10);
              onMaxCombinedSizeChange(Number.isFinite(v) ? v : 0);
            }}
            disabled={loading}
          />
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

      {loading && progress && (
        <div className="merge-progress">
          <span className="merge-progress-text">{progress}</span>
        </div>
      )}

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
