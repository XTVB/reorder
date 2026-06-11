import { useEffect, useMemo, useRef, useState } from "react";
import { useDismissOnOutside } from "../../hooks/useDismissOnOutside.ts";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useNNQueryStore } from "../../stores/nnQueryStore.ts";
import type { NNAggregation, NNFilter } from "../../types.ts";
import { cn } from "../../utils/helpers.ts";
import { ImageThumb } from "../shared/ImageThumb.tsx";
import { Modal } from "../shared/Modal.tsx";

const FILTER_OPTIONS: { key: NNFilter; label: string }[] = [
  { key: "any", label: "Any" },
  { key: "not-in-group", label: "Not grouped" },
  { key: "in-group", label: "Grouped" },
];

const AGG_OPTIONS: { key: NNAggregation; label: string; title: string }[] = [
  { key: "centroid", label: "Avg", title: "Average distance — uses the query centroid" },
  { key: "min", label: "Best match", title: "Best match — min distance over the query set" },
];

const TOPN_PRESETS = [10, 25, 50, 100, 200];

export function NNResultsModal() {
  const open = useNNQueryStore((s) => s.open);
  const queryLabel = useNNQueryStore((s) => s.queryLabel);
  const filter = useNNQueryStore((s) => s.filter);
  const topN = useNNQueryStore((s) => s.topN);
  const aggregation = useNNQueryStore((s) => s.aggregation);
  const loading = useNNQueryStore((s) => s.loading);
  const progress = useNNQueryStore((s) => s.progress);
  const error = useNNQueryStore((s) => s.error);
  const results = useNNQueryStore((s) => s.results);
  const usedModels = useNNQueryStore((s) => s.usedModels);
  const patchesBlended = useNNQueryStore((s) => s.patchesBlended);
  const modalSelection = useSelectionStore((s) => s.contexts.nn);

  const close = useNNQueryStore((s) => s.close);
  const setFilter = useNNQueryStore((s) => s.setFilter);
  const setTopN = useNNQueryStore((s) => s.setTopN);
  const setAggregation = useNNQueryStore((s) => s.setAggregation);
  const toggleResultSelected = useNNQueryStore((s) => s.toggleResultSelected);
  const rangeSelectResults = useNNQueryStore((s) => s.rangeSelectResults);
  const clearModalSelection = useNNQueryStore((s) => s.clearModalSelection);
  const createClusterFromSelected = useNNQueryStore((s) => s.createClusterFromSelected);
  const addSelectedToGroup = useNNQueryStore((s) => s.addSelectedToGroup);
  const addSelectedToSourceCluster = useNNQueryStore((s) => s.addSelectedToSourceCluster);
  const sourceClusterLabel = useNNQueryStore((s) => s.sourceClusterLabel);

  const lightboxOpen = useLightboxStore((s) => s.open);

  const resultFilenames = useMemo(() => results.map((r) => r.filename), [results]);

  useEffect(() => {
    if (!open || lightboxOpen) return;
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "Escape") {
        e.preventDefault();
        close();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [open, close, lightboxOpen]);

  if (!open) return null;

  const title = (
    <>
      <span className="modal-title-main">Nearest neighbors</span>
      <span className="modal-title-context">— {queryLabel}</span>
      <button
        type="button"
        className="btn btn-icon modal-close-btn"
        onClick={close}
        aria-label="Close"
      >
        ×
      </button>
    </>
  );

  const hasResults = results.length > 0;

  return (
    <Modal
      title={title}
      className="image-picker-modal"
      headerClassName="image-picker-header"
      bodyClassName="image-picker-body"
      onClose={close}
      footer={
        <NNFooter
          selectionCount={modalSelection.size}
          onClose={close}
          onClearSelection={clearModalSelection}
          onCreateCluster={createClusterFromSelected}
          onAddToGroup={addSelectedToGroup}
          sourceClusterLabel={sourceClusterLabel}
          onAddToSourceCluster={addSelectedToSourceCluster}
        />
      }
    >
      <div className="image-picker-toolbar">
        <SegmentedControl
          label="Filter"
          options={FILTER_OPTIONS}
          value={filter}
          onChange={setFilter}
        />
        <SegmentedControl
          label="Aggregate"
          options={AGG_OPTIONS}
          value={aggregation}
          onChange={setAggregation}
        />
        <label className="nn-topn-control">
          <span>Top</span>
          <select value={topN} onChange={(e) => setTopN(parseInt(e.target.value, 10))}>
            {TOPN_PRESETS.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        {patchesBlended && <span className="nn-patches-pill">Patches on</span>}
        {usedModels.length > 0 && (
          <span className="nn-models-pill" title={`Models: ${usedModels.join(", ")}`}>
            {usedModels.length}× model
          </span>
        )}
        {loading && <span className="nn-progress">{progress || "Loading..."}</span>}
        {error && <span className="nn-error">{error}</span>}
      </div>

      {!loading && !error && !hasResults && (
        <div className="image-picker-empty">No matches for the current filter.</div>
      )}
      {hasResults && (
        <div className="image-thumb-grid">
          {results.map((r, i) => (
            <ImageThumb
              key={r.filename}
              filename={r.filename}
              isSelected={modalSelection.has(r.filename)}
              showSelectButton
              onSelect={() => toggleResultSelected(r.filename)}
              onRangeSelect={() => rangeSelectResults(r.filename)}
              lightboxImages={resultFilenames}
              lightboxIndex={i}
              bottomLeft={
                <span className="image-thumb-pill">
                  {r.distance != null ? r.distance.toFixed(3) : "—"}
                </span>
              }
              bottomRight={
                r.inGroupName ? (
                  <span className="image-thumb-pill image-thumb-pill-group" title={r.inGroupName}>
                    {r.inGroupName}
                  </span>
                ) : null
              }
            />
          ))}
        </div>
      )}
    </Modal>
  );
}

function SegmentedControl<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: { key: T; label: string; title?: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="nn-segmented" role="group" aria-label={label}>
      <span className="nn-segmented-label">{label}</span>
      {options.map((opt) => (
        <button
          key={opt.key}
          type="button"
          className={cn("nn-segment", value === opt.key && "nn-segment-active")}
          onClick={() => onChange(opt.key)}
          title={opt.title}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function NNFooter({
  selectionCount,
  onClose,
  onClearSelection,
  onCreateCluster,
  onAddToGroup,
  sourceClusterLabel,
  onAddToSourceCluster,
}: {
  selectionCount: number;
  onClose: () => void;
  onClearSelection: () => void;
  onCreateCluster: () => void;
  onAddToGroup: (groupId: string) => Promise<void>;
  sourceClusterLabel: string | null;
  onAddToSourceCluster: () => Promise<void>;
}) {
  const groups = useGroupStore((s) => s.groups);
  const [groupPickerOpen, setGroupPickerOpen] = useState(false);
  const [query, setQuery] = useState("");
  const pickerRef = useRef<HTMLDivElement>(null);

  useDismissOnOutside(pickerRef, groupPickerOpen, () => setGroupPickerOpen(false));

  const filteredGroups = useMemo(() => {
    if (!query) return groups;
    const lq = query.toLowerCase();
    return groups.filter((g) => g.name.toLowerCase().includes(lq));
  }, [groups, query]);

  const hasSelection = selectionCount > 0;

  async function handlePickGroup(groupId: string) {
    setGroupPickerOpen(false);
    setQuery("");
    await onAddToGroup(groupId);
  }

  return (
    <>
      <span className="modal-footer-status">
        {hasSelection
          ? `${selectionCount} selected`
          : "⌘ click to select • ⇧ click for range • click to zoom"}
      </span>
      {hasSelection && (
        <button type="button" className="btn btn-small" onClick={onClearSelection}>
          Clear
        </button>
      )}
      {sourceClusterLabel && (
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => {
            onAddToSourceCluster();
          }}
          disabled={!hasSelection}
          title={`Add the selected images directly to ${sourceClusterLabel}`}
        >
          Add to "{sourceClusterLabel}"
        </button>
      )}
      <div className="nn-group-picker-wrap" ref={pickerRef}>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => setGroupPickerOpen((v) => !v)}
          disabled={!hasSelection || groups.length === 0}
        >
          Add selected to group…
        </button>
        {groupPickerOpen && (
          <div className="group-picker-dropdown nn-group-picker-dropdown">
            <input
              type="text"
              autoFocus
              placeholder="Search groups…"
              className="group-picker-search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <div className="group-picker-list">
              {filteredGroups.length === 0 ? (
                <div className="group-picker-empty">No matches</div>
              ) : (
                filteredGroups.map((g) => (
                  <button
                    type="button"
                    key={g.id}
                    className="group-picker-item"
                    onClick={() => handlePickGroup(g.id)}
                  >
                    <span className="group-picker-item-name">{g.name}</span>
                    <span className="group-picker-item-count">{g.images.length}</span>
                  </button>
                ))
              )}
            </div>
          </div>
        )}
      </div>
      <button
        type="button"
        className="btn btn-create"
        onClick={onCreateCluster}
        disabled={!hasSelection}
      >
        Create cluster from selected
      </button>
      <button type="button" className="btn" onClick={onClose}>
        Close
      </button>
    </>
  );
}
