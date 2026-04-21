import { useEffect, useMemo, useRef, useState } from "react";
import { useDismissOnOutside } from "../../hooks/useDismissOnOutside.ts";
import { useClusterStore } from "../../stores/clusterStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useNNQueryStore } from "../../stores/nnQueryStore.ts";
import type { ImageInfo, NNAggregation, NNFilter } from "../../types.ts";
import { cn, imageUrl } from "../../utils/helpers.ts";
import { Lightbox } from "../Lightbox.tsx";
import { Modal } from "../Modal.tsx";

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
  const modalSelection = useNNQueryStore((s) => s.modalSelection);

  const close = useNNQueryStore((s) => s.close);
  const setFilter = useNNQueryStore((s) => s.setFilter);
  const setTopN = useNNQueryStore((s) => s.setTopN);
  const setAggregation = useNNQueryStore((s) => s.setAggregation);
  const toggleResultSelected = useNNQueryStore((s) => s.toggleResultSelected);
  const clearModalSelection = useNNQueryStore((s) => s.clearModalSelection);
  const createClusterFromSelected = useNNQueryStore((s) => s.createClusterFromSelected);
  const addSelectedToGroup = useNNQueryStore((s) => s.addSelectedToGroup);

  const inScope = useClusterStore((s) => !!s.clusterData?.scope);

  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  // Index may become invalid when `results` shrinks (e.g. filter toggle).
  useEffect(() => {
    setLightboxIndex((i) => (i == null ? i : i < results.length ? i : null));
  }, [results.length]);

  useEffect(() => {
    if (!open || lightboxIndex != null) return;
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "Escape") {
        e.preventDefault();
        close();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [open, close, lightboxIndex]);

  const lightboxImages: ImageInfo[] = useMemo(
    () => results.map((r) => ({ filename: r.filename })),
    [results],
  );

  if (!open) return null;

  const title = (
    <>
      <span className="modal-title-main">Nearest neighbors</span>
      <span className="nn-query-label">— {queryLabel}</span>
      {inScope && (
        <span className="nn-scope-pill" title="Restricted to current scope">
          In scope
        </span>
      )}
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
    <>
      <Modal
        title={title}
        className="nn-results-modal"
        headerClassName="nn-results-header"
        bodyClassName="nn-results-body"
        onClose={close}
        footer={
          <NNFooter
            selectionCount={modalSelection.size}
            onClose={close}
            onClearSelection={clearModalSelection}
            onCreateCluster={createClusterFromSelected}
            onAddToGroup={addSelectedToGroup}
          />
        }
      >
        <div className="nn-toolbar">
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
          <div className="nn-empty">No matches for the current filter.</div>
        )}
        {hasResults && (
          <div className="nn-results-grid">
            {results.map((r, i) => (
              <NNCard
                key={r.filename}
                filename={r.filename}
                distance={r.distance}
                inGroupName={r.inGroupName}
                selected={modalSelection.has(r.filename)}
                onToggleSelect={() => toggleResultSelected(r.filename)}
                onOpenLightbox={() => setLightboxIndex(i)}
              />
            ))}
          </div>
        )}
      </Modal>
      {lightboxIndex != null && (
        <Lightbox
          images={lightboxImages}
          initialIndex={lightboxIndex}
          onClose={() => setLightboxIndex(null)}
        />
      )}
    </>
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

function NNCard({
  filename,
  distance,
  inGroupName,
  selected,
  onToggleSelect,
  onOpenLightbox,
}: {
  filename: string;
  distance: number;
  inGroupName: string | null;
  selected: boolean;
  onToggleSelect: () => void;
  onOpenLightbox: () => void;
}) {
  return (
    <div className={cn("nn-card", selected && "nn-card-selected")}>
      <button
        type="button"
        className="nn-card-select"
        onClick={(e) => {
          e.stopPropagation();
          onToggleSelect();
        }}
        aria-label={selected ? "Deselect" : "Select"}
        title={selected ? "Deselect" : "Select"}
      >
        <span className="nn-card-check" aria-hidden>
          {selected ? "✓" : ""}
        </span>
      </button>
      <button
        type="button"
        className="nn-card-image"
        onClick={onOpenLightbox}
        title={`Open ${filename}`}
        aria-label={`Open ${filename}`}
      >
        <img src={imageUrl(filename)} alt="" loading="lazy" decoding="async" draggable={false} />
        <span className="nn-card-dist">{distance.toFixed(3)}</span>
      </button>
      <div className="nn-card-meta">
        <span className="nn-card-name" title={filename}>
          {filename}
        </span>
        {inGroupName && (
          <span className="nn-card-group" title={`In group: ${inGroupName}`}>
            {inGroupName}
          </span>
        )}
      </div>
    </div>
  );
}

function NNFooter({
  selectionCount,
  onClose,
  onClearSelection,
  onCreateCluster,
  onAddToGroup,
}: {
  selectionCount: number;
  onClose: () => void;
  onClearSelection: () => void;
  onCreateCluster: () => void;
  onAddToGroup: (groupId: string) => Promise<void>;
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
      <span className="nn-footer-status">
        {hasSelection ? `${selectionCount} selected` : "Click ✓ to select, click image to zoom"}
      </span>
      {hasSelection && (
        <button type="button" className="btn btn-small" onClick={onClearSelection}>
          Clear
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
