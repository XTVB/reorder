import { useEffect, useMemo } from "react";
import { useConstraintsStore } from "../../stores/constraintsStore.ts";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import {
  findClusterEverywhere,
  useExpandStore,
  useListStore,
} from "../../stores/modes/cluster/index.ts";
import type { ClusterResultData } from "../../types.ts";
import { ImageThumb } from "../shared/ImageThumb.tsx";
import { Modal } from "../shared/Modal.tsx";
import { RejectButton } from "./RejectButton.tsx";

const SLIDER_MIN = 0.5;
const SLIDER_MAX = 4;
const RENDER_CAP = 400;

function logToMultiplier(value: number): number {
  const t = Math.min(1, Math.max(0, value));
  const lo = Math.log(SLIDER_MIN);
  const hi = Math.log(SLIDER_MAX);
  return Math.exp(lo + (hi - lo) * t);
}

function multiplierToLog(mult: number): number {
  const m = Math.min(SLIDER_MAX, Math.max(SLIDER_MIN, mult));
  const lo = Math.log(SLIDER_MIN);
  const hi = Math.log(SLIDER_MAX);
  return (Math.log(m) - lo) / (hi - lo);
}

export function ExpandModal() {
  const expand = useExpandStore((s) => s.expand);
  const clusterData = useListStore((s) => s.clusterData);
  const splitChildren = useListStore((s) => s.splitChildren);
  const checked = useSelectionStore((s) => s.contexts.expand);
  const groups = useGroupStore((s) => s.groups);
  const closeExpand = useExpandStore((s) => s.closeExpand);
  const toggleExpandFile = useExpandStore((s) => s.toggleExpandFile);
  const rangeSelectExpandFile = useExpandStore((s) => s.rangeSelectExpandFile);
  const setThreshold = useExpandStore((s) => s.setExpandThreshold);
  const setIncludeConfirmedGroups = useExpandStore((s) => s.setExpandIncludeConfirmedGroups);
  const confirmExpand = useExpandStore((s) => s.confirmExpand);

  const lightboxOpen = useLightboxStore((s) => s.open);

  const cannotLinkIndex = useConstraintsStore((s) => s.index);
  const lockedGroupIds = useConstraintsStore((s) => s.lockedGroupIds);
  const addCannotLink = useConstraintsStore((s) => s.addImageGroupCannotLink);

  // Build a filename → current cluster map. We look at top-level + split
  // children too so the label reflects the user's current view, not just
  // confirmed groups.
  const fileCluster = useMemo(() => {
    const map = new Map<string, { id: string; name: string; isConfirmedGroup: boolean }>();
    if (!clusterData) return map;
    function add(c: ClusterResultData) {
      const isConf = !!c.confirmedGroup;
      const name = c.confirmedGroup?.name ?? c.autoName;
      for (const f of c.images) {
        if (!map.has(f)) map.set(f, { id: c.id, name, isConfirmedGroup: isConf });
      }
    }
    for (const c of clusterData.clusters) add(c);
    for (const k of Object.values(splitChildren)) {
      add(k.childA);
      add(k.childB);
    }
    for (const g of groups) {
      for (const f of g.images) {
        if (!map.has(f)) map.set(f, { id: g.id, name: g.name, isConfirmedGroup: true });
      }
    }
    return map;
  }, [clusterData, splitChildren, groups]);

  const source = useMemo(
    () =>
      expand && clusterData
        ? findClusterEverywhere(clusterData.clusters, splitChildren, expand.sourceClusterId)
        : null,
    [expand, clusterData, splitChildren],
  );

  const allFiltered = useMemo(() => {
    if (!expand || !source) return [];
    const sourceGroupId = source.confirmedGroup?.id;
    const ref = expand.p90Intra > 0 ? expand.p90Intra : 0.1;
    const threshold = ref * expand.thresholdMultiplier;
    return expand.candidates.filter((c) => {
      if (c.distance > threshold) return false;
      if (!expand.includeConfirmedGroups) {
        const cur = fileCluster.get(c.filename);
        if (cur?.isConfirmedGroup) return false;
      }
      if (sourceGroupId && cannotLinkIndex.get(c.filename)?.has(sourceGroupId)) return false;
      return true;
    });
  }, [expand, source, fileCluster, cannotLinkIndex]);

  const allFilenames = useMemo(() => allFiltered.map((c) => c.filename), [allFiltered]);

  useEffect(() => {
    if (!expand || lightboxOpen) return;
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        if (e.key === "Escape") closeExpand();
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        closeExpand();
      } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        confirmExpand();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [expand, closeExpand, confirmExpand, lightboxOpen]);

  if (!expand || !clusterData || !source) return null;

  // ClusterCard hides "expand…" for locked groups, but the modal may already
  // be open from before the lock — close it.
  const sourceGroupId = source.confirmedGroup?.id;
  if (sourceGroupId && lockedGroupIds.has(sourceGroupId)) {
    closeExpand();
    return null;
  }

  const onReject = sourceGroupId
    ? (filename: string) => {
        void addCannotLink(filename, sourceGroupId);
      }
    : undefined;

  const totalChecked = checked.size;
  const sliderValue = multiplierToLog(expand.thresholdMultiplier);
  const sourceName = source.confirmedGroup?.name ?? source.autoName;

  const title = (
    <>
      <span className="modal-title-main">Expand</span>
      <span className="modal-title-context">— {sourceName}</span>
      <span className="modal-title-meta">{source.images.length} images</span>
      <button
        type="button"
        className="btn btn-icon modal-close-btn"
        onClick={closeExpand}
        aria-label="Close"
      >
        ×
      </button>
    </>
  );

  const footer = (
    <>
      <span className="modal-footer-status">
        {totalChecked > 0
          ? `${totalChecked} selected`
          : "⌘ click to select • ⇧ click for range • click to zoom"}
        {" · "}showing {allFiltered.length} of {expand.candidates.length}
        {expand.p90Intra > 0 && (
          <>
            {" · "}1× = {expand.p90Intra.toFixed(3)}
          </>
        )}
      </span>
      <button className="btn" onClick={closeExpand}>
        Cancel
      </button>
      <button
        type="button"
        className="btn btn-primary"
        onClick={confirmExpand}
        disabled={totalChecked === 0}
      >
        Add to "{sourceName}" ({totalChecked})
      </button>
    </>
  );

  const sliced = allFiltered.length > RENDER_CAP ? allFiltered.slice(0, RENDER_CAP) : allFiltered;

  return (
    <Modal
      title={title}
      onClose={closeExpand}
      footer={footer}
      className="image-picker-modal"
      headerClassName="image-picker-header"
      bodyClassName="image-picker-body"
    >
      <div className="image-picker-toolbar image-picker-toolbar-stack">
        <div className="expand-slider-row">
          <label className="expand-slider-label">Density threshold</label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={sliderValue}
            onChange={(e) => setThreshold(logToMultiplier(parseFloat(e.target.value)))}
            className="expand-slider"
          />
          <span className="expand-slider-value">{expand.thresholdMultiplier.toFixed(2)}× p90</span>
        </div>
        <div className="expand-slider-tickrow">
          <span>0.5×</span>
          <span className="expand-slider-tick-strict">stricter than self</span>
          <span>1×</span>
          <span className="expand-slider-tick-loose">pulls in outliers →</span>
          <span>4×</span>
        </div>
        <label className="expand-toggle">
          <input
            type="checkbox"
            checked={expand.includeConfirmedGroups}
            onChange={(e) => setIncludeConfirmedGroups(e.target.checked)}
          />
          <span>Include images currently in confirmed groups</span>
        </label>
      </div>

      {expand.loading ? (
        <div className="image-picker-empty">Computing distances…</div>
      ) : allFiltered.length === 0 ? (
        <div className="image-picker-empty">
          No images within threshold. Try a higher slider value.
        </div>
      ) : (
        <div className="image-thumb-grid">
          {sliced.map((c, i) => {
            const cur = fileCluster.get(c.filename);
            return (
              <ImageThumb
                key={c.filename}
                filename={c.filename}
                isSelected={checked.has(c.filename)}
                showSelectButton
                onSelect={() => toggleExpandFile(c.filename)}
                onRangeSelect={() => rangeSelectExpandFile(c.filename, allFilenames)}
                lightboxImages={allFilenames}
                lightboxIndex={i}
                topRight={
                  onReject ? <RejectButton filename={c.filename} onReject={onReject} /> : undefined
                }
                bottomLeft={<span className="image-thumb-pill">{c.distance.toFixed(3)}</span>}
                bottomRight={
                  cur ? (
                    <span className="image-thumb-pill image-thumb-pill-group" title={cur.name}>
                      {cur.isConfirmedGroup ? `🔒 ${cur.name}` : cur.name}
                    </span>
                  ) : undefined
                }
              />
            );
          })}
          {allFiltered.length > RENDER_CAP && (
            <div className="image-thumb-grid-more">
              +{allFiltered.length - RENDER_CAP} more not shown
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}
