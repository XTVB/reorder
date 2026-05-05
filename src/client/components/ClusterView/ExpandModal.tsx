import { useEffect, useMemo } from "react";
import { findClusterEverywhere, useClusterStore } from "../../stores/clusterStore.ts";
import { useConstraintsStore } from "../../stores/constraintsStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import type { ClusterResultData, ExpandCandidate } from "../../types.ts";
import { cn, imageUrl } from "../../utils/helpers.ts";
import { Modal } from "../Modal.tsx";
import { RejectButton } from "./RejectButton.tsx";

const SLIDER_MIN = 0.5;
const SLIDER_MAX = 4;

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
  const expand = useClusterStore((s) => s.expand);
  const clusterData = useClusterStore((s) => s.clusterData);
  const splitChildren = useClusterStore((s) => s.splitChildren);
  const groups = useGroupStore((s) => s.groups);
  const closeExpand = useClusterStore((s) => s.closeExpand);
  const toggleExpandFile = useClusterStore((s) => s.toggleExpandFile);
  const setThreshold = useClusterStore((s) => s.setExpandThreshold);
  const setIncludeConfirmedGroups = useClusterStore((s) => s.setExpandIncludeConfirmedGroups);
  const confirmExpand = useClusterStore((s) => s.confirmExpand);

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
    // Also fold in confirmed group memberships from the group store (fallback for
    // images not currently visible in any cluster).
    for (const g of groups) {
      for (const f of g.images) {
        if (!map.has(f)) map.set(f, { id: g.id, name: g.name, isConfirmedGroup: true });
      }
    }
    return map;
  }, [clusterData, splitChildren, groups]);

  useEffect(() => {
    if (!expand) return;
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
  }, [expand, closeExpand, confirmExpand]);

  if (!expand || !clusterData) return null;

  const source = findClusterEverywhere(clusterData.clusters, splitChildren, expand.sourceClusterId);
  if (!source) return null;

  // ClusterCard hides "expand…" for locked groups, but the modal may already
  // be open from before the lock — close it.
  const sourceGroupId = source.confirmedGroup?.id;
  if (sourceGroupId && lockedGroupIds.has(sourceGroupId)) {
    closeExpand();
    return null;
  }

  const ref = expand.p90Intra > 0 ? expand.p90Intra : 0.1;
  const threshold = ref * expand.thresholdMultiplier;

  const allFiltered = expand.candidates.filter((c) => {
    if (c.distance > threshold) return false;
    if (!expand.includeConfirmedGroups) {
      const cur = fileCluster.get(c.filename);
      if (cur?.isConfirmedGroup) return false;
    }
    if (sourceGroupId && cannotLinkIndex.get(c.filename)?.has(sourceGroupId)) return false;
    return true;
  });

  const onReject = sourceGroupId
    ? (filename: string) => {
        void addCannotLink(filename, sourceGroupId);
      }
    : undefined;

  const totalChecked = expand.checked.size;
  const sliderValue = multiplierToLog(expand.thresholdMultiplier);

  // Per spec: the commit button lives at the top of the modal. We render it
  // inside the title row so the user always sees the action without scrolling.
  const title = (
    <div className="expand-modal-title-row">
      <div className="expand-modal-title">
        <div>
          Expand{" "}
          <span className="expand-modal-source">
            {source.confirmedGroup?.name ?? source.autoName}
          </span>
        </div>
        <div className="expand-modal-subtitle">{source.images.length} images currently</div>
      </div>
      <button
        className="btn btn-primary expand-modal-commit"
        onClick={confirmExpand}
        disabled={totalChecked === 0}
      >
        add selected ({totalChecked})
      </button>
    </div>
  );

  const footer = (
    <>
      <div className="expand-footer-status">
        Showing {allFiltered.length} of {expand.candidates.length} candidates
        {expand.p90Intra > 0 && (
          <span>
            {" · "}1× threshold = {expand.p90Intra.toFixed(3)}
          </span>
        )}
      </div>
      <button className="btn" onClick={closeExpand}>
        cancel
      </button>
    </>
  );

  return (
    <Modal
      title={title}
      onClose={closeExpand}
      footer={footer}
      className="expand-modal"
      bodyClassName="expand-modal-body"
    >
      <div className="expand-controls">
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
        <div className="expand-empty">Computing distances…</div>
      ) : allFiltered.length === 0 ? (
        <div className="expand-empty">No images within threshold. Try a higher slider value.</div>
      ) : (
        <ExpandGrid
          filtered={allFiltered}
          checked={expand.checked}
          fileCluster={fileCluster}
          onToggle={toggleExpandFile}
          onReject={onReject}
        />
      )}
    </Modal>
  );
}

function ExpandGrid({
  filtered,
  checked,
  fileCluster,
  onToggle,
  onReject,
}: {
  filtered: ExpandCandidate[];
  checked: Set<string>;
  fileCluster: Map<string, { id: string; name: string; isConfirmedGroup: boolean }>;
  onToggle: (filename: string) => void;
  onReject?: (filename: string) => void;
}) {
  // Cap visible candidates for perf — sliders should remain responsive.
  const RENDER_CAP = 400;
  const sliced = filtered.length > RENDER_CAP ? filtered.slice(0, RENDER_CAP) : filtered;

  return (
    <div className="expand-grid">
      {sliced.map((c) => {
        const cur = fileCluster.get(c.filename);
        const isChecked = checked.has(c.filename);
        return (
          <div
            key={c.filename}
            className={cn("expand-cell", isChecked && "expand-cell-checked")}
            onClick={() => onToggle(c.filename)}
          >
            <input
              type="checkbox"
              className="expand-cell-check"
              checked={isChecked}
              onChange={() => onToggle(c.filename)}
              onClick={(e) => e.stopPropagation()}
            />
            {onReject && <RejectButton filename={c.filename} onReject={onReject} />}
            <img
              src={imageUrl(c.filename)}
              loading="lazy"
              decoding="async"
              alt={c.filename}
              className="expand-cell-img"
            />
            <div className="expand-cell-meta">
              <div className="expand-cell-dist">{c.distance.toFixed(3)}</div>
              <div className="expand-cell-cluster" title={cur?.name ?? ""}>
                {cur ? (cur.isConfirmedGroup ? `🔒 ${cur.name}` : cur.name) : "—"}
              </div>
            </div>
          </div>
        );
      })}
      {filtered.length > RENDER_CAP && (
        <div className="expand-cell-more">+{filtered.length - RENDER_CAP} more not shown</div>
      )}
    </div>
  );
}
