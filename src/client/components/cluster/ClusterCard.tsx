import type React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useConstraintsStore } from "../../stores/constraintsStore.ts";
import { useListStore } from "../../stores/modes/cluster/index.ts";
import { useNNQueryStore } from "../../stores/nnQueryStore.ts";
import type { ClusterMetrics, ClusterResultData } from "../../types.ts";
import { cn, imageUrl } from "../../utils/helpers.ts";
import { AskClaudeButton } from "../shared/AskClaudeButton.tsx";
import { RejectButton } from "./RejectButton.tsx";

interface Props {
  cluster: ClusterResultData;
  collapsed: boolean;
  mergeSelected: boolean;
  focused: boolean;
  selectedImages: Set<string>;
  isCurrentSearchMatch?: boolean;
  searchMatchFilenames?: Set<string>;
  metrics?: ClusterMetrics;
  splitExpanded?: boolean;
  /** Visual nesting depth from the parent (0 = top-level). */
  depth?: number;
  onToggleCollapse: () => void;
  onMergeSelect: (e: React.MouseEvent) => void;
  onImageSelect: (filename: string) => void;
  onImageRangeSelect: (index: number) => void;
  onAccept: () => void;
  onAddToGroup: () => void;
  onDismiss: () => void;
  onOpenLightbox: (index: number) => void;
  onOpenCompare: () => void;
  onToggleSplit: () => void;
  onOpenExpand: () => void;
}

function fmtDistance(v: number | undefined): string {
  if (v === undefined) return "—";
  if (!Number.isFinite(v) || v < 0) return "∞";
  if (v === 0) return "0";
  return v < 0.01 ? v.toFixed(3) : v.toFixed(2);
}

function fmtStability(v: number | undefined): string {
  if (v === undefined || !Number.isFinite(v) || v <= 0) return "—";
  return v.toFixed(2);
}

export function ClusterCard({
  cluster,
  collapsed,
  mergeSelected,
  focused,
  selectedImages,
  isCurrentSearchMatch,
  searchMatchFilenames,
  metrics,
  splitExpanded,
  depth = 0,
  onToggleCollapse,
  onMergeSelect,
  onImageSelect,
  onImageRangeSelect,
  onAccept,
  onAddToGroup,
  onDismiss,
  onOpenLightbox,
  onOpenCompare,
  onToggleSplit,
  onOpenExpand,
}: Props) {
  const hasGroup = !!cluster.confirmedGroup;
  const groupId = cluster.confirmedGroup?.id;

  const cannotLinkIndex = useConstraintsStore((s) => s.index);
  const lockedGroupIds = useConstraintsStore((s) => s.lockedGroupIds);
  const addCannotLink = useConstraintsStore((s) => s.addImageGroupCannotLink);
  const toggleGroupLock = useConstraintsStore((s) => s.toggleGroupLock);
  const isLocked = !!groupId && lockedGroupIds.has(groupId);

  const confirmedSet = useMemo(
    () => (hasGroup ? new Set(cluster.confirmedGroup!.images) : new Set<string>()),
    [hasGroup, cluster.confirmedGroup?.images],
  );

  const rawSuggestedImages = useMemo(
    () => cluster.images.filter((f) => !confirmedSet.has(f)),
    [cluster.images, confirmedSet],
  );

  const suggestedImages = useMemo(() => {
    if (!groupId) return rawSuggestedImages;
    return rawSuggestedImages.filter((f) => !cannotLinkIndex.get(f)?.has(groupId));
  }, [rawSuggestedImages, groupId, cannotLinkIndex]);

  const imageIndex = useMemo(() => new Map(cluster.images.map((f, i) => [f, i])), [cluster.images]);

  const isFullyGrouped = hasGroup && suggestedImages.length === 0;

  const statusClass = hasGroup ? "confirmed" : "suggested";

  const cardClass = cn(
    "cluster-card",
    mergeSelected && "merge-selected",
    focused && "focused",
    isFullyGrouped && "fully-grouped",
    isCurrentSearchMatch && "is-search-match",
    depth > 0 && "cluster-card-child",
    splitExpanded && "cluster-card-split-open",
  );

  const splitDisabled = cluster.images.length < 3;

  function renderThumbs(files: string[], confirmed: boolean) {
    const rejectFn =
      !confirmed && groupId
        ? (filename: string) => {
            void addCannotLink(filename, groupId);
          }
        : undefined;
    return files.map((f) => (
      <ThumbCard
        key={f}
        filename={f}
        index={imageIndex.get(f)!}
        isConfirmed={confirmed}
        isSelected={selectedImages.has(`${cluster.id}:${f}`)}
        isSearchMatch={searchMatchFilenames?.has(f) ?? false}
        onSelect={onImageSelect}
        onRangeSelect={onImageRangeSelect}
        onOpenLightbox={onOpenLightbox}
        onReject={rejectFn}
      />
    ));
  }

  return (
    <div
      className={cardClass}
      onClick={onMergeSelect}
      style={depth > 0 ? { marginLeft: `${Math.min(depth, 4) * 24}px` } : undefined}
    >
      <div
        className="cluster-header"
        onClick={(e) => {
          e.stopPropagation();
          onToggleCollapse();
        }}
      >
        <span
          className={`cluster-chevron ${collapsed ? "" : "cluster-chevron-open"}`}
          aria-hidden
        />
        <span className={`cluster-status-dot ${statusClass}`} />

        {isFullyGrouped && <span className="cluster-check">✓</span>}

        <EditableName
          name={hasGroup ? cluster.confirmedGroup!.name : cluster.autoName}
          clusterId={cluster.id}
          editable={!hasGroup}
        />

        <span className="cluster-count">{cluster.images.length} images</span>

        <div className="cluster-metrics" onClick={(e) => e.stopPropagation()}>
          <span className="cluster-metric" title="Cohesion — max intra-pair distance">
            c:<span className="cluster-metric-val">{fmtDistance(metrics?.cohesion)}</span>
          </span>
          <span className="cluster-metric-sep">·</span>
          <span
            className="cluster-metric"
            title="Isolation — distance at which it would merge into its parent"
          >
            i:
            <span className="cluster-metric-val">{fmtDistance(metrics?.isolation)}</span>
          </span>
          <span className="cluster-metric-sep">·</span>
          <span className="cluster-metric" title="Stability — (death − birth) / death">
            s:<span className="cluster-metric-val">{fmtStability(metrics?.stability)}</span>
          </span>
        </div>

        {!collapsed &&
          cluster.autoTags.slice(0, 4).map((t) => (
            <span key={t.term} className="cluster-tag" title={`z=${t.z.toFixed(1)}`}>
              {t.term}
            </span>
          ))}

        <div className="cluster-actions" onClick={(e) => e.stopPropagation()}>
          {!hasGroup && (
            <button className="btn btn-small btn-create" onClick={onAccept}>
              Create Group
            </button>
          )}
          {hasGroup && !isLocked && suggestedImages.length > 0 && (
            <button className="btn btn-small btn-add" onClick={onAddToGroup}>
              Add {suggestedImages.length} to Group
            </button>
          )}
          <button
            className="btn btn-small btn-tree-nav"
            onClick={onOpenCompare}
            title="Open compare-mode to merge with similar clusters"
          >
            merge…
          </button>
          <button
            className="btn btn-small btn-tree-nav"
            onClick={onToggleSplit}
            disabled={splitDisabled}
            title={
              splitDisabled
                ? "Need at least 3 images to split"
                : splitExpanded
                  ? "Collapse split"
                  : "Split into two children"
            }
          >
            {splitExpanded ? "collapse" : "split"}
          </button>
          {!isLocked && (
            <button
              className="btn btn-small btn-tree-nav"
              onClick={onOpenExpand}
              title="Pull in nearby images from outside this cluster"
            >
              expand…
            </button>
          )}
          <button
            className="btn btn-small"
            onClick={() => useNNQueryStore.getState().openForCluster(cluster)}
            title="Find images similar to this cluster (f)"
          >
            Find Similar
          </button>
          <AskClaudeButton images={cluster.images} name={cluster.autoName || cluster.id} />
          {hasGroup && groupId && (
            <button
              className={cn("btn btn-small btn-lock", isLocked && "btn-lock-active")}
              onClick={() => {
                void toggleGroupLock(groupId);
              }}
              title={
                isLocked
                  ? "Unlock — allow new suggestions on next re-cluster"
                  : "Lock — never suggest new additions for this group"
              }
            >
              {isLocked ? "🔒" : "🔓"}
            </button>
          )}
          <button className="btn btn-small btn-dismiss" onClick={onDismiss}>
            ×
          </button>
        </div>
      </div>

      {!collapsed && (
        <div className="cluster-body">
          {hasGroup && (
            <div className="cluster-section cluster-section-confirmed">
              <div className="cluster-section-label">
                Group: {cluster.confirmedGroup!.name} ({cluster.confirmedGroup!.images.length})
              </div>
              <div className="cluster-thumbs">
                {renderThumbs(cluster.confirmedGroup!.images, true)}
              </div>
            </div>
          )}

          {!isLocked && (suggestedImages.length > 0 || !hasGroup) && (
            <div className="cluster-section cluster-section-suggested">
              {hasGroup && (
                <div className="cluster-section-label">
                  Suggested additions ({suggestedImages.length})
                </div>
              )}
              <div className="cluster-thumbs">
                {renderThumbs(hasGroup ? suggestedImages : cluster.images, false)}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ThumbCard({
  filename,
  index,
  isConfirmed,
  isSelected,
  isSearchMatch,
  onSelect,
  onRangeSelect,
  onOpenLightbox,
  onReject,
}: {
  filename: string;
  index: number;
  isConfirmed: boolean;
  isSelected: boolean;
  isSearchMatch: boolean;
  onSelect: (f: string) => void;
  onRangeSelect: (i: number) => void;
  onOpenLightbox: (i: number) => void;
  onReject?: (filename: string) => void;
}) {
  const thumbClass = cn(
    "cluster-thumb",
    isConfirmed ? "confirmed" : "suggested",
    isSelected && "selected",
    isSearchMatch && "search-match",
  );

  return (
    <div
      className={thumbClass}
      onClick={(e) => {
        e.stopPropagation();
        if (e.shiftKey) {
          onRangeSelect(index);
        } else if (e.metaKey || e.ctrlKey) {
          onSelect(filename);
        } else {
          onOpenLightbox(index);
        }
      }}
    >
      <img
        src={imageUrl(filename)}
        loading="lazy"
        decoding="async"
        alt={filename}
        draggable={false}
      />
      <span className="cluster-thumb-name">{filename}</span>
      {onReject && <RejectButton filename={filename} onReject={onReject} />}
    </div>
  );
}

function EditableName({
  name,
  clusterId,
  editable,
}: {
  name: string;
  clusterId: string;
  editable: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(name);
  const inputRef = useRef<HTMLInputElement>(null);
  const renameCluster = useListStore((s) => s.renameCluster);

  useEffect(() => {
    setValue(name);
  }, [name]);
  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  if (!editable || !editing) {
    return (
      <span
        className={`cluster-name ${editable ? "editable" : ""}`}
        onClick={(e) => {
          if (editable) {
            e.stopPropagation();
            setEditing(true);
          }
        }}
        title={editable ? "Click to rename" : undefined}
      >
        {name}
      </span>
    );
  }

  return (
    <input
      ref={inputRef}
      className="cluster-name-input"
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onBlur={() => {
        renameCluster(clusterId, value);
        setEditing(false);
      }}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === "Enter") {
          renameCluster(clusterId, value);
          setEditing(false);
        }
        if (e.key === "Escape") {
          setValue(name);
          setEditing(false);
        }
      }}
      onClick={(e) => e.stopPropagation()}
    />
  );
}
