import type React from "react";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useConstraintsStore } from "../../stores/constraintsStore.ts";
import { useListStore } from "../../stores/modes/cluster/index.ts";
import {
  getSuggestedAdditions,
  type ImageSection,
} from "../../stores/modes/cluster/tree-helpers.ts";
import { useNNQueryStore } from "../../stores/nnQueryStore.ts";
import { useTrashStore } from "../../stores/trashStore.ts";
import type { ClusterMetrics, ClusterResultData } from "../../types.ts";
import { cn } from "../../utils/helpers.ts";
import { AskClaudeButton } from "../shared/AskClaudeButton.tsx";
import { ImageThumb } from "../shared/ImageThumb.tsx";
import { TrashBadge } from "../shared/TrashBadge.tsx";
import { TrashIcon } from "../shared/TrashIcon.tsx";
import { RejectButton } from "./RejectButton.tsx";

interface Props {
  cluster: ClusterResultData;
  collapsed: boolean;
  mergeSelected: boolean;
  selectedImages: Set<string>;
  markedTrashIds: Set<string>;
  isCurrentSearchMatch?: boolean;
  searchMatchFilenames?: Set<string>;
  metrics?: ClusterMetrics;
  splitExpanded?: boolean;
  /** Visual nesting depth from the parent (0 = top-level). */
  depth?: number;
  onToggleCollapse: () => void;
  onMergeSelect: (e: React.MouseEvent) => void;
  onImageSelect: (filename: string, section: ImageSection) => void;
  onImageRangeSelect: (filename: string, section: ImageSection) => void;
  onAccept: () => void;
  onAddToGroup: () => void;
  onDismiss: () => void;
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

export const ClusterCard = memo(function ClusterCard({
  cluster,
  collapsed,
  mergeSelected,
  selectedImages,
  markedTrashIds,
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

  const suggestedImages = useMemo(
    () => getSuggestedAdditions(cluster, cannotLinkIndex),
    [cluster, cannotLinkIndex],
  );

  const isFullyGrouped = hasGroup && suggestedImages.length === 0;

  const allImagesMarked =
    cluster.images.length > 0 && cluster.images.every((f) => markedTrashIds.has(f));

  const statusClass = hasGroup ? "confirmed" : "suggested";

  const cardClass = cn(
    "cluster-card",
    mergeSelected && "merge-selected",
    isFullyGrouped && "fully-grouped",
    isCurrentSearchMatch && "is-search-match",
    depth > 0 && "cluster-card-child",
    splitExpanded && "cluster-card-split-open",
    allImagesMarked && "cluster-card-marked-trash",
  );

  const splitDisabled = cluster.images.length < 3;

  function handleToggleMarkAll() {
    useTrashStore.getState().toggleMany(cluster.images);
  }

  const handleConfirmedSelect = useCallback(
    (f: string) => onImageSelect(f, "confirmed"),
    [onImageSelect],
  );
  const handleConfirmedRange = useCallback(
    (f: string) => onImageRangeSelect(f, "confirmed"),
    [onImageRangeSelect],
  );
  const handleSuggestedSelect = useCallback(
    (f: string) => onImageSelect(f, "suggested"),
    [onImageSelect],
  );
  const handleSuggestedRange = useCallback(
    (f: string) => onImageRangeSelect(f, "suggested"),
    [onImageRangeSelect],
  );
  const rejectSuggested = useMemo(
    () =>
      groupId
        ? (filename: string) => {
            void addCannotLink(filename, groupId);
          }
        : undefined,
    [groupId, addCannotLink],
  );

  function renderThumbs(files: string[], confirmed: boolean) {
    return files.map((f, i) => (
      <ThumbCard
        key={f}
        filename={f}
        files={files}
        index={i}
        isConfirmed={confirmed}
        isSelected={selectedImages.has(`${cluster.id}:${f}`)}
        isSearchMatch={searchMatchFilenames?.has(f) ?? false}
        isMarkedForTrash={markedTrashIds.has(f)}
        onSelect={confirmed ? handleConfirmedSelect : handleSuggestedSelect}
        onRangeSelect={confirmed ? handleConfirmedRange : handleSuggestedRange}
        onReject={confirmed ? undefined : rejectSuggested}
      />
    ));
  }

  return (
    <div
      className={cardClass}
      onClick={onMergeSelect}
      style={depth > 0 ? { marginLeft: `${Math.min(depth, 4) * 24}px` } : undefined}
    >
      {allImagesMarked && <TrashBadge />}
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
            title="Find images similar to this cluster"
          >
            Find Similar
          </button>
          <AskClaudeButton images={cluster.images} name={cluster.autoName || cluster.id} />
          <button
            className="btn btn-small btn-icon"
            onClick={handleToggleMarkAll}
            title={allImagesMarked ? "Unmark cluster" : "Mark cluster for deletion"}
            aria-label={allImagesMarked ? "Unmark cluster" : "Mark cluster for deletion"}
          >
            <TrashIcon size={14} variant={allImagesMarked ? "minus" : "plus"} />
          </button>
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
});

const ThumbCard = memo(function ThumbCard({
  filename,
  files,
  index,
  isConfirmed,
  isSelected,
  isSearchMatch,
  isMarkedForTrash,
  onSelect,
  onRangeSelect,
  onReject,
}: {
  filename: string;
  files: string[];
  index: number;
  isConfirmed: boolean;
  isSelected: boolean;
  isSearchMatch: boolean;
  isMarkedForTrash: boolean;
  onSelect: (f: string) => void;
  onRangeSelect: (f: string) => void;
  onReject?: (filename: string) => void;
}) {
  return (
    <ImageThumb
      filename={filename}
      isSelected={isSelected}
      isSearchMatch={isSearchMatch}
      isMarkedForTrash={isMarkedForTrash}
      variant={isConfirmed ? "confirmed" : "suggested"}
      onSelect={onSelect}
      onRangeSelect={onRangeSelect}
      lightboxImages={files}
      lightboxIndex={index}
      footer={<span className="image-thumb-name">{filename}</span>}
      topRight={onReject ? <RejectButton filename={filename} onReject={onReject} /> : undefined}
    />
  );
});

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
