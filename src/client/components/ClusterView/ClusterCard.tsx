import type React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useClusterStore } from "../../stores/clusterStore.ts";
import type { ClusterResultData } from "../../types.ts";
import { cn, imageUrl } from "../../utils/helpers.ts";
import { AskClaudeButton } from "../AskClaudeButton.tsx";

interface Props {
  cluster: ClusterResultData;
  collapsed: boolean;
  mergeSelected: boolean;
  focused: boolean;
  selectedImages: Set<string>;
  onToggleCollapse: () => void;
  onMergeSelect: (e: React.MouseEvent) => void;
  onImageSelect: (filename: string) => void;
  onImageRangeSelect: (index: number) => void;
  onAccept: () => void;
  onAddToGroup: () => void;
  onDismiss: () => void;
  onOpenLightbox: (index: number) => void;
}

export function ClusterCard({
  cluster,
  collapsed,
  mergeSelected,
  focused,
  selectedImages,
  onToggleCollapse,
  onMergeSelect,
  onImageSelect,
  onImageRangeSelect,
  onAccept,
  onAddToGroup,
  onDismiss,
  onOpenLightbox,
}: Props) {
  const hasGroup = !!cluster.confirmedGroup;

  const confirmedSet = useMemo(
    () => (hasGroup ? new Set(cluster.confirmedGroup!.images) : new Set<string>()),
    [hasGroup, cluster.confirmedGroup?.images],
  );

  const suggestedImages = useMemo(
    () => cluster.images.filter((f) => !confirmedSet.has(f)),
    [cluster.images, confirmedSet],
  );

  const imageIndex = useMemo(() => new Map(cluster.images.map((f, i) => [f, i])), [cluster.images]);

  const isFullyGrouped = hasGroup && suggestedImages.length === 0;

  const statusClass = hasGroup ? "confirmed" : "suggested";

  const cardClass = cn(
    "cluster-card",
    mergeSelected && "merge-selected",
    focused && "focused",
    isFullyGrouped && "fully-grouped",
  );

  function renderThumbs(files: string[], confirmed: boolean) {
    return files.map((f) => (
      <ThumbCard
        key={f}
        filename={f}
        index={imageIndex.get(f)!}
        isConfirmed={confirmed}
        isSelected={selectedImages.has(`${cluster.id}:${f}`)}
        onSelect={onImageSelect}
        onRangeSelect={onImageRangeSelect}
        onOpenLightbox={onOpenLightbox}
      />
    ));
  }

  return (
    <div className={cardClass} onClick={onMergeSelect}>
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
          {hasGroup && suggestedImages.length > 0 && (
            <button className="btn btn-small btn-add" onClick={onAddToGroup}>
              Add {suggestedImages.length} to Group
            </button>
          )}
          <AskClaudeButton images={cluster.images} name={cluster.autoName || cluster.id} />
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

          {(suggestedImages.length > 0 || !hasGroup) && (
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
  onSelect,
  onRangeSelect,
  onOpenLightbox,
}: {
  filename: string;
  index: number;
  isConfirmed: boolean;
  isSelected: boolean;
  onSelect: (f: string) => void;
  onRangeSelect: (i: number) => void;
  onOpenLightbox: (i: number) => void;
}) {
  const thumbClass = cn(
    "cluster-thumb",
    isConfirmed ? "confirmed" : "suggested",
    isSelected && "selected",
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
  const renameCluster = useClusterStore((s) => s.renameCluster);

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
