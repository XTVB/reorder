import type React from "react";
import type { ImageGroup, ImageInfo } from "../types.ts";
import { ExpandedGroupItem } from "./ExpandedGroupItem.tsx";

interface FloatingPopoverContentProps {
  displayName: string;
  imageCount: number;
  actions: React.ReactNode;
  children: React.ReactNode;
  /** Forwarded to the outer .group-popover div (e.g., for DnD data attributes). */
  rootProps?: React.HTMLAttributes<HTMLDivElement>;
}

/**
 * Generic shell for the group floating popover: header (name + count + actions)
 * and a children slot for the image grid. Reused by the reorder-page group
 * popover and the merge-suggestions portal popover.
 */
export function FloatingPopoverContent({
  displayName,
  imageCount,
  actions,
  children,
  rootProps,
}: FloatingPopoverContentProps) {
  return (
    <div className="group-popover" {...rootProps} onClick={(e) => e.stopPropagation()}>
      <div className="group-popover-header">
        <span className="group-popover-name">{displayName}</span>
        <span className="group-popover-count">
          {imageCount} image{imageCount !== 1 ? "s" : ""}
        </span>
        <div className="group-popover-actions">{actions}</div>
      </div>
      <div className="group-popover-grid">{children}</div>
    </div>
  );
}

interface PopoverShellProps {
  id: string;
  dataAttr: string;
  displayName: string;
  images: string[];
  imageMap: Map<string, ImageInfo>;
  selectedIds: Set<string>;
  isMultiDragging: boolean;
  activeId: string | null;
  actions: React.ReactNode;
  onRemove: (filename: string) => void;
  onCardClick: (id: string, e: React.MouseEvent) => void;
}

export function PopoverShell({
  id,
  dataAttr,
  displayName,
  images,
  imageMap,
  selectedIds,
  isMultiDragging,
  activeId,
  actions,
  onRemove,
  onCardClick,
}: PopoverShellProps) {
  return (
    <FloatingPopoverContent
      displayName={displayName}
      imageCount={images.length}
      actions={actions}
      rootProps={{ [dataAttr]: id } as React.HTMLAttributes<HTMLDivElement>}
    >
      {images.map((fn) => {
        const img = imageMap.get(fn);
        if (!img) return null;
        return (
          <ExpandedGroupItem
            key={fn}
            image={img}
            isSelected={selectedIds.has(fn)}
            isGhost={isMultiDragging && selectedIds.has(fn) && fn !== activeId}
            onRemove={() => onRemove(fn)}
            onCardClick={onCardClick}
          />
        );
      })}
    </FloatingPopoverContent>
  );
}

interface GroupPopoverProps {
  group: ImageGroup;
  imageMap: Map<string, ImageInfo>;
  selectedIds: Set<string>;
  isMultiDragging: boolean;
  activeId: string | null;
  onRename: (groupId: string) => void;
  onDelete: (groupId: string) => void;
  onCollapse: () => void;
  onRemoveFromGroup: (groupId: string, filename: string) => void;
  onCardClick: (id: string, e: React.MouseEvent) => void;
}

export function GroupPopover({
  group,
  imageMap,
  selectedIds,
  isMultiDragging,
  activeId,
  onRename,
  onDelete,
  onCollapse,
  onRemoveFromGroup,
  onCardClick,
}: GroupPopoverProps) {
  return (
    <PopoverShell
      id={group.id}
      dataAttr="data-group-popover"
      displayName={group.name}
      images={group.images}
      imageMap={imageMap}
      selectedIds={selectedIds}
      isMultiDragging={isMultiDragging}
      activeId={activeId}
      actions={
        <>
          <button className="btn btn-small btn-secondary" onClick={() => onRename(group.id)}>
            Rename
          </button>
          <button className="btn btn-small btn-danger" onClick={() => onDelete(group.id)}>
            Dissolve
          </button>
          <button className="btn btn-small btn-secondary" onClick={onCollapse}>
            Close
          </button>
        </>
      }
      onRemove={(fn) => onRemoveFromGroup(group.id, fn)}
      onCardClick={onCardClick}
    />
  );
}
