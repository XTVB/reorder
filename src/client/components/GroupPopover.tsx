import React from "react";
import type { ImageInfo, ImageGroup } from "../types.ts";
import { ExpandedGroupItem } from "./ExpandedGroupItem.tsx";

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
  const dataProps = { [dataAttr]: id };
  return (
    <div className="group-popover" {...dataProps} onClick={(e) => e.stopPropagation()}>
      <div className="group-popover-header">
        <span className="group-popover-name">{displayName}</span>
        <span className="group-popover-count">{images.length} image{images.length !== 1 ? "s" : ""}</span>
        <div className="group-popover-actions">{actions}</div>
      </div>
      <div className="group-popover-grid">
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
      </div>
    </div>
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
          <button className="btn btn-small btn-secondary" onClick={() => onRename(group.id)}>Rename</button>
          <button className="btn btn-small btn-danger" onClick={() => onDelete(group.id)}>Dissolve</button>
          <button className="btn btn-small btn-secondary" onClick={onCollapse}>Close</button>
        </>
      }
      onRemove={(fn) => onRemoveFromGroup(group.id, fn)}
      onCardClick={onCardClick}
    />
  );
}
