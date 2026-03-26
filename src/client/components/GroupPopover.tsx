import React from "react";
import type { ImageInfo, ImageGroup } from "../types.ts";
import { ExpandedGroupItem } from "./ExpandedGroupItem.tsx";

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
    <div className="group-popover" data-group-popover={group.id} onClick={(e) => e.stopPropagation()}>
      <div className="group-popover-header">
        <span className="group-popover-name">{group.name}</span>
        <span className="group-popover-count">{group.images.length} image{group.images.length !== 1 ? "s" : ""}</span>
        <div className="group-popover-actions">
          <button className="btn btn-small btn-secondary" onClick={() => onRename(group.id)}>Rename</button>
          <button className="btn btn-small btn-danger" onClick={() => onDelete(group.id)}>Dissolve</button>
          <button className="btn btn-small btn-secondary" onClick={onCollapse}>Close</button>
        </div>
      </div>
      <div className="group-popover-grid">
        {group.images.map((fn) => {
          const img = imageMap.get(fn);
          if (!img) return null;
          return (
            <ExpandedGroupItem
              key={fn}
              image={img}
              isSelected={selectedIds.has(fn)}
              isGhost={isMultiDragging && selectedIds.has(fn) && fn !== activeId}
              onRemove={() => onRemoveFromGroup(group.id, fn)}
              onCardClick={onCardClick}
            />
          );
        })}
      </div>
    </div>
  );
}
