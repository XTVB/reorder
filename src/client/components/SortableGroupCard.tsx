import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import type React from "react";
import { memo, useCallback, useEffect, useRef } from "react";
import type { ImageGroup } from "../types.ts";
import { cn, toGroupSortId, wasJustDragged } from "../utils/helpers.ts";
import { GroupThumbGrid } from "./GroupThumbGrid.tsx";

export const SortableGroupCard = memo(function SortableGroupCard({
  group,
  gridIndex,
  isDropTarget,
  isExpanded,
  isFrozen,
  isSelected,
  isGhost,
  isSearchMatch,
  isCurrentSearchMatch,
  onClick,
  popover,
}: {
  group: ImageGroup;
  gridIndex: number;
  isDropTarget: boolean;
  isExpanded: boolean;
  isFrozen: boolean;
  isSelected: boolean;
  isGhost: boolean;
  isSearchMatch?: boolean;
  isCurrentSearchMatch?: boolean;
  onClick: (e: React.MouseEvent) => void;
  popover?: React.ReactNode;
}) {
  const {
    attributes,
    listeners,
    setNodeRef: setSortRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: toGroupSortId(group.id) });

  const cardRef = useRef<HTMLDivElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const setNodeRef = useCallback(
    (node: HTMLDivElement | null) => {
      setSortRef(node);
      (cardRef as React.MutableRefObject<HTMLDivElement | null>).current = node;
    },
    [setSortRef],
  );

  useEffect(() => {
    if (!isExpanded || !cardRef.current || !popoverRef.current) return;
    const card = cardRef.current.getBoundingClientRect();
    const pop = popoverRef.current;
    const popWidth = pop.offsetWidth;
    const cardCenterX = card.left + card.width / 2;
    let left = cardCenterX - popWidth / 2;
    left = Math.max(16, Math.min(left, window.innerWidth - popWidth - 16));
    pop.style.left = `${left - card.left}px`;
  }, [isExpanded]);

  const frozenTransform = useRef<string | undefined>(undefined);
  if (isFrozen && frozenTransform.current === undefined) {
    frozenTransform.current = CSS.Transform.toString(transform);
  } else if (!isFrozen) {
    frozenTransform.current = undefined;
  }

  const style = {
    transform: isFrozen
      ? (frozenTransform.current ?? CSS.Transform.toString(transform))
      : CSS.Transform.toString(transform),
    transition: isFrozen ? "none" : transition,
  };

  function handleClick(e: React.MouseEvent) {
    if (wasJustDragged()) return;
    e.stopPropagation();
    onClick(e);
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={cn(
        "card",
        "group-card",
        isDragging && "card-overlay",
        isDropTarget && "group-drop-target",
        isExpanded && "group-expanded-card",
        isSelected && "card-selected",
        isGhost && "card-ghost",
        isSearchMatch && "card-search-match",
        isCurrentSearchMatch && "card-search-current",
      )}
      data-group-id={group.id}
      onClick={handleClick}
      {...attributes}
      {...listeners}
    >
      <GroupThumbGrid images={group.images} />
      <div className="card-info">
        <span className="card-badge">{gridIndex + 1}</span>
        <span className="card-name" title={group.name}>
          {group.name}
        </span>
        <span className="group-count">{group.images.length}</span>
      </div>
      {popover && (
        <div ref={popoverRef} className="group-popover-anchor">
          {popover}
        </div>
      )}
    </div>
  );
});
