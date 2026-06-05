import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import type React from "react";
import { memo, useCallback, useEffect, useRef } from "react";
import type { FolderGroup, ImageGroup } from "../../types.ts";
import {
  cn,
  stripFolderNumber,
  toFolderSortId,
  toGroupSortId,
  wasJustDragged,
} from "../../utils/helpers.ts";
import { GroupThumbGrid } from "../shared/GroupThumbGrid.tsx";
import { TrashBadge } from "../shared/TrashBadge.tsx";

type GroupVariantProps = {
  variant: "group";
  group: ImageGroup;
};

type FolderVariantProps = {
  variant: "folder";
  folder: FolderGroup;
};

type CommonProps = {
  gridIndex: number;
  isDropTarget: boolean;
  isExpanded: boolean;
  isFrozen: boolean;
  isSelected: boolean;
  isGhost: boolean;
  isSearchMatch?: boolean;
  isCurrentSearchMatch?: boolean;
  isMarkedForTrash?: boolean;
  onClick: (e: React.MouseEvent) => void;
  popover?: React.ReactNode;
};

type Props = CommonProps & (GroupVariantProps | FolderVariantProps);

export const SortableContainerCard = memo(function SortableContainerCard(props: Props) {
  const {
    gridIndex,
    isDropTarget,
    isExpanded,
    isFrozen,
    isSelected,
    isGhost,
    isSearchMatch,
    isCurrentSearchMatch,
    isMarkedForTrash,
    onClick,
    popover,
  } = props;

  const sortId =
    props.variant === "group" ? toGroupSortId(props.group.id) : toFolderSortId(props.folder.name);
  const images = props.variant === "group" ? props.group.images : props.folder.images;
  const displayName =
    props.variant === "group"
      ? props.group.name
      : stripFolderNumber(props.folder.name) || props.folder.name;
  const titleText = props.variant === "group" ? props.group.name : props.folder.name;
  const dataAttr =
    props.variant === "group"
      ? { "data-group-id": props.group.id }
      : { "data-folder-name": props.folder.name };

  const {
    attributes,
    listeners,
    setNodeRef: setSortRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: sortId });

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

    // Clamp height to the viewport so the final members stay reachable.
    // Prefer placing below the card; flip above when there's more room there.
    const gap = 8;
    const margin = 16;
    const spaceBelow = window.innerHeight - card.bottom - gap - margin;
    const spaceAbove = card.top - gap - margin;
    if (spaceAbove > spaceBelow) {
      pop.style.top = "auto";
      pop.style.bottom = "calc(100% + 8px)";
      pop.style.setProperty("--popover-max-height", `${Math.max(spaceAbove, 0)}px`);
    } else {
      pop.style.top = "calc(100% + 8px)";
      pop.style.bottom = "auto";
      pop.style.setProperty("--popover-max-height", `${Math.max(spaceBelow, 0)}px`);
    }
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
        isMarkedForTrash && "card-marked-trash",
      )}
      {...dataAttr}
      onClick={handleClick}
      {...attributes}
      {...listeners}
    >
      <GroupThumbGrid images={images} />
      {isMarkedForTrash && <TrashBadge />}
      <div className="card-info">
        <span className="card-badge">{gridIndex + 1}</span>
        <span className="card-name" title={titleText}>
          {displayName}
        </span>
        <span className="group-count">{images.length}</span>
      </div>
      {popover && (
        <div ref={popoverRef} className="group-popover-anchor">
          {popover}
        </div>
      )}
    </div>
  );
});
