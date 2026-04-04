import type React from "react";
import type { FolderGroup, ImageInfo } from "../types.ts";
import { stripFolderNumber } from "../utils/helpers.ts";
import { PopoverShell } from "./GroupPopover.tsx";

interface FolderPopoverProps {
  folder: FolderGroup;
  imageMap: Map<string, ImageInfo>;
  selectedIds: Set<string>;
  isMultiDragging: boolean;
  activeId: string | null;
  onRename: (folderName: string) => void;
  onDissolve: (folderName: string) => void;
  onCollapse: () => void;
  onRemoveFromFolder: (folderName: string, filename: string) => void;
  onCardClick: (id: string, e: React.MouseEvent) => void;
}

export function FolderPopover({
  folder,
  imageMap,
  selectedIds,
  isMultiDragging,
  activeId,
  onRename,
  onDissolve,
  onCollapse,
  onRemoveFromFolder,
  onCardClick,
}: FolderPopoverProps) {
  return (
    <PopoverShell
      id={folder.name}
      dataAttr="data-folder-popover"
      displayName={stripFolderNumber(folder.name) || folder.name}
      images={folder.images}
      imageMap={imageMap}
      selectedIds={selectedIds}
      isMultiDragging={isMultiDragging}
      activeId={activeId}
      actions={
        <>
          <button className="btn btn-small btn-secondary" onClick={() => onRename(folder.name)}>
            Rename
          </button>
          <button className="btn btn-small btn-danger" onClick={() => onDissolve(folder.name)}>
            Dissolve
          </button>
          <button className="btn btn-small btn-secondary" onClick={onCollapse}>
            Close
          </button>
        </>
      }
      onRemove={(fn) => onRemoveFromFolder(folder.name, fn)}
      onCardClick={onCardClick}
    />
  );
}
