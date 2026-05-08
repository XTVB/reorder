import type React from "react";
import type { FolderGroup, ImageInfo } from "../../types.ts";
import { stripFolderNumber } from "../../utils/helpers.ts";
import { PopoverShell } from "./GroupPopover.tsx";

interface FolderPopoverProps {
  folder: FolderGroup;
  imageMap: Map<string, ImageInfo>;
  selectedIds: Set<string>;
  isMultiDragging: boolean;
  activeId: string | null;
  lightboxImages: string[];
  onRename: (folderName: string) => void;
  onDissolve: (folderName: string) => void;
  onCollapse: () => void;
  onRemoveFromFolder: (folderName: string, filename: string) => void;
  onSelect: (filename: string, e: React.MouseEvent) => void;
  onRangeSelect: (filename: string, e: React.MouseEvent) => void;
}

export function FolderPopover({
  folder,
  imageMap,
  selectedIds,
  isMultiDragging,
  activeId,
  lightboxImages,
  onRename,
  onDissolve,
  onCollapse,
  onRemoveFromFolder,
  onSelect,
  onRangeSelect,
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
      lightboxImages={lightboxImages}
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
      onSelect={onSelect}
      onRangeSelect={onRangeSelect}
    />
  );
}
