import { useRef, useState } from "react";
import { useDismissOnOutside } from "../../hooks/useDismissOnOutside.ts";
import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useSessionStore } from "../../stores/core/sessionStore.ts";
import { useFolderStore } from "../../stores/folderStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";

export function ReorderToolbarOverflow() {
  const folderModeEnabled = useFolderStore((s) => s.folderModeEnabled);
  const setFolderModeEnabled = useFolderStore((s) => s.setFolderModeEnabled);
  const fetchFolders = useFolderStore((s) => s.fetchFolders);

  const groups = useGroupStore((s) => s.groups);
  const groupsEnabled = useGroupStore((s) => s.groupsEnabled);
  const toggleGroupsEnabled = useGroupStore((s) => s.toggleGroupsEnabled);
  const updateGroups = useGroupStore((s) => s.updateGroups);
  const collapseGroup = useGroupStore((s) => s.collapseGroup);
  const fetchGroups = useGroupStore((s) => s.fetchGroups);

  const fetchImages = useImageStore((s) => s.fetchImages);
  const clear = useSelectionStore((s) => s.clear);

  const numberedFolderPrefix = useSessionStore((s) => s.numberedFolderPrefix);
  const setNumberedFolderPrefix = useSessionStore((s) => s.setNumberedFolderPrefix);

  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useDismissOnOutside(containerRef, open, () => setOpen(false));

  async function toggleFolderMode() {
    const next = !folderModeEnabled;
    setFolderModeEnabled(next);
    clear("reorder");
    setOpen(false);
    if (next) {
      await fetchFolders();
    } else {
      await fetchImages();
      await fetchGroups();
    }
  }

  function toggleGroups() {
    toggleGroupsEnabled();
    clear("reorder");
    setOpen(false);
  }

  function clearGroups() {
    updateGroups(() => []);
    collapseGroup();
    setOpen(false);
  }

  const folderModeDisabled = !folderModeEnabled && groups.length > 0;
  const showGroupsControls = !folderModeEnabled;

  return (
    <div className="overflow-menu" ref={containerRef}>
      <button
        className="overflow-menu-trigger"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label="More view options"
        title="More view options"
      >
        <span aria-hidden="true">⋮</span>
      </button>
      {open && (
        <div className="overflow-menu-panel" role="menu">
          <button
            className="overflow-menu-item"
            onClick={toggleFolderMode}
            disabled={folderModeDisabled}
            title={folderModeDisabled ? "Clear groups first to enable folder mode" : undefined}
          >
            <span className="overflow-menu-check">{folderModeEnabled ? "✓" : ""}</span>
            Folder mode
          </button>
          <button
            className="overflow-menu-item"
            onClick={() => setNumberedFolderPrefix(!numberedFolderPrefix)}
            title="When on, created folders are prefixed with a sequence number (e.g. 001 - Beach). When off, only the title is used."
          >
            <span className="overflow-menu-check">{numberedFolderPrefix ? "✓" : ""}</span>
            Number folder names
          </button>
          {showGroupsControls && (
            <>
              <button className="overflow-menu-item" onClick={toggleGroups}>
                <span className="overflow-menu-check">{groupsEnabled ? "✓" : ""}</span>
                Show groups
              </button>
              {groups.length > 0 && (
                <>
                  <div className="overflow-menu-divider" />
                  <button className="overflow-menu-item overflow-menu-danger" onClick={clearGroups}>
                    <span className="overflow-menu-check" />
                    Clear all groups
                  </button>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
