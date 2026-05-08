import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useSessionStore } from "../../stores/core/sessionStore.ts";
import { useFolderStore } from "../../stores/folderStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import { OverflowMenu, OverflowMenuDivider, OverflowMenuItem } from "../shared/OverflowMenu.tsx";

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

  const folderModeDisabled = !folderModeEnabled && groups.length > 0;
  const showGroupsControls = !folderModeEnabled;

  async function toggleFolderMode() {
    const next = !folderModeEnabled;
    setFolderModeEnabled(next);
    clear("reorder");
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
  }

  function clearGroups() {
    updateGroups(() => []);
    collapseGroup();
  }

  return (
    <OverflowMenu label="More view options" checkable>
      <OverflowMenuItem
        onClick={toggleFolderMode}
        disabled={folderModeDisabled}
        checked={folderModeEnabled}
        title={folderModeDisabled ? "Clear groups first to enable folder mode" : undefined}
      >
        Folder mode
      </OverflowMenuItem>
      <OverflowMenuItem
        onClick={() => setNumberedFolderPrefix(!numberedFolderPrefix)}
        checked={numberedFolderPrefix}
        title="When on, created folders are prefixed with a sequence number (e.g. 001 - Beach). When off, only the title is used."
      >
        Number folder names
      </OverflowMenuItem>
      {showGroupsControls && (
        <>
          <OverflowMenuItem onClick={toggleGroups} checked={groupsEnabled}>
            Show groups
          </OverflowMenuItem>
          {groups.length > 0 && (
            <>
              <OverflowMenuDivider />
              <OverflowMenuItem danger onClick={clearGroups}>
                Clear all groups
              </OverflowMenuItem>
            </>
          )}
        </>
      )}
    </OverflowMenu>
  );
}
