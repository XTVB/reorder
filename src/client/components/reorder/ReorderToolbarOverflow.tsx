import { useModalStore } from "../../stores/core/modalStore.ts";
import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useSessionStore } from "../../stores/core/sessionStore.ts";
import { useFolderStore } from "../../stores/folderStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import { OverflowMenu, OverflowMenuDivider, OverflowMenuItem } from "../shared/OverflowMenu.tsx";
import { applyJsonOrder } from "./ReorderToolbar.tsx";

export function ReorderToolbarOverflow() {
  const folderModeEnabled = useFolderStore((s) => s.folderModeEnabled);
  const setFolderModeEnabled = useFolderStore((s) => s.setFolderModeEnabled);
  const flattenFolders = useFolderStore((s) => s.flattenFolders);
  const setFlattenFolders = useFolderStore((s) => s.setFlattenFolders);
  const fetchFolders = useFolderStore((s) => s.fetchFolders);
  const openModal = useModalStore((s) => s.openModal);
  const saving = useSessionStore((s) => s.saving);

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
    <OverflowMenu label="More view options" checkable align="right">
      {!folderModeEnabled && (
        <>
          {groups.length > 0 && (
            <OverflowMenuItem
              onClick={() => openModal("namingRules")}
              title="Compose group names from title/subtitle/short_sub templates"
            >
              Naming rules…
            </OverflowMenuItem>
          )}
          <OverflowMenuItem
            onClick={() => void applyJsonOrder()}
            disabled={saving}
            title="Rename files on disk so groups appear in the order listed in the groups JSON (ungrouped files at end)"
          >
            Apply JSON order
          </OverflowMenuItem>
          <OverflowMenuDivider />
        </>
      )}
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
      {folderModeEnabled && (
        <OverflowMenuItem
          onClick={() => {
            setFlattenFolders(!flattenFolders);
            clear("reorder");
          }}
          checked={flattenFolders}
          title="Show every image from all subfolders as one continuous stream instead of folder cards — select across folders to copy paths"
        >
          Flatten folders
        </OverflowMenuItem>
      )}
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
