import { useEffect, useMemo, useState } from "react";
import { postJson } from "../../api/client.ts";
import { useModalStore } from "../../stores/core/modalStore.ts";
import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useSessionStore } from "../../stores/core/sessionStore.ts";
import { useToastStore } from "../../stores/core/toastStore.ts";
import { useFolderStore } from "../../stores/folderStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import { useTrashStore } from "../../stores/trashStore.ts";
import type { ImageGroup, OrganizeMapping, RenameMapping, SaveResponse } from "../../types.ts";
import {
  generateContactSheetsBatch,
  getErrorMessage,
  reorderImagesByGroups,
  selectedImageFilenames as selectedImageFilenamesFromIds,
  stripFolderNumber,
} from "../../utils/helpers.ts";
import { reverseSelection } from "../../utils/reverseSelection.ts";
import { GroupPicker } from "../shared/GroupPicker.tsx";
import { TrashIcon } from "../shared/TrashIcon.tsx";

export function ReorderToolbar() {
  const images = useImageStore((s) => s.images);
  const hasChanges = useImageStore((s) => s.hasChanges);
  const fetchImages = useImageStore((s) => s.fetchImages);

  const folderModeEnabled = useFolderStore((s) => s.folderModeEnabled);
  const folders = useFolderStore((s) => s.folders);
  const fetchFolders = useFolderStore((s) => s.fetchFolders);
  const folderHasChanges = useFolderStore((s) => s.hasChanges);

  const selectedIds = useSelectionStore((s) => s.contexts.reorder);

  const groups = useGroupStore((s) => s.groups);
  const groupsEnabled = useGroupStore((s) => s.groupsEnabled);
  const fetchGroups = useGroupStore((s) => s.fetchGroups);

  const saving = useSessionStore((s) => s.saving);
  const canUndo = useSessionStore((s) => s.canUndo);
  const numberedFolderPrefix = useSessionStore((s) => s.numberedFolderPrefix);
  const showToast = useToastStore((s) => s.showToast);
  const setSaving = useSessionStore((s) => s.setSaving);
  const openModal = useModalStore((s) => s.openModal);
  const openSlideshow = useSessionStore((s) => s.openSlideshow);
  const setPreviewRenames = useSessionStore((s) => s.setPreviewRenames);
  const setOrganizeMappings = useSessionStore((s) => s.setOrganizeMappings);
  const checkUndo = useSessionStore((s) => s.checkUndo);
  const setHeaderSubtitle = useSessionStore((s) => s.setHeaderSubtitle);

  const createGroupFromSelection = useGroupStore((s) => s.createGroupFromSelection);
  const addImagesToGroupAction = useGroupStore((s) => s.addImagesToGroup);
  const flushPending = useGroupStore((s) => s.flushPending);

  const markedTrashIds = useSelectionStore((s) => s.contexts.trash);
  const [generatingSheets, setGeneratingSheets] = useState(false);

  const selectedImageFilenames = useMemo(
    () => selectedImageFilenamesFromIds(selectedIds),
    [selectedIds],
  );
  const selectionAllMarked =
    selectedImageFilenames.length > 0 &&
    selectedImageFilenames.every((fn) => markedTrashIds.has(fn));
  const selectionTrashLabel = selectionAllMarked
    ? "Unmark selection from deletion"
    : "Mark selection for deletion";

  function handleSelectionMarkTrash() {
    useTrashStore.getState().toggleMany(selectedImageFilenames);
  }

  // Update header subtitle when counts change
  // biome-ignore lint/correctness/useExhaustiveDependencies: setHeaderSubtitle is a stable Zustand action
  useEffect(() => {
    let subtitle: string;
    if (folderModeEnabled) {
      subtitle =
        selectedIds.size > 0
          ? `${selectedIds.size} selected`
          : `${folders.length} folder${folders.length !== 1 ? "s" : ""} — drag to reorder`;
    } else {
      subtitle =
        selectedIds.size > 0
          ? `${selectedIds.size} selected — drag to move`
          : `${images.length} image${images.length !== 1 ? "s" : ""} — drag to reorder`;
    }
    setHeaderSubtitle(subtitle);
    return () => setHeaderSubtitle("");
  }, [folderModeEnabled, selectedIds.size, folders.length, images.length]);

  async function refreshState() {
    if (folderModeEnabled) {
      await Promise.all([fetchFolders(), checkUndo()]);
    } else {
      await Promise.all([fetchImages(), checkUndo(), fetchGroups()]);
    }
  }

  async function handleSaveClick() {
    try {
      const data = await postJson<{ renames: RenameMapping[] }>("/api/preview", {
        order: images.map((i) => i.filename),
      });
      setPreviewRenames(data.renames);
      openModal("preview");
    } catch (err) {
      showToast(getErrorMessage(err, "Failed to preview"), "error");
    }
  }

  async function handleApplyJsonOrder() {
    setSaving(true);
    try {
      await flushPending();
      const data = await postJson<SaveResponse>("/api/reorder-by-groups", {});
      if (!data.success) throw new Error("Reorder failed");
      const renames: RenameMapping[] = data.renames ?? [];
      useTrashStore.getState().remap(renames);
      const effective = renames.filter((r) => r.from !== r.to).length;
      const warnings: string[] = data.warnings ?? [];
      if (warnings.length > 0) {
        showToast(
          `Applied JSON order: ${effective} renamed, ${warnings.length} warning${warnings.length === 1 ? "" : "s"}`,
          "warning",
        );
      } else {
        showToast(`Applied JSON order: ${effective} renamed`, "success");
      }
      await refreshState();
    } catch (err) {
      showToast(getErrorMessage(err, "Apply JSON order failed"), "error");
    } finally {
      setSaving(false);
    }
  }

  async function handleUndo() {
    setSaving(true);
    try {
      await flushPending();
      const data = await postJson<{ renames?: RenameMapping[] }>("/api/undo", {});
      useTrashStore.getState().remap(data.renames ?? []);
      showToast("Undo successful", "success");
      await refreshState();
    } catch (err) {
      showToast(getErrorMessage(err, "Undo failed"), "error");
    } finally {
      setSaving(false);
    }
  }

  async function handleOrganizeClick() {
    try {
      const data = await postJson<{ mappings: OrganizeMapping[] }>("/api/organize/preview", {
        groups: groups.map((g) => ({ name: g.name, images: g.images })),
        order: images.map((i) => i.filename),
        numbered: useSessionStore.getState().numberedFolderPrefix,
      });
      setOrganizeMappings(data.mappings);
      openModal("organize");
    } catch (err) {
      showToast(getErrorMessage(err, "Failed to preview"), "error");
    }
  }

  async function handleFolderSave() {
    setSaving(true);
    try {
      const {
        folders: currentFolders,
        rootImages: currentRoot,
        fetchFolders: refreshFolders,
      } = useFolderStore.getState();
      const body = {
        folders: currentFolders.map((f) => ({
          title: stripFolderNumber(f.name) || f.name,
          images: f.images,
        })),
        rootImages: currentRoot,
        numbered: useSessionStore.getState().numberedFolderPrefix,
      };
      const data = await postJson<{ success: boolean }>("/api/folders/save", body);
      if (!data.success) throw new Error("Folder save failed");
      showToast("Folders saved successfully", "success");
      await refreshFolders();
    } catch (err) {
      showToast(getErrorMessage(err, "Folder save failed"), "error");
    } finally {
      setSaving(false);
    }
  }

  async function handleContactSheets() {
    setGeneratingSheets(true);
    try {
      const pad = String(groups.length).length;
      const results = await generateContactSheetsBatch(
        groups.map((g, i) => ({
          filenames: g.images,
          clusterName: `${String(i + 1).padStart(pad, "0")}-${g.name}`,
        })),
      );
      await navigator.clipboard.writeText(results.map((r) => r.path).join("\n"));
      showToast(
        `Copied ${results.length} contact sheet path${results.length === 1 ? "" : "s"}`,
        "success",
      );
    } catch (err) {
      showToast(getErrorMessage(err, "Failed to generate contact sheets"), "error");
    } finally {
      setGeneratingSheets(false);
    }
  }

  function sortGroupsByGalleryOrder(): ImageGroup[] {
    const imageIndex = new Map(images.map((img, i) => [img.filename, i]));
    return [...groups].sort((a, b) => {
      const aIdx = a.images.reduce(
        (min, fn) => Math.min(min, imageIndex.get(fn) ?? Infinity),
        Infinity,
      );
      const bIdx = b.images.reduce(
        (min, fn) => Math.min(min, imageIndex.get(fn) ?? Infinity),
        Infinity,
      );
      return aIdx - bIdx;
    });
  }

  function handleGroupsToTop() {
    const sortedGroups = sortGroupsByGalleryOrder();
    const { imageMap, setImages } = useImageStore.getState();
    setImages(reorderImagesByGroups(images, imageMap, sortedGroups));
  }

  async function handleSaveJsonOrder() {
    const sorted = sortGroupsByGalleryOrder();
    const unchanged = sorted.every((g, i) => g.id === groups[i]?.id);
    if (unchanged) {
      showToast("JSON order already matches gallery", "success");
      return;
    }
    useGroupStore.getState().updateGroups(() => sorted);
    try {
      await flushPending();
      showToast("Saved group order to JSON", "success");
    } catch (err) {
      showToast(getErrorMessage(err, "Save group order failed"), "error");
    }
  }

  // In folder mode, toggling "numbered" alone is a meaningful change even when
  // no folders were reordered/renamed — surface it via the Save button.
  const folderNumberingMismatch =
    folderModeEnabled &&
    folders.length > 0 &&
    folders.some((f) => (stripFolderNumber(f.name) !== f.name) !== numberedFolderPrefix);

  const hasSelectionActions = selectedIds.size > 0;
  const showUndo = !folderModeEnabled && canUndo;
  const showSlideshow = !folderModeEnabled && images.length > 0;
  const trashCount = !folderModeEnabled ? markedTrashIds.size : 0;
  const showTrashButton = trashCount > 0;

  return (
    <>
      {hasSelectionActions && (
        <div className="toolbar-group">
          <button className="btn btn-secondary" onClick={() => openModal("paths")}>
            Paths
          </button>
          {selectedIds.size >= 2 && (
            <button
              className="btn btn-secondary btn-icon"
              onClick={() => reverseSelection()}
              title="Reverse the order of the selection (R)"
              aria-label="Reverse selection order"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" role="presentation">
                <path
                  d="M7 4v13M7 17l-3-3M7 17l3-3M17 20V7M17 7l-3 3M17 7l3 3"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          )}
          {!folderModeEnabled && groupsEnabled && (
            <>
              <button className="btn btn-secondary" onClick={createGroupFromSelection}>
                Group
              </button>
              {groups.length > 0 && (
                <GroupPicker
                  groups={groups}
                  onSelect={(groupId: string) =>
                    addImagesToGroupAction(groupId, [
                      ...useSelectionStore.getState().contexts.reorder,
                    ])
                  }
                />
              )}
            </>
          )}
          {!folderModeEnabled && selectedImageFilenames.length > 0 && (
            <button
              className="btn btn-secondary btn-icon"
              onClick={handleSelectionMarkTrash}
              title={`${selectionTrashLabel} (D)`}
              aria-label={selectionTrashLabel}
            >
              <TrashIcon size={18} variant={selectionAllMarked ? "minus" : "plus"} />
            </button>
          )}
        </div>
      )}
      {!folderModeEnabled && (
        <div className="toolbar-group">
          <button className="btn btn-secondary" onClick={handleGroupsToTop}>
            Groups to Top
          </button>
          <button
            className="btn btn-secondary"
            onClick={handleApplyJsonOrder}
            disabled={saving}
            title="Rename files on disk so groups appear in the order listed in .reorder-groups.json (ungrouped files at end)"
          >
            Apply Order
          </button>
          <button
            className="btn btn-secondary"
            onClick={handleSaveJsonOrder}
            disabled={saving}
            title="Reorder groups in .reorder-groups.json to match the current gallery order (no file renames)"
          >
            Save Order
          </button>
          <button
            className="btn btn-secondary btn-icon"
            onClick={handleContactSheets}
            disabled={generatingSheets}
            title={
              generatingSheets
                ? "Generating contact sheets..."
                : "Generate a contact sheet for each group and copy paths to clipboard"
            }
            aria-label="Generate contact sheets"
          >
            {generatingSheets ? (
              <svg
                className="btn-spinner"
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                role="presentation"
              >
                <circle
                  cx="12"
                  cy="12"
                  r="9"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeOpacity="0.25"
                />
                <path
                  d="M21 12a9 9 0 0 0-9-9"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" role="presentation">
                <rect
                  x="3"
                  y="3"
                  width="7"
                  height="7"
                  rx="1"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <rect
                  x="14"
                  y="3"
                  width="7"
                  height="7"
                  rx="1"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <rect
                  x="3"
                  y="14"
                  width="7"
                  height="7"
                  rx="1"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <rect
                  x="14"
                  y="14"
                  width="7"
                  height="7"
                  rx="1"
                  stroke="currentColor"
                  strokeWidth="2"
                />
              </svg>
            )}
          </button>
          <button
            className="btn btn-secondary btn-icon"
            onClick={() => openModal("review")}
            title="Review groups"
            aria-label="Review groups"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" role="presentation">
              <path
                d="M4 5l2 2 3-3M4 12l2 2 3-3M4 19l2 2 3-3"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M13 5h7M13 12h7M13 19h7"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </button>
          <button
            className="btn btn-secondary btn-icon"
            onClick={() => openModal("createGroups")}
            title="Sort ungrouped photos into new groups"
            aria-label="Create groups from ungrouped photos"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" role="presentation">
              <path
                d="M3 7a2 2 0 0 1 2-2h3l2 2h6a2 2 0 0 1 2 2v3"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M3 7v10a2 2 0 0 0 2 2h8"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M18 15v6M15 18h6"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </button>
          <button
            className="btn btn-secondary btn-icon"
            onClick={handleOrganizeClick}
            disabled={saving}
            title="Organize groups into folders"
            aria-label="Organize groups into folders"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" role="presentation">
              <path
                d="M3 5a1 1 0 0 1 1-1h5l2 3h9a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinejoin="round"
              />
              <path
                d="M12 11v5M9.5 13.5h5"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>
      )}
      {(showUndo || showSlideshow || showTrashButton) && (
        <div className="toolbar-group">
          {showUndo && (
            <button
              className="btn btn-ghost-danger"
              onClick={handleUndo}
              disabled={saving}
              title="Undo last save"
            >
              Undo
            </button>
          )}
          {showSlideshow && (
            <button
              className="btn btn-secondary btn-icon"
              onClick={() => {
                const sel = useSelectionStore.getState().contexts.reorder;
                let startIdx = 0;
                if (sel.size > 0) {
                  const firstSelected = images.findIndex((img) => sel.has(img.filename));
                  if (firstSelected !== -1) startIdx = firstSelected;
                }
                openSlideshow(startIdx);
              }}
              title="Slideshow (full-screen viewer with autoplay)"
              aria-label="Slideshow"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" role="presentation">
                <path
                  d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          )}
          <button
            className="btn btn-secondary btn-icon"
            onClick={refreshState}
            disabled={saving}
            title="Refresh"
            aria-label="Refresh"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" role="presentation">
              <path
                d="M20 11A8 8 0 1 0 18.3 17M20 5v6h-6"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          {showTrashButton && (
            <button
              className="btn btn-secondary btn-icon toolbar-trash-btn"
              onClick={() => openModal("trash")}
              title={`Review ${trashCount} file${trashCount === 1 ? "" : "s"} marked for deletion`}
              aria-label={`Review ${trashCount} marked for deletion`}
            >
              <TrashIcon size={16} />
              <span className="toolbar-trash-count">{trashCount}</span>
            </button>
          )}
        </div>
      )}
      {folderModeEnabled ? (
        <button
          className="btn btn-primary"
          onClick={handleFolderSave}
          disabled={(!folderHasChanges && !folderNumberingMismatch) || saving}
        >
          {saving ? "Saving..." : "Save Folders"}
        </button>
      ) : (
        <button
          className="btn btn-primary"
          onClick={handleSaveClick}
          disabled={!hasChanges || saving}
        >
          {saving ? "Saving..." : "Save"}
        </button>
      )}
    </>
  );
}
