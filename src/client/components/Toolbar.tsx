import { useEffect, useRef, useState } from "react";
import { useDismissOnOutside } from "../hooks/useDismissOnOutside.ts";
import { useGroupOperations } from "../hooks/useGroupOperations.ts";
import { useFolderStore } from "../stores/folderStore.ts";
import { flushGroupPersist, useGroupStore } from "../stores/groupStore.ts";
import { useImageStore } from "../stores/imageStore.ts";
import { useSelectionStore } from "../stores/selectionStore.ts";
import { useUIStore } from "../stores/uiStore.ts";
import type { ImageGroup } from "../types.ts";
import {
  generateContactSheetsBatch,
  getErrorMessage,
  postJson,
  reorderImagesByGroups,
  stripFolderNumber,
} from "../utils/helpers.ts";
import { GroupPicker } from "./GroupPicker.tsx";

export function Toolbar() {
  const images = useImageStore((s) => s.images);
  const hasChanges = useImageStore((s) => s.hasChanges);
  const fetchImages = useImageStore((s) => s.fetchImages);

  const folderModeEnabled = useFolderStore((s) => s.folderModeEnabled);
  const folders = useFolderStore((s) => s.folders);
  const fetchFolders = useFolderStore((s) => s.fetchFolders);
  const folderHasChanges = useFolderStore((s) => s.hasChanges);

  const selectedIds = useSelectionStore((s) => s.selectedIds);

  const groups = useGroupStore((s) => s.groups);
  const groupsEnabled = useGroupStore((s) => s.groupsEnabled);
  const fetchGroups = useGroupStore((s) => s.fetchGroups);

  const saving = useUIStore((s) => s.saving);
  const canUndo = useUIStore((s) => s.canUndo);
  const showToast = useUIStore((s) => s.showToast);
  const setSaving = useUIStore((s) => s.setSaving);
  const setShowPreview = useUIStore((s) => s.setShowPreview);
  const setShowPaths = useUIStore((s) => s.setShowPaths);
  const setShowOrganize = useUIStore((s) => s.setShowOrganize);
  const setShowReview = useUIStore((s) => s.setShowReview);
  const openSlideshow = useUIStore((s) => s.openSlideshow);
  const setPreviewRenames = useUIStore((s) => s.setPreviewRenames);
  const setOrganizeMappings = useUIStore((s) => s.setOrganizeMappings);
  const checkUndo = useUIStore((s) => s.checkUndo);
  const setHeaderSubtitle = useUIStore((s) => s.setHeaderSubtitle);

  const groupOps = useGroupOperations();

  const [generatingSheets, setGeneratingSheets] = useState(false);

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
      const res = await postJson("/api/preview", { order: images.map((i) => i.filename) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setPreviewRenames(data.renames);
      setShowPreview(true);
    } catch (err) {
      showToast(getErrorMessage(err, "Failed to preview"), "error");
    }
  }

  async function handleApplyJsonOrder() {
    setSaving(true);
    try {
      await flushGroupPersist();
      const res = await postJson("/api/reorder-by-groups", {});
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error ?? "Reorder failed");
      const effective = (data.renames ?? []).filter(
        (r: { from: string; to: string }) => r.from !== r.to,
      ).length;
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
      await flushGroupPersist();
      const res = await postJson("/api/undo", {});
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
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
      const res = await postJson("/api/organize/preview", {
        groups: groups.map((g) => ({ name: g.name, images: g.images })),
        order: images.map((i) => i.filename),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setOrganizeMappings(data.mappings);
      setShowOrganize(true);
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
      };
      const res = await postJson("/api/folders/save", body);
      const data = await res.json();
      if (!data.success) throw new Error(data.error ?? "Folder save failed");
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
      await flushGroupPersist();
      showToast("Saved group order to JSON", "success");
    } catch (err) {
      showToast(getErrorMessage(err, "Save group order failed"), "error");
    }
  }

  const hasSelectionActions = selectedIds.size > 0;
  const hasGroupManagement = !folderModeEnabled && groups.length > 0;
  const showUndo = !folderModeEnabled && canUndo;
  const showSlideshow = !folderModeEnabled && images.length > 0;

  return (
    <>
      {hasSelectionActions && (
        <div className="toolbar-group">
          <button className="btn btn-secondary" onClick={() => setShowPaths(true)}>
            Paths
          </button>
          {!folderModeEnabled && groupsEnabled && (
            <>
              <button className="btn btn-secondary" onClick={groupOps.handleCreateGroup}>
                Group
              </button>
              {groups.length > 0 && (
                <GroupPicker
                  groups={groups}
                  onSelect={(groupId) =>
                    groupOps.addImagesToGroup(groupId, [
                      ...useSelectionStore.getState().selectedIds,
                    ])
                  }
                />
              )}
            </>
          )}
        </div>
      )}
      {hasGroupManagement && (
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
            onClick={() => setShowReview(true)}
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
      {(showUndo || showSlideshow) && (
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
                const sel = useSelectionStore.getState().selectedIds;
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
        </div>
      )}
      {folderModeEnabled ? (
        <button
          className="btn btn-primary"
          onClick={handleFolderSave}
          disabled={!folderHasChanges || saving}
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

export function ToolbarOverflowMenu() {
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
  const clearSelection = useSelectionStore((s) => s.clearSelection);

  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useDismissOnOutside(containerRef, open, () => setOpen(false));

  async function toggleFolderMode() {
    const next = !folderModeEnabled;
    setFolderModeEnabled(next);
    clearSelection();
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
    clearSelection();
    setOpen(false);
  }

  function clearGroups() {
    updateGroups(() => []);
    collapseGroup();
    setOpen(false);
  }

  const folderModeDisabled = !folderModeEnabled && groups.length > 0;

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
          {!folderModeEnabled && (
            <button className="overflow-menu-item" onClick={toggleGroups}>
              <span className="overflow-menu-check">{groupsEnabled ? "✓" : ""}</span>
              Show groups
            </button>
          )}
          {!folderModeEnabled && groups.length > 0 && (
            <>
              <div className="overflow-menu-divider" />
              <button className="overflow-menu-item overflow-menu-danger" onClick={clearGroups}>
                <span className="overflow-menu-check" />
                Clear all groups
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
