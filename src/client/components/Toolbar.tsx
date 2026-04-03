import React, { useEffect } from "react";
import { useImageStore } from "../stores/imageStore.ts";
import { useSelectionStore } from "../stores/selectionStore.ts";
import { useGroupStore, flushGroupPersist } from "../stores/groupStore.ts";
import { useFolderStore } from "../stores/folderStore.ts";
import { useUIStore } from "../stores/uiStore.ts";
import { getErrorMessage, postJson, stripFolderNumber } from "../utils/helpers.ts";
import { useGroupOperations } from "../hooks/useGroupOperations.ts";
import { GroupPicker } from "./GroupPicker.tsx";

export function Toolbar() {
  const images = useImageStore((s) => s.images);
  const hasChanges = useImageStore((s) => s.hasChanges);
  const fetchImages = useImageStore((s) => s.fetchImages);

  const folderModeEnabled = useFolderStore((s) => s.folderModeEnabled);
  const folders = useFolderStore((s) => s.folders);
  const setFolderModeEnabled = useFolderStore((s) => s.setFolderModeEnabled);
  const fetchFolders = useFolderStore((s) => s.fetchFolders);
  const folderHasChanges = useFolderStore((s) => s.hasChanges);

  const selectedIds = useSelectionStore((s) => s.selectedIds);
  const clearSelection = useSelectionStore((s) => s.clearSelection);

  const groups = useGroupStore((s) => s.groups);
  const groupsEnabled = useGroupStore((s) => s.groupsEnabled);
  const toggleGroupsEnabled = useGroupStore((s) => s.toggleGroupsEnabled);
  const updateGroups = useGroupStore((s) => s.updateGroups);
  const collapseGroup = useGroupStore((s) => s.collapseGroup);
  const fetchGroups = useGroupStore((s) => s.fetchGroups);

  const saving = useUIStore((s) => s.saving);
  const canUndo = useUIStore((s) => s.canUndo);
  const showToast = useUIStore((s) => s.showToast);
  const setSaving = useUIStore((s) => s.setSaving);
  const setShowPreview = useUIStore((s) => s.setShowPreview);
  const setShowPaths = useUIStore((s) => s.setShowPaths);
  const setShowOrganize = useUIStore((s) => s.setShowOrganize);
  const setPreviewRenames = useUIStore((s) => s.setPreviewRenames);
  const setOrganizeMappings = useUIStore((s) => s.setOrganizeMappings);
  const checkUndo = useUIStore((s) => s.checkUndo);
  const setHeaderSubtitle = useUIStore((s) => s.setHeaderSubtitle);

  const groupOps = useGroupOperations();

  // Update header subtitle when counts change
  useEffect(() => {
    let subtitle: string;
    if (folderModeEnabled) {
      subtitle = selectedIds.size > 0
        ? `${selectedIds.size} selected`
        : `${folders.length} folder${folders.length !== 1 ? "s" : ""} — drag to reorder`;
    } else {
      subtitle = selectedIds.size > 0
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

  function handleToggleGroups() {
    toggleGroupsEnabled();
    clearSelection();
  }

  async function handleToggleFolderMode() {
    const next = !folderModeEnabled;
    setFolderModeEnabled(next);
    clearSelection();
    if (next) {
      await fetchFolders();
    } else {
      await fetchImages();
      await fetchGroups();
    }
  }

  function handleClearGroups() {
    updateGroups(() => []);
    collapseGroup();
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
      const { folders: currentFolders, rootImages: currentRoot, fetchFolders: refreshFolders } = useFolderStore.getState();
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

  const hasSelectionActions = selectedIds.size > 0;
  const hasGroupManagement = !folderModeEnabled && groups.length > 0;

  return (
    <>
      {hasSelectionActions && (
        <button className="btn btn-secondary" onClick={() => setShowPaths(true)}>
          Paths ({selectedIds.size})
        </button>
      )}
      {!folderModeEnabled && groupsEnabled && selectedIds.size > 0 && (
        <>
          <button className="btn btn-secondary" onClick={groupOps.handleCreateGroup}>
            Group ({selectedIds.size})
          </button>
          {groups.length > 0 && (
            <GroupPicker
              groups={groups}
              onSelect={(groupId) => groupOps.addImagesToGroup(groupId, [...useSelectionStore.getState().selectedIds])}
              selectedCount={selectedIds.size}
            />
          )}
        </>
      )}
      {hasSelectionActions && <span className="header-separator" />}
      <button
        className={`btn ${folderModeEnabled ? "btn-primary" : "btn-secondary"}`}
        onClick={handleToggleFolderMode}
        disabled={!folderModeEnabled && groups.length > 0}
        title={!folderModeEnabled && groups.length > 0 ? "Clear groups first to enable folder mode" : undefined}
      >
        Folders {folderModeEnabled ? "On" : "Off"}
      </button>
      {!folderModeEnabled && (
        <button
          className={`btn ${groupsEnabled ? "btn-primary" : "btn-secondary"}`}
          onClick={handleToggleGroups}
        >
          Groups {groupsEnabled ? "On" : "Off"}
        </button>
      )}
      {hasGroupManagement && (
        <>
          <span className="header-separator" />
          <button className="btn btn-secondary" onClick={handleOrganizeClick} disabled={saving}>
            Organize Folders
          </button>
          <button className="btn btn-danger" onClick={handleClearGroups}>
            Clear Groups
          </button>
        </>
      )}
      <span className="header-separator" />
      {!folderModeEnabled && canUndo && (
        <button className="btn btn-danger" onClick={handleUndo} disabled={saving}>
          Undo
        </button>
      )}
      <button className="btn btn-secondary" onClick={refreshState} disabled={saving}>
        Refresh
      </button>
      <span className="header-separator" />
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
          {saving ? "Saving..." : "Save Order"}
        </button>
      )}
    </>
  );
}
