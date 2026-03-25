import React from "react";
import { useImageStore } from "../stores/imageStore.ts";
import { useSelectionStore } from "../stores/selectionStore.ts";
import { useGroupStore } from "../stores/groupStore.ts";
import { useUIStore } from "../stores/uiStore.ts";
import { getErrorMessage, postJson } from "../utils/helpers.ts";

export function Toolbar({ onCreateGroup }: { onCreateGroup: () => void }) {
  const images = useImageStore((s) => s.images);
  const hasChanges = useImageStore((s) => s.hasChanges);
  const fetchImages = useImageStore((s) => s.fetchImages);

  const selectedIds = useSelectionStore((s) => s.selectedIds);
  const clearSelection = useSelectionStore((s) => s.clearSelection);

  const groups = useGroupStore((s) => s.groups);
  const groupsEnabled = useGroupStore((s) => s.groupsEnabled);
  const toggleGroupsEnabled = useGroupStore((s) => s.toggleGroupsEnabled);
  const updateGroups = useGroupStore((s) => s.updateGroups);
  const collapseGroup = useGroupStore((s) => s.collapseGroup);

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
  const bumpCacheNonce = useUIStore((s) => s.bumpCacheNonce);

  async function refreshState() {
    await Promise.all([fetchImages(), checkUndo()]);
  }

  function handleToggleGroups() {
    toggleGroupsEnabled();
    clearSelection();
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
      const res = await fetch("/api/undo", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      bumpCacheNonce();
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

  return (
    <header className="header">
      <div className="header-info">
        <div className="header-title">Reorder Images</div>
        <div className="header-subtitle">
          {selectedIds.size > 0
            ? `${selectedIds.size} selected — drag to move`
            : `${images.length} image${images.length !== 1 ? "s" : ""} — drag to reorder`}
        </div>
      </div>
      <div className="header-actions">
        {selectedIds.size > 0 && (
          <button className="btn btn-secondary" onClick={() => setShowPaths(true)}>
            Paths ({selectedIds.size})
          </button>
        )}
        {groupsEnabled && selectedIds.size > 0 && (
          <button className="btn btn-secondary" onClick={onCreateGroup}>
            Group ({selectedIds.size})
          </button>
        )}
        <button
          className={`btn ${groupsEnabled ? "btn-primary" : "btn-secondary"}`}
          onClick={handleToggleGroups}
        >
          Groups {groupsEnabled ? "On" : "Off"}
        </button>
        {groups.length > 0 && (
          <>
            <button className="btn btn-secondary" onClick={handleOrganizeClick} disabled={saving}>
              Organize Folders
            </button>
            <button className="btn btn-danger" onClick={handleClearGroups}>
              Clear Groups
            </button>
          </>
        )}
        {canUndo && (
          <button className="btn btn-danger" onClick={handleUndo} disabled={saving}>
            Undo
          </button>
        )}
        <button className="btn btn-secondary" onClick={refreshState} disabled={saving}>
          Refresh
        </button>
        <button
          className="btn btn-primary"
          onClick={handleSaveClick}
          disabled={!hasChanges || saving}
        >
          {saving ? "Saving..." : "Save Order"}
        </button>
      </div>
    </header>
  );
}
