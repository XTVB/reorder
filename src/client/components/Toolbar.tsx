import React, { useRef, useState } from "react";
import { useImageStore } from "../stores/imageStore.ts";
import { useSelectionStore } from "../stores/selectionStore.ts";
import { useGroupStore, flushGroupPersist } from "../stores/groupStore.ts";
import { useFolderStore } from "../stores/folderStore.ts";
import { useUIStore } from "../stores/uiStore.ts";
import { useTagStore } from "../stores/tagStore.ts";
import { getErrorMessage, postJson } from "../utils/helpers.ts";
import type { AppMode } from "../types.ts";
import { GroupPicker } from "./GroupPicker.tsx";

const MODES: { key: AppMode; label: string }[] = [
  { key: "reorder", label: "Reorder" },
  { key: "tags", label: "Tags" },
  { key: "merge", label: "Merge" },
];

export function Toolbar({ onCreateGroup, onAddToGroup, onFolderSave }: { onCreateGroup?: () => void; onAddToGroup?: (groupId: string) => void; onFolderSave?: () => void }) {
  const reimportRef = useRef<HTMLInputElement>(null);
  const [reimporting, setReimporting] = useState(false);
  const hasDb = useTagStore((s) => s.hasDb);
  const ingestFile = useTagStore((s) => s.ingestFile);
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

  const appMode = useUIStore((s) => s.appMode);
  const setAppMode = useUIStore((s) => s.setAppMode);
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

  async function refreshState() {
    if (folderModeEnabled) {
      await Promise.all([fetchFolders(), checkUndo()]);
    } else {
      await Promise.all([fetchImages(), checkUndo(), fetchGroups()]);
    }
  }

  async function handleReimport(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setReimporting(true);
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const result = await ingestFile(data);
      showToast(`Reimported ${result.ingested} images (${result.skipped} skipped)`, "success");
    } catch (err) {
      showToast(getErrorMessage(err, "Reimport failed"), "error");
    } finally {
      setReimporting(false);
      if (reimportRef.current) reimportRef.current.value = "";
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

  return (
    <header className="header">
      <div className="mode-toggle">
        {MODES.map((m) => (
          <button
            key={m.key}
            className={`mode-toggle-btn ${appMode === m.key ? "mode-toggle-active" : ""}`}
            onClick={() => {
              if (appMode !== m.key) {
                clearSelection();
                setAppMode(m.key);
              }
            }}
          >
            {m.label}
          </button>
        ))}
      </div>
      <div className="header-info">
        <div className="header-title">
          {appMode === "reorder" ? "Reorder Images" : appMode === "tags" ? "Tag Explorer" : "Merge Groups"}
        </div>
        <div className="header-subtitle">
          {appMode === "reorder" ? (
            folderModeEnabled
              ? (selectedIds.size > 0
                ? `${selectedIds.size} selected`
                : `${folders.length} folder${folders.length !== 1 ? "s" : ""} — drag to reorder`)
              : selectedIds.size > 0
              ? `${selectedIds.size} selected — drag to move`
              : `${images.length} image${images.length !== 1 ? "s" : ""} — drag to reorder`
          ) : appMode === "tags" ? (
            selectedIds.size > 0
              ? `${selectedIds.size} selected`
              : `${images.length} images`
          ) : (
            `${groups.length} groups`
          )}
        </div>
      </div>
      {appMode === "tags" && (
        <div className="header-actions">
          {selectedIds.size > 0 && (
            <button className="btn btn-secondary" onClick={onCreateGroup}>
              Group ({selectedIds.size})
            </button>
          )}
          {hasDb && (
            <>
              <input ref={reimportRef} type="file" accept=".json" style={{ display: "none" }} onChange={handleReimport} />
              <button className="btn btn-secondary" onClick={() => reimportRef.current?.click()} disabled={reimporting}>
                {reimporting ? "Reimporting..." : "Reimport Tags"}
              </button>
            </>
          )}
        </div>
      )}
      {appMode === "reorder" && (
        <div className="header-actions">
          {selectedIds.size > 0 && (
            <button className="btn btn-secondary" onClick={() => setShowPaths(true)}>
              Paths ({selectedIds.size})
            </button>
          )}
          {!folderModeEnabled && groupsEnabled && selectedIds.size > 0 && (
            <>
              <button className="btn btn-secondary" onClick={onCreateGroup}>
                Group ({selectedIds.size})
              </button>
              {groups.length > 0 && onAddToGroup && (
                <GroupPicker groups={groups} onSelect={onAddToGroup} selectedCount={selectedIds.size} />
              )}
            </>
          )}
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
          {!folderModeEnabled && groups.length > 0 && (
            <>
              <button className="btn btn-secondary" onClick={handleOrganizeClick} disabled={saving}>
                Organize Folders
              </button>
              <button className="btn btn-danger" onClick={handleClearGroups}>
                Clear Groups
              </button>
            </>
          )}
          {!folderModeEnabled && canUndo && (
            <button className="btn btn-danger" onClick={handleUndo} disabled={saving}>
              Undo
            </button>
          )}
          <button className="btn btn-secondary" onClick={refreshState} disabled={saving}>
            Refresh
          </button>
          {folderModeEnabled ? (
            <button
              className="btn btn-primary"
              onClick={onFolderSave}
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
        </div>
      )}
    </header>
  );
}
