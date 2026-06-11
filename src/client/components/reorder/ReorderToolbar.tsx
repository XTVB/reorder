import { useEffect, useMemo, useState } from "react";
import { postJson } from "../../api/client.ts";
import { consumeSSE, startSSE } from "../../api/sse.ts";
import { useModalStore } from "../../stores/core/modalStore.ts";
import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useSessionStore } from "../../stores/core/sessionStore.ts";
import { useToastStore } from "../../stores/core/toastStore.ts";
import { useFolderStore } from "../../stores/folderStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import { useMergeSuggestionsStore } from "../../stores/mergeSuggestionsStore.ts";
import { useListStore } from "../../stores/modes/cluster/listStore.ts";
import { useTrashStore } from "../../stores/trashStore.ts";
import type {
  GroupOrderMode,
  ImageGroup,
  OrganizeMapping,
  RenameMapping,
  SaveResponse,
} from "../../types.ts";
import {
  groupedFilenameSet,
  groupsInGalleryOrder,
  withLockedGroupsInPlace,
} from "../../utils/groups.ts";
import {
  generateContactSheetsBatch,
  getErrorMessage,
  reorderImagesByGroups,
  selectedImageFilenames as selectedImageFilenamesFromIds,
  stripFolderNumber,
} from "../../utils/helpers.ts";
import { reorderImagesWithinSlots, reorderSubsetWithinSlots } from "../../utils/reorder.ts";
import { reverseSelection } from "../../utils/reverseSelection.ts";
import { beginSortTransition } from "../../utils/sortFlip.ts";
import { GroupPicker } from "../shared/GroupPicker.tsx";
import { TrashIcon } from "../shared/TrashIcon.tsx";

const SORT_MODE_KEY = "reorder-similarity-sort-mode";
const MINIMAL_LOCALITY_KEY = "reorder-similarity-minimal-locality";
const SORT_TARGET_KEY = "reorder-similarity-sort-target";

type SortTarget = "groups" | "ungrouped";

const SORT_TARGET_TITLES: Record<SortTarget, string> = {
  groups:
    "Sort the groups: similar groups end up adjacent; ungrouped images move along with the consolidation. Locked groups (L) keep their slot",
  ungrouped:
    "Sort only the loose ungrouped images: they swap among their own slots so similar ones sit together; every group stays exactly where it is",
};

const SORT_MODE_TITLES: Record<GroupOrderMode, string> = {
  chain:
    "Chain: greedy nearest-neighbor walk from the first group, then 2-opt segment reversal — each group follows its closest match",
  tree: "Tree: cluster the groups hierarchically and order the leaves optimally — families of related groups stay together as blocks",
  spectral:
    "Spectral: arrange all groups along the dominant similarity gradient (Fiedler vector) — one global axis through the collection",
  minimal:
    "Minimal: keep the current order, only making small local moves (max 5 positions) where similarity clearly improves",
};

function loadSortMode(): GroupOrderMode {
  const v = localStorage.getItem(SORT_MODE_KEY);
  return v === "chain" || v === "tree" || v === "spectral" || v === "minimal" ? v : "tree";
}

function loadMinimalLocality(): number {
  const v = parseInt(localStorage.getItem(MINIMAL_LOCALITY_KEY) ?? "", 10);
  return Number.isFinite(v) && v >= 1 ? v : 5;
}

function loadSortTarget(): SortTarget {
  return localStorage.getItem(SORT_TARGET_KEY) === "ungrouped" ? "ungrouped" : "groups";
}

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
  const [sortingBySimilarity, setSortingBySimilarity] = useState(false);
  const [sortMode, setSortMode] = useState<GroupOrderMode>(loadSortMode);
  const [minimalLocality, setMinimalLocality] = useState<number>(loadMinimalLocality);
  const [sortTarget, setSortTarget] = useState<SortTarget>(loadSortTarget);
  // Last progress message from a running similarity sort, shown as the header
  // subtitle so long computations (big ungrouped sets) have visible feedback.
  const [sortProgress, setSortProgress] = useState<string | null>(null);

  function handleSortModeChange(mode: GroupOrderMode) {
    localStorage.setItem(SORT_MODE_KEY, mode);
    setSortMode(mode);
  }

  function handleSortTargetChange(target: SortTarget) {
    localStorage.setItem(SORT_TARGET_KEY, target);
    setSortTarget(target);
  }

  function handleMinimalLocalityChange(v: number) {
    localStorage.setItem(MINIMAL_LOCALITY_KEY, String(v));
    setMinimalLocality(v);
  }

  const selectedImageFilenames = useMemo(
    () => selectedImageFilenamesFromIds(selectedIds),
    [selectedIds],
  );

  const ungroupedCount = useMemo(() => {
    const grouped = groupedFilenameSet(groups);
    return images.reduce((acc, i) => acc + (grouped.has(i.filename) ? 0 : 1), 0);
  }, [groups, images]);
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
    if (sortProgress) {
      subtitle = sortProgress;
    } else if (folderModeEnabled) {
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
  }, [folderModeEnabled, selectedIds.size, folders.length, images.length, sortProgress]);

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
    return groupsInGalleryOrder(groups, images);
  }

  function handleGroupsToTop() {
    const sortedGroups = sortGroupsByGalleryOrder();
    const { imageMap, setImages } = useImageStore.getState();
    beginSortTransition();
    setImages(reorderImagesByGroups(images, imageMap, sortedGroups));
  }

  /** POST to similarity-order and collect the orderedIds result. */
  async function fetchSimilarityOrder(body: Record<string, unknown>): Promise<string[] | null> {
    const start = await startSSE("/api/groups/similarity-order", body);
    if (start.kind === "conflict") {
      showToast(start.message, "error");
      return null;
    }
    const outcome: { orderedIds: string[] | null; error: string | null } = {
      orderedIds: null,
      error: null,
    };
    await consumeSSE<{ orderedIds: string[] }>(start.response, {
      onProgress: (message) => setSortProgress(message),
      onResult: (data) => {
        outcome.orderedIds = data.orderedIds;
      },
      onError: (error) => {
        outcome.error = error;
      },
    });
    if (outcome.error) throw new Error(outcome.error);
    if (!outcome.orderedIds) throw new Error("Stream ended unexpectedly");
    return outcome.orderedIds;
  }

  async function sortGroupsBySimilarity() {
    const { method, fullResolution } = useMergeSuggestionsStore.getState();
    const weights = useListStore.getState().weights;
    // The sort is based on the current gallery order, not the JSON order:
    // minimal mode preserves it, chain starts from its first group, and
    // tree/spectral orient their axis toward that group.
    const galleryOrderedIds = sortGroupsByGalleryOrder().map((g) => g.id);
    const orderedIds = await fetchSimilarityOrder({
      method,
      fullResolution,
      weights,
      mode: sortMode,
      anchorId: galleryOrderedIds[0],
      orderedGroupIds: galleryOrderedIds,
      ...(sortMode === "minimal" && { minimalLocality }),
    });
    if (!orderedIds) return;

    const pos = new Map(orderedIds.map((id, i) => [id, i]));
    const current = useGroupStore.getState().groups;
    const sorted = [...current].sort(
      (a, b) => (pos.get(a.id) ?? Infinity) - (pos.get(b.id) ?? Infinity),
    );
    const { images: imgs, imageMap, setImages } = useImageStore.getState();
    // Locked groups keep their current gallery slot; the similarity order
    // fills in around them.
    const finalOrder = withLockedGroupsInPlace(groupsInGalleryOrder(current, imgs), sorted);
    beginSortTransition();
    setImages(reorderImagesByGroups(imgs, imageMap, finalOrder));
    const unchanged =
      orderedIds.length === galleryOrderedIds.length &&
      orderedIds.every((id, i) => id === galleryOrderedIds[i]);
    const lockedCount = current.filter((g) => g.locked).length;
    showToast(
      unchanged
        ? "Group order already optimal"
        : `Sorted ${finalOrder.length} groups by similarity${
            lockedCount > 0 ? ` (${lockedCount} locked in place)` : ""
          }`,
      "success",
    );
  }

  async function sortUngroupedBySimilarity() {
    const weights = useListStore.getState().weights;
    // With images selected, sort just the selection among its own slots —
    // lets big collections be sorted in batches, and a selection from inside
    // a group sorts within that group. Otherwise sort all ungrouped images.
    const selectionScope = selectedImageFilenames.length > 0;
    let targetOrder: string[];
    if (selectionScope) {
      const sel = new Set(selectedImageFilenames);
      targetOrder = images.filter((i) => sel.has(i.filename)).map((i) => i.filename);
      if (targetOrder.length < 3) {
        showToast("Select at least 3 images to sort a selection", "error");
        return;
      }
    } else {
      const grouped = groupedFilenameSet(useGroupStore.getState().groups);
      targetOrder = images.filter((i) => !grouped.has(i.filename)).map((i) => i.filename);
      if (targetOrder.length < 3) {
        showToast("Not enough ungrouped images to sort", "error");
        return;
      }
    }
    const orderedIds = await fetchSimilarityOrder({
      target: "ungrouped",
      weights,
      mode: sortMode,
      orderedImageFilenames: targetOrder,
      ...(sortMode === "minimal" && { minimalLocality }),
    });
    if (!orderedIds) return;

    const { images: imgs, setImages } = useImageStore.getState();
    beginSortTransition();
    setImages(reorderImagesWithinSlots(imgs, orderedIds));
    if (selectionScope) {
      // Sorted members that live inside a group reorder within that group's
      // own image list too, so the group popover (and eventual rename order)
      // reflects the new sequence.
      useGroupStore.getState().updateGroups((prev) => {
        let changed = false;
        const next = prev.map((g) => {
          const reordered = reorderSubsetWithinSlots(g.images, orderedIds);
          if (reordered.every((fn, i) => fn === g.images[i])) return g;
          changed = true;
          return { ...g, images: reordered };
        });
        return changed ? next : prev;
      });
    }
    // Compare against the current order of the images the server actually
    // covered (filenames without embeddings are dropped from the response).
    const coveredSet = new Set(orderedIds);
    const coveredCurrent = targetOrder.filter((fn) => coveredSet.has(fn));
    const unchanged =
      orderedIds.length === coveredCurrent.length &&
      orderedIds.every((fn, i) => fn === coveredCurrent[i]);
    showToast(
      unchanged
        ? `${selectionScope ? "Selection" : "Ungrouped"} order already optimal`
        : `Sorted ${orderedIds.length} ${selectionScope ? "selected" : "ungrouped"} images by similarity`,
      "success",
    );
  }

  // Gallery-only reorder (like Groups to Top): similar groups (or, with the
  // Ungrouped target, the loose images) end up adjacent, using the
  // merge-suggestions pairwise similarity. Nothing is persisted — Save renames
  // files and Save Order writes the group order to JSON.
  async function handleSortBySimilarity() {
    setSortingBySimilarity(true);
    setSortProgress("Starting similarity sort...");
    try {
      // The similarity computation reads groups from disk — flush pending edits.
      await flushPending();
      if (sortTarget === "ungrouped") await sortUngroupedBySimilarity();
      else await sortGroupsBySimilarity();
    } catch (err) {
      showToast(getErrorMessage(err, "Sort by similarity failed"), "error");
    } finally {
      setSortingBySimilarity(false);
      setSortProgress(null);
    }
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
          {(groups.length >= 2 || ungroupedCount >= 3 || selectedImageFilenames.length >= 3) && (
            <>
              <button
                className="btn btn-secondary"
                onClick={handleSortBySimilarity}
                disabled={
                  sortingBySimilarity ||
                  saving ||
                  (sortTarget === "groups"
                    ? groups.length < 2
                    : selectedImageFilenames.length > 0
                      ? selectedImageFilenames.length < 3
                      : ungroupedCount < 3)
                }
                title={`${
                  sortTarget === "ungrouped" && selectedImageFilenames.length > 0
                    ? `Sort only the ${selectedImageFilenames.length} selected images: they swap among their own slots (including within their groups); everything else stays put`
                    : SORT_TARGET_TITLES[sortTarget]
                }. Not persisted — use Save and Save Order to keep it`}
              >
                {sortingBySimilarity ? "Sorting..." : "Sort Similar"}
              </button>
              <select
                className="toolbar-select"
                value={sortTarget}
                onChange={(e) => handleSortTargetChange(e.target.value as SortTarget)}
                disabled={sortingBySimilarity}
                title={SORT_TARGET_TITLES[sortTarget]}
                aria-label="Sort Similar target"
              >
                <option value="groups" title={SORT_TARGET_TITLES.groups}>
                  Groups
                </option>
                <option value="ungrouped" title={SORT_TARGET_TITLES.ungrouped}>
                  {selectedImageFilenames.length > 0 ? "Selection" : "Ungrouped"}
                </option>
              </select>
              <select
                className="toolbar-select"
                value={sortMode}
                onChange={(e) => handleSortModeChange(e.target.value as GroupOrderMode)}
                disabled={sortingBySimilarity}
                title={SORT_MODE_TITLES[sortMode]}
                aria-label="Sort Similar algorithm"
              >
                <option value="tree" title={SORT_MODE_TITLES.tree}>
                  Tree
                </option>
                <option value="chain" title={SORT_MODE_TITLES.chain}>
                  Chain
                </option>
                <option value="spectral" title={SORT_MODE_TITLES.spectral}>
                  Spectral
                </option>
                <option value="minimal" title={SORT_MODE_TITLES.minimal}>
                  Minimal
                </option>
              </select>
              {sortMode === "minimal" && (
                <input
                  type="number"
                  className="toolbar-input-number"
                  value={minimalLocality}
                  min={1}
                  max={50}
                  disabled={sortingBySimilarity}
                  title="Max positions a group may move from its original slot (Minimal mode)"
                  aria-label="Minimal locality"
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (Number.isFinite(v) && v >= 1) handleMinimalLocalityChange(v);
                  }}
                />
              )}
            </>
          )}
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
