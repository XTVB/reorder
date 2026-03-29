import React, { useMemo, useState, useCallback } from "react";
import { Toolbar } from "../Toolbar.tsx";
import { Toast } from "../Toast.tsx";
import { useTagStore } from "../../stores/tagStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useUIStore } from "../../stores/uiStore.ts";
import { cn, imageUrl, resolveFilterMode } from "../../utils/helpers.ts";
import { computeGroupTagProfiles, computeSimilarityMatrix } from "../../utils/tagSimilarity.ts";
import { applyFilters } from "../../utils/tagIndex.ts";
import { FilterBar } from "../TagExplorer/FilterBar.tsx";
import { TagBrowserPanel } from "../TagExplorer/TagBrowserPanel.tsx";
import { useTagViewInit } from "../../hooks/useTagViewInit.ts";

type MergeMode = "filter" | "similarity";

export function MergeView() {
  useTagViewInit();

  const showToast = useUIStore((s) => s.showToast);

  const hasDb = useTagStore((s) => s.hasDb);
  const loading = useTagStore((s) => s.loading);
  const indexReady = useTagStore((s) => s.indexReady);
  const tagData = useTagStore((s) => s.tagData);
  const invertedIndex = useTagStore((s) => s.invertedIndex);

  const mergeFilters = useTagStore((s) => s.mergeFilters);
  const addMergeFilter = useTagStore((s) => s.addMergeFilter);
  const removeMergeFilter = useTagStore((s) => s.removeMergeFilter);
  const setMergeFilterMode = useTagStore((s) => s.setMergeFilterMode);
  const clearMergeFilters = useTagStore((s) => s.clearMergeFilters);
  const selectedGroupIds = useTagStore((s) => s.selectedGroupIds);
  const toggleGroupSelection = useTagStore((s) => s.toggleGroupSelection);
  const clearGroupSelection = useTagStore((s) => s.clearGroupSelection);

  const images = useImageStore((s) => s.images);
  const groups = useGroupStore((s) => s.groups);
  const updateGroups = useGroupStore((s) => s.updateGroups);

  const [mode, setMode] = useState<MergeMode>("filter");
  const [expandedGroupId, setExpandedGroupId] = useState<string | null>(null);
  const [tagBrowserOpen, setTagBrowserOpen] = useState(false);

  const groupMap = useMemo(
    () => new Map(groups.map((g) => [g.id, g])),
    [groups],
  );

  const allFilenames = useMemo(() => images.map((i) => i.filename), [images]);

  const filteredImageSet = useMemo(() => {
    if (mergeFilters.length === 0) return null;
    const result = applyFilters(allFilenames, invertedIndex, mergeFilters, null, "all");
    return new Set(result);
  }, [allFilenames, invertedIndex, mergeFilters]);

  const groupsWithMatchCounts = useMemo(() => {
    return groups.map((g) => {
      if (!filteredImageSet) return { group: g, matchCount: g.images.length, total: g.images.length };
      const matchCount = g.images.filter((fn) => filteredImageSet.has(fn)).length;
      return { group: g, matchCount, total: g.images.length };
    }).sort((a, b) => b.matchCount - a.matchCount);
  }, [groups, filteredImageSet]);

  const similarityPairs = useMemo(() => {
    if (!indexReady || mode !== "similarity") return [];
    const profiles = computeGroupTagProfiles(groups, tagData);
    return computeSimilarityMatrix(profiles);
  }, [groups, tagData, indexReady, mode]);

  const handleMerge = useCallback(() => {
    if (selectedGroupIds.size < 2) return;
    const selected = [...selectedGroupIds];
    const first = groupMap.get(selected[0]);
    const name = prompt("Name for merged group:", first?.name ?? "Merged Group");
    if (!name) return;

    const allImages: string[] = [];
    const seen = new Set<string>();
    for (const gid of selected) {
      const g = groupMap.get(gid);
      if (!g) continue;
      for (const fn of g.images) {
        if (!seen.has(fn)) { seen.add(fn); allImages.push(fn); }
      }
    }

    updateGroups((prev) => {
      const remaining = prev.filter((g) => !selectedGroupIds.has(g.id));
      return [...remaining, { id: crypto.randomUUID(), name, images: allImages }];
    });

    clearGroupSelection();
    showToast(`Merged ${selected.length} groups into "${name}"`, "success");
  }, [selectedGroupIds, groupMap, updateGroups, clearGroupSelection, showToast]);

  const handleTagClick = useCallback((category: string, value: string, e: React.MouseEvent) => {
    const existing = mergeFilters.find((f) => f.category === category && f.value === value);
    if (existing) {
      removeMergeFilter(category, value);
    } else {
      addMergeFilter(category, value, resolveFilterMode(e));
    }
  }, [mergeFilters, addMergeFilter, removeMergeFilter]);

  if (!hasDb || !indexReady) {
    return (
      <>
        <Toolbar />
        <div className="merge-view">
          <div className="tag-explorer-empty">
            {loading ? (
              <>
                <div className="spinner" />
                Loading tags...
              </>
            ) : (
              <p>No tags database found. Switch to Tags view to ingest a tags file first.</p>
            )}
          </div>
        </div>
        <Toast />
      </>
    );
  }

  return (
    <>
      <Toolbar />
      <div className="merge-view">
        <div className="merge-toolbar">
          <div className="merge-toolbar-left">
            {mode === "filter" ? (
              <FilterBar
                filters={mergeFilters}
                onRemove={removeMergeFilter}
                onModeChange={setMergeFilterMode}
                onClear={clearMergeFilters}
              />
            ) : (
              <span style={{ fontSize: 13, color: "var(--text-muted)" }}>
                Groups ranked by tag similarity (Jaccard)
              </span>
            )}
          </div>
          <div style={{ display: "flex", gap: 4, flexShrink: 0 }}>
            {mode === "filter" && (
              <button
                className={cn("btn", tagBrowserOpen ? "btn-primary" : "btn-secondary")}
                style={{ padding: "6px 12px", fontSize: "12px" }}
                onClick={() => setTagBrowserOpen(!tagBrowserOpen)}
              >
                Tags
              </button>
            )}
            <div className="mode-toggle">
              <button
                className={cn("mode-toggle-btn", mode === "filter" && "mode-toggle-active")}
                onClick={() => setMode("filter")}
              >
                Filter
              </button>
              <button
                className={cn("mode-toggle-btn", mode === "similarity" && "mode-toggle-active")}
                onClick={() => setMode("similarity")}
              >
                Similarity
              </button>
            </div>
          </div>
        </div>

        <div className="merge-view-split">
          <div className="merge-view-body">
            {mode === "filter" ? (
              groupsWithMatchCounts.map(({ group, matchCount, total }) => {
                const isSelected = selectedGroupIds.has(group.id);
                const isDimmed = mergeFilters.length > 0 && matchCount === 0;
                const isExpanded = expandedGroupId === group.id;

                return (
                  <div key={group.id}>
                    <div
                      className={cn(
                        "merge-group-row",
                        isSelected && "merge-group-row-selected",
                        isDimmed && "group-list-item-dimmed",
                      )}
                      onClick={() => setExpandedGroupId(isExpanded ? null : group.id)}
                    >
                      <input
                        type="checkbox"
                        className="merge-group-checkbox"
                        checked={isSelected}
                        onChange={(e) => { e.stopPropagation(); toggleGroupSelection(group.id); }}
                        onClick={(e) => e.stopPropagation()}
                      />
                      <span className="merge-group-name">{group.name}</span>
                      <div className="merge-group-thumbs">
                        {group.images.slice(0, 4).map((fn) => (
                          <img key={fn} src={imageUrl(fn)} alt="" loading="lazy" />
                        ))}
                      </div>
                      <span className="merge-group-stats">
                        {mergeFilters.length > 0 ? (
                          <span className={matchCount > 0 ? "group-list-item-match" : ""}>
                            {matchCount}/{total}
                          </span>
                        ) : (
                          `${total} images`
                        )}
                      </span>
                    </div>
                    {isExpanded && (
                      <div style={{ padding: "8px 16px 16px 48px" }}>
                        <div className="focused-group-grid" style={{ maxHeight: 200 }}>
                          {group.images.map((fn) => (
                            <img key={fn} src={imageUrl(fn)} alt={fn} loading="lazy" />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              similarityPairs.length === 0 ? (
                <div style={{ padding: 32, textAlign: "center", color: "var(--text-muted)" }}>
                  {groups.length < 2 ? "Need at least 2 groups to compute similarity." : "No similar group pairs found."}
                </div>
              ) : (
                similarityPairs.map((pair, idx) => {
                  const gA = groupMap.get(pair.groupA);
                  const gB = groupMap.get(pair.groupB);
                  if (!gA || !gB) return null;
                  const bothSelected = selectedGroupIds.has(pair.groupA) && selectedGroupIds.has(pair.groupB);

                  return (
                    <div
                      key={idx}
                      className={cn("merge-similarity-pair", bothSelected && "merge-group-row-selected")}
                      onClick={() => {
                        toggleGroupSelection(pair.groupA);
                        toggleGroupSelection(pair.groupB);
                      }}
                    >
                      <span className="merge-similarity-score">
                        {Math.round(pair.score * 100)}%
                      </span>
                      <div className="merge-similarity-names">
                        <strong>{gA.name}</strong> ↔ <strong>{gB.name}</strong>
                        <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                          {pair.sharedTags.slice(0, 8).map((t) => t.split(":")[1]?.replace(/_/g, " ")).join(", ")}
                          {pair.sharedTags.length > 8 && ` +${pair.sharedTags.length - 8} more`}
                        </div>
                      </div>
                      <span className="merge-group-stats">
                        {gA.images.length} + {gB.images.length} images
                      </span>
                    </div>
                  );
                })
              )
            )}
          </div>

          {tagBrowserOpen && mode === "filter" && (
            <div className="tag-right-panel">
              <TagBrowserPanel
                invertedIndex={invertedIndex}
                filteredFilenames={allFilenames}
                filters={mergeFilters}
                onTagClick={handleTagClick}
              />
            </div>
          )}
        </div>

        {selectedGroupIds.size >= 2 && (
          <div className="merge-actions">
            <span style={{ fontSize: 13 }}>
              {selectedGroupIds.size} groups selected
            </span>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn btn-secondary" onClick={clearGroupSelection}>
                Clear Selection
              </button>
              <button className="btn btn-primary" onClick={handleMerge}>
                Merge {selectedGroupIds.size} Groups
              </button>
            </div>
          </div>
        )}
      </div>

      <Toast />
    </>
  );
}
