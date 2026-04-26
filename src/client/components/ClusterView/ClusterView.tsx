import { useVirtualizer } from "@tanstack/react-virtual";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRemeasureVirtualRows } from "../../hooks/useRemeasureVirtualRows.ts";
import { filenamesFromSelectedImages, useClusterStore } from "../../stores/clusterStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useNNQueryStore } from "../../stores/nnQueryStore.ts";
import { useUIStore } from "../../stores/uiStore.ts";
import type { ClusterData, ClusterResultData } from "../../types.ts";
import { Lightbox } from "../Lightbox.tsx";
import { SearchOverlay, useSearchOverlayState } from "../SearchBar.tsx";
import { ClusterCard } from "./ClusterCard.tsx";
import { MergeBar } from "./MergeBar.tsx";
import { NNResultsModal } from "./NNResultsModal.tsx";
import { ScopeBanner } from "./ScopeBanner.tsx";

function getClusterSubtitle(
  clusterData: ClusterData | null,
  visibleCount: number,
  groupCount: number,
  loading: boolean,
): string {
  if (clusterData?.scope) {
    const { scope } = clusterData;
    return `Scoped: ${scope.groupIds.length} groups · ${scope.nImages} images`;
  }
  if (clusterData) return `${visibleCount} clusters — ${groupCount} groups`;
  return loading ? "Loading..." : "Run clustering to start";
}

const EMPTY_CLUSTERS: ClusterResultData[] = [];

export function ClusterView() {
  const clusterData = useClusterStore((s) => s.clusterData);
  const loading = useClusterStore((s) => s.loading);
  const mergeSelection = useClusterStore((s) => s.mergeSelection);
  const selectedImages = useClusterStore((s) => s.selectedImages);
  const collapsedClusters = useClusterStore((s) => s.collapsedClusters);
  const lightbox = useClusterStore((s) => s.lightbox);
  const treeStale = useClusterStore((s) => s.treeStale);
  const focusedClusterId = useClusterStore((s) => s.focusedClusterId);
  const fetchClusters = useClusterStore((s) => s.fetchClusters);
  const runScopedCluster = useClusterStore((s) => s.runScopedCluster);
  const toggleMergeSelect = useClusterStore((s) => s.toggleMergeSelect);
  const clearMergeSelection = useClusterStore((s) => s.clearMergeSelection);
  const mergeSelectedClusters = useClusterStore((s) => s.mergeSelectedClusters);
  const toggleImageSelect = useClusterStore((s) => s.toggleImageSelect);
  const rangeSelectImages = useClusterStore((s) => s.rangeSelectImages);
  const clearImageSelection = useClusterStore((s) => s.clearImageSelection);
  const splitSelected = useClusterStore((s) => s.splitSelected);
  const dismissCluster = useClusterStore((s) => s.dismissCluster);
  const toggleCollapsed = useClusterStore((s) => s.toggleCollapsed);
  const openLightbox = useClusterStore((s) => s.openLightbox);
  const closeLightbox = useClusterStore((s) => s.closeLightbox);
  const acceptCluster = useClusterStore((s) => s.acceptCluster);
  const addToGroup = useClusterStore((s) => s.addToGroup);
  const loadCachedClusters = useClusterStore((s) => s.loadCachedClusters);
  const moveFocus = useClusterStore((s) => s.moveFocus);

  const groups = useGroupStore((s) => s.groups);
  const fetchGroups = useGroupStore((s) => s.fetchGroups);
  const setHeaderSubtitle = useUIStore((s) => s.setHeaderSubtitle);

  const unsortedClusters = clusterData?.clusters ?? EMPTY_CLUSTERS;

  // Sort: suggestions to existing groups first, then by original order
  const visibleClusters = React.useMemo(() => {
    if (unsortedClusters.length === 0) return unsortedClusters;
    const withGroup: ClusterResultData[] = [];
    const withoutGroup: ClusterResultData[] = [];
    const sectionById = new Map<string, "group" | "none">();
    for (const c of unsortedClusters) {
      const inGroup =
        !!c.confirmedGroup || (c.splitFrom != null && sectionById.get(c.splitFrom) === "group");
      sectionById.set(c.id, inGroup ? "group" : "none");
      if (inGroup) withGroup.push(c);
      else withoutGroup.push(c);
    }
    if (withGroup.length === 0) return unsortedClusters;
    return [...withGroup, ...withoutGroup];
  }, [unsortedClusters]);

  // Ensure groups are loaded before any cluster operations can modify them
  // biome-ignore lint/correctness/useExhaustiveDependencies: mount-only — fetchGroups is a stable Zustand action
  useEffect(() => {
    fetchGroups();
  }, []);

  // Auto-load cached clusters on mount
  // biome-ignore lint/correctness/useExhaustiveDependencies: mount-only — loadCachedClusters is a stable Zustand action
  useEffect(() => {
    loadCachedClusters();
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: setHeaderSubtitle is a stable Zustand action
  useEffect(() => {
    setHeaderSubtitle(
      getClusterSubtitle(clusterData, visibleClusters.length, groups.length, loading),
    );
    return () => setHeaderSubtitle("");
  }, [visibleClusters.length, groups.length, clusterData, loading]);

  // Virtualization
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: visibleClusters.length,
    getScrollElement: () => scrollContainerRef.current,
    getItemKey: (index) => visibleClusters[index]?.id ?? index,
    estimateSize: (index) => {
      const cluster = visibleClusters[index];
      if (!cluster || collapsedClusters.has(cluster.id)) return 56;
      const containerWidth = scrollContainerRef.current?.clientWidth ?? 960;
      const cols = Math.max(1, Math.floor(containerWidth / 168)); // 160px min + 8px gap
      const thumbRows = Math.ceil(cluster.images.length / cols);
      // .cluster-thumbs has max-height: 520px, so cap the grid contribution.
      const thumbGridHeight = Math.min(520, thumbRows * 176);
      // +~70px when rendering both confirmed + suggested sections, else ~32px padding.
      const sectionsOverhead = cluster.confirmedGroup ? 70 : 32;
      return 56 + thumbGridHeight + sectionsOverhead;
    },
    measureElement: (el) => el.getBoundingClientRect().height,
    overscan: 3,
  });

  useRemeasureVirtualRows(virtualizer, scrollContainerRef, [visibleClusters, collapsedClusters]);

  const search = useSearchOverlayState();
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);

  const normalizedQuery = useMemo(() => search.query.trim().toLowerCase(), [search.query]);

  const matchRowIndices = useMemo(() => {
    if (!normalizedQuery) return [] as number[];
    const indices: number[] = [];
    for (let i = 0; i < visibleClusters.length; i++) {
      const c = visibleClusters[i]!;
      const nameHit =
        c.autoName.toLowerCase().includes(normalizedQuery) ||
        (c.confirmedGroup?.name.toLowerCase().includes(normalizedQuery) ?? false);
      const fileHit = !nameHit && c.images.some((f) => f.toLowerCase().includes(normalizedQuery));
      if (nameHit || fileHit) indices.push(i);
    }
    return indices;
  }, [visibleClusters, normalizedQuery]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: query is the intentional trigger; body only resets the cursor
  useEffect(() => {
    setCurrentMatchIndex(0);
  }, [normalizedQuery]);

  const clampedMatchIndex =
    matchRowIndices.length === 0 ? 0 : Math.min(currentMatchIndex, matchRowIndices.length - 1);
  const currentMatchRowIndex = matchRowIndices[clampedMatchIndex];
  const currentMatchClusterId =
    currentMatchRowIndex !== undefined ? (visibleClusters[currentMatchRowIndex]?.id ?? null) : null;

  const currentMatchFilenames = useMemo(() => {
    if (currentMatchRowIndex === undefined || !normalizedQuery) return undefined;
    const cluster = visibleClusters[currentMatchRowIndex];
    if (!cluster) return undefined;
    const matches = new Set<string>();
    for (const f of cluster.images) {
      if (f.toLowerCase().includes(normalizedQuery)) matches.add(f);
    }
    return matches;
  }, [currentMatchRowIndex, visibleClusters, normalizedQuery]);

  useEffect(() => {
    if (currentMatchRowIndex !== undefined) {
      virtualizer.scrollToIndex(currentMatchRowIndex, { align: "center" });
    }
  }, [currentMatchRowIndex, virtualizer]);

  const goNextMatch = useCallback(() => {
    if (matchRowIndices.length === 0) return;
    setCurrentMatchIndex((i) => (i + 1) % matchRowIndices.length);
  }, [matchRowIndices.length]);

  const goPrevMatch = useCallback(() => {
    if (matchRowIndices.length === 0) return;
    setCurrentMatchIndex((i) => (i - 1 + matchRowIndices.length) % matchRowIndices.length);
  }, [matchRowIndices.length]);

  // Focus scrolling via virtualizer
  // biome-ignore lint/correctness/useExhaustiveDependencies: focusedClusterId is the intentional trigger; virtualizer/visibleClusters are stable between renders
  useEffect(() => {
    if (focusedClusterId) {
      const idx = visibleClusters.findIndex((c) => c.id === focusedClusterId);
      if (idx !== -1) virtualizer.scrollToIndex(idx, { align: "auto" });
    }
  }, [focusedClusterId]);

  // Keyboard shortcuts
  // biome-ignore lint/correctness/useExhaustiveDependencies: all handlers are stable Zustand actions; visibleClusters/focusedClusterId read via closure are intentionally not deps to avoid re-registering on every cluster change
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (useClusterStore.getState().lightbox) return;

      switch (e.key) {
        case "Escape":
          if (mergeSelection.size > 0) clearMergeSelection();
          else if (selectedImages.size > 0) clearImageSelection();
          break;
        case "ArrowDown":
          if (e.metaKey) break; // let browser handle Cmd+Down (scroll to bottom)
          e.preventDefault();
          moveFocus(1);
          break;
        case "j":
          e.preventDefault();
          moveFocus(1);
          break;
        case "ArrowUp":
          if (e.metaKey) break; // let browser handle Cmd+Up (scroll to top)
          e.preventDefault();
          moveFocus(-1);
          break;
        case "k":
          e.preventDefault();
          moveFocus(-1);
          break;
        case "Enter": {
          const focused = focusedClusterId;
          if (focused) toggleCollapsed(focused);
          break;
        }
        case "g":
        case "G": {
          const cluster = visibleClusters.find((c) => c.id === focusedClusterId);
          if (cluster && !cluster.confirmedGroup) acceptCluster(cluster);
          break;
        }
        case "d":
        case "D": {
          if (focusedClusterId) {
            dismissCluster(focusedClusterId);
            moveFocus(1);
          }
          break;
        }
        case "a":
        case "A": {
          const cluster = visibleClusters.find((c) => c.id === focusedClusterId);
          if (cluster?.confirmedGroup) addToGroup(cluster);
          break;
        }
        case "f":
        case "F": {
          // Uppercase "F" or shift-modified "f" → NN for the cross-cluster selection.
          // Lowercase "f" without shift → NN for the currently focused cluster.
          const wantSelection = e.shiftKey || e.key === "F";
          if (wantSelection) {
            if (selectedImages.size === 0) break;
            useNNQueryStore
              .getState()
              .openForSelection(filenamesFromSelectedImages(selectedImages));
          } else {
            const cluster = visibleClusters.find((c) => c.id === focusedClusterId);
            if (cluster) useNNQueryStore.getState().openForCluster(cluster);
          }
          break;
        }
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [mergeSelection.size, selectedImages.size, focusedClusterId, visibleClusters]);

  function renderLightbox() {
    if (!lightbox) return null;
    const cluster = clusterData?.clusters.find((c) => c.id === lightbox.clusterId);
    if (!cluster) return null;
    return (
      <Lightbox
        images={cluster.images.map((f) => ({ filename: f }))}
        initialIndex={lightbox.imageIndex}
        onClose={closeLightbox}
      />
    );
  }

  const virtualItems = virtualizer.getVirtualItems();

  return (
    <div className="cluster-view">
      <ScopeBanner />
      <SearchOverlay
        isOpen={search.isOpen}
        query={search.query}
        setQuery={search.setQuery}
        open={search.open}
        close={search.close}
        matchCount={matchRowIndices.length}
        currentMatchIndex={clampedMatchIndex}
        onNext={goNextMatch}
        onPrev={goPrevMatch}
        placeholder="Search clusters or filenames..."
      />
      {mergeSelection.size > 0 && (
        <MergeBar
          selection={mergeSelection}
          clusters={clusterData?.clusters ?? []}
          onMerge={mergeSelectedClusters}
          onCancel={clearMergeSelection}
          onRemove={toggleMergeSelect}
        />
      )}

      {!clusterData && !loading ? (
        <div className="cluster-empty-state">
          <div className="cluster-empty-icon">&#x2728;</div>
          <div className="cluster-empty-title">No clusters yet</div>
          <div className="cluster-empty-desc">
            Click "Run Clustering" to analyze images and group them by visual similarity
          </div>
        </div>
      ) : (
        <div ref={scrollContainerRef} className="cluster-list">
          <div style={{ height: virtualizer.getTotalSize(), width: "100%", position: "relative" }}>
            {virtualItems.map((virtualItem) => {
              const cluster = visibleClusters[virtualItem.index];
              if (!cluster) return null;
              return (
                <div
                  key={cluster.id}
                  data-index={virtualItem.index}
                  ref={virtualizer.measureElement}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    transform: `translateY(${virtualItem.start}px)`,
                  }}
                >
                  <ClusterCard
                    cluster={cluster}
                    collapsed={collapsedClusters.has(cluster.id)}
                    mergeSelected={mergeSelection.has(cluster.id)}
                    focused={focusedClusterId === cluster.id}
                    selectedImages={selectedImages}
                    isCurrentSearchMatch={cluster.id === currentMatchClusterId}
                    searchMatchFilenames={
                      cluster.id === currentMatchClusterId ? currentMatchFilenames : undefined
                    }
                    onToggleCollapse={() => toggleCollapsed(cluster.id)}
                    onMergeSelect={(e) => {
                      if (e.metaKey || e.ctrlKey) toggleMergeSelect(cluster.id);
                    }}
                    onImageSelect={(filename) => toggleImageSelect(cluster.id, filename)}
                    onImageRangeSelect={(index) => rangeSelectImages(cluster.id, index)}
                    onAccept={() => acceptCluster(cluster)}
                    onAddToGroup={() => addToGroup(cluster)}
                    onDismiss={() => dismissCluster(cluster.id)}
                    onOpenLightbox={(index) => openLightbox(cluster.id, index)}
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {selectedImages.size > 0 && (
        <div className="cluster-selection-bar">
          <span>
            {selectedImages.size} image{selectedImages.size > 1 ? "s" : ""} selected
          </span>
          <button className="btn btn-accent" onClick={splitSelected}>
            Split to New Cluster
          </button>
          <button
            className="btn btn-secondary"
            onClick={() =>
              useNNQueryStore
                .getState()
                .openForSelection(filenamesFromSelectedImages(selectedImages))
            }
            title="Find the nearest images to this selection (Shift+F)"
          >
            Find Nearest
          </button>
          <button className="btn" onClick={clearImageSelection}>
            Deselect
          </button>
        </div>
      )}

      {treeStale && clusterData && (
        <div className="cluster-stale-banner">
          Groups changed —{" "}
          <button
            className="btn btn-small btn-primary"
            onClick={() =>
              clusterData.scope
                ? runScopedCluster(clusterData.scope.groupIds, { nClusters: clusterData.nClusters })
                : fetchClusters(clusterData.nClusters)
            }
          >
            {clusterData.scope ? "Re-run scoped" : "Re-run clustering"}
          </button>{" "}
          to incorporate new groups as seeds
        </div>
      )}

      {renderLightbox()}
      <NNResultsModal />
    </div>
  );
}
