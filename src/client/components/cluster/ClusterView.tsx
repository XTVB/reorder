import { useVirtualizer } from "@tanstack/react-virtual";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRemeasureVirtualRows } from "../../hooks/useRemeasureVirtualRows.ts";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useSessionStore } from "../../stores/core/sessionStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import {
  clusterShapeHash,
  filenamesFromSelectedImages,
  useCompareStore,
  useExpandStore,
  useInteractionsStore,
  useListStore,
  useMetricsStore,
  useSplitStore,
} from "../../stores/modes/cluster/index.ts";
import { useNNQueryStore } from "../../stores/nnQueryStore.ts";
import type { ClusterData, ClusterResultData, SplitChildren } from "../../types.ts";
import { Lightbox } from "../shared/Lightbox.tsx";
import { SearchOverlay, useSearchOverlayState } from "../shared/SearchBar.tsx";
import { ClusterCard } from "./ClusterCard.tsx";
import { ComparePanel } from "./ComparePanel.tsx";
import { ExpandModal } from "./ExpandModal.tsx";
import { MergeBar } from "./MergeBar.tsx";
import { NNResultsModal } from "./NNResultsModal.tsx";
import { ScopeBanner } from "./ScopeBanner.tsx";

interface DisplayEntry {
  cluster: ClusterResultData;
  depth: number;
}

function flattenWithSplits(
  clusters: ClusterResultData[],
  splitChildren: Record<string, SplitChildren>,
): DisplayEntry[] {
  const out: DisplayEntry[] = [];
  function visit(c: ClusterResultData, depth: number) {
    out.push({ cluster: c, depth });
    const kids = splitChildren[c.id];
    if (kids) {
      visit(kids.childA, depth + 1);
      visit(kids.childB, depth + 1);
    }
  }
  for (const c of clusters) visit(c, 0);
  return out;
}

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
  const clusterData = useListStore((s) => s.clusterData);
  const loading = useListStore((s) => s.loading);
  const collapsedClusters = useListStore((s) => s.collapsedClusters);
  const treeStale = useListStore((s) => s.treeStale);
  const focusedClusterId = useListStore((s) => s.focusedClusterId);
  const splitChildren = useListStore((s) => s.splitChildren);
  const fetchClusters = useListStore((s) => s.fetchClusters);
  const runScopedCluster = useListStore((s) => s.runScopedCluster);
  const dismissCluster = useListStore((s) => s.dismissCluster);
  const toggleCollapsed = useListStore((s) => s.toggleCollapsed);
  const loadCachedClusters = useListStore((s) => s.loadCachedClusters);
  const moveFocus = useListStore((s) => s.moveFocus);

  const mergeSelection = useSelectionStore((s) => s.contexts["cluster:merge"]);
  const selectedImages = useSelectionStore((s) => s.contexts["cluster:images"]);

  const toggleMergeSelect = useInteractionsStore((s) => s.toggleMergeSelect);
  const clearMergeSelection = useInteractionsStore((s) => s.clearMergeSelection);
  const mergeSelectedClusters = useInteractionsStore((s) => s.mergeSelectedClusters);
  const toggleImageSelect = useInteractionsStore((s) => s.toggleImageSelect);
  const rangeSelectImages = useInteractionsStore((s) => s.rangeSelectImages);
  const clearImageSelection = useInteractionsStore((s) => s.clearImageSelection);
  const splitSelected = useInteractionsStore((s) => s.splitSelected);
  const acceptCluster = useInteractionsStore((s) => s.acceptCluster);
  const addToGroup = useInteractionsStore((s) => s.addToGroup);

  const metrics = useMetricsStore((s) => s.metrics);
  const refreshMetrics = useMetricsStore((s) => s.refreshMetrics);

  const compare = useCompareStore((s) => s.compare);
  const openCompare = useCompareStore((s) => s.openCompare);

  const expand = useExpandStore((s) => s.expand);
  const openExpand = useExpandStore((s) => s.openExpand);

  const toggleSplit = useSplitStore((s) => s.toggleSplit);

  const lightboxOpen = useLightboxStore((s) => s.open && s.source === "cluster");
  const lightboxIndex = useLightboxStore((s) => s.index);
  const lightboxFilenames = useLightboxStore((s) => s.filenames);
  const closeLightbox = useLightboxStore((s) => s.close);

  const groups = useGroupStore((s) => s.groups);
  const fetchGroups = useGroupStore((s) => s.fetchGroups);
  const setHeaderSubtitle = useSessionStore((s) => s.setHeaderSubtitle);

  const unsortedClusters = clusterData?.clusters ?? EMPTY_CLUSTERS;

  // Sort: suggestions to existing groups first, then by original order
  const sortedTopLevel = React.useMemo(() => {
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

  const displayEntries = React.useMemo(
    () => flattenWithSplits(sortedTopLevel, splitChildren),
    [sortedTopLevel, splitChildren],
  );
  const visibleClusters = React.useMemo(
    () => displayEntries.map((e) => e.cluster),
    [displayEntries],
  );

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

  const clusterShape = clusterShapeHash(clusterData);
  // biome-ignore lint/correctness/useExhaustiveDependencies: refreshMetrics is a stable Zustand action; clusterShape captures membership changes — we deliberately ignore identity-only mutations like rename
  useEffect(() => {
    if (!clusterData) return;
    refreshMetrics();
  }, [clusterShape, splitChildren]);

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

  // biome-ignore lint/correctness/useExhaustiveDependencies: handlers are stable Zustand actions; current state is read via getState() inside the handler so the listener can attach once on mount
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (useLightboxStore.getState().open) return;

      const list = useListStore.getState();
      const sel = useSelectionStore.getState();
      const focusedId = list.focusedClusterId;
      const currentClusters = list.clusterData?.clusters ?? [];
      const mergeSize = sel.contexts["cluster:merge"].size;
      const selectedSize = sel.contexts["cluster:images"].size;

      switch (e.key) {
        case "Escape":
          if (mergeSize > 0) clearMergeSelection();
          else if (selectedSize > 0) clearImageSelection();
          break;
        case "ArrowDown":
          if (e.metaKey) break;
          e.preventDefault();
          moveFocus(1);
          break;
        case "j":
          e.preventDefault();
          moveFocus(1);
          break;
        case "ArrowUp":
          if (e.metaKey) break;
          e.preventDefault();
          moveFocus(-1);
          break;
        case "k":
          e.preventDefault();
          moveFocus(-1);
          break;
        case "Enter": {
          if (focusedId) toggleCollapsed(focusedId);
          break;
        }
        case "g":
        case "G": {
          const cluster = currentClusters.find((c) => c.id === focusedId);
          if (cluster && !cluster.confirmedGroup) acceptCluster(cluster);
          break;
        }
        case "d":
        case "D": {
          if (focusedId) {
            dismissCluster(focusedId);
            moveFocus(1);
          }
          break;
        }
        case "a":
        case "A": {
          const cluster = currentClusters.find((c) => c.id === focusedId);
          if (cluster?.confirmedGroup) addToGroup(cluster);
          break;
        }
        case "f":
        case "F": {
          const wantSelection = e.shiftKey || e.key === "F";
          if (wantSelection) {
            const selectedSet = sel.contexts["cluster:images"];
            if (selectedSet.size === 0) break;
            useNNQueryStore.getState().openForSelection(filenamesFromSelectedImages(selectedSet));
          } else {
            const cluster = currentClusters.find((c) => c.id === focusedId);
            if (cluster) useNNQueryStore.getState().openForCluster(cluster);
          }
          break;
        }
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  function renderLightbox() {
    if (!lightboxOpen) return null;
    return (
      <Lightbox
        filenames={lightboxFilenames}
        initialIndex={lightboxIndex}
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
              const entry = displayEntries[virtualItem.index];
              const depth = entry?.depth ?? 0;
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
                    metrics={metrics[cluster.id]}
                    splitExpanded={!!splitChildren[cluster.id]}
                    depth={depth}
                    onToggleCollapse={() => toggleCollapsed(cluster.id)}
                    onMergeSelect={(e) => {
                      if (e.metaKey || e.ctrlKey) toggleMergeSelect(cluster.id);
                    }}
                    onImageSelect={(filename, section) =>
                      toggleImageSelect(cluster.id, filename, section)
                    }
                    onImageRangeSelect={(filename, section) =>
                      rangeSelectImages(cluster.id, filename, section)
                    }
                    onAccept={() => acceptCluster(cluster)}
                    onAddToGroup={() => addToGroup(cluster)}
                    onDismiss={() => dismissCluster(cluster.id)}
                    onOpenCompare={() => openCompare(cluster.id)}
                    onToggleSplit={() => toggleSplit(cluster.id)}
                    onOpenExpand={() => openExpand(cluster.id)}
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
      {compare && <ComparePanel />}
      {expand && <ExpandModal />}
    </div>
  );
}
