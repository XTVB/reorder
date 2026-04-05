import { useVirtualizer } from "@tanstack/react-virtual";
import React, { useEffect, useRef } from "react";
import { useRemeasureVirtualRows } from "../../hooks/useRemeasureVirtualRows.ts";
import { useClusterStore } from "../../stores/clusterStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useUIStore } from "../../stores/uiStore.ts";
import type { ClusterResultData } from "../../types.ts";
import { postJson } from "../../utils/helpers.ts";
import { Lightbox } from "../Lightbox.tsx";
import { ClusterCard } from "./ClusterCard.tsx";
import { MergeBar } from "./MergeBar.tsx";

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
  const showToast = useUIStore((s) => s.showToast);
  const setHeaderSubtitle = useUIStore((s) => s.setHeaderSubtitle);

  const unsortedClusters = clusterData?.clusters ?? EMPTY_CLUSTERS;

  // Sort: suggestions to existing groups first, then by original order
  const visibleClusters = React.useMemo(() => {
    if (unsortedClusters.length === 0) return unsortedClusters;
    const withGroup: ClusterResultData[] = [];
    const withoutGroup: ClusterResultData[] = [];
    for (const c of unsortedClusters) {
      if (c.confirmedGroup) withGroup.push(c);
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

  // Update header subtitle
  // biome-ignore lint/correctness/useExhaustiveDependencies: setHeaderSubtitle is a stable Zustand action
  useEffect(() => {
    const subtitle = clusterData
      ? `${visibleClusters.length} clusters — ${groups.length} groups`
      : loading
        ? "Loading..."
        : "Run clustering to start";
    setHeaderSubtitle(subtitle);
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
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [mergeSelection.size, selectedImages.size, focusedClusterId, visibleClusters]);

  async function handleAskClaude(clusterId: string, images: string[], autoName: string) {
    try {
      const res = await postJson("/api/cluster/contact-sheet", {
        filenames: images,
        clusterName: autoName || clusterId,
      });
      const result: { path?: string } = await res.json();
      if (result.path) {
        await navigator.clipboard.writeText(result.path);
        showToast(`Contact sheet saved — path copied to clipboard`, "success");
      }
    } catch (err) {
      showToast(`Failed to generate contact sheet: ${err}`, "error");
    }
  }

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
                    onToggleCollapse={() => toggleCollapsed(cluster.id)}
                    onMergeSelect={(e) => {
                      if (e.metaKey || e.ctrlKey) toggleMergeSelect(cluster.id);
                    }}
                    onImageSelect={(filename) => toggleImageSelect(cluster.id, filename)}
                    onImageRangeSelect={(index) => rangeSelectImages(cluster.id, index)}
                    onAccept={() => acceptCluster(cluster)}
                    onAddToGroup={() => addToGroup(cluster)}
                    onAskClaude={() =>
                      handleAskClaude(cluster.id, cluster.images, cluster.autoName)
                    }
                    onDismiss={() => dismissCluster(cluster.id)}
                    onSplit={splitSelected}
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
          <button className="btn" onClick={clearImageSelection}>
            Deselect
          </button>
        </div>
      )}

      {treeStale && (
        <div className="cluster-stale-banner">
          Groups changed —{" "}
          <button
            className="btn btn-small btn-primary"
            onClick={() => fetchClusters(clusterData?.nClusters)}
          >
            Re-run clustering
          </button>{" "}
          to incorporate new groups as seeds
        </div>
      )}

      {renderLightbox()}
    </div>
  );
}
