import React, { useEffect, useMemo } from "react";
import { useClusterStore } from "../../stores/clusterStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import { useUIStore } from "../../stores/uiStore.ts";
import { ClusterCard } from "./ClusterCard.tsx";
import { ClusterToolbar } from "./ClusterToolbar.tsx";
import { MergeBar } from "./MergeBar.tsx";
import { Lightbox } from "../Lightbox.tsx";
import { postJson } from "../../utils/helpers.ts";
import { consolidateBlock } from "../../utils/reorder.ts";
import type { ClusterResultData } from "../../types.ts";

export function ClusterView() {
  const clusterData = useClusterStore((s) => s.clusterData);
  const loading = useClusterStore((s) => s.loading);
  const progress = useClusterStore((s) => s.progress);
  const mergeSelection = useClusterStore((s) => s.mergeSelection);
  const selectedImages = useClusterStore((s) => s.selectedImages);
  const collapsedClusters = useClusterStore((s) => s.collapsedClusters);
  const lightbox = useClusterStore((s) => s.lightbox);
  const treeStale = useClusterStore((s) => s.treeStale);
  const focusedClusterId = useClusterStore((s) => s.focusedClusterId);
  const fetchClusters = useClusterStore((s) => s.fetchClusters);
  const recutClusters = useClusterStore((s) => s.recutClusters);
  const toggleMergeSelect = useClusterStore((s) => s.toggleMergeSelect);
  const clearMergeSelection = useClusterStore((s) => s.clearMergeSelection);
  const mergeSelectedClusters = useClusterStore((s) => s.mergeSelectedClusters);
  const toggleImageSelect = useClusterStore((s) => s.toggleImageSelect);
  const rangeSelectImages = useClusterStore((s) => s.rangeSelectImages);
  const clearImageSelection = useClusterStore((s) => s.clearImageSelection);
  const splitSelected = useClusterStore((s) => s.splitSelected);
  const dismissCluster = useClusterStore((s) => s.dismissCluster);
  const toggleCollapsed = useClusterStore((s) => s.toggleCollapsed);
  const expandAll = useClusterStore((s) => s.expandAll);
  const collapseAll = useClusterStore((s) => s.collapseAll);
  const openLightbox = useClusterStore((s) => s.openLightbox);
  const closeLightbox = useClusterStore((s) => s.closeLightbox);
  const markTreeStale = useClusterStore((s) => s.markTreeStale);
  const moveFocus = useClusterStore((s) => s.moveFocus);

  const updateGroups = useGroupStore((s) => s.updateGroups);
  const images = useImageStore((s) => s.images);
  const setImages = useImageStore((s) => s.setImages);
  const showToast = useUIStore((s) => s.showToast);

  const visibleClusters = useMemo(
    () => clusterData?.clusters ?? [],
    [clusterData?.clusters],
  );

  const hasError = progress.startsWith("Error:");

  useEffect(() => {
    if (focusedClusterId) {
      document.getElementById(`cluster-${focusedClusterId}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [focusedClusterId]);

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
        case "j":
          e.preventDefault();
          moveFocus(1);
          break;
        case "ArrowUp":
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
          const cluster = visibleClusters.find(c => c.id === focusedClusterId);
          if (cluster && !cluster.confirmedGroup) handleAcceptCluster(cluster);
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
          const cluster = visibleClusters.find(c => c.id === focusedClusterId);
          if (cluster?.confirmedGroup) handleAddToGroup(cluster);
          break;
        }
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [mergeSelection.size, selectedImages.size, focusedClusterId, visibleClusters]);

  function handleAcceptCluster(cluster: ClusterResultData) {
    const name = cluster.autoName || `Cluster ${cluster.id}`;
    updateGroups((prev) => [
      ...prev,
      { id: crypto.randomUUID(), name, images: cluster.images },
    ]);
    const consolidated = consolidateBlock(images, new Set(cluster.images));
    setImages(consolidated);
    showToast(`Created group "${name}" with ${cluster.images.length} images`, "success");
    dismissCluster(cluster.id);
    markTreeStale();
  }

  function handleAcceptAll(minSize: number) {
    if (!clusterData) return;
    const eligible = clusterData.clusters
      .filter(c => !c.confirmedGroup && c.images.length >= minSize);

    if (eligible.length === 0) {
      showToast("No eligible clusters to accept", "warning");
      return;
    }

    const totalImages = eligible.reduce((n, c) => n + c.images.length, 0);
    if (!confirm(`Create ${eligible.length} groups from ${eligible.length} clusters (${totalImages} images total)?`)) {
      return;
    }

    const newGroups = eligible.map(c => {
      dismissCluster(c.id);
      return { id: crypto.randomUUID(), name: c.autoName || `Cluster ${c.id}`, images: c.images };
    });

    updateGroups((prev) => [...prev, ...newGroups]);
    showToast(`Created ${newGroups.length} groups`, "success");
    markTreeStale();
  }

  function handleAddToGroup(cluster: ClusterResultData) {
    if (!cluster.confirmedGroup) return;
    const groupId = cluster.confirmedGroup.id;
    const confirmedSet = new Set(cluster.confirmedGroup.images);
    const suggested = cluster.images.filter(f => !confirmedSet.has(f));

    const selectedInCluster = [...selectedImages]
      .filter(key => key.startsWith(`${cluster.id}:`))
      .map(key => key.slice(cluster.id.length + 1))
      .filter(f => !confirmedSet.has(f));
    const toAdd = selectedInCluster.length > 0 ? selectedInCluster : suggested;

    updateGroups((prev) =>
      prev.map(g =>
        g.id === groupId
          ? { ...g, images: [...g.images, ...toAdd] }
          : g
      )
    );
    showToast(`Added ${toAdd.length} images to "${cluster.confirmedGroup.name}"`, "success");
    clearImageSelection();
    markTreeStale();
  }

  async function handleAskClaude(cluster: ClusterResultData) {
    try {
      const res = await postJson("/api/cluster/contact-sheet", {
        filenames: cluster.images,
        clusterName: cluster.autoName || cluster.id,
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
    const cluster = clusterData?.clusters.find(c => c.id === lightbox.clusterId);
    if (!cluster) return null;
    return (
      <Lightbox
        images={cluster.images.map(f => ({ filename: f }))}
        initialIndex={lightbox.imageIndex}
        onClose={closeLightbox}
      />
    );
  }

  return (
    <div className="cluster-view">
      <ClusterToolbar
        loading={loading}
        progress={progress}
        nClusters={clusterData?.nClusters ?? 200}
        suggestedCounts={clusterData?.suggestedCounts ?? []}
        totalClusters={visibleClusters.length}
        hasError={hasError}
        onRun={fetchClusters}
        onRecut={recutClusters}
        onExpandAll={expandAll}
        onCollapseAll={collapseAll}
        onAcceptAll={handleAcceptAll}
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

      <div className="cluster-list">
        {visibleClusters.map((cluster) => (
          <ClusterCard
            key={cluster.id}
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
            onAccept={() => handleAcceptCluster(cluster)}
            onAddToGroup={() => handleAddToGroup(cluster)}
            onAskClaude={() => handleAskClaude(cluster)}
            onDismiss={() => dismissCluster(cluster.id)}
            onSplit={splitSelected}
            onOpenLightbox={(index) => openLightbox(cluster.id, index)}
          />
        ))}
      </div>

      {selectedImages.size > 0 && (
        <div className="cluster-selection-bar">
          <span>{selectedImages.size} image{selectedImages.size > 1 ? "s" : ""} selected</span>
          <button className="btn btn-accent" onClick={splitSelected}>Split to New Cluster</button>
          <button className="btn" onClick={clearImageSelection}>Deselect</button>
        </div>
      )}

      {treeStale && (
        <div className="cluster-stale-banner">
          Groups changed — <button className="btn btn-small btn-primary" onClick={() => fetchClusters(clusterData?.nClusters)}>Re-run clustering</button> to incorporate new groups as seeds
        </div>
      )}

      {renderLightbox()}
    </div>
  );
}
