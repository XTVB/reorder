import { create } from "zustand";
import type { ClusterData, ClusterResultData, ImportClusterInput, WeightConfig } from "../types.ts";
import { getErrorMessage, postJson } from "../utils/helpers.ts";
import { consolidateBlock } from "../utils/reorder.ts";
import { consumeSSE } from "../utils/sse.ts";
import { useGroupStore } from "./groupStore.ts";
import { useImageStore } from "./imageStore.ts";
import { useUIStore } from "./uiStore.ts";

interface ClusterState {
  clusterData: ClusterData | null;
  loading: boolean;
  progress: string;
  mergeSelection: Set<string>; // cluster IDs selected for merging
  selectedImages: Set<string>; // keys: "clusterId:filename"
  collapsedClusters: Set<string>;
  lastClickedImage: { clusterId: string; index: number } | null;
  lightbox: { clusterId: string; imageIndex: number } | null;
  treeStale: boolean;
  focusedClusterId: string | null;
  weights: WeightConfig;
  usePatches: boolean;

  setWeights: (w: WeightConfig) => void;
  setUsePatches: (v: boolean) => void;
  fetchClusters: (nClusters?: number) => Promise<void>;
  recutClusters: (nClusters: number) => Promise<void>;
  recutByThreshold: (threshold: number) => Promise<void>;
  recutAdaptive: (minClusterSize: number) => Promise<void>;
  toggleMergeSelect: (clusterId: string) => void;
  clearMergeSelection: () => void;
  toggleImageSelect: (clusterId: string, filename: string) => void;
  rangeSelectImages: (clusterId: string, index: number) => void;
  clearImageSelection: () => void;
  dismissCluster: (clusterId: string) => void;
  toggleCollapsed: (clusterId: string) => void;
  expandAll: () => void;
  collapseAll: () => void;

  openLightbox: (clusterId: string, imageIndex: number) => void;
  closeLightbox: () => void;
  markTreeStale: () => void;
  renameCluster: (clusterId: string, name: string) => void;

  setFocusedCluster: (id: string | null) => void;
  moveFocus: (direction: 1 | -1) => void;

  mergeSelectedClusters: () => void;
  splitSelected: () => void;
  acceptCluster: (cluster: ClusterResultData) => void;
  acceptAllClusters: (minSize: number) => void;
  addToGroup: (cluster: ClusterResultData) => void;
  loadCachedClusters: () => Promise<void>;
  importClusters: (payload: { clusters: ImportClusterInput[] }) => Promise<void>;
  clearImportedClusters: () => Promise<void>;
}

function applyClusterResult(data: ClusterData): Partial<ClusterState> {
  const collapsed = new Set<string>();
  for (const c of data.clusters) {
    if (c.confirmedGroup && c.images.length === c.confirmedGroup.images.length) {
      collapsed.add(c.id);
    }
  }
  return {
    clusterData: data,
    loading: false,
    progress: "",
    mergeSelection: new Set(),
    selectedImages: new Set(),
    collapsedClusters: collapsed,
  };
}

/** Parse "clusterId:filename" composite key */
function parseImageKey(key: string): { clusterId: string; filename: string } {
  const sep = key.indexOf(":");
  return { clusterId: key.slice(0, sep), filename: key.slice(sep + 1) };
}

export const useClusterStore = create<ClusterState>((set, get) => ({
  clusterData: null,
  loading: false,
  progress: "",
  mergeSelection: new Set(),
  selectedImages: new Set(),
  collapsedClusters: new Set(),
  lastClickedImage: null,
  lightbox: null,
  treeStale: false,
  focusedClusterId: null,
  weights: { pecore_g: 1.0, color: 0.5 },
  usePatches: false,

  setWeights: (w) => set({ weights: w, treeStale: true }),
  setUsePatches: (v) => set({ usePatches: v, treeStale: true }),

  fetchClusters: async (nClusters = 200) => {
    set({ loading: true, progress: "Starting clustering..." });
    try {
      const { weights, usePatches } = get();
      const response = await fetch("/api/cluster", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nClusters, weights, usePatches }),
      });

      if (response.status === 409) {
        set({ progress: "Clustering already in progress..." });
        return;
      }

      let result: ClusterData | null = null;
      await consumeSSE(response, {
        onProgress: (message) => set({ progress: message }),
        onResult: (data) => {
          result = data as ClusterData;
        },
        onError: (error) => {
          set({ loading: false, progress: `Error: ${error}` });
        },
      });

      if (result) {
        set({ ...applyClusterResult(result), treeStale: false });
      } else if (!get().progress.startsWith("Error:")) {
        set({ loading: false, progress: "No results returned" });
      }
    } catch (err) {
      set({ loading: false, progress: `Error: ${err}` });
    }
  },

  recutClusters: async (nClusters: number) => {
    set({ loading: true, progress: "Re-cutting tree..." });
    try {
      const res = await postJson("/api/cluster/recut", { nClusters });
      const result: ClusterData = await res.json();
      set(applyClusterResult(result));
    } catch (err) {
      set({ loading: false, progress: `Error: ${err}` });
    }
  },

  recutByThreshold: async (threshold: number) => {
    set({ loading: true, progress: "Re-cutting tree..." });
    try {
      const res = await postJson("/api/cluster/recut", { threshold });
      const result: ClusterData = await res.json();
      set(applyClusterResult(result));
    } catch (err) {
      set({ loading: false, progress: `Error: ${err}` });
    }
  },

  recutAdaptive: async (minClusterSize: number) => {
    set({ loading: true, progress: "Adaptive re-cut..." });
    try {
      const res = await postJson("/api/cluster/recut", { minClusterSize });
      const result: ClusterData = await res.json();
      set(applyClusterResult(result));
    } catch (err) {
      set({ loading: false, progress: `Error: ${err}` });
    }
  },

  toggleMergeSelect: (clusterId) => {
    const sel = new Set(get().mergeSelection);
    if (sel.has(clusterId)) sel.delete(clusterId);
    else sel.add(clusterId);
    set({ mergeSelection: sel });
  },

  clearMergeSelection: () => set({ mergeSelection: new Set() }),

  toggleImageSelect: (clusterId, filename) => {
    const sel = new Set(get().selectedImages);
    const key = `${clusterId}:${filename}`;
    if (sel.has(key)) sel.delete(key);
    else sel.add(key);
    const cluster = get().clusterData?.clusters.find((c) => c.id === clusterId);
    const index = cluster?.images.indexOf(filename) ?? -1;
    set({ selectedImages: sel, lastClickedImage: { clusterId, index } });
  },

  rangeSelectImages: (clusterId, toIndex) => {
    const { lastClickedImage, clusterData } = get();
    if (!lastClickedImage || lastClickedImage.clusterId !== clusterId || !clusterData) return;
    const cluster = clusterData.clusters.find((c) => c.id === clusterId);
    if (!cluster) return;

    const from = Math.min(lastClickedImage.index, toIndex);
    const to = Math.max(lastClickedImage.index, toIndex);
    const sel = new Set(get().selectedImages);
    for (let i = from; i <= to; i++) {
      const f = cluster.images[i];
      if (f) sel.add(`${clusterId}:${f}`);
    }
    set({ selectedImages: sel });
  },

  clearImageSelection: () => set({ selectedImages: new Set(), lastClickedImage: null }),

  dismissCluster: (clusterId) => {
    const { clusterData } = get();
    if (!clusterData) return;
    set({
      clusterData: {
        ...clusterData,
        clusters: clusterData.clusters.filter((c) => c.id !== clusterId),
      },
    });
  },

  toggleCollapsed: (clusterId) => {
    const collapsed = new Set(get().collapsedClusters);
    if (collapsed.has(clusterId)) collapsed.delete(clusterId);
    else collapsed.add(clusterId);
    set({ collapsedClusters: collapsed });
  },

  expandAll: () => set({ collapsedClusters: new Set() }),

  collapseAll: () => {
    const data = get().clusterData;
    if (!data) return;
    set({ collapsedClusters: new Set(data.clusters.map((c) => c.id)) });
  },

  openLightbox: (clusterId, imageIndex) => set({ lightbox: { clusterId, imageIndex } }),
  closeLightbox: () => set({ lightbox: null }),
  markTreeStale: () => set({ treeStale: true }),

  setFocusedCluster: (id) => set({ focusedClusterId: id }),
  moveFocus: (direction) => {
    const { clusterData, focusedClusterId } = get();
    if (!clusterData || clusterData.clusters.length === 0) return;
    const clusters = clusterData.clusters;
    if (!focusedClusterId) {
      set({ focusedClusterId: clusters[direction === 1 ? 0 : clusters.length - 1]!.id });
      return;
    }
    const idx = clusters.findIndex((c) => c.id === focusedClusterId);
    const next = idx + direction;
    if (next >= 0 && next < clusters.length) {
      set({ focusedClusterId: clusters[next]!.id });
    }
  },

  renameCluster: (clusterId, name) => {
    const { clusterData } = get();
    if (!clusterData) return;
    set({
      clusterData: {
        ...clusterData,
        clusters: clusterData.clusters.map((c) =>
          c.id === clusterId ? { ...c, autoName: name } : c,
        ),
      },
    });
  },

  mergeSelectedClusters: () => {
    const { mergeSelection, clusterData } = get();
    if (!clusterData || mergeSelection.size < 2) return;

    const selected = clusterData.clusters.filter((c) => mergeSelection.has(c.id));
    if (selected.length < 2) return;
    selected.sort((a, b) => b.images.length - a.images.length);
    const target = selected.find((c) => c.confirmedGroup) ?? selected[0];
    if (!target) return;
    const sources = selected.filter((c) => c.id !== target.id);

    const seen = new Set(target.images);
    const mergedImages = [...target.images];
    for (const src of sources) {
      for (const f of src.images) {
        if (!seen.has(f)) {
          seen.add(f);
          mergedImages.push(f);
        }
      }
    }

    const sourceIds = new Set(sources.map((s) => s.id));
    const newClusters = clusterData.clusters
      .filter((c) => !sourceIds.has(c.id))
      .map((c) => (c.id === target.id ? { ...c, images: mergedImages.sort() } : c));

    set({
      clusterData: { ...clusterData, clusters: newClusters },
      mergeSelection: new Set(),
    });
  },

  splitSelected: () => {
    const { selectedImages, clusterData } = get();
    if (!clusterData || selectedImages.size === 0) return;

    const byCluster = new Map<string, string[]>();
    for (const key of selectedImages) {
      const { clusterId, filename } = parseImageKey(key);
      if (!byCluster.has(clusterId)) byCluster.set(clusterId, []);
      byCluster.get(clusterId)!.push(filename);
    }

    const newClusters = [...clusterData.clusters];
    for (const [clusterId, filenames] of byCluster) {
      const sourceIdx = newClusters.findIndex((c) => c.id === clusterId);
      if (sourceIdx === -1) continue;
      const source = newClusters[sourceIdx];
      if (!source || filenames.length >= source.images.length) continue;

      const removeSet = new Set(filenames);
      const remaining = source.images.filter((f) => !removeSet.has(f));
      const splitFiles = filenames.sort();

      newClusters[sourceIdx] = { ...source, images: remaining };
      const newCluster: ClusterResultData = {
        id: `split_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        autoName: "Split",
        autoTags: [],
        images: splitFiles,
        confirmedGroup: null,
      };
      newClusters.splice(sourceIdx + 1, 0, newCluster);
    }

    set({
      clusterData: { ...clusterData, clusters: newClusters },
      selectedImages: new Set(),
      lastClickedImage: null,
    });
  },

  acceptCluster: (cluster) => {
    const name = cluster.autoName || `Cluster ${cluster.id}`;
    const { updateGroups, groupsLoaded } = useGroupStore.getState();
    const { images, setImages } = useImageStore.getState();
    const { showToast } = useUIStore.getState();

    if (!groupsLoaded) {
      showToast("Groups still loading — please wait", "warning");
      return;
    }

    updateGroups((prev) => [...prev, { id: crypto.randomUUID(), name, images: cluster.images }]);
    setImages(consolidateBlock(images, new Set(cluster.images)));
    showToast(`Created group "${name}" with ${cluster.images.length} images`, "success");
    get().dismissCluster(cluster.id);
    set({ treeStale: true });
  },

  acceptAllClusters: (minSize) => {
    const { clusterData } = get();
    if (!clusterData) return;
    const { showToast } = useUIStore.getState();
    const { groupsLoaded } = useGroupStore.getState();

    if (!groupsLoaded) {
      showToast("Groups still loading — please wait", "warning");
      return;
    }

    const eligible = clusterData.clusters.filter(
      (c) => !c.confirmedGroup && c.images.length >= minSize,
    );

    if (eligible.length === 0) {
      showToast("No eligible clusters to accept", "warning");
      return;
    }

    const totalImages = eligible.reduce((n, c) => n + c.images.length, 0);
    if (!confirm(`Create ${eligible.length} groups from ${totalImages} images?`)) {
      return;
    }

    const dismissIds = new Set(eligible.map((c) => c.id));
    const newGroups = eligible.map((c) => ({
      id: crypto.randomUUID(),
      name: c.autoName || `Cluster ${c.id}`,
      images: c.images,
    }));

    const { updateGroups } = useGroupStore.getState();
    const { images, setImages } = useImageStore.getState();
    updateGroups((prev) => [...prev, ...newGroups]);
    const allAccepted = new Set(eligible.flatMap((c) => c.images));
    setImages(consolidateBlock(images, allAccepted));
    showToast(`Created ${newGroups.length} groups`, "success");
    set({
      clusterData: {
        ...clusterData,
        clusters: clusterData.clusters.filter((c) => !dismissIds.has(c.id)),
      },
      treeStale: true,
    });
  },

  addToGroup: (cluster) => {
    if (!cluster.confirmedGroup) return;
    const { groupsLoaded } = useGroupStore.getState();
    if (!groupsLoaded) {
      useUIStore.getState().showToast("Groups still loading — please wait", "warning");
      return;
    }
    const groupId = cluster.confirmedGroup.id;
    const confirmedSet = new Set(cluster.confirmedGroup.images);
    const suggested = cluster.images.filter((f) => !confirmedSet.has(f));

    const { selectedImages, clusterData } = get();
    const selectedInCluster = [...selectedImages]
      .filter((key) => key.startsWith(`${cluster.id}:`))
      .map((key) => key.slice(cluster.id.length + 1))
      .filter((f) => !confirmedSet.has(f));
    const toAdd = selectedInCluster.length > 0 ? selectedInCluster : suggested;

    const { updateGroups } = useGroupStore.getState();
    const { showToast } = useUIStore.getState();
    updateGroups((prev) =>
      prev.map((g) => (g.id === groupId ? { ...g, images: [...g.images, ...toAdd] } : g)),
    );
    showToast(`Added ${toAdd.length} images to "${cluster.confirmedGroup.name}"`, "success");

    // Update the cluster's confirmedGroup snapshot so it reflects the new state
    const newConfirmedImages = [...cluster.confirmedGroup.images, ...toAdd];
    const updatedClusters = clusterData
      ? clusterData.clusters.map((c) =>
          c.id === cluster.id
            ? { ...c, confirmedGroup: { ...cluster.confirmedGroup!, images: newConfirmedImages } }
            : c,
        )
      : [];

    // Auto-collapse if all images are now confirmed
    const collapsed = new Set(get().collapsedClusters);
    const allConfirmed = newConfirmedImages.length >= cluster.images.length;
    if (allConfirmed) collapsed.add(cluster.id);

    set({
      clusterData: clusterData ? { ...clusterData, clusters: updatedClusters } : null,
      collapsedClusters: collapsed,
      selectedImages: new Set(),
      lastClickedImage: null,
      treeStale: true,
    });
  },

  loadCachedClusters: async () => {
    try {
      const res = await fetch("/api/cluster/cache-status");
      const { cached, imported } = await res.json();
      if (get().clusterData || get().loading) return;
      if (imported) {
        const importedRes = await fetch("/api/cluster/imported");
        if (importedRes.ok) {
          const data: ClusterData = await importedRes.json();
          set(applyClusterResult(data));
          return;
        }
      }
      if (cached) {
        await get().recutClusters(200);
      }
    } catch {}
  },

  importClusters: async (payload) => {
    set({ loading: true, progress: "Importing clusters..." });
    try {
      const res = await postJson("/api/cluster/import", payload);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        set({ loading: false, progress: `Error: ${err.error ?? "import failed"}` });
        return;
      }
      const result: ClusterData = await res.json();
      set(applyClusterResult(result));
      useUIStore.getState().showToast(`Imported ${result.clusters.length} clusters`, "success");
    } catch (err) {
      set({ loading: false, progress: `Error: ${getErrorMessage(err, "import failed")}` });
    }
  },

  clearImportedClusters: async () => {
    try {
      await fetch("/api/cluster/imported", { method: "DELETE" });
      set({
        clusterData: null,
        collapsedClusters: new Set(),
        selectedImages: new Set(),
        mergeSelection: new Set(),
      });
      useUIStore.getState().showToast("Cleared imported clusters", "success");
    } catch (err) {
      useUIStore.getState().showToast(getErrorMessage(err, "Failed to clear"), "error");
    }
  },
}));
