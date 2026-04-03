import { create } from "zustand";
import type { ClusterData, ClusterResultData } from "../types.ts";
import { postJson } from "../utils/helpers.ts";

interface ClusterState {
  clusterData: ClusterData | null;
  loading: boolean;
  progress: string;
  mergeSelection: Set<string>;      // cluster IDs selected for merging
  selectedImages: Set<string>;       // keys: "clusterId:filename"
  collapsedClusters: Set<string>;
  lastClickedImage: { clusterId: string; index: number } | null;
  lightbox: { clusterId: string; imageIndex: number } | null;
  treeStale: boolean;
  focusedClusterId: string | null;

  fetchClusters: (nClusters?: number) => Promise<void>;
  recutClusters: (nClusters: number) => Promise<void>;
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

  fetchClusters: async (nClusters = 200) => {
    set({ loading: true, progress: "Starting clustering..." });
    try {
      const response = await fetch("/api/cluster", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nClusters }),
      });

      if (response.status === 409) {
        set({ progress: "Clustering already in progress..." });
        return;
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let result: ClusterData | null = null;
      let eventType = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7);
          } else if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6));
            if (eventType === "progress") {
              set({ progress: data.message });
            } else if (eventType === "result") {
              result = data;
            } else if (eventType === "error") {
              set({ loading: false, progress: `Error: ${data.error}` });
              return;
            }
            eventType = "";
          } else if (line === "") {
            eventType = "";
          }
        }
      }

      if (result) {
        set({ ...applyClusterResult(result), treeStale: false });
      } else {
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
    const cluster = get().clusterData?.clusters.find(c => c.id === clusterId);
    const index = cluster?.images.indexOf(filename) ?? -1;
    set({ selectedImages: sel, lastClickedImage: { clusterId, index } });
  },

  rangeSelectImages: (clusterId, toIndex) => {
    const { lastClickedImage, clusterData } = get();
    if (!lastClickedImage || lastClickedImage.clusterId !== clusterId || !clusterData) return;
    const cluster = clusterData.clusters.find(c => c.id === clusterId);
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
        clusters: clusterData.clusters.filter(c => c.id !== clusterId),
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
    set({ collapsedClusters: new Set(data.clusters.map(c => c.id)) });
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
    const idx = clusters.findIndex(c => c.id === focusedClusterId);
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
        clusters: clusterData.clusters.map(c =>
          c.id === clusterId ? { ...c, autoName: name } : c
        ),
      },
    });
  },

  mergeSelectedClusters: () => {
    const { mergeSelection, clusterData } = get();
    if (!clusterData || mergeSelection.size < 2) return;

    const selected = clusterData.clusters.filter(c => mergeSelection.has(c.id));
    if (selected.length < 2) return;
    selected.sort((a, b) => b.images.length - a.images.length);
    const target = selected.find(c => c.confirmedGroup) ?? selected[0];
    if (!target) return;
    const sources = selected.filter(c => c.id !== target.id);

    const seen = new Set(target.images);
    const mergedImages = [...target.images];
    for (const src of sources) {
      for (const f of src.images) {
        if (!seen.has(f)) { seen.add(f); mergedImages.push(f); }
      }
    }

    const sourceIds = new Set(sources.map(s => s.id));
    const newClusters = clusterData.clusters
      .filter(c => !sourceIds.has(c.id))
      .map(c => c.id === target.id ? { ...c, images: mergedImages.sort() } : c);

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

    let newClusters = [...clusterData.clusters];
    for (const [clusterId, filenames] of byCluster) {
      const sourceIdx = newClusters.findIndex(c => c.id === clusterId);
      if (sourceIdx === -1) continue;
      const source = newClusters[sourceIdx];
      if (!source || filenames.length >= source.images.length) continue;

      const removeSet = new Set(filenames);
      const remaining = source.images.filter(f => !removeSet.has(f));
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
}));
