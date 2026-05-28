// Cluster list state — clusterData, settings, recut/import pipelines,
// collapse helpers, and low-level mutators used by other cluster sub-stores.

import { create } from "zustand";
import { deleteJson, getJson, postJson } from "../../../api/client.ts";
import { consumeSSE, startSSE } from "../../../api/sse.ts";
import type {
  ClusterData,
  ClusterResultData,
  ImportClusterInput,
  SplitChildren,
  WeightConfig,
} from "../../../types.ts";
import { getErrorMessage } from "../../../utils/helpers.ts";
import { useToastStore } from "../../core/toastStore.ts";

interface ListState {
  clusterData: ClusterData | null;
  loading: boolean;
  progress: string;
  treeStale: boolean;
  weights: WeightConfig;
  usePatches: boolean;
  useRerank: boolean;
  /** Blend strength for re-rank distance (0=cosine only, 1=rerank only). */
  rerankBlend: number;
  collapsedClusters: Set<string>;
  /** Children produced by an inline split, keyed by parent cluster id. Presence == expanded. */
  splitChildren: Record<string, SplitChildren>;

  // Settings
  setWeights: (w: WeightConfig) => void;
  setUsePatches: (v: boolean) => void;
  setUseRerank: (v: boolean) => void;
  setRerankBlend: (v: number) => void;

  // Pipeline
  fetchClusters: (nClusters?: number) => Promise<void>;
  recut: (params: {
    nClusters?: number;
    threshold?: number;
    minClusterSize?: number;
  }) => Promise<void>;
  loadCachedClusters: () => Promise<void>;
  importClusters: (payload: { clusters: ImportClusterInput[] }) => Promise<void>;
  clearImportedClusters: () => Promise<void>;

  // Inline insertion (NN flow)
  insertClusterFromFilenames: (name: string, filenames: string[], afterClusterId?: string) => void;

  // Collapse / tree-stale
  toggleCollapsed: (clusterId: string) => void;
  expandAll: () => void;
  collapseAll: () => void;
  markTreeStale: () => void;
  renameCluster: (clusterId: string, name: string) => void;
  dismissCluster: (clusterId: string) => void;

  // Low-level mutators (used by interactionsStore / compareStore / expandStore)
  applyClusterResult: (data: ClusterData) => void;
  applyClusterReplace: (id: string, next: ClusterResultData) => void;
  applyClusterRemoval: (ids: Set<string>) => void;
  stripImagesFromOtherClusters: (keepId: string, filenames: Set<string>) => void;
  setClusterData: (data: ClusterData | null) => void;
  setSplitChildren: (children: Record<string, SplitChildren>) => void;
}

function applyClusterResultPayload(data: ClusterData): Partial<ListState> {
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
    collapsedClusters: collapsed,
    splitChildren: {},
  };
}

export const useListStore = create<ListState>((set, get) => {
  async function doRecut(url: string, body: unknown, msg = "Re-cutting tree...") {
    set({ loading: true, progress: msg });
    try {
      const result = await postJson<ClusterData>(url, body);
      get().applyClusterResult(result);
    } catch (err) {
      set({ loading: false, progress: `Error: ${getErrorMessage(err, "recut failed")}` });
    }
  }

  return {
    clusterData: null,
    loading: false,
    progress: "",
    treeStale: false,
    weights: { pecore_g: 1.0, color: 0.5, learned_proj: 0.45 },
    usePatches: false,
    useRerank: true,
    rerankBlend: 0.7,
    collapsedClusters: new Set(),
    splitChildren: {},

    setWeights: (w) => set({ weights: w, treeStale: true }),
    setUsePatches: (v) => {
      if (get().usePatches === v) return;
      set({ usePatches: v, treeStale: true });
    },
    setUseRerank: (v) => {
      if (get().useRerank === v) return;
      set({ useRerank: v, treeStale: true });
    },
    setRerankBlend: (v) => {
      if (get().rerankBlend === v) return;
      set({ rerankBlend: v, treeStale: true });
    },

    applyClusterResult: (data) => {
      set(applyClusterResultPayload(data));
    },

    setClusterData: (data) => set({ clusterData: data }),
    setSplitChildren: (children) => set({ splitChildren: children }),

    applyClusterReplace: (id, next) => {
      const { clusterData, splitChildren } = get();
      if (!clusterData) return;
      if (clusterData.clusters.some((c) => c.id === id)) {
        set({
          clusterData: {
            ...clusterData,
            clusters: clusterData.clusters.map((c) => (c.id === id ? next : c)),
          },
        });
        return;
      }
      const newKids = { ...splitChildren };
      for (const [parentId, kids] of Object.entries(splitChildren)) {
        if (kids.childA.id === id) {
          newKids[parentId] = { ...kids, childA: next };
          set({ splitChildren: newKids });
          return;
        }
        if (kids.childB.id === id) {
          newKids[parentId] = { ...kids, childB: next };
          set({ splitChildren: newKids });
          return;
        }
      }
    },

    applyClusterRemoval: (ids) => {
      const { clusterData, splitChildren } = get();
      if (!clusterData) return;
      const newTop = clusterData.clusters.filter((c) => !ids.has(c.id));
      const newKids = { ...splitChildren };
      for (const [parentId, kids] of Object.entries(splitChildren)) {
        if (ids.has(parentId) || ids.has(kids.childA.id) || ids.has(kids.childB.id)) {
          const stack = [parentId, kids.childA.id, kids.childB.id];
          while (stack.length) {
            const id = stack.pop()!;
            const k = newKids[id];
            if (k) {
              stack.push(k.childA.id, k.childB.id);
              delete newKids[id];
            }
          }
        }
      }
      set({
        clusterData: { ...clusterData, clusters: newTop },
        splitChildren: newKids,
      });
    },

    stripImagesFromOtherClusters: (keepId, filenames) => {
      const { clusterData, splitChildren } = get();
      if (!clusterData) return;
      const newClusters = clusterData.clusters.map((c) => {
        if (c.id === keepId) return c;
        if (!c.images.some((f) => filenames.has(f))) return c;
        return { ...c, images: c.images.filter((f) => !filenames.has(f)) };
      });
      const newKids = { ...splitChildren };
      for (const [parentId, kids] of Object.entries(splitChildren)) {
        let { childA, childB } = kids;
        if (childA.id !== keepId && childA.images.some((f) => filenames.has(f))) {
          childA = { ...childA, images: childA.images.filter((f) => !filenames.has(f)) };
        }
        if (childB.id !== keepId && childB.images.some((f) => filenames.has(f))) {
          childB = { ...childB, images: childB.images.filter((f) => !filenames.has(f)) };
        }
        if (childA !== kids.childA || childB !== kids.childB) {
          newKids[parentId] = { childA, childB };
        }
      }
      set({
        clusterData: { ...clusterData, clusters: newClusters },
        splitChildren: newKids,
      });
    },

    fetchClusters: async (nClusters = 200) => {
      set({ loading: true, progress: "Starting clustering..." });
      try {
        const { weights, usePatches, useRerank, rerankBlend } = get();
        const start = await startSSE("/api/cluster", {
          nClusters,
          weights,
          usePatches,
          useRerank,
          rerankBlend,
        });

        if (start.kind === "conflict") {
          set({ progress: "Clustering already in progress..." });
          return;
        }

        let result: ClusterData | null = null;
        await consumeSSE(start.response, {
          onProgress: (message) => set({ progress: message }),
          onResult: (data) => {
            result = data as ClusterData;
          },
          onError: (error) => {
            set({ loading: false, progress: `Error: ${error}` });
          },
        });

        if (result) {
          set({ ...applyClusterResultPayload(result), treeStale: false });
        } else if (!get().progress.startsWith("Error:")) {
          set({ loading: false, progress: "No results returned" });
        }
      } catch (err) {
        set({ loading: false, progress: `Error: ${getErrorMessage(err, "fetch failed")}` });
      }
    },

    recut: (params) => {
      const body: Record<string, unknown> = {};
      let msg = "Re-cutting tree...";
      if (params.minClusterSize != null) {
        body.minClusterSize = params.minClusterSize;
        msg = "Adaptive re-cut...";
      } else if (params.threshold != null) {
        body.threshold = params.threshold;
      } else {
        body.nClusters = params.nClusters ?? 200;
      }
      return doRecut("/api/cluster/recut", body, msg);
    },

    dismissCluster: (clusterId) => {
      get().applyClusterRemoval(new Set([clusterId]));
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

    markTreeStale: () => set({ treeStale: true }),

    renameCluster: (clusterId, name) => {
      const { clusterData } = get();
      if (!clusterData) return;
      set({
        clusterData: {
          ...clusterData,
          clusters: clusterData.clusters.map((c) => (c.id === clusterId ? { ...c, name } : c)),
        },
      });
    },

    loadCachedClusters: async () => {
      if (get().clusterData || get().loading) return;
      try {
        const { cached, imported } = await getJson<{
          cached: boolean;
          imported: ClusterData | null;
        }>("/api/cluster/cache-status");
        if (get().clusterData || get().loading) return;
        if (imported) {
          get().applyClusterResult(imported);
          return;
        }
        if (cached) {
          await get().recut({ nClusters: 200 });
        }
      } catch (err) {
        console.warn("Failed to load cached clusters:", err);
      }
    },

    importClusters: async (payload) => {
      set({ loading: true, progress: "Importing clusters..." });
      try {
        const result = await postJson<ClusterData>("/api/cluster/import", payload);
        get().applyClusterResult(result);
        useToastStore
          .getState()
          .showToast(`Imported ${result.clusters.length} clusters`, "success");
      } catch (err) {
        set({ loading: false, progress: `Error: ${getErrorMessage(err, "import failed")}` });
      }
    },

    clearImportedClusters: async () => {
      try {
        await deleteJson<{ ok: boolean }>("/api/cluster/imported");
        set({
          clusterData: null,
          collapsedClusters: new Set(),
        });
        useToastStore.getState().showToast("Cleared imported clusters", "success");
      } catch (err) {
        useToastStore.getState().showToast(getErrorMessage(err, "Failed to clear"), "error");
      }
    },

    insertClusterFromFilenames: (name, filenames, afterClusterId) => {
      const { clusterData } = get();
      if (!clusterData || filenames.length === 0) return;
      const newCluster: ClusterResultData = {
        id: `nn_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        name: name || "From NN",
        images: [...new Set(filenames)],
        confirmedGroup: null,
      };
      const newClusters = [...clusterData.clusters];
      const afterIdx = afterClusterId ? newClusters.findIndex((c) => c.id === afterClusterId) : -1;
      newClusters.splice(afterIdx >= 0 ? afterIdx + 1 : 0, 0, newCluster);
      set({ clusterData: { ...clusterData, clusters: newClusters } });
    },
  };
});
