import { create } from "zustand";
import type {
  ClusterResultData,
  NNAggregation,
  NNFilter,
  NNQueryResponse,
  NNResult,
} from "../types.ts";
import { getErrorMessage } from "../utils/helpers.ts";
import { consumeSSE } from "../utils/sse.ts";
import { useClusterStore } from "./clusterStore.ts";
import { flushGroupPersist, useGroupStore } from "./groupStore.ts";
import { useUIStore } from "./uiStore.ts";

type QuerySource =
  | { kind: "cluster"; clusterId: string; images: string[] }
  | { kind: "selection"; images: string[] };

interface NNQueryState {
  open: boolean;
  queryLabel: string;
  querySource: QuerySource | null;

  filter: NNFilter;
  topN: number;
  aggregation: NNAggregation;

  loading: boolean;
  progress: string;
  error: string | null;
  results: NNResult[];
  usedModels: string[];
  patchesBlended: boolean;

  modalSelection: Set<string>;

  openForCluster: (cluster: ClusterResultData) => void;
  openForSelection: (filenames: string[]) => void;
  close: () => void;

  setFilter: (f: NNFilter) => void;
  setTopN: (n: number) => void;
  setAggregation: (a: NNAggregation) => void;

  fetch: () => Promise<void>;

  toggleResultSelected: (filename: string) => void;
  clearModalSelection: () => void;

  createClusterFromSelected: () => void;
  addSelectedToGroup: (groupId: string) => Promise<void>;
}

let _debounceTimer: ReturnType<typeof setTimeout> | null = null;

function debouncedFetch(delayMs = 150) {
  if (_debounceTimer) clearTimeout(_debounceTimer);
  _debounceTimer = setTimeout(() => {
    _debounceTimer = null;
    useNNQueryStore.getState().fetch();
  }, delayMs);
}

export const useNNQueryStore = create<NNQueryState>((set, get) => ({
  open: false,
  queryLabel: "",
  querySource: null,
  filter: "any",
  topN: 50,
  aggregation: "centroid",
  loading: false,
  progress: "",
  error: null,
  results: [],
  usedModels: [],
  patchesBlended: false,
  modalSelection: new Set(),

  openForCluster: (cluster) => {
    const label = cluster.autoName
      ? `Cluster "${cluster.autoName}" (${cluster.images.length})`
      : `${cluster.images.length} images`;
    set({
      open: true,
      queryLabel: label,
      querySource: { kind: "cluster", clusterId: cluster.id, images: cluster.images },
      results: [],
      error: null,
      modalSelection: new Set(),
    });
    get().fetch();
  },

  openForSelection: (filenames) => {
    const deduped = [...new Set(filenames)];
    set({
      open: true,
      queryLabel: `${deduped.length} selected image${deduped.length === 1 ? "" : "s"}`,
      querySource: { kind: "selection", images: deduped },
      results: [],
      error: null,
      modalSelection: new Set(),
    });
    get().fetch();
  },

  close: () => {
    if (_debounceTimer) {
      clearTimeout(_debounceTimer);
      _debounceTimer = null;
    }
    set({
      open: false,
      querySource: null,
      queryLabel: "",
      results: [],
      error: null,
      loading: false,
      progress: "",
      modalSelection: new Set(),
    });
  },

  setFilter: (f) => {
    set({ filter: f });
    debouncedFetch();
  },
  setTopN: (n) => {
    set({ topN: Math.max(1, Math.min(n, 500)) });
    debouncedFetch();
  },
  setAggregation: (a) => {
    set({ aggregation: a });
    debouncedFetch();
  },

  fetch: async () => {
    const { querySource, filter, topN, aggregation } = get();
    if (!querySource || querySource.images.length === 0) return;

    const { weights, usePatches, clusterData } = useClusterStore.getState();
    const restrictToFilenames = clusterData?.scope?.subsetFilenames;

    set({ loading: true, progress: "Running NN query...", error: null });
    try {
      const response = await fetch("/api/cluster/nn-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          queryFilenames: querySource.images,
          topN,
          filter,
          aggregation,
          weights,
          usePatches,
          restrictToFilenames,
        }),
      });
      if (response.status === 409) {
        set({ loading: false, progress: "", error: "Clustering in progress — retry shortly" });
        return;
      }

      let result: NNQueryResponse | null = null;
      let errMsg: string | null = null;
      await consumeSSE(response, {
        onProgress: (message) => set({ progress: message }),
        onResult: (data) => {
          result = data as NNQueryResponse;
        },
        onError: (error) => {
          errMsg = error;
        },
      });

      if (errMsg) {
        set({ loading: false, progress: "", error: errMsg });
        return;
      }
      if (result) {
        const r = result as NNQueryResponse;
        set({
          loading: false,
          progress: "",
          error: null,
          results: r.results,
          usedModels: r.usedModels,
          patchesBlended: r.patchesBlended,
        });
      } else {
        set({ loading: false, progress: "", error: "No results returned" });
      }
    } catch (err) {
      set({
        loading: false,
        progress: "",
        error: getErrorMessage(err, "NN query failed"),
      });
    }
  },

  toggleResultSelected: (filename) => {
    const sel = new Set(get().modalSelection);
    if (sel.has(filename)) sel.delete(filename);
    else sel.add(filename);
    set({ modalSelection: sel });
  },

  clearModalSelection: () => set({ modalSelection: new Set() }),

  createClusterFromSelected: () => {
    const { modalSelection, queryLabel } = get();
    if (modalSelection.size === 0) return;
    const insertClusterFromFilenames = useClusterStore.getState().insertClusterFromFilenames;
    const label = `NN: ${queryLabel}`.slice(0, 60);
    insertClusterFromFilenames(label, [...modalSelection]);
    useUIStore
      .getState()
      .showToast(`Created cluster with ${modalSelection.size} images`, "success");
    get().close();
  },

  addSelectedToGroup: async (groupId) => {
    const { modalSelection } = get();
    if (modalSelection.size === 0) return;
    const { groups, updateGroups, groupsLoaded } = useGroupStore.getState();
    const { showToast } = useUIStore.getState();

    if (!groupsLoaded) {
      showToast("Groups still loading — please wait", "warning");
      return;
    }
    const group = groups.find((g) => g.id === groupId);
    if (!group) {
      showToast("Group not found", "error");
      return;
    }

    const existing = new Set(group.images);
    const toAdd = [...modalSelection].filter((f) => !existing.has(f));
    if (toAdd.length === 0) {
      showToast("All selected images are already in that group", "warning");
      return;
    }

    updateGroups((prev) =>
      prev.map((g) => (g.id === groupId ? { ...g, images: [...g.images, ...toAdd] } : g)),
    );
    showToast(`Added ${toAdd.length} to "${group.name}"`, "success");

    // Flush so subsequent badge lookups see the new membership.
    await flushGroupPersist();
    set({ modalSelection: new Set() });
    await get().fetch();
  },
}));
