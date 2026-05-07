import { create } from "zustand";
import { consumeSSE, startSSE } from "../api/sse.ts";
import type {
  ClusterResultData,
  NNAggregation,
  NNFilter,
  NNQueryResponse,
  NNResult,
} from "../types.ts";
import { addFilenamesToGroup } from "../utils/groups.ts";
import { getErrorMessage } from "../utils/helpers.ts";
import { useSelectionStore } from "./core/selectionStore.ts";
import { useToastStore } from "./core/toastStore.ts";
import { useGroupStore } from "./groupStore.ts";
import { dropCannotLinkAgainstGroup } from "./modes/cluster/interactionsStore.ts";
import { useListStore } from "./modes/cluster/listStore.ts";
import { findClusterEverywhere } from "./modes/cluster/tree-helpers.ts";

type QuerySource =
  | { kind: "cluster"; clusterId: string; images: string[] }
  | { kind: "selection"; images: string[] };

interface NNQueryState {
  open: boolean;
  queryLabel: string;
  querySource: QuerySource | null;
  sourceClusterLabel: string | null;

  filter: NNFilter;
  topN: number;
  aggregation: NNAggregation;

  loading: boolean;
  progress: string;
  error: string | null;
  results: NNResult[];
  usedModels: string[];
  patchesBlended: boolean;

  openForCluster: (cluster: ClusterResultData) => void;
  openForSelection: (filenames: string[]) => void;
  close: () => void;

  setFilter: (f: NNFilter) => void;
  setTopN: (n: number) => void;
  setAggregation: (a: NNAggregation) => void;

  fetch: () => Promise<void>;

  toggleResultSelected: (filename: string) => void;
  rangeSelectResults: (filename: string) => void;
  clearModalSelection: () => void;

  createClusterFromSelected: () => void;
  addSelectedToGroup: (groupId: string) => Promise<void>;
  addSelectedToSourceCluster: () => Promise<void>;
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
  sourceClusterLabel: null,
  filter: "any",
  topN: 50,
  aggregation: "centroid",
  loading: false,
  progress: "",
  error: null,
  results: [],
  usedModels: [],
  patchesBlended: false,

  openForCluster: (cluster) => {
    const label = cluster.autoName
      ? `Cluster "${cluster.autoName}" (${cluster.images.length})`
      : `${cluster.images.length} images`;
    useSelectionStore.getState().clear("nn");
    set({
      open: true,
      queryLabel: label,
      querySource: { kind: "cluster", clusterId: cluster.id, images: cluster.images },
      sourceClusterLabel: cluster.confirmedGroup?.name ?? cluster.autoName ?? "this cluster",
      results: [],
      error: null,
    });
    get().fetch();
  },

  openForSelection: (filenames) => {
    const deduped = [...new Set(filenames)];
    useSelectionStore.getState().clear("nn");
    set({
      open: true,
      queryLabel: `${deduped.length} selected image${deduped.length === 1 ? "" : "s"}`,
      querySource: { kind: "selection", images: deduped },
      sourceClusterLabel: null,
      results: [],
      error: null,
    });
    get().fetch();
  },

  close: () => {
    if (_debounceTimer) {
      clearTimeout(_debounceTimer);
      _debounceTimer = null;
    }
    useSelectionStore.getState().clear("nn");
    set({
      open: false,
      querySource: null,
      queryLabel: "",
      sourceClusterLabel: null,
      results: [],
      error: null,
      loading: false,
      progress: "",
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

    const { weights, usePatches, clusterData } = useListStore.getState();
    const restrictToFilenames = clusterData?.scope?.subsetFilenames;

    set({ loading: true, progress: "Running NN query...", error: null });
    try {
      const start = await startSSE("/api/cluster/nn-query", {
        queryFilenames: querySource.images,
        topN,
        filter,
        aggregation,
        weights,
        usePatches,
        restrictToFilenames,
      });
      if (start.kind === "conflict") {
        set({ loading: false, progress: "", error: "Clustering in progress — retry shortly" });
        return;
      }

      let result: NNQueryResponse | null = null;
      let errMsg: string | null = null;
      await consumeSSE(start.response, {
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
    useSelectionStore.getState().toggle("nn", filename);
  },

  rangeSelectResults: (filename) => {
    const filenames = get().results.map((r) => r.filename);
    useSelectionStore.getState().rangeSelect("nn", filenames, filename);
  },

  clearModalSelection: () => {
    useSelectionStore.getState().clear("nn");
  },

  createClusterFromSelected: () => {
    const modalSelection = useSelectionStore.getState().contexts.nn;
    const { queryLabel } = get();
    if (modalSelection.size === 0) return;
    const insertClusterFromFilenames = useListStore.getState().insertClusterFromFilenames;
    const label = `NN: ${queryLabel}`.slice(0, 60);
    insertClusterFromFilenames(label, [...modalSelection]);
    useToastStore
      .getState()
      .showToast(`Created cluster with ${modalSelection.size} images`, "success");
    get().close();
  },

  addSelectedToGroup: async (groupId) => {
    const modalSelection = useSelectionStore.getState().contexts.nn;
    if (modalSelection.size === 0) return;
    const { groups, updateGroups, groupsLoaded } = useGroupStore.getState();
    const { showToast } = useToastStore.getState();

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

    updateGroups((prev) => addFilenamesToGroup(prev, groupId, toAdd));
    showToast(`Added ${toAdd.length} to "${group.name}"`, "success");

    // Flush so subsequent badge lookups see the new membership.
    await useGroupStore.getState().flushPending();
    useSelectionStore.getState().clear("nn");
    await get().fetch();
  },

  addSelectedToSourceCluster: async () => {
    const { querySource } = get();
    if (!querySource || querySource.kind !== "cluster") return;

    const modalSelection = useSelectionStore.getState().contexts.nn;
    if (modalSelection.size === 0) return;

    const { showToast } = useToastStore.getState();
    const list = useListStore.getState();
    if (!list.clusterData) return;

    const cluster = findClusterEverywhere(
      list.clusterData.clusters,
      list.splitChildren,
      querySource.clusterId,
    );
    if (!cluster) {
      showToast("Source cluster no longer exists", "error");
      return;
    }

    const existingImages = new Set(cluster.images);
    const selectedNew = [...modalSelection].filter((f) => !existingImages.has(f));

    let next: ClusterResultData;
    let toastMsg: string;
    let postCommit: (() => Promise<void>) | null = null;

    if (cluster.confirmedGroup) {
      const { groups, updateGroups, groupsLoaded, flushPending } = useGroupStore.getState();
      if (!groupsLoaded) {
        showToast("Groups still loading — please wait", "warning");
        return;
      }
      const group = groups.find((g) => g.id === cluster.confirmedGroup!.id);
      if (!group) {
        showToast("Confirmed group not found", "error");
        return;
      }
      const inGroup = new Set(group.images);
      const toAddToGroup = [...modalSelection].filter((f) => !inGroup.has(f));
      if (toAddToGroup.length === 0) {
        showToast("All selected images are already in this group", "warning");
        return;
      }

      updateGroups((prev) => addFilenamesToGroup(prev, group.id, toAddToGroup));
      dropCannotLinkAgainstGroup(toAddToGroup, group.id);

      next = {
        ...cluster,
        images: [...cluster.images, ...selectedNew],
        confirmedGroup: { ...cluster.confirmedGroup, images: [...group.images, ...toAddToGroup] },
      };
      toastMsg = `Added ${toAddToGroup.length} to "${group.name}"`;
      postCommit = flushPending;
    } else {
      if (selectedNew.length === 0) {
        showToast("All selected images are already in this cluster", "warning");
        return;
      }
      next = { ...cluster, images: [...cluster.images, ...selectedNew] };
      toastMsg = `Added ${selectedNew.length} to cluster "${cluster.autoName || cluster.id}"`;
    }

    useListStore.getState().applyClusterReplace(cluster.id, next);
    useListStore.setState({ treeStale: true });
    showToast(toastMsg, "success");
    if (postCommit) await postCommit();

    useSelectionStore.getState().clear("nn");
    await get().fetch();
  },
}));
