import { create } from "zustand";
import type {
  ClusterData,
  ClusterMetrics,
  ClusterResultData,
  ExpandResult,
  ImportClusterInput,
  SplitChildren,
  WeightConfig,
} from "../types.ts";
import { getErrorMessage, postJson } from "../utils/helpers.ts";
import { consolidateBlock } from "../utils/reorder.ts";
import { consumeSSE } from "../utils/sse.ts";
import { useConstraintsStore } from "./constraintsStore.ts";
import { flushGroupPersist, useGroupStore } from "./groupStore.ts";
import { useImageStore } from "./imageStore.ts";
import { useUIStore } from "./uiStore.ts";

/**
 * When images are added to a confirmed group, any cannot-link constraint
 * between those images and that group becomes stale (the user just told us
 * the image *does* belong). Fire-and-forget cleanup; failures are non-fatal
 * because the Rust pre-merge will absorb the image into the group anyway.
 */
function dropCannotLinkAgainstGroup(filenames: string[], groupId: string) {
  const store = useConstraintsStore.getState();
  for (const f of filenames) {
    if (store.isCannotLinked(f, groupId)) {
      store.removeImageGroupCannotLink(f, groupId).catch(() => {});
    }
  }
}

/** Compare-mode (merge…) state — open candidate stack against a source cluster. */
export interface CompareState {
  sourceClusterId: string;
  /** Ordered list of candidate cluster IDs (search-added at the front). */
  candidateOrder: string[];
  /** Centroid distance per candidate (for hover/tooltip). */
  candidateDistance: Record<string, number>;
  /** IDs added via the search field — kept distinct so they always show. */
  manuallyAdded: Set<string>;
  /** Checked candidates included in the merge. */
  checked: Set<string>;
  includeConfirmedGroups: boolean;
  loading: boolean;
  /** Set when merge is awaiting a winner pick across multiple confirmed groups. */
  pendingWinnerChoice: { confirmedGroupIds: string[] } | null;
}

/** Expand-modal state — staging set of images to pull into a source cluster. */
export interface ExpandState {
  sourceClusterId: string;
  /** Candidates pre-sorted ascending by distance. */
  candidates: { filename: string; distance: number }[];
  p90Intra: number;
  maxDistance: number;
  /** Slider position, multiplier of p90Intra. Range 0.5–4. */
  thresholdMultiplier: number;
  includeConfirmedGroups: boolean;
  checked: Set<string>;
  loading: boolean;
}

interface ClusterState {
  clusterData: ClusterData | null;
  /** Snapshot of the pre-scope global clusterData, restored on exitScope. */
  globalClusterData: ClusterData | null;
  loading: boolean;
  progress: string;
  mergeSelection: Set<string>;
  /** Composite keys `"clusterId:filename"` — parse with parseImageKey. */
  selectedImages: Set<string>;
  collapsedClusters: Set<string>;
  lastClickedImage: { clusterId: string; index: number } | null;
  lightbox: { clusterId: string; imageIndex: number } | null;
  treeStale: boolean;
  focusedClusterId: string | null;
  weights: WeightConfig;
  usePatches: boolean;
  useRerank: boolean;
  /** Blend strength for re-rank distance (0=cosine only, 1=rerank only). */
  rerankBlend: number;

  // Tree-navigation state
  /** Per-cluster {cohesion, isolation, stability} keyed by cluster id. */
  metrics: Record<string, ClusterMetrics>;
  /** Children produced by an inline split, keyed by parent cluster id. Presence == expanded. */
  splitChildren: Record<string, SplitChildren>;
  compare: CompareState | null;
  expand: ExpandState | null;

  setWeights: (w: WeightConfig) => void;
  setUsePatches: (v: boolean) => void;
  setUseRerank: (v: boolean) => void;
  setRerankBlend: (v: number) => void;
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

  runScopedCluster: (groupIds: string[], opts?: { nClusters?: number }) => Promise<void>;
  exitScope: () => void;
  recutScopedByN: (n: number) => Promise<void>;
  recutScopedByThreshold: (t: number) => Promise<void>;
  recutScopedAdaptive: (minSize: number) => Promise<void>;

  insertClusterFromFilenames: (name: string, filenames: string[]) => void;

  // Tree-navigation actions
  refreshMetrics: () => Promise<void>;
  openCompare: (sourceClusterId: string) => Promise<void>;
  closeCompare: () => void;
  toggleCompareCandidate: (id: string) => void;
  setCompareIncludeConfirmedGroups: (v: boolean) => void;
  addClusterToCompare: (id: string) => void;
  confirmMerge: () => void;
  /** Commit the merge after the user picks which confirmed group is the winner. */
  commitMergeWithWinner: (winnerGroupId: string) => void;
  cancelWinnerChoice: () => void;
  toggleSplit: (parentId: string) => Promise<void>;
  collapseSplit: (parentId: string) => void;
  openExpand: (sourceClusterId: string) => Promise<void>;
  closeExpand: () => void;
  toggleExpandFile: (filename: string) => void;
  setExpandThreshold: (multiplier: number) => void;
  setExpandIncludeConfirmedGroups: (v: boolean) => void;
  confirmExpand: () => void;
}

// Re-entrancy guard for refreshMetrics. Module-scoped so it doesn't leak into
// the publicly observable store shape.
let metricsInFlight = false;

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
    metrics: {},
    splitChildren: {},
    compare: null,
    expand: null,
  };
}

/** Parse a "clusterId:filename" composite key from `selectedImages`. */
export function parseImageKey(key: string): { clusterId: string; filename: string } {
  const sep = key.indexOf(":");
  return { clusterId: key.slice(0, sep), filename: key.slice(sep + 1) };
}

/** Extract deduped filenames from a `selectedImages` Set (composite keys). */
export function filenamesFromSelectedImages(selection: Iterable<string>): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const key of selection) {
    const { filename } = parseImageKey(key);
    if (!seen.has(filename)) {
      seen.add(filename);
      out.push(filename);
    }
  }
  return out;
}

export const useClusterStore = create<ClusterState>((set, get) => {
  async function doRecut(url: string, body: unknown, msg = "Re-cutting tree...") {
    set({ loading: true, progress: msg });
    try {
      const res = await postJson(url, body);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        set({ loading: false, progress: `Error: ${err.error ?? "recut failed"}` });
        return;
      }
      const result = (await res.json()) as ClusterData;
      set(applyClusterResult(result));
    } catch (err) {
      set({ loading: false, progress: `Error: ${getErrorMessage(err, "recut failed")}` });
    }
  }

  /** Replace a cluster in clusterData OR in splitChildren, wherever it lives. */
  function applyClusterReplace(id: string, next: ClusterResultData) {
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
  }

  /** Remove a set of cluster IDs from clusterData AND any matching split children. */
  function applyClusterRemoval(ids: Set<string>) {
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
  }

  /** Remove `filenames` from every cluster except the one with `keepId`. */
  function stripImagesFromOtherClusters(keepId: string, filenames: Set<string>) {
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
  }

  function commitMergeIntoGroup(participants: ClusterResultData[], winnerGroupId: string) {
    const merged = unionImages(participants);
    const losingGroupIds = new Set<string>();
    let winnerName = "";
    for (const c of participants) {
      const cg = c.confirmedGroup;
      if (!cg) continue;
      if (cg.id === winnerGroupId) winnerName = cg.name;
      else losingGroupIds.add(cg.id);
    }

    const { updateGroups } = useGroupStore.getState();
    updateGroups((prev) => {
      const next = prev
        .filter((g) => !losingGroupIds.has(g.id))
        .map((g) => {
          if (g.id !== winnerGroupId) return g;
          const seen = new Set(g.images);
          const additions = merged.filter((f) => !seen.has(f));
          return { ...g, images: [...g.images, ...additions] };
        });
      if (!winnerName) {
        const w = next.find((g) => g.id === winnerGroupId);
        if (w) winnerName = w.name;
      }
      return next;
    });

    dropCannotLinkAgainstGroup(merged, winnerGroupId);
    applyClusterRemoval(new Set(participants.map((c) => c.id)));
    set({ compare: null, treeStale: true });
    useUIStore
      .getState()
      .showToast(
        losingGroupIds.size > 0
          ? `Merged ${losingGroupIds.size + 1} groups into "${winnerName}" (${merged.length} images)`
          : `Extended "${winnerName}" with ${merged.length} images`,
        "success",
      );
  }

  return {
    clusterData: null,
    globalClusterData: null,
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
    useRerank: true,
    rerankBlend: 0.7,

    metrics: {},
    splitChildren: {},
    compare: null,
    expand: null,

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

    fetchClusters: async (nClusters = 200) => {
      set({ loading: true, progress: "Starting clustering..." });
      try {
        const { weights, usePatches, useRerank, rerankBlend } = get();
        const response = await fetch("/api/cluster", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ nClusters, weights, usePatches, useRerank, rerankBlend }),
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
        set({ loading: false, progress: `Error: ${getErrorMessage(err, "fetch failed")}` });
      }
    },

    recutClusters: (nClusters) => doRecut("/api/cluster/recut", { nClusters }),
    recutByThreshold: (threshold) => doRecut("/api/cluster/recut", { threshold }),
    recutAdaptive: (minClusterSize) =>
      doRecut("/api/cluster/recut", { minClusterSize }, "Adaptive re-cut..."),

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
      const cluster = findClusterEverywhere(
        get().clusterData?.clusters ?? [],
        get().splitChildren,
        clusterId,
      );
      const index = cluster?.images.indexOf(filename) ?? -1;
      set({ selectedImages: sel, lastClickedImage: { clusterId, index } });
    },

    rangeSelectImages: (clusterId, toIndex) => {
      const { lastClickedImage, clusterData, splitChildren } = get();
      if (!lastClickedImage || lastClickedImage.clusterId !== clusterId || !clusterData) return;
      const cluster = findClusterEverywhere(clusterData.clusters, splitChildren, clusterId);
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
      applyClusterRemoval(new Set([clusterId]));
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
          splitFrom: source.id,
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

      const { selectedImages } = get();
      const selectedInCluster: string[] = [];
      for (const key of selectedImages) {
        const { clusterId, filename } = parseImageKey(key);
        if (clusterId === cluster.id && !confirmedSet.has(filename)) {
          selectedInCluster.push(filename);
        }
      }
      const toAdd = selectedInCluster.length > 0 ? selectedInCluster : suggested;

      const { updateGroups } = useGroupStore.getState();
      const { showToast } = useUIStore.getState();
      updateGroups((prev) =>
        prev.map((g) => (g.id === groupId ? { ...g, images: [...g.images, ...toAdd] } : g)),
      );
      dropCannotLinkAgainstGroup(toAdd, groupId);
      showToast(`Added ${toAdd.length} images to "${cluster.confirmedGroup.name}"`, "success");

      const newConfirmedImages = [...cluster.confirmedGroup.images, ...toAdd];
      const next: ClusterResultData = {
        ...cluster,
        confirmedGroup: { ...cluster.confirmedGroup, images: newConfirmedImages },
      };
      applyClusterReplace(cluster.id, next);

      // Auto-collapse if all images are now confirmed
      const collapsed = new Set(get().collapsedClusters);
      const allConfirmed = newConfirmedImages.length >= cluster.images.length;
      if (allConfirmed) collapsed.add(cluster.id);

      set({
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
      } catch (err) {
        console.warn("Failed to load cached clusters:", err);
      }
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

    runScopedCluster: async (groupIds, opts) => {
      if (groupIds.length === 0) return;
      // Flush pending group persists so Rust sees latest .reorder-groups.json
      await flushGroupPersist();

      const prev = get().clusterData;
      // Preserve the pre-scope global view (unless we're already in scope).
      const globalBackup = prev?.scope ? get().globalClusterData : prev;

      set({
        loading: true,
        progress: "Starting scoped clustering...",
        globalClusterData: globalBackup,
      });
      try {
        const { weights } = get();
        const response = await fetch("/api/cluster/scoped", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ groupIds, nClusters: opts?.nClusters, weights }),
        });
        if (response.status === 409) {
          set({ loading: false, progress: "Clustering already in progress" });
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
          set({ loading: false, progress: "No scoped results returned" });
        }
      } catch (err) {
        set({
          loading: false,
          progress: `Error: ${getErrorMessage(err, "scoped cluster failed")}`,
        });
      }
    },

    exitScope: () => {
      const backup = get().globalClusterData;
      if (!backup) {
        set({ clusterData: null, globalClusterData: null });
        return;
      }
      set({
        clusterData: backup,
        globalClusterData: null,
        mergeSelection: new Set(),
        selectedImages: new Set(),
        lastClickedImage: null,
      });
    },

    recutScopedByN: async (n) => {
      const scopeKey = get().clusterData?.scope?.scopeKey;
      if (!scopeKey) return;
      await doRecut(
        "/api/cluster/scoped/recut",
        { scopeKey, nClusters: n },
        "Re-cutting scoped tree...",
      );
    },
    recutScopedByThreshold: async (threshold) => {
      const scopeKey = get().clusterData?.scope?.scopeKey;
      if (!scopeKey) return;
      await doRecut(
        "/api/cluster/scoped/recut",
        { scopeKey, threshold },
        "Re-cutting scoped tree...",
      );
    },
    recutScopedAdaptive: async (minClusterSize) => {
      const scopeKey = get().clusterData?.scope?.scopeKey;
      if (!scopeKey) return;
      await doRecut(
        "/api/cluster/scoped/recut",
        { scopeKey, minClusterSize },
        "Adaptive scoped re-cut...",
      );
    },

    insertClusterFromFilenames: (name, filenames) => {
      const { clusterData } = get();
      if (!clusterData || filenames.length === 0) return;
      const deduped = [...new Set(filenames)];
      const newCluster: ClusterResultData = {
        id: `nn_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        autoName: name || "From NN",
        autoTags: [],
        images: deduped,
        confirmedGroup: null,
      };
      set({
        clusterData: { ...clusterData, clusters: [newCluster, ...clusterData.clusters] },
        focusedClusterId: newCluster.id,
      });
    },

    refreshMetrics: async () => {
      const { clusterData, splitChildren, weights } = get();
      if (!clusterData || metricsInFlight) return;
      const all = collectAllClusters(clusterData.clusters, splitChildren);
      if (all.length === 0) return;
      metricsInFlight = true;
      try {
        const res = await postJson("/api/cluster/tree-nav/metrics", {
          clusters: all.map((c) => ({ id: c.id, images: c.images })),
          weights,
        });
        if (!res.ok) return;
        const body = (await res.json()) as { metrics: Record<string, ClusterMetrics> };
        set({ metrics: body.metrics });
      } catch {
        // metrics are best-effort — drop on error
      } finally {
        metricsInFlight = false;
      }
    },

    openCompare: async (sourceClusterId) => {
      const { clusterData, splitChildren, weights } = get();
      if (!clusterData) return;
      const source = findClusterEverywhere(clusterData.clusters, splitChildren, sourceClusterId);
      if (!source) return;

      set({
        compare: {
          sourceClusterId,
          candidateOrder: [],
          candidateDistance: {},
          manuallyAdded: new Set(),
          checked: new Set(),
          includeConfirmedGroups: true,
          loading: true,
          pendingWinnerChoice: null,
        },
      });

      const allClusters = collectAllClusters(clusterData.clusters, splitChildren).filter(
        (c) => c.id !== sourceClusterId,
      );
      try {
        const res = await postJson("/api/cluster/tree-nav/merge-candidates", {
          sourceImages: source.images,
          candidates: allClusters.map((c) => ({ id: c.id, images: c.images })),
          weights,
        });
        if (!res.ok) {
          set({ compare: null });
          useUIStore.getState().showToast("Failed to load merge candidates", "error");
          return;
        }
        const body = (await res.json()) as { scores: { id: string; distance: number }[] };
        const distance: Record<string, number> = {};
        for (const s of body.scores) distance[s.id] = s.distance;
        const candidateOrder = body.scores.map((s) => s.id);

        const compare = get().compare;
        if (!compare || compare.sourceClusterId !== sourceClusterId) return;
        set({
          compare: {
            ...compare,
            candidateOrder,
            candidateDistance: distance,
            loading: false,
          },
        });
      } catch (err) {
        set({ compare: null });
        useUIStore.getState().showToast(getErrorMessage(err, "Compare failed"), "error");
      }
    },

    closeCompare: () => set({ compare: null }),

    toggleCompareCandidate: (id) => {
      const { compare } = get();
      if (!compare) return;
      const checked = new Set(compare.checked);
      if (checked.has(id)) checked.delete(id);
      else checked.add(id);
      set({ compare: { ...compare, checked } });
    },

    setCompareIncludeConfirmedGroups: (v) => {
      const { compare } = get();
      if (!compare) return;
      // Drop checks for clusters that disappear when toggling off
      let checked = compare.checked;
      if (!v) {
        const { clusterData, splitChildren } = get();
        if (clusterData) {
          checked = new Set(compare.checked);
          for (const id of compare.checked) {
            const c = findClusterEverywhere(clusterData.clusters, splitChildren, id);
            if (c?.confirmedGroup) checked.delete(id);
          }
        }
      }
      set({ compare: { ...compare, includeConfirmedGroups: v, checked } });
    },

    addClusterToCompare: (id) => {
      const { compare, clusterData, splitChildren } = get();
      if (!compare || !clusterData) return;
      if (id === compare.sourceClusterId) return;
      const target = findClusterEverywhere(clusterData.clusters, splitChildren, id);
      if (!target) return;
      const order = compare.candidateOrder.filter((x) => x !== id);
      order.unshift(id);
      const manuallyAdded = new Set(compare.manuallyAdded);
      manuallyAdded.add(id);
      set({ compare: { ...compare, candidateOrder: order, manuallyAdded } });
    },

    confirmMerge: () => {
      const { compare, clusterData, splitChildren } = get();
      if (!compare || !clusterData) return;
      const source = findClusterEverywhere(
        clusterData.clusters,
        splitChildren,
        compare.sourceClusterId,
      );
      if (!source) return;
      if (compare.checked.size === 0) {
        set({ compare: null });
        return;
      }

      const allParticipants = collectMergeParticipants(
        clusterData.clusters,
        splitChildren,
        source,
        compare.checked,
      );
      const confirmedParticipants = allParticipants.filter((c) => !!c.confirmedGroup);

      // Multiple confirmed groups → defer commit until the user picks a winner.
      if (confirmedParticipants.length >= 2) {
        set({
          compare: {
            ...compare,
            pendingWinnerChoice: {
              confirmedGroupIds: confirmedParticipants.map((c) => c.confirmedGroup!.id),
            },
          },
        });
        return;
      }

      // Single confirmed group: extend it with the union of all participants' images.
      if (confirmedParticipants.length === 1) {
        const cg = confirmedParticipants[0]!.confirmedGroup!;
        commitMergeIntoGroup(allParticipants, cg.id);
        return;
      }

      // No confirmed groups: union into the source cluster, drop the others.
      const merged = unionImages(allParticipants);
      const removeIds = new Set(allParticipants.filter((c) => c.id !== source.id).map((c) => c.id));
      const newSource: ClusterResultData = { ...source, images: merged.sort() };
      applyClusterReplace(source.id, newSource);
      applyClusterRemoval(removeIds);
      set({ compare: null });
    },

    commitMergeWithWinner: (winnerGroupId) => {
      const { compare, clusterData, splitChildren } = get();
      if (!compare || !clusterData || !compare.pendingWinnerChoice) return;
      const source = findClusterEverywhere(
        clusterData.clusters,
        splitChildren,
        compare.sourceClusterId,
      );
      if (!source) return;
      const allParticipants = collectMergeParticipants(
        clusterData.clusters,
        splitChildren,
        source,
        compare.checked,
      );
      commitMergeIntoGroup(allParticipants, winnerGroupId);
    },

    cancelWinnerChoice: () => {
      const { compare } = get();
      if (!compare) return;
      set({ compare: { ...compare, pendingWinnerChoice: null } });
    },

    toggleSplit: async (parentId) => {
      const { splitChildren, clusterData } = get();
      if (!clusterData) return;

      if (splitChildren[parentId]) {
        return get().collapseSplit(parentId);
      }

      const parent = findClusterEverywhere(clusterData.clusters, splitChildren, parentId);
      if (!parent || parent.images.length < 2) return;

      let kids: SplitChildren;
      try {
        const res = await postJson("/api/cluster/tree-nav/split", { images: parent.images });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          useUIStore.getState().showToast(err.error ?? "Cannot split this cluster", "warning");
          return;
        }
        kids = (await res.json()) as SplitChildren;
      } catch (err) {
        useUIStore.getState().showToast(getErrorMessage(err, "Split failed"), "error");
        return;
      }
      kids = {
        childA: { ...kids.childA, splitFrom: parentId },
        childB: { ...kids.childB, splitFrom: parentId },
      };
      set({ splitChildren: { ...get().splitChildren, [parentId]: kids } });
    },

    collapseSplit: (parentId) => {
      const newChildren = { ...get().splitChildren };
      const stack = [parentId];
      while (stack.length) {
        const id = stack.pop()!;
        const kids = newChildren[id];
        if (kids) stack.push(kids.childA.id, kids.childB.id);
        delete newChildren[id];
      }
      set({ splitChildren: newChildren });
    },

    openExpand: async (sourceClusterId) => {
      const { clusterData, splitChildren, weights } = get();
      if (!clusterData) return;
      const source = findClusterEverywhere(clusterData.clusters, splitChildren, sourceClusterId);
      if (!source) return;

      set({
        expand: {
          sourceClusterId,
          candidates: [],
          p90Intra: 0,
          maxDistance: 0,
          thresholdMultiplier: 1.5,
          includeConfirmedGroups: false,
          checked: new Set(),
          loading: true,
        },
      });

      try {
        const res = await postJson("/api/cluster/tree-nav/expand-candidates", {
          sourceImages: source.images,
          weights,
        });
        if (!res.ok) {
          set({ expand: null });
          useUIStore.getState().showToast("Failed to compute expand candidates", "error");
          return;
        }
        const body = (await res.json()) as ExpandResult;
        const expand = get().expand;
        if (!expand || expand.sourceClusterId !== sourceClusterId) return;
        set({
          expand: {
            ...expand,
            candidates: body.candidates,
            p90Intra: body.p90Intra,
            maxDistance: body.maxDistance,
            loading: false,
          },
        });
      } catch (err) {
        set({ expand: null });
        useUIStore.getState().showToast(getErrorMessage(err, "Expand failed"), "error");
      }
    },

    closeExpand: () => set({ expand: null }),

    toggleExpandFile: (filename) => {
      const { expand } = get();
      if (!expand) return;
      const checked = new Set(expand.checked);
      if (checked.has(filename)) checked.delete(filename);
      else checked.add(filename);
      set({ expand: { ...expand, checked } });
    },

    setExpandThreshold: (multiplier) => {
      const { expand } = get();
      if (!expand) return;
      set({ expand: { ...expand, thresholdMultiplier: multiplier } });
    },

    setExpandIncludeConfirmedGroups: (v) => {
      const { expand } = get();
      if (!expand) return;
      set({ expand: { ...expand, includeConfirmedGroups: v } });
    },

    confirmExpand: () => {
      const { expand, clusterData, splitChildren } = get();
      if (!expand || !clusterData || expand.checked.size === 0) {
        if (expand) set({ expand: null });
        return;
      }
      const source = findClusterEverywhere(
        clusterData.clusters,
        splitChildren,
        expand.sourceClusterId,
      );
      if (!source) {
        set({ expand: null });
        return;
      }

      const checked = expand.checked;
      const newSource: ClusterResultData = {
        ...source,
        images: dedupeAppend(source.images, [...checked]),
      };
      applyClusterReplace(source.id, newSource);
      stripImagesFromOtherClusters(source.id, checked);

      set({ expand: null });
      useUIStore
        .getState()
        .showToast(`Added ${checked.size} images to "${source.autoName}"`, "success");
    },
  };
});

// ── Helpers ────────────────────────────────────────────────────────────────

/** Recursively collect every visible cluster (top-level + every split child). */
function collectAllClusters(
  topLevel: ClusterResultData[],
  splitChildren: Record<string, SplitChildren>,
): ClusterResultData[] {
  const out: ClusterResultData[] = [];
  function visit(c: ClusterResultData) {
    out.push(c);
    const kids = splitChildren[c.id];
    if (kids) {
      visit(kids.childA);
      visit(kids.childB);
    }
  }
  for (const c of topLevel) visit(c);
  return out;
}

/** Locate a cluster by id in either the top-level array or any split children entry. */
export function findClusterEverywhere(
  topLevel: ClusterResultData[],
  splitChildren: Record<string, SplitChildren>,
  id: string,
): ClusterResultData | undefined {
  const top = topLevel.find((c) => c.id === id);
  if (top) return top;
  for (const kids of Object.values(splitChildren)) {
    if (kids.childA.id === id) return kids.childA;
    if (kids.childB.id === id) return kids.childB;
  }
  return undefined;
}

function dedupeAppend(base: string[], adds: string[]): string[] {
  const seen = new Set(base);
  const out = [...base];
  for (const a of adds) {
    if (!seen.has(a)) {
      seen.add(a);
      out.push(a);
    }
  }
  return out;
}

/** Resolve {source, ...checkedClusters} from compare-state ids, deduped by id. */
function collectMergeParticipants(
  clusters: ClusterResultData[],
  splitChildren: Record<string, SplitChildren>,
  source: ClusterResultData,
  checkedIds: Set<string>,
): ClusterResultData[] {
  const out = [source];
  const seenIds = new Set([source.id]);
  for (const id of checkedIds) {
    if (seenIds.has(id)) continue;
    const c = findClusterEverywhere(clusters, splitChildren, id);
    if (c) {
      out.push(c);
      seenIds.add(id);
    }
  }
  return out;
}

/** Compute the deduped union of all images across participants in input order. */
function unionImages(participants: ClusterResultData[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const c of participants) {
    for (const f of c.images) {
      if (!seen.has(f)) {
        seen.add(f);
        out.push(f);
      }
    }
  }
  return out;
}
