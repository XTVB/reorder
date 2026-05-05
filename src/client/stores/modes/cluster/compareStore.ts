// Compare-mode panel state. The `checked` set of candidate cluster IDs lives
// in selectionStore "compare" context; everything else (candidate ordering,
// distances, manual additions, winner-choice modal flag) is local to this store.

import { create } from "zustand";
import { postJson } from "../../../api/client.ts";
import { getErrorMessage } from "../../../utils/helpers.ts";
import { useSelectionStore } from "../../core/selectionStore.ts";
import { useToastStore } from "../../core/toastStore.ts";
import { commitMergeIntoGroup } from "./interactionsStore.ts";
import { useListStore } from "./listStore.ts";
import {
  collectAllClusters,
  collectMergeParticipants,
  findClusterEverywhere,
  unionImages,
} from "./tree-helpers.ts";

/** Compare-mode (merge…) state — open candidate stack against a source cluster. */
export interface CompareState {
  sourceClusterId: string;
  /** Ordered list of candidate cluster IDs (search-added at the front). */
  candidateOrder: string[];
  /** Centroid distance per candidate (for hover/tooltip). */
  candidateDistance: Record<string, number>;
  /** IDs added via the search field — kept distinct so they always show. */
  manuallyAdded: Set<string>;
  includeConfirmedGroups: boolean;
  loading: boolean;
  /** Set when merge is awaiting a winner pick across multiple confirmed groups. */
  pendingWinnerChoice: { confirmedGroupIds: string[] } | null;
}

interface CompareSlice {
  compare: CompareState | null;

  openCompare: (sourceClusterId: string) => Promise<void>;
  closeCompare: () => void;
  toggleCompareCandidate: (id: string) => void;
  setCompareIncludeConfirmedGroups: (v: boolean) => void;
  addClusterToCompare: (id: string) => void;
  confirmMerge: () => void;
  /** Commit the merge after the user picks which confirmed group is the winner. */
  commitMergeWithWinner: (winnerGroupId: string) => void;
  cancelWinnerChoice: () => void;
}

export const useCompareStore = create<CompareSlice>((set, get) => ({
  compare: null,

  openCompare: async (sourceClusterId) => {
    const list = useListStore.getState();
    const { clusterData, splitChildren, weights } = list;
    if (!clusterData) return;
    const source = findClusterEverywhere(clusterData.clusters, splitChildren, sourceClusterId);
    if (!source) return;

    // Reset selection context for this compare session.
    useSelectionStore.getState().clear("compare");

    set({
      compare: {
        sourceClusterId,
        candidateOrder: [],
        candidateDistance: {},
        manuallyAdded: new Set(),
        includeConfirmedGroups: true,
        loading: true,
        pendingWinnerChoice: null,
      },
    });

    const allClusters = collectAllClusters(clusterData.clusters, splitChildren).filter(
      (c) => c.id !== sourceClusterId,
    );
    try {
      const body = await postJson<{ scores: { id: string; distance: number }[] }>(
        "/api/cluster/tree-nav/merge-candidates",
        {
          sourceImages: source.images,
          candidates: allClusters.map((c) => ({ id: c.id, images: c.images })),
          weights,
        },
      );
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
      useToastStore.getState().showToast(getErrorMessage(err, "Compare failed"), "error");
    }
  },

  closeCompare: () => {
    useSelectionStore.getState().clear("compare");
    set({ compare: null });
  },

  toggleCompareCandidate: (id) => {
    useSelectionStore.getState().toggle("compare", id);
  },

  setCompareIncludeConfirmedGroups: (v) => {
    const { compare } = get();
    if (!compare) return;
    // Drop checks for clusters that disappear when toggling off
    if (!v) {
      const list = useListStore.getState();
      if (list.clusterData) {
        const sel = useSelectionStore.getState();
        const checked = sel.contexts.compare;
        const toRemove: string[] = [];
        for (const id of checked) {
          const c = findClusterEverywhere(list.clusterData.clusters, list.splitChildren, id);
          if (c?.confirmedGroup) toRemove.push(id);
        }
        if (toRemove.length > 0) sel.remove("compare", toRemove);
      }
    }
    set({ compare: { ...compare, includeConfirmedGroups: v } });
  },

  addClusterToCompare: (id) => {
    const { compare } = get();
    const list = useListStore.getState();
    if (!compare || !list.clusterData) return;
    if (id === compare.sourceClusterId) return;
    const target = findClusterEverywhere(list.clusterData.clusters, list.splitChildren, id);
    if (!target) return;
    const order = compare.candidateOrder.filter((x) => x !== id);
    order.unshift(id);
    const manuallyAdded = new Set(compare.manuallyAdded);
    manuallyAdded.add(id);
    set({ compare: { ...compare, candidateOrder: order, manuallyAdded } });
  },

  confirmMerge: () => {
    const { compare } = get();
    const list = useListStore.getState();
    if (!compare || !list.clusterData) return;
    const source = findClusterEverywhere(
      list.clusterData.clusters,
      list.splitChildren,
      compare.sourceClusterId,
    );
    if (!source) return;
    const checked = useSelectionStore.getState().contexts.compare;
    if (checked.size === 0) {
      get().closeCompare();
      return;
    }

    const allParticipants = collectMergeParticipants(
      list.clusterData.clusters,
      list.splitChildren,
      source,
      checked,
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
      get().closeCompare();
      return;
    }

    // No confirmed groups: union into the source cluster, drop the others.
    const merged = unionImages(allParticipants);
    const removeIds = new Set(allParticipants.filter((c) => c.id !== source.id).map((c) => c.id));
    const newSource = { ...source, images: merged.sort() };
    list.applyClusterReplace(source.id, newSource);
    list.applyClusterRemoval(removeIds);
    get().closeCompare();
  },

  commitMergeWithWinner: (winnerGroupId) => {
    const { compare } = get();
    const list = useListStore.getState();
    if (!compare || !list.clusterData || !compare.pendingWinnerChoice) return;
    const source = findClusterEverywhere(
      list.clusterData.clusters,
      list.splitChildren,
      compare.sourceClusterId,
    );
    if (!source) return;
    const checked = useSelectionStore.getState().contexts.compare;
    const allParticipants = collectMergeParticipants(
      list.clusterData.clusters,
      list.splitChildren,
      source,
      checked,
    );
    commitMergeIntoGroup(allParticipants, winnerGroupId);
    get().closeCompare();
  },

  cancelWinnerChoice: () => {
    const { compare } = get();
    if (!compare) return;
    set({ compare: { ...compare, pendingWinnerChoice: null } });
  },
}));
