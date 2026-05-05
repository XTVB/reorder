// Expand-modal state. The `checked` set of filenames lives in selectionStore
// "expand" context; everything else (candidates, threshold, source id) is local.

import { create } from "zustand";
import { postJson } from "../../../api/client.ts";
import type { ExpandResult } from "../../../types.ts";
import { getErrorMessage } from "../../../utils/helpers.ts";
import { useSelectionStore } from "../../core/selectionStore.ts";
import { useToastStore } from "../../core/toastStore.ts";
import { useListStore } from "./listStore.ts";
import { dedupeAppend, findClusterEverywhere } from "./tree-helpers.ts";

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
  loading: boolean;
}

interface ExpandSlice {
  expand: ExpandState | null;

  openExpand: (sourceClusterId: string) => Promise<void>;
  closeExpand: () => void;
  toggleExpandFile: (filename: string) => void;
  setExpandThreshold: (multiplier: number) => void;
  setExpandIncludeConfirmedGroups: (v: boolean) => void;
  confirmExpand: () => void;
}

export const useExpandStore = create<ExpandSlice>((set, get) => ({
  expand: null,

  openExpand: async (sourceClusterId) => {
    const list = useListStore.getState();
    const { clusterData, splitChildren, weights } = list;
    if (!clusterData) return;
    const source = findClusterEverywhere(clusterData.clusters, splitChildren, sourceClusterId);
    if (!source) return;

    useSelectionStore.getState().clear("expand");

    set({
      expand: {
        sourceClusterId,
        candidates: [],
        p90Intra: 0,
        maxDistance: 0,
        thresholdMultiplier: 1.5,
        includeConfirmedGroups: false,
        loading: true,
      },
    });

    try {
      const body = await postJson<ExpandResult>("/api/cluster/tree-nav/expand-candidates", {
        sourceImages: source.images,
        weights,
      });
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
      useToastStore.getState().showToast(getErrorMessage(err, "Expand failed"), "error");
    }
  },

  closeExpand: () => {
    useSelectionStore.getState().clear("expand");
    set({ expand: null });
  },

  toggleExpandFile: (filename) => {
    useSelectionStore.getState().toggle("expand", filename);
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
    const { expand } = get();
    const list = useListStore.getState();
    const checked = useSelectionStore.getState().contexts.expand;
    if (!expand || !list.clusterData || checked.size === 0) {
      if (expand) get().closeExpand();
      return;
    }
    const source = findClusterEverywhere(
      list.clusterData.clusters,
      list.splitChildren,
      expand.sourceClusterId,
    );
    if (!source) {
      get().closeExpand();
      return;
    }

    const newSource = {
      ...source,
      images: dedupeAppend(source.images, [...checked]),
    };
    list.applyClusterReplace(source.id, newSource);
    list.stripImagesFromOtherClusters(source.id, checked);

    const count = checked.size;
    get().closeExpand();
    useToastStore.getState().showToast(`Added ${count} images to "${source.autoName}"`, "success");
  },
}));
