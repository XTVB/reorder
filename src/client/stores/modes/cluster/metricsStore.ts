// Tree-nav per-cluster metrics (cohesion, isolation, stability). Refresh is
// debounced by a re-entrancy guard so concurrent calls collapse into one.

import { create } from "zustand";
import { postJson } from "../../../api/client.ts";
import type { ClusterMetrics } from "../../../types.ts";
import { useListStore } from "./listStore.ts";
import { collectAllClusters } from "./tree-helpers.ts";

interface MetricsSlice {
  /** Per-cluster {cohesion, isolation, stability} keyed by cluster id. */
  metrics: Record<string, ClusterMetrics>;
  refreshMetrics: () => Promise<void>;
}

// Re-entrancy guard: collapses concurrent refreshMetrics() calls into one.
let metricsInFlight = false;

export const useMetricsStore = create<MetricsSlice>((set) => ({
  metrics: {},

  refreshMetrics: async () => {
    const list = useListStore.getState();
    const { clusterData, splitChildren, weights } = list;
    if (!clusterData || metricsInFlight) return;
    const all = collectAllClusters(clusterData.clusters, splitChildren);
    if (all.length === 0) return;
    metricsInFlight = true;
    try {
      const body = await postJson<{ metrics: Record<string, ClusterMetrics> }>(
        "/api/cluster/tree-nav/metrics",
        {
          clusters: all.map((c) => ({ id: c.id, images: c.images })),
          weights,
        },
      );
      set({ metrics: body.metrics });
    } catch {
      // metrics are best-effort — drop on error
    } finally {
      metricsInFlight = false;
    }
  },
}));
