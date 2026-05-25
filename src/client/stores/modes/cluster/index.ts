// Cluster mode sub-store barrel + cross-store wiring. Importing this module
// installs a single subscription that clears expand/metrics/selection state
// whenever the cluster *shape* changes (id set or per-cluster image counts) —
// renames and other identity-only mutations are ignored.

import type { ClusterData } from "../../../types.ts";
import { useSelectionStore } from "../../core/selectionStore.ts";
import { useExpandStore } from "./expandStore.ts";
import { useInteractionsStore } from "./interactionsStore.ts";
import { useListStore } from "./listStore.ts";
import { useMetricsStore } from "./metricsStore.ts";

export { type ExpandState, useExpandStore } from "./expandStore.ts";
export { commitMergeIntoGroup, useInteractionsStore } from "./interactionsStore.ts";
export { useListStore } from "./listStore.ts";
export { useMetricsStore } from "./metricsStore.ts";
export { useSplitStore } from "./splitStore.ts";
export {
  collectAllClusters,
  dedupeAppend,
  filenamesFromSelectedImages,
  findClusterEverywhere,
  parseImageKey,
  unionImages,
} from "./tree-helpers.ts";

export function clusterShapeHash(d: ClusterData | null): string {
  if (!d) return "";
  return d.clusters.map((c) => `${c.id}:${c.images.length}`).join("|");
}

let _prevShape = clusterShapeHash(useListStore.getState().clusterData);
useListStore.subscribe((state) => {
  const next = clusterShapeHash(state.clusterData);
  if (next === _prevShape) return;
  _prevShape = next;
  if (useExpandStore.getState().expand !== null) {
    useExpandStore.setState({ expand: null });
  }
  if (Object.keys(useMetricsStore.getState().metrics).length > 0) {
    useMetricsStore.setState({ metrics: {} });
  }
  const sel = useSelectionStore.getState();
  if (sel.contexts["cluster:images"].size > 0) sel.clear("cluster:images");
  if (sel.contexts["cluster:merge"].size > 0) sel.clear("cluster:merge");
  if (sel.contexts.expand.size > 0) sel.clear("expand");
  if (useInteractionsStore.getState().lastClickedImage !== null) {
    useInteractionsStore.setState({ lastClickedImage: null });
  }
});
