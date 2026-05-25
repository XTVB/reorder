// Cluster interaction actions: image / merge selection, accept, addToGroup,
// mergeSelectedClusters / splitSelected. Selection state lives in
// core/selectionStore — this store only provides the per-cluster semantics
// (composite "clusterId:filename" keys, lastClickedImage anchor, etc.).

import { create } from "zustand";
import type { ClusterResultData } from "../../../types.ts";
import { addFilenamesToGroup, appendNewGroup, appendNewGroups } from "../../../utils/groups.ts";
import { consolidateBlock } from "../../../utils/reorder.ts";
import { useConstraintsStore } from "../../constraintsStore.ts";
import { useSelectionStore } from "../../core/selectionStore.ts";
import { useToastStore } from "../../core/toastStore.ts";
import { useGroupStore } from "../../groupStore.ts";
import { useImageStore } from "../../imageStore.ts";
import { useListStore } from "./listStore.ts";
import {
  findClusterEverywhere,
  getSectionList,
  getSuggestedAdditions,
  type ImageSection,
  parseImageKey,
  unionImages,
} from "./tree-helpers.ts";

/**
 * When images are added to a confirmed group, any cannot-link constraint
 * between those images and that group becomes stale. Fire-and-forget.
 */
export function dropCannotLinkAgainstGroup(filenames: string[], groupId: string) {
  const store = useConstraintsStore.getState();
  const pairs = filenames
    .filter((f) => store.isCannotLinked(f, groupId))
    .map((filename) => ({ filename, groupId }));
  if (pairs.length === 0) return;
  store.removeImageGroupCannotLink(pairs).catch(() => {});
}

export function commitMergeIntoGroup(participants: ClusterResultData[], winnerGroupId: string) {
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
    const next = addFilenamesToGroup(
      prev.filter((g) => !losingGroupIds.has(g.id)),
      winnerGroupId,
      merged,
    );
    if (!winnerName) {
      const w = next.find((g) => g.id === winnerGroupId);
      if (w) winnerName = w.name;
    }
    return next;
  });

  dropCannotLinkAgainstGroup(merged, winnerGroupId);
  useListStore.getState().applyClusterRemoval(new Set(participants.map((c) => c.id)));
  useListStore.setState({ treeStale: true });
  useToastStore
    .getState()
    .showToast(
      losingGroupIds.size > 0
        ? `Merged ${losingGroupIds.size + 1} groups into "${winnerName}" (${merged.length} images)`
        : `Extended "${winnerName}" with ${merged.length} images`,
      "success",
    );
}

interface InteractionsState {
  /**
   * Anchor for shift+click range select. Scoped to a section so a range
   * doesn't sweep through confirmed images sandwiched between two suggested
   * clicks. Filename (not index) survives cluster mutations between clicks.
   */
  lastClickedImage: { clusterId: string; section: ImageSection; filename: string } | null;

  toggleMergeSelect: (clusterId: string) => void;
  clearMergeSelection: () => void;
  toggleImageSelect: (clusterId: string, filename: string, section: ImageSection) => void;
  rangeSelectImages: (clusterId: string, filename: string, section: ImageSection) => void;
  clearImageSelection: () => void;

  acceptCluster: (cluster: ClusterResultData) => void;
  acceptAllClusters: (minSize: number) => void;
  addToGroup: (cluster: ClusterResultData) => void;

  mergeSelectedClusters: () => void;
  splitSelected: () => void;
}

export const useInteractionsStore = create<InteractionsState>((set, get) => ({
  lastClickedImage: null,

  toggleMergeSelect: (clusterId) => {
    useSelectionStore.getState().toggle("cluster:merge", clusterId);
  },

  clearMergeSelection: () => {
    useSelectionStore.getState().clear("cluster:merge");
  },

  toggleImageSelect: (clusterId, filename, section) => {
    useSelectionStore.getState().toggle("cluster:images", `${clusterId}:${filename}`);
    set({ lastClickedImage: { clusterId, section, filename } });
  },

  rangeSelectImages: (clusterId, toFilename, section) => {
    const { lastClickedImage } = get();
    const list = useListStore.getState();
    if (
      !lastClickedImage ||
      lastClickedImage.clusterId !== clusterId ||
      lastClickedImage.section !== section ||
      !list.clusterData
    )
      return;
    const cluster = findClusterEverywhere(list.clusterData.clusters, list.splitChildren, clusterId);
    if (!cluster) return;

    const sectionList = getSectionList(cluster, section, useConstraintsStore.getState().index);
    const fromIndex = sectionList.indexOf(lastClickedImage.filename);
    const toIndex = sectionList.indexOf(toFilename);
    if (fromIndex < 0 || toIndex < 0) return;

    const lo = Math.min(fromIndex, toIndex);
    const hi = Math.max(fromIndex, toIndex);
    const ids: string[] = [];
    for (let i = lo; i <= hi; i++) {
      ids.push(`${clusterId}:${sectionList[i]!}`);
    }
    useSelectionStore.getState().add("cluster:images", ids);
  },

  clearImageSelection: () => {
    useSelectionStore.getState().clear("cluster:images");
    set({ lastClickedImage: null });
  },

  acceptCluster: (cluster) => {
    const name = cluster.autoName || `Cluster ${cluster.id}`;
    const { updateGroups, groupsLoaded } = useGroupStore.getState();
    const { images, setImages } = useImageStore.getState();
    const { showToast } = useToastStore.getState();

    if (!groupsLoaded) {
      showToast("Groups still loading — please wait", "warning");
      return;
    }

    updateGroups((prev) =>
      appendNewGroup(prev, { id: crypto.randomUUID(), name, images: cluster.images }),
    );
    setImages(consolidateBlock(images, new Set(cluster.images)));
    showToast(`Created group "${name}" with ${cluster.images.length} images`, "success");
    useListStore.getState().dismissCluster(cluster.id);
    useListStore.setState({ treeStale: true });
  },

  acceptAllClusters: (minSize) => {
    const { clusterData } = useListStore.getState();
    if (!clusterData) return;
    const { showToast } = useToastStore.getState();
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
    updateGroups((prev) => appendNewGroups(prev, newGroups));
    const allAccepted = new Set(eligible.flatMap((c) => c.images));
    setImages(consolidateBlock(images, allAccepted));
    showToast(`Created ${newGroups.length} groups`, "success");
    useListStore.setState({
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
      useToastStore.getState().showToast("Groups still loading — please wait", "warning");
      return;
    }
    const groupId = cluster.confirmedGroup.id;
    const confirmedSet = new Set(cluster.confirmedGroup.images);
    const suggested = getSuggestedAdditions(cluster, useConstraintsStore.getState().index);

    const selectedImages = useSelectionStore.getState().contexts["cluster:images"];
    const selectedInCluster: string[] = [];
    for (const key of selectedImages) {
      const { clusterId, filename } = parseImageKey(key);
      if (clusterId === cluster.id && !confirmedSet.has(filename)) {
        selectedInCluster.push(filename);
      }
    }
    const toAdd = selectedInCluster.length > 0 ? selectedInCluster : suggested;

    const { updateGroups } = useGroupStore.getState();
    const { showToast } = useToastStore.getState();
    updateGroups((prev) => addFilenamesToGroup(prev, groupId, toAdd));
    dropCannotLinkAgainstGroup(toAdd, groupId);
    showToast(`Added ${toAdd.length} images to "${cluster.confirmedGroup.name}"`, "success");

    const newConfirmedImages = [...cluster.confirmedGroup.images, ...toAdd];
    const next: ClusterResultData = {
      ...cluster,
      confirmedGroup: { ...cluster.confirmedGroup, images: newConfirmedImages },
    };
    useListStore.getState().applyClusterReplace(cluster.id, next);

    // Auto-collapse if all images are now confirmed
    const list = useListStore.getState();
    const collapsed = new Set(list.collapsedClusters);
    const allConfirmed = newConfirmedImages.length >= cluster.images.length;
    if (allConfirmed) collapsed.add(cluster.id);

    useListStore.setState({ collapsedClusters: collapsed, treeStale: true });
    useSelectionStore.getState().clear("cluster:images");
    set({ lastClickedImage: null });
  },

  mergeSelectedClusters: () => {
    const list = useListStore.getState();
    const mergeSelection = useSelectionStore.getState().contexts["cluster:merge"];
    const { clusterData } = list;
    if (!clusterData || mergeSelection.size < 2) return;

    const selected = clusterData.clusters.filter((c) => mergeSelection.has(c.id));
    if (selected.length < 2) return;
    selected.sort((a, b) => b.images.length - a.images.length);
    const target = selected.find((c) => c.confirmedGroup) ?? selected[0];
    if (!target) return;
    const sources = selected.filter((c) => c.id !== target.id);

    const mergedImages = unionImages([target, ...sources]);

    const sourceIds = new Set(sources.map((s) => s.id));
    const newClusters = clusterData.clusters
      .filter((c) => !sourceIds.has(c.id))
      .map((c) => (c.id === target.id ? { ...c, images: mergedImages.sort() } : c));

    useListStore.setState({ clusterData: { ...clusterData, clusters: newClusters } });
    useSelectionStore.getState().clear("cluster:merge");
  },

  splitSelected: () => {
    const list = useListStore.getState();
    const selectedImages = useSelectionStore.getState().contexts["cluster:images"];
    const { clusterData } = list;
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
        images: splitFiles,
        confirmedGroup: null,
        splitFrom: source.id,
      };
      newClusters.splice(sourceIdx + 1, 0, newCluster);
    }

    useListStore.setState({ clusterData: { ...clusterData, clusters: newClusters } });
    useSelectionStore.getState().clear("cluster:images");
    set({ lastClickedImage: null });
  },
}));
