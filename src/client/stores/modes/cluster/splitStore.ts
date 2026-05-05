// Inline split tree-nav actions. The `splitChildren` map itself lives in
// listStore (because clusterData and splitChildren are co-traversed by the
// tree-walk helpers); this store owns only the open/collapse actions.

import { create } from "zustand";
import { postJson } from "../../../api/client.ts";
import type { SplitChildren } from "../../../types.ts";
import { getErrorMessage } from "../../../utils/helpers.ts";
import { useToastStore } from "../../core/toastStore.ts";
import { useListStore } from "./listStore.ts";
import { findClusterEverywhere } from "./tree-helpers.ts";

interface SplitSlice {
  toggleSplit: (parentId: string) => Promise<void>;
  collapseSplit: (parentId: string) => void;
}

export const useSplitStore = create<SplitSlice>(() => ({
  toggleSplit: async (parentId) => {
    const list = useListStore.getState();
    const { splitChildren, clusterData } = list;
    if (!clusterData) return;

    if (splitChildren[parentId]) {
      useSplitStore.getState().collapseSplit(parentId);
      return;
    }

    const parent = findClusterEverywhere(clusterData.clusters, splitChildren, parentId);
    if (!parent || parent.images.length < 2) return;

    let kids: SplitChildren;
    try {
      kids = await postJson<SplitChildren>("/api/cluster/tree-nav/split", {
        images: parent.images,
      });
    } catch (err) {
      useToastStore
        .getState()
        .showToast(getErrorMessage(err, "Cannot split this cluster"), "warning");
      return;
    }
    kids = {
      childA: { ...kids.childA, splitFrom: parentId },
      childB: { ...kids.childB, splitFrom: parentId },
    };
    useListStore.setState({
      splitChildren: { ...useListStore.getState().splitChildren, [parentId]: kids },
    });
  },

  collapseSplit: (parentId) => {
    const list = useListStore.getState();
    const newChildren = { ...list.splitChildren };
    const stack = [parentId];
    while (stack.length) {
      const id = stack.pop()!;
      const kids = newChildren[id];
      if (kids) stack.push(kids.childA.id, kids.childB.id);
      delete newChildren[id];
    }
    useListStore.setState({ splitChildren: newChildren });
  },
}));
