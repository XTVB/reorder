import { useCallback, useRef } from "react";
import { useImageStore } from "../stores/imageStore.ts";
import { useGroupStore } from "../stores/groupStore.ts";
import { useSelectionStore } from "../stores/selectionStore.ts";
import { consolidateBlock, repositionBlock } from "../utils/reorder.ts";

export function useGroupOperations() {
  function addImagesToGroup(groupId: string, filenames: string[]) {
    const { groups, updateGroups } = useGroupStore.getState();
    const { images, setImages } = useImageStore.getState();
    const { removeFromSelection } = useSelectionStore.getState();

    const fileSet = new Set(filenames);
    const newGroups = groups.map((g) => {
      const cleaned = g.images.filter((fn) => !fileSet.has(fn));
      if (g.id === groupId) return { ...g, images: [...cleaned, ...filenames] };
      return { ...g, images: cleaned };
    });

    updateGroups(() => newGroups);
    const targetGroup = newGroups.find((g) => g.id === groupId);
    if (!targetGroup) return;

    setImages((() => {
      const allGroupImages = new Set(targetGroup.images);
      const toMove = images.filter((i) => fileSet.has(i.filename));
      const rest = images.filter((i) => !fileSet.has(i.filename));
      let lastIdx = -1;
      for (let i = 0; i < rest.length; i++) {
        if (allGroupImages.has(rest[i]!.filename)) lastIdx = i;
      }
      if (lastIdx === -1) return images;
      const out = [...rest];
      out.splice(lastIdx + 1, 0, ...toMove);
      return out;
    })());

    removeFromSelection(filenames);
  }

  function handleGroupReorder(groupId: string, newOrder: string[]) {
    const { updateGroups } = useGroupStore.getState();
    const { images, setImages } = useImageStore.getState();
    updateGroups((prev) =>
      prev.map((g) => (g.id === groupId ? { ...g, images: newOrder } : g))
    );
    setImages(repositionBlock(images, newOrder));
  }

  function handleRemoveFromGroup(groupId: string, filename: string) {
    const { groupMap, collapseGroup, updateGroups } = useGroupStore.getState();
    const group = groupMap.get(groupId);
    if (group && group.images.length <= 1) collapseGroup();
    updateGroups((prev) =>
      prev.map((g) =>
        g.id === groupId
          ? { ...g, images: g.images.filter((fn) => fn !== filename) }
          : g
      )
    );
  }

  function handleRenameGroup(groupId: string) {
    const { groupMap, updateGroups } = useGroupStore.getState();
    const group = groupMap.get(groupId);
    if (!group) return;
    const name = prompt("New group name:", group.name);
    if (!name?.trim()) return;
    updateGroups((prev) =>
      prev.map((g) => g.id === groupId ? { ...g, name: name.trim() } : g)
    );
  }

  function handleDeleteGroup(groupId: string) {
    const { expandedGroupId, collapseGroup, updateGroups } = useGroupStore.getState();
    updateGroups((prev) => prev.filter((g) => g.id !== groupId));
    if (expandedGroupId === groupId) collapseGroup();
  }

  const handleCreateGroupImpl = () => {
    const { selectedIds, clearSelection } = useSelectionStore.getState();
    const { images, setImages } = useImageStore.getState();
    const { updateGroups } = useGroupStore.getState();

    if (selectedIds.size === 0) return;
    const name = prompt("Enter group name:");
    if (!name?.trim()) return;

    const id = crypto.randomUUID();
    const selectedInOrder = images
      .filter((i) => selectedIds.has(i.filename))
      .map((i) => i.filename);

    setImages(consolidateBlock(images, selectedIds));
    updateGroups((prev) => [
      ...prev,
      { id, name: name.trim(), images: selectedInOrder },
    ]);
    clearSelection();
  };
  const createGroupRef = useRef(handleCreateGroupImpl);
  createGroupRef.current = handleCreateGroupImpl;
  const handleCreateGroup = useCallback(() => createGroupRef.current(), []);

  return {
    handleCreateGroup,
    addImagesToGroup,
    handleGroupReorder,
    handleRemoveFromGroup,
    handleRenameGroup,
    handleDeleteGroup,
  };
}
