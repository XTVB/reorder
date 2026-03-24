import { create } from "zustand";

interface DndState {
  activeId: string | null;
  dragOverGroupId: string | null;
  frozenGroupId: string | null;

  setActiveId: (id: string | null) => void;
  setDragOverGroupId: (id: string | null) => void;
  setFrozenGroupId: (id: string | null) => void;
  clearDrag: () => void;
}

export const useDndStore = create<DndState>((set) => ({
  activeId: null,
  dragOverGroupId: null,
  frozenGroupId: null,

  setActiveId: (id) => set({ activeId: id }),
  setDragOverGroupId: (id) => set({ dragOverGroupId: id }),
  setFrozenGroupId: (id) => set({ frozenGroupId: id }),
  clearDrag: () => set({ activeId: null, dragOverGroupId: null, frozenGroupId: null }),
}));
