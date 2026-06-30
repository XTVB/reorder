// Centralized modal open/close state. Replaces the per-modal `show*` booleans
// previously bagged inside uiStore.

import { create } from "zustand";

export type ModalName =
  | "preview"
  | "organize"
  | "review"
  | "namingRules"
  | "createGroups"
  | "paths"
  | "trash"
  | "groupPicker";

interface ModalState {
  open: Record<ModalName, boolean>;
  openModal: (name: ModalName) => void;
  closeModal: (name: ModalName) => void;
  toggleModal: (name: ModalName) => void;
  setModalOpen: (name: ModalName, open: boolean) => void;
  isOpen: (name: ModalName) => boolean;
}

const INITIAL: Record<ModalName, boolean> = {
  preview: false,
  organize: false,
  review: false,
  namingRules: false,
  createGroups: false,
  paths: false,
  trash: false,
  groupPicker: false,
};

export const useModalStore = create<ModalState>((set, get) => ({
  open: { ...INITIAL },

  openModal: (name) => set({ open: { ...get().open, [name]: true } }),
  closeModal: (name) => set({ open: { ...get().open, [name]: false } }),
  toggleModal: (name) => set({ open: { ...get().open, [name]: !get().open[name] } }),
  setModalOpen: (name, open) => set({ open: { ...get().open, [name]: open } }),
  isOpen: (name) => get().open[name],
}));
