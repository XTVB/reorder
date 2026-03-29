import { create } from "zustand";
import type { ActiveFilter, FilterMode, ImageTagData, ClothingOption } from "../types.ts";
import { buildInvertedIndex, applyFilters, type InvertedIndex } from "../utils/tagIndex.ts";
import { useImageStore } from "./imageStore.ts";
import { useGroupStore } from "./groupStore.ts";
import { postJson } from "../utils/helpers.ts";

interface TagState {
  // Data
  tagData: Map<string, ImageTagData>;
  invertedIndex: InvertedIndex;
  clothingOptions: ClothingOption[];
  indexReady: boolean;
  loading: boolean;
  hasDb: boolean;
  dbImageCount: number;

  // Filters
  filters: ActiveFilter[];
  filteredFilenames: string[];

  // Scope
  scope: "all" | "ungrouped";

  // UI state
  focusedGroupId: string | null;
  detailFilename: string | null;
  sidebarOpen: boolean;

  // Merge view state
  mergeFilters: ActiveFilter[];
  selectedGroupIds: Set<string>;

  // Actions
  checkDbStatus: () => Promise<void>;
  loadAllTags: () => Promise<void>;
  addFilter: (category: string, value: string, mode?: FilterMode) => void;
  removeFilter: (category: string, value: string) => void;
  setFilterMode: (category: string, value: string, mode: FilterMode) => void;
  clearFilters: () => void;
  setScope: (scope: "all" | "ungrouped") => void;
  setFocusedGroup: (groupId: string | null) => void;
  setDetailFilename: (filename: string | null) => void;
  toggleSidebar: () => void;
  recomputeFiltered: () => void;
  ingestFile: (data: unknown) => Promise<{ ingested: number; skipped: number }>;

  // Merge view actions
  addMergeFilter: (category: string, value: string, mode?: FilterMode) => void;
  removeMergeFilter: (category: string, value: string) => void;
  setMergeFilterMode: (category: string, value: string, mode: FilterMode) => void;
  clearMergeFilters: () => void;
  toggleGroupSelection: (groupId: string) => void;
  clearGroupSelection: () => void;
}

function computeGroupedFilenames(): Set<string> {
  const groups = useGroupStore.getState().groups;
  const set = new Set<string>();
  for (const g of groups) {
    for (const fn of g.images) set.add(fn);
  }
  return set;
}

function getOrderedFilenames(): string[] {
  return useImageStore.getState().images.map((img) => img.filename);
}

function applyStructuredClothingFilters(
  filenames: string[],
  filters: ActiveFilter[],
  tagData: Map<string, ImageTagData>,
): string[] {
  const structured = filters.filter((f) => f.category === "__clothing_structured");
  if (structured.length === 0) return filenames;

  return filenames.filter((fn) => {
    const data = tagData.get(fn);
    if (!data) return false;
    return structured.every((f) => {
      const [piece, color] = f.value.split("|");
      return data.clothing.some(
        (ci) =>
          ci.piece === piece &&
          (!color || ci.colors.includes(color)),
      );
    });
  });
}

function updateFilter(
  filters: ActiveFilter[],
  category: string,
  value: string,
  mode: FilterMode,
): ActiveFilter[] {
  const exists = filters.find((f) => f.category === category && f.value === value);
  if (exists) {
    return filters.map((f) =>
      f.category === category && f.value === value ? { ...f, mode } : f,
    );
  }
  return [...filters, { category, value, mode }];
}

function recomputeAndSet(
  get: () => TagState,
  set: (partial: Partial<TagState>) => void,
): void {
  const { filters, scope, invertedIndex, tagData, filteredFilenames: prev } = get();
  const allFilenames = getOrderedFilenames();
  const groupedFilenames = scope === "ungrouped" ? computeGroupedFilenames() : null;

  let result = applyFilters(allFilenames, invertedIndex, filters, groupedFilenames, scope);
  result = applyStructuredClothingFilters(result, filters, tagData);

  if (result.length === prev.length && result.every((fn, i) => fn === prev[i])) return;
  set({ filteredFilenames: result });
}

function removeFilterEntry(
  filters: ActiveFilter[],
  category: string,
  value: string,
): ActiveFilter[] {
  return filters.filter((f) => !(f.category === category && f.value === value));
}

export const useTagStore = create<TagState>((set, get) => ({
  tagData: new Map(),
  invertedIndex: new Map(),
  clothingOptions: [],
  indexReady: false,
  loading: false,
  hasDb: false,
  dbImageCount: 0,

  filters: [],
  filteredFilenames: [],

  scope: "ungrouped",

  focusedGroupId: null,
  detailFilename: null,
  sidebarOpen: true,

  mergeFilters: [],
  selectedGroupIds: new Set(),

  checkDbStatus: async () => {
    try {
      const res = await fetch("/api/tags/status");
      const { hasDb, imageCount } = await res.json();
      set({ hasDb, dbImageCount: imageCount });
    } catch {
      set({ hasDb: false, dbImageCount: 0 });
    }
  },

  loadAllTags: async () => {
    set({ loading: true });
    try {
      const [tagsRes, clothingRes] = await Promise.all([
        fetch("/api/tags/all"),
        fetch("/api/tags/clothing-structured"),
      ]);

      if (!tagsRes.ok) {
        set({ loading: false });
        return;
      }

      const { images } = (await tagsRes.json()) as { images: ImageTagData[] };
      const clothingOptions: ClothingOption[] = clothingRes.ok
        ? await clothingRes.json()
        : [];

      const tagData = new Map(images.map((img) => [img.filename, img]));
      const invertedIndex = buildInvertedIndex(images);

      set({
        tagData,
        invertedIndex,
        clothingOptions,
        indexReady: true,
        loading: false,
        hasDb: true,
        dbImageCount: images.length,
      });

      // Initial filter computation
      get().recomputeFiltered();
    } catch {
      set({ loading: false });
    }
  },

  addFilter: (category, value, mode = "AND") => {
    set({ filters: updateFilter(get().filters, category, value, mode) });
    recomputeAndSet(get, set);
  },

  removeFilter: (category, value) => {
    set({ filters: removeFilterEntry(get().filters, category, value) });
    recomputeAndSet(get, set);
  },

  setFilterMode: (category, value, mode) => {
    set({ filters: get().filters.map((f) =>
      f.category === category && f.value === value ? { ...f, mode } : f,
    ) });
    recomputeAndSet(get, set);
  },

  clearFilters: () => {
    set({ filters: [] });
    recomputeAndSet(get, set);
  },

  setScope: (scope) => {
    set({ scope });
    recomputeAndSet(get, set);
  },

  setFocusedGroup: (groupId) => set({ focusedGroupId: groupId }),
  setDetailFilename: (filename) => set({ detailFilename: filename }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  recomputeFiltered: () => recomputeAndSet(get, set),

  ingestFile: async (data) => {
    const res = await postJson("/api/tags/ingest", { data });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error ?? "Ingest failed");
    }
    const result = await res.json();
    // Reload tags after ingest
    await get().loadAllTags();
    return result;
  },

  // Merge view
  addMergeFilter: (category, value, mode = "AND") => {
    set({ mergeFilters: updateFilter(get().mergeFilters, category, value, mode) });
  },

  removeMergeFilter: (category, value) => {
    set({ mergeFilters: removeFilterEntry(get().mergeFilters, category, value) });
  },

  setMergeFilterMode: (category, value, mode) => {
    const mergeFilters = get().mergeFilters.map((f) =>
      f.category === category && f.value === value ? { ...f, mode } : f,
    );
    set({ mergeFilters });
  },

  clearMergeFilters: () => set({ mergeFilters: [] }),

  toggleGroupSelection: (groupId) => {
    const next = new Set(get().selectedGroupIds);
    if (next.has(groupId)) next.delete(groupId);
    else next.add(groupId);
    set({ selectedGroupIds: next });
  },

  clearGroupSelection: () => set({ selectedGroupIds: new Set() }),
}));
