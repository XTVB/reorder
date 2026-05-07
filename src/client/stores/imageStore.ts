import { create } from "zustand";
import { getJson } from "../api/client.ts";
import type { ImageInfo, ImagesResponse } from "../types.ts";

interface ImageState {
  images: ImageInfo[];
  originalOrder: string[];
  imageMap: Map<string, ImageInfo>;
  hasChanges: boolean;
  loading: boolean;
  // Bumped on fetchImages (after save/undo) to bust the browser's in-memory image cache.
  // HTTP cache correctness is handled by ETag/no-cache; this only changes the URL so
  // <img> tags don't serve stale in-memory copies after filenames are reassigned.
  imageVersion: number;

  setImages: (images: ImageInfo[]) => void;
  fetchImages: () => Promise<void>;
  applyDeletions: (deleted: string[]) => void;
}

export const useImageStore = create<ImageState>((set, get) => ({
  images: [],
  originalOrder: [],
  imageMap: new Map(),
  hasChanges: false,
  loading: true,
  imageVersion: 0,

  setImages: (images) => {
    const { originalOrder, imageMap: existingMap } = get();
    // Skip Map rebuild when only order changed (drag reorder) — same filenames, different positions
    const needsMapRebuild =
      images.length !== existingMap.size || images.some((i) => !existingMap.has(i.filename));
    const imageMap = needsMapRebuild ? new Map(images.map((i) => [i.filename, i])) : existingMap;
    const hasChanges =
      images.length > 0 &&
      (images.length !== originalOrder.length ||
        images.some((img, i) => img.filename !== originalOrder[i]));
    set({ images, imageMap, hasChanges });
  },

  fetchImages: async () => {
    try {
      const { images: data } = await getJson<ImagesResponse>("/api/images");
      const order = data.map((d) => d.filename);
      const imageMap = new Map(data.map((i) => [i.filename, i]));
      set({
        images: data,
        originalOrder: order,
        imageMap,
        hasChanges: false,
        loading: false,
        imageVersion: get().imageVersion + 1,
      });
    } catch {
      set({ loading: false });
      throw new Error("Failed to load images");
    }
  },

  // Prune deleted filenames in-place, preserving any pending reorder.
  // Used after /api/delete so the user's unsaved reorder isn't reset by a full refetch.
  applyDeletions: (deleted) => {
    if (deleted.length === 0) return;
    const { images, originalOrder, imageMap } = get();
    const deletedSet = new Set(deleted);
    const nextImages = images.filter((i) => !deletedSet.has(i.filename));
    if (nextImages.length === images.length) return;
    const nextOriginalOrder = originalOrder.filter((fn) => !deletedSet.has(fn));
    const nextMap = new Map(imageMap);
    for (const fn of deleted) nextMap.delete(fn);
    const hasChanges =
      nextImages.length > 0 &&
      (nextImages.length !== nextOriginalOrder.length ||
        nextImages.some((img, i) => img.filename !== nextOriginalOrder[i]));
    set({
      images: nextImages,
      originalOrder: nextOriginalOrder,
      imageMap: nextMap,
      hasChanges,
    });
  },
}));
