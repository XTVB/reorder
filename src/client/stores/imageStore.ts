import { create } from "zustand";
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
      const res = await fetch("/api/images");
      if (!res.ok) throw new Error("Failed to load images");
      const { images: data }: ImagesResponse = await res.json();
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
}));
