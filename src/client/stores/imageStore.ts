import { create } from "zustand";
import type { ImageInfo, ImagesResponse } from "../types.ts";

interface ImageState {
  images: ImageInfo[];
  originalOrder: string[];
  imageMap: Map<string, ImageInfo>;
  hasChanges: boolean;
  loading: boolean;
  cacheNonce: string;

  setImages: (images: ImageInfo[]) => void;
  fetchImages: () => Promise<void>;
  resetOriginalOrder: () => void;
}

function deriveState(images: ImageInfo[], originalOrder: string[]) {
  return {
    imageMap: new Map(images.map((i) => [i.filename, i])),
    hasChanges: images.length > 0 && (images.length !== originalOrder.length || images.some((img, i) => img.filename !== originalOrder[i])),
  };
}

export const useImageStore = create<ImageState>((set, get) => ({
  images: [],
  originalOrder: [],
  imageMap: new Map(),
  hasChanges: false,
  loading: true,
  cacheNonce: "",

  setImages: (images) => {
    const { originalOrder } = get();
    set({ images, ...deriveState(images, originalOrder) });
  },

  fetchImages: async () => {
    try {
      const res = await fetch("/api/images");
      if (!res.ok) throw new Error("Failed to load images");
      const { images: data, cacheNonce }: ImagesResponse = await res.json();
      const order = data.map((d) => d.filename);
      set({
        images: data,
        originalOrder: order,
        loading: false,
        cacheNonce,
        ...deriveState(data, order),
      });
    } catch {
      set({ loading: false });
      throw new Error("Failed to load images");
    }
  },

  resetOriginalOrder: () => {
    const { images } = get();
    const order = images.map((i) => i.filename);
    set({ originalOrder: order, hasChanges: false });
  },
}));
