import { create } from "zustand";
import type { FolderGroup, FolderData, ImageInfo } from "../types.ts";
import { useImageStore } from "./imageStore.ts";
import { postJson, stripFolderNumber } from "../utils/helpers.ts";

const FOLDER_MODE_KEY = "reorder-folder-mode";

/**
 * In folder mode, every image is identified by its original compound path
 * (e.g. "001 - Day 1/003.jpg"). This is stable across local mutations and
 * tells the server where the file lives on disk.
 *
 * folder.images stores these compound paths. Root images are bare filenames.
 */

interface FolderState {
  /** Current desired state (may differ from disk after local edits) */
  folders: FolderGroup[];
  rootImages: string[];   // compound paths (bare for root-native files)
  folderMap: Map<string, FolderGroup>;

  /** Snapshot of disk state at last fetch */
  diskFolders: FolderGroup[];
  diskRootImages: string[];

  folderModeEnabled: boolean;
  expandedFolderName: string | null;
  folderModeLoaded: boolean;
  hasChanges: boolean;

  fetchFolders: () => Promise<void>;
  setFolderModeEnabled: (enabled: boolean) => void;
  expandFolder: (name: string | null) => void;
  collapseFolder: () => void;

  /** Local mutations — no server calls */
  reorderFolders: (orderedNames: string[]) => void;
  renameFolder: (folderName: string, newTitle: string) => void;
  dissolveFolder: (folderName: string) => void;
  moveImages: (compoundPaths: string[], toFolder: string) => void;
  reorderWithinFolder: (folderName: string, newOrder: string[]) => void;
}

function allImages(folders: FolderGroup[], rootImages: string[]): ImageInfo[] {
  const out: ImageInfo[] = [];
  for (const f of folders) {
    for (const fn of f.images) out.push({ filename: fn });
  }
  for (const fn of rootImages) out.push({ filename: fn });
  return out;
}

function buildFolderMap(folders: FolderGroup[]): Map<string, FolderGroup> {
  return new Map(folders.map((f) => [f.name, f]));
}

function computeHasChanges(
  folders: FolderGroup[], rootImages: string[],
  diskFolders: FolderGroup[], diskRootImages: string[],
): boolean {
  if (folders.length !== diskFolders.length) return true;
  if (rootImages.length !== diskRootImages.length) return true;
  for (let i = 0; i < folders.length; i++) {
    const a = folders[i]!, b = diskFolders[i]!;
    if (a.name !== b.name) return true;
    if (a.images.length !== b.images.length) return true;
    for (let j = 0; j < a.images.length; j++) {
      if (a.images[j] !== b.images[j]) return true;
    }
  }
  for (let i = 0; i < rootImages.length; i++) {
    if (rootImages[i] !== diskRootImages[i]) return true;
  }
  return false;
}

/** After local folder mutations, sync the imageStore with the new flat list */
function syncImageStore(folders: FolderGroup[], rootImages: string[]) {
  const images = allImages(folders, rootImages);
  const { originalOrder } = useImageStore.getState();
  const imageMap = new Map(images.map((i) => [i.filename, i]));
  const hasChanges = images.length !== originalOrder.length ||
    images.some((img, i) => img.filename !== originalOrder[i]);
  useImageStore.setState({ images, imageMap, hasChanges });
}

/** Set folders + derived folderMap + hasChanges in one call */
function setFolders(
  set: (s: Partial<FolderState>) => void,
  get: () => FolderState,
  folders: FolderGroup[],
  rootImages: string[],
  extra?: Partial<FolderState>,
) {
  const { diskFolders, diskRootImages } = get();
  set({
    folders,
    rootImages,
    folderMap: buildFolderMap(folders),
    hasChanges: computeHasChanges(folders, rootImages, diskFolders, diskRootImages),
    ...extra,
  });
  syncImageStore(folders, rootImages);
}

export const useFolderStore = create<FolderState>((set, get) => ({
  folders: [],
  rootImages: [],
  folderMap: new Map(),
  diskFolders: [],
  diskRootImages: [],
  folderModeEnabled: localStorage.getItem(FOLDER_MODE_KEY) === "true",
  expandedFolderName: null,
  folderModeLoaded: false,
  hasChanges: false,

  fetchFolders: async () => {
    try {
      const res = await fetch("/api/folders");
      const data: FolderData = await res.json();
      const folders: FolderGroup[] = data.folders.map((f) => ({
        name: f.name,
        images: f.images.map((fn) => `${f.name}/${fn}`),
      }));
      const rootImages = data.rootImages;
      const diskFolders = folders.map((f) => ({ ...f, images: [...f.images] }));
      const diskRootImages = [...rootImages];
      set({
        folders,
        rootImages,
        folderMap: buildFolderMap(folders),
        diskFolders,
        diskRootImages,
        folderModeLoaded: true,
        hasChanges: false,
      });
      const images = allImages(folders, rootImages);
      const { imageVersion } = useImageStore.getState();
      useImageStore.setState({
        images,
        originalOrder: images.map((i) => i.filename),
        imageMap: new Map(images.map((i) => [i.filename, i])),
        hasChanges: false,
        loading: false,
        imageVersion: imageVersion + 1,
      });
    } catch {
      set({ folderModeLoaded: true });
    }
  },

  setFolderModeEnabled: (enabled) => {
    localStorage.setItem(FOLDER_MODE_KEY, String(enabled));
    set({ folderModeEnabled: enabled, expandedFolderName: null });
  },

  expandFolder: (name) => set({ expandedFolderName: name }),
  collapseFolder: () => set({ expandedFolderName: null }),

  reorderFolders: (orderedNames) => {
    const { folders, rootImages } = get();
    const byName = new Map(folders.map((f) => [f.name, f]));
    const reordered = orderedNames.map((n) => byName.get(n)!).filter(Boolean);
    setFolders(set, get, reordered, rootImages);
  },

  renameFolder: (folderName, newTitle) => {
    const { folders, rootImages, expandedFolderName } = get();
    const match = folderName.match(/^(\d+\s*-\s*)/);
    const prefix = match ? match[1] : "";
    const newName = prefix ? `${prefix}${newTitle}` : newTitle;
    const updated = folders.map((f) =>
      f.name === folderName ? { ...f, name: newName } : f
    );
    setFolders(set, get, updated, rootImages, {
      expandedFolderName: expandedFolderName === folderName ? newName : expandedFolderName,
    });
  },

  dissolveFolder: (folderName) => {
    const { folders, rootImages, expandedFolderName } = get();
    const folder = folders.find((f) => f.name === folderName);
    if (!folder) return;
    const updatedFolders = folders.filter((f) => f.name !== folderName);
    const updatedRoot = [...rootImages, ...folder.images];
    setFolders(set, get, updatedFolders, updatedRoot, {
      expandedFolderName: expandedFolderName === folderName ? null : expandedFolderName,
    });
  },

  moveImages: (compoundPaths, toFolder) => {
    const { folders, rootImages } = get();
    const moving = new Set(compoundPaths);

    let updatedFolders = folders.map((f) => ({
      ...f,
      images: f.images.filter((fn) => !moving.has(fn)),
    }));
    let updatedRoot = rootImages.filter((fn) => !moving.has(fn));

    if (toFolder === "") {
      updatedRoot = [...updatedRoot, ...compoundPaths];
    } else {
      updatedFolders = updatedFolders.map((f) => {
        if (f.name !== toFolder) return f;
        return { ...f, images: [...f.images, ...compoundPaths] };
      });
    }

    updatedFolders = updatedFolders.filter((f) => f.images.length > 0);
    setFolders(set, get, updatedFolders, updatedRoot);
  },

  reorderWithinFolder: (folderName, newOrder) => {
    const { folders, rootImages } = get();
    const updated = folders.map((f) =>
      f.name === folderName ? { ...f, images: newOrder } : f
    );
    setFolders(set, get, updated, rootImages);
  },
}));
