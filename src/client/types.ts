export interface ImageInfo {
  filename: string;
}

export interface RenameMapping {
  from: string;
  to: string;
}

export interface Toast {
  message: string;
  type: "success" | "error";
}

export interface OrganizeMapping {
  folder: string;
  files: RenameMapping[];
}

export interface ImageGroup {
  id: string;
  name: string;
  images: string[];
}

export interface DirResponse {
  dir: string;
}

export interface ImagesResponse {
  images: ImageInfo[];
}

export interface CanUndoResponse {
  canUndo: boolean;
}

export interface SaveResponse {
  success: boolean;
  renames: RenameMapping[];
}

export type GridItem =
  | { type: "image"; filename: string }
  | { type: "group"; groupId: string }
  | { type: "group-image"; groupId: string; filename: string };

// Tag system types
export type FilterMode = "AND" | "OR" | "NOT";

export interface ActiveFilter {
  category: string;
  value: string;
  mode: FilterMode;
}

export interface ClothingItemData {
  piece: string;
  colors: string[];
  styles: string[];
}

export interface ImageTagData {
  filename: string;
  tags: Record<string, string[]>;
  clothing: ClothingItemData[];
}

export type ClothingOption = ClothingItemData;

export type AppMode = "reorder" | "tags" | "merge";
