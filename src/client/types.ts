export interface ImageInfo {
  filename: string;
}

export interface RenameMapping {
  from: string;
  to: string;
}

export interface Toast {
  message: string;
  type: "success" | "error" | "warning";
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
  warnings?: string[];
}

export interface FolderGroup {
  name: string;
  images: string[];
}

export interface FolderData {
  folders: FolderGroup[];
  rootImages: string[];
}

export type GridItem =
  | { type: "image"; filename: string }
  | { type: "group"; groupId: string }
  | { type: "group-image"; groupId: string; filename: string }
  | { type: "folder"; folderName: string }
  | { type: "folder-image"; folderName: string; filename: string };

export type AppMode = "reorder" | "cluster";

// Cluster types
export interface ClusterResultData {
  id: string;
  autoName: string;
  autoTags: { term: string; z: number }[];
  images: string[];
  confirmedGroup: { id: string; name: string; images: string[] } | null;
}

export interface ClusterData {
  clusters: ClusterResultData[];
  suggestedCounts: number[];
  nClusters: number;
}
