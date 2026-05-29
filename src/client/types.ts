// Client-only types. Cross-cutting shapes live in src/shared/types.ts and
// are re-exported here so existing client imports keep working.

export type {
  CanUndoResponse,
  ClusterData,
  ClusterMetrics,
  ClusterResultData,
  DirResponse,
  DistanceProfile,
  ExpandCandidate,
  ExpandResult,
  FolderData,
  FolderGroup,
  ImageGroup,
  ImageInfo,
  ImagesResponse,
  ImportClusterInput,
  LinkageMethod,
  MergeSuggestionRow,
  MergeSuggestionSimilar,
  MergeSuggestionsResponse,
  NNAggregation,
  NNFilter,
  NNQueryRequest,
  NNQueryResponse,
  NNResult,
  RenameMapping,
  SaveResponse,
  SplitChildren,
  WeightConfig,
} from "../shared/types.ts";

export interface Toast {
  message: string;
  type: "success" | "error" | "warning";
}

export interface OrganizeMapping {
  folder: string;
  files: import("../shared/types.ts").RenameMapping[];
}

export type GridItem =
  | { type: "image"; filename: string }
  | { type: "group"; groupId: string }
  | { type: "group-image"; groupId: string; filename: string }
  | { type: "folder"; folderName: string }
  | { type: "folder-image"; folderName: string; filename: string };

export type AppMode = "reorder" | "cluster" | "merge-suggestions";
