// Cross-cutting types used by both client and server.

export interface ImageInfo {
  filename: string;
}

export interface RenameMapping {
  from: string;
  to: string;
}

export interface ImageGroup {
  id: string;
  name: string;
  images: string[];
}

export interface FolderGroup {
  name: string;
  images: string[];
}

export interface FolderData {
  folders: FolderGroup[];
  rootImages: string[];
}

// API response shapes

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

// Cluster types

export interface WeightConfig {
  color?: number;
  dinov3?: number;
  pecore_g?: number;
  learned_proj?: number;
}

export interface ClusterResultData {
  id: string;
  autoName: string;
  images: string[];
  confirmedGroup: { id: string; name: string; images: string[] } | null;
  splitFrom?: string;
}

export interface DistanceProfile {
  distances: number[];
  nAfterPremerge: number;
  nGroups: number;
}

export interface ClusterData {
  clusters: ClusterResultData[];
  suggestedCounts: number[];
  nClusters: number;
  distanceProfile?: DistanceProfile;
}

export interface ImportClusterInput {
  name: string;
  images: string[];
}

// Tree-navigation operation types

export interface ClusterMetrics {
  cohesion: number;
  isolation: number; // -1 sentinel from server when Infinity
  stability: number;
}

export interface ExpandCandidate {
  filename: string;
  distance: number;
}

export interface ExpandResult {
  candidates: ExpandCandidate[];
  p90Intra: number;
  maxDistance: number;
}

export interface SplitChildren {
  childA: ClusterResultData;
  childB: ClusterResultData;
}

// Nearest-neighbor query types

export type NNAggregation = "centroid" | "min";
export type NNFilter = "any" | "in-group" | "not-in-group";

export interface NNResult {
  filename: string;
  distance: number;
  inGroupId: string | null;
  inGroupName: string | null;
}

export interface NNQueryRequest {
  queryFilenames: string[];
  topN?: number;
  filter?: NNFilter;
  aggregation?: NNAggregation;
  weights: WeightConfig;
  usePatches: boolean;
  restrictToFilenames?: string[];
  excludeQuery?: boolean;
}

export interface NNQueryResponse {
  results: NNResult[];
  usedModels: string[];
  queryCount: number;
  patchesBlended: boolean;
}

// Merge suggestion types

export interface MergeSuggestionSimilar {
  groupId: string;
  groupName: string;
  groupImages: string[];
  distance: number;
}

export interface MergeSuggestionRow {
  refGroupId: string;
  refGroupName: string;
  refGroupImages: string[];
  similar: MergeSuggestionSimilar[];
}

export interface MergeSuggestionsResponse {
  suggestions: MergeSuggestionRow[];
  computeTimeMs: number;
}
