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
  /**
   * Category/subcategory labels this group was explicitly bucketed into during
   * a grouping-sort Apply, e.g. ["Keep", "Keep - Top"]. Used for metadata
   * search / resuming a partial categorisation; absent on never-categorised
   * groups. See `explicitGroupTags` / `configOwnedTags` in reviewConfigs.ts for
   * how a config builds and merges/replaces its slice of these tags.
   */
  tags?: string[];
  /**
   * Locked groups keep their gallery slot when a sort is applied (Sort
   * Similar / Review Apply Order) — only unlocked groups move around them.
   * Toggled with L on selected groups; absent means unlocked.
   */
  locked?: boolean;
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

/** Agglomerative linkage method for the cluster tree. */
export type LinkageMethod = "ward" | "average" | "complete";

/**
 * Algorithm for ordering groups by pairwise similarity (reorder page's
 * Sort Similar): "chain" = greedy nearest-neighbor + 2-opt, "tree" =
 * average-linkage clustering + optimal leaf ordering, "spectral" = Fiedler-
 * vector seriation, "minimal" = keep the current order, applying only small
 * local moves where similarity clearly improves.
 */
export type GroupOrderMode = "chain" | "tree" | "spectral" | "minimal";

export interface ClusterResultData {
  id: string;
  name: string;
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
  // null when the embedding is incomparable (corrupt/missing row → non-finite distance).
  distance: number | null;
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
