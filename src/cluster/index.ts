// Public API of the cluster module — used by server/, nn-query, and the
// rest of the codebase. Internal helpers (cache clearers, tree-cut primitives,
// resolved-constraint writers, etc.) stay private inside their sub-modules.

export type { Constraints, RejectedMergePair } from "./constraints.ts";
export {
  loadConstraints,
  mergePairKey,
  mutateConstraints,
  normalizeMergePair,
  pruneDanglingConstraints,
  writeResolvedRejectedPairsFile,
} from "./constraints.ts";
export { generateContactSheet } from "./contact-sheets.ts";
export {
  ensurePatchDistMatrix,
  loadPatchDistMatrix,
  removeRerankDistMatrix,
} from "./distance-matrices.ts";
export type { ModelEmbedding, ModelKey } from "./embeddings.ts";
export {
  activeModelsFromWeights,
  cachedHashMapping,
  l2Norm,
  loadModelEmbedding,
  MODEL_KEYS,
  ModelMissingError,
} from "./embeddings.ts";
export { orderGroupsBySimilarity } from "./group-ordering.ts";
export { orderImagesBySimilarity } from "./image-ordering.ts";
export {
  buildImportedResult,
  clearImportedClusters,
  loadImportedClusters,
  saveImportedClusters,
} from "./imported.ts";
export {
  cancelClusterJob,
  getClusterAbortSignal,
  isClusterJobRunning,
  setClusterJobRunning,
} from "./job-mutex.ts";
export { removeLinkageTree } from "./linkage.ts";
export { computeMergeSuggestions, type MergeMethod } from "./merge-suggestions.ts";
export {
  extractFeatures,
  runFullCluster,
  runLinkageOnly,
  runRecut,
  runRecutAdaptive,
  runRecutByThreshold,
} from "./pipeline.ts";
export { broadcastProgress, getLastProgress, subscribeProgress } from "./progress.ts";

import { clearPatchDistMatrixCache } from "./distance-matrices.ts";
import { clearEmbeddingsCache } from "./embeddings.ts";
import { clearTreeCache } from "./linkage.ts";

/** Clear every in-memory cache held by the cluster modules. Called when the
 * underlying caches/embeddings/tree change on disk. */
export function invalidateClusterCache(): void {
  clearEmbeddingsCache();
  clearTreeCache();
  clearPatchDistMatrixCache();
}
