// Imported clusters: a JSON file at .reorder-cache/imported_clusters.json
// that takes precedence over linkage-tree re-cuts when present. Used when the
// user supplies pre-grouped clusters directly (e.g. from an external tool)
// and wants the UI to surface them without running the pipeline.

import { existsSync, mkdirSync } from "node:fs";
import { unlink } from "node:fs/promises";
import { writeJsonAtomic } from "../fs/atomic-json.ts";
import { loadGroups } from "../fs/groups.ts";
import { cacheDir, importedClustersPath } from "../fs/paths.ts";
import type {
  ClusterData,
  ClusterResultData,
  ImageGroup,
  ImportClusterInput,
} from "../shared/types.ts";
import { computeSuggestedCounts } from "./pipeline.ts";

export { importedClustersPath };

export function hasImportedClusters(targetDir: string): boolean {
  return existsSync(importedClustersPath(targetDir));
}

export async function loadImportedClusters(targetDir: string): Promise<ClusterData | null> {
  try {
    return (await Bun.file(importedClustersPath(targetDir)).json()) as ClusterData;
  } catch {
    return null;
  }
}

export async function saveImportedClusters(targetDir: string, data: ClusterData): Promise<void> {
  mkdirSync(cacheDir(targetDir), { recursive: true });
  await writeJsonAtomic(importedClustersPath(targetDir), data);
}

export async function clearImportedClusters(targetDir: string): Promise<void> {
  await unlink(importedClustersPath(targetDir)).catch(() => {});
}

export async function buildImportedResult(
  targetDir: string,
  input: ImportClusterInput[],
): Promise<ClusterData> {
  const imgToGroup = new Map<string, ImageGroup>();
  for (const g of loadGroups(targetDir)) {
    for (const f of g.images) imgToGroup.set(f, g);
  }

  const clusters: ClusterResultData[] = input.map((c, i) => {
    const sortedImages = [...c.images].sort();
    const confirmed = sortedImages.find((f) => imgToGroup.has(f));
    const group = confirmed ? (imgToGroup.get(confirmed) ?? null) : null;
    return {
      id: `imported_${i}`,
      name: c.name,
      images: sortedImages,
      confirmedGroup: group ? { id: group.id, name: group.name, images: group.images } : null,
    };
  });

  const totalImages = clusters.reduce((n, c) => n + c.images.length, 0);
  return {
    clusters,
    suggestedCounts: computeSuggestedCounts(totalImages),
    nClusters: clusters.length,
  };
}
