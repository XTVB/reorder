// Tree-walk helpers shared across cluster sub-stores. Used by listStore,
// interactionsStore, expandStore, and consumers.

import type { ClusterResultData, SplitChildren } from "../../../types.ts";

/** Which section of a cluster card the user is interacting with. */
export type ImageSection = "confirmed" | "suggested";

/** Recursively collect every visible cluster (top-level + every split child). */
export function collectAllClusters(
  topLevel: ClusterResultData[],
  splitChildren: Record<string, SplitChildren>,
): ClusterResultData[] {
  const out: ClusterResultData[] = [];
  function visit(c: ClusterResultData) {
    out.push(c);
    const kids = splitChildren[c.id];
    if (kids) {
      visit(kids.childA);
      visit(kids.childB);
    }
  }
  for (const c of topLevel) visit(c);
  return out;
}

/** Locate a cluster by id in either the top-level array or any split children entry. */
export function findClusterEverywhere(
  topLevel: ClusterResultData[],
  splitChildren: Record<string, SplitChildren>,
  id: string,
): ClusterResultData | undefined {
  const top = topLevel.find((c) => c.id === id);
  if (top) return top;
  for (const kids of Object.values(splitChildren)) {
    if (kids.childA.id === id) return kids.childA;
    if (kids.childB.id === id) return kids.childB;
  }
  return undefined;
}

/** Concatenate `adds` onto `base`, skipping duplicates already in `base`. */
export function dedupeAppend(base: string[], adds: string[]): string[] {
  const seen = new Set(base);
  const out = [...base];
  for (const a of adds) {
    if (!seen.has(a)) {
      seen.add(a);
      out.push(a);
    }
  }
  return out;
}

/** Compute the deduped union of all images across participants in input order. */
export function unionImages(participants: ClusterResultData[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const c of participants) {
    for (const f of c.images) {
      if (!seen.has(f)) {
        seen.add(f);
        out.push(f);
      }
    }
  }
  return out;
}

/** Parse a "clusterId:filename" composite key from selection contexts. */
export function parseImageKey(key: string): { clusterId: string; filename: string } {
  const sep = key.indexOf(":");
  return { clusterId: key.slice(0, sep), filename: key.slice(sep + 1) };
}

/**
 * Single source of truth for the suggested section: keep aligned with what
 * the cluster card renders, since this also defines what "Add N to Group" adds.
 */
export function getSuggestedAdditions(
  cluster: ClusterResultData,
  cannotLinkIndex: Map<string, Set<string>>,
): string[] {
  const groupId = cluster.confirmedGroup?.id;
  if (!groupId) return cluster.images;
  const confirmedSet = new Set(cluster.confirmedGroup!.images);
  return cluster.images.filter(
    (f) => !confirmedSet.has(f) && !cannotLinkIndex.get(f)?.has(groupId),
  );
}

/** The image list belonging to one section of a cluster card. */
export function getSectionList(
  cluster: ClusterResultData,
  section: ImageSection,
  cannotLinkIndex: Map<string, Set<string>>,
): string[] {
  if (section === "confirmed") return cluster.confirmedGroup?.images ?? [];
  return getSuggestedAdditions(cluster, cannotLinkIndex);
}

/** Extract deduped filenames from a cluster:image composite-key Set. */
export function filenamesFromSelectedImages(selection: Iterable<string>): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const key of selection) {
    const { filename } = parseImageKey(key);
    if (!seen.has(filename)) {
      seen.add(filename);
      out.push(filename);
    }
  }
  return out;
}
