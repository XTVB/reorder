// Tree-walk helpers shared across cluster sub-stores. Used by listStore,
// interactionsStore, compareStore, expandStore, and consumers.

import type { ClusterResultData, SplitChildren } from "../../../types.ts";

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

/** Resolve {source, ...checkedClusters} from compare-state ids, deduped by id. */
export function collectMergeParticipants(
  clusters: ClusterResultData[],
  splitChildren: Record<string, SplitChildren>,
  source: ClusterResultData,
  checkedIds: Set<string>,
): ClusterResultData[] {
  const out = [source];
  const seenIds = new Set([source.id]);
  for (const id of checkedIds) {
    if (seenIds.has(id)) continue;
    const c = findClusterEverywhere(clusters, splitChildren, id);
    if (c) {
      out.push(c);
      seenIds.add(id);
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
