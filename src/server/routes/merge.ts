// /api/merge-suggestions — pairwise group similarity. Two methods: DINOv3 patch
// matching ("patches") or the weighted CLS-embedding blend ("embeddings").
// /api/groups/similarity-order — same pairwise scores, reduced to a group
// ordering (greedy nearest-neighbor chain) for the reorder page's sort button.

import type { MergeSuggestionSimilar } from "../../client/types.ts";
import type { MergeMethod } from "../../cluster/index.ts";
import {
  computeMergeSuggestions,
  loadConstraints,
  mergePairKey,
  orderGroupsBySimilarity,
  orderImagesBySimilarity,
} from "../../cluster/index.ts";
import { loadGroups } from "../../fs/index.ts";
import type { GroupOrderMode, WeightConfig } from "../../shared/types.ts";
import { sseResponse } from "../middleware/sse.ts";
import type { RouteHandler } from "../types.ts";

export const mergeRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;
  if (req.method !== "POST") return null;

  if (path === "/api/groups/similarity-order") {
    const body = (await req.json()) as {
      method?: MergeMethod;
      fullResolution?: boolean;
      weights?: WeightConfig;
      mode?: GroupOrderMode;
      anchorId?: string;
      minimalLocality?: number;
      /** Group ids in the client's current gallery order — the base ordering. */
      orderedGroupIds?: string[];
      /** What to order: groups (default) or individual images. */
      target?: "groups" | "ungrouped";
      /**
       * Image filenames in the client's current gallery order. The client
       * decides scope: all ungrouped images by default, or an explicit
       * selection (which may include in-group images so they sort within
       * their group).
       */
      orderedImageFilenames?: string[];
    };
    const method: MergeMethod = body.method === "patches" ? "patches" : "embeddings";
    const fullResolution = body.fullResolution ?? false;
    const mode: GroupOrderMode =
      body.mode === "tree" || body.mode === "spectral" || body.mode === "minimal"
        ? body.mode
        : "chain";

    if (body.target === "ungrouped") {
      return sseResponse(async (send) => {
        send("result", {
          orderedIds: await orderImagesBySimilarity(
            targetDir,
            body.orderedImageFilenames ?? [],
            body.weights ?? {},
            { mode, minimalLocality: body.minimalLocality },
            (msg) => send("progress", { message: msg }),
          ),
        });
      });
    }

    return sseResponse(async (send) => {
      // minScore 0 and no combined-size cap: the ordering needs every pair
      // score, not just the ones the merge-suggestions UI would surface.
      const entries = await computeMergeSuggestions(targetDir, 0, {
        method,
        fullResolution,
        maxCombinedSize: 0,
        weights: body.weights,
        onProgress: (msg) => send("progress", { message: msg }),
      });
      // Base ordering: the client's gallery order where provided, reconciled
      // against disk (unknown ids dropped, missing ids appended in disk
      // order). Minimal mode preserves this order; the others use it for
      // tie-breaks.
      const diskIds = loadGroups(targetDir).map((g) => g.id);
      const diskIdSet = new Set(diskIds);
      const clientIds: string[] = [];
      const clientIdSet = new Set<string>();
      for (const id of body.orderedGroupIds ?? []) {
        if (diskIdSet.has(id) && !clientIdSet.has(id)) {
          clientIds.push(id);
          clientIdSet.add(id);
        }
      }
      const groupIds = [...clientIds, ...diskIds.filter((id) => !clientIdSet.has(id))];
      send("result", {
        orderedIds: orderGroupsBySimilarity(groupIds, entries, {
          mode,
          anchorId: body.anchorId,
          minimalLocality: body.minimalLocality,
        }),
      });
    });
  }

  if (path !== "/api/merge-suggestions") return null;

  const body = (await req.json()) as {
    threshold?: number;
    maxPerGroup?: number;
    method?: MergeMethod;
    fullResolution?: boolean;
    maxCombinedSize?: number;
    weights?: WeightConfig;
  };
  const threshold = body.threshold ?? 0.65;
  const maxPerGroup = body.maxPerGroup ?? 8;
  const method: MergeMethod = body.method === "patches" ? "patches" : "embeddings";
  const fullResolution = body.fullResolution ?? false;
  const maxCombinedSize = Math.max(0, Math.floor(body.maxCombinedSize ?? 0));

  return sseResponse(async (send) => {
    const startTime = performance.now();

    const entries = await computeMergeSuggestions(targetDir, threshold, {
      method,
      fullResolution,
      maxCombinedSize,
      weights: body.weights,
      onProgress: (msg) => send("progress", { message: msg }),
    });

    const groupMap = new Map(loadGroups(targetDir).map((g) => [g.id, g]));
    const rejectedKeys = new Set(
      loadConstraints(targetDir).rejectedMergePairs.map((p) => mergePairKey(p.groupA, p.groupB)),
    );
    const rowMap = new Map<string, { refGroupId: string; similar: MergeSuggestionSimilar[] }>();

    for (const d of entries) {
      const gA = groupMap.get(d.groupA);
      const gB = groupMap.get(d.groupB);
      if (!gA || !gB) continue;
      if (rejectedKeys.has(mergePairKey(d.groupA, d.groupB))) continue;

      // 1 - patchMedian so lower = more similar, matching the Ward-distance semantics used by other UI.
      const displayDist = 1 - d.patchMedian;

      for (const [srcId, other] of [
        [d.groupA, gB],
        [d.groupB, gA],
      ] as const) {
        let row = rowMap.get(srcId);
        if (!row) {
          row = { refGroupId: srcId, similar: [] };
          rowMap.set(srcId, row);
        }
        row.similar.push({
          groupId: other.id,
          groupName: other.name,
          groupImages: other.images,
          distance: displayDist,
        });
      }
    }

    const suggestions = Array.from(rowMap.values())
      .map((row) => {
        const g = groupMap.get(row.refGroupId)!;
        row.similar.sort((a, b) => a.distance - b.distance);
        row.similar = row.similar.slice(0, maxPerGroup);
        return {
          refGroupId: row.refGroupId,
          refGroupName: g.name,
          refGroupImages: g.images,
          similar: row.similar,
        };
      })
      .sort((a, b) => a.similar[0]!.distance - b.similar[0]!.distance);

    const computeTimeMs = Math.round(performance.now() - startTime);
    send("result", { suggestions, computeTimeMs });
  });
};
