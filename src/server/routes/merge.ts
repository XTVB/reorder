// /api/merge-suggestions — pairwise group similarity. Two methods: DINOv3 patch
// matching ("patches") or the weighted CLS-embedding blend ("embeddings").
// /api/groups/similarity-order — same pairwise scores, reduced to a group
// ordering (greedy nearest-neighbor chain) for the reorder page's sort button.

import type { MergeSuggestionSimilar } from "../../client/types.ts";
import type { MergeMethod } from "../../cluster/index.ts";
import {
  computeMergeSuggestions,
  orderGroupsBySimilarity,
  orderImagesBatch,
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
      /** Stable mode: force this many sets instead of the automatic gap cut. */
      stableClusters?: number;
      /** Gather mode: minimum cost improvement (blended distance) to move at all. */
      gatherMinGain?: number;
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
      body.mode === "tree" ||
      body.mode === "spectral" ||
      body.mode === "minimal" ||
      body.mode === "stable" ||
      body.mode === "gather"
        ? body.mode
        : "chain";

    if (body.target === "ungrouped") {
      return sseResponse(async (send) => {
        const [result] = await orderImagesBatch(
          targetDir,
          [{ id: "ungrouped", filenames: body.orderedImageFilenames ?? [] }],
          body.weights ?? {},
          {
            mode,
            minimalLocality: body.minimalLocality,
            stableClusters: body.stableClusters,
            gatherMinGain: body.gatherMinGain,
          },
          (msg) => send("progress", { message: msg }),
        );
        const { orderedIds, skipped, clusters, moved } = result ?? {
          orderedIds: [],
          skipped: 0,
        };
        send("result", { orderedIds, skipped, clusters, moved });
      });
    }

    return sseResponse(async (send) => {
      // minScore 0, no combined-size cap, rejected pairs kept: the ordering
      // needs every pair score, not just the ones the merge-suggestions UI
      // would surface — a rejected merge is still a similar pair.
      const entries = await computeMergeSuggestions(targetDir, 0, {
        method,
        fullResolution,
        maxCombinedSize: 0,
        weights: body.weights,
        includeRejected: true,
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
      let info: { clusters?: number; moved?: number } = {};
      send("result", {
        orderedIds: orderGroupsBySimilarity(groupIds, entries, {
          mode,
          anchorId: body.anchorId,
          minimalLocality: body.minimalLocality,
          stableClusters: body.stableClusters,
          gatherMinGain: body.gatherMinGain,
          onModeInfo: (i) => {
            info = i;
          },
        }),
        clusters: info.clusters,
        moved: info.moved,
      });
    });
  }

  // Batch image ordering: order the contents of several groups independently in
  // a single Rust invocation (one job per group). Used by the reorder page's
  // "Group contents" sort target.
  if (path === "/api/groups/similarity-order-batch") {
    const body = (await req.json()) as {
      weights?: WeightConfig;
      mode?: GroupOrderMode;
      minimalLocality?: number;
      stableClusters?: number;
      gatherMinGain?: number;
      jobs?: { id: string; orderedImageFilenames: string[] }[];
    };
    const mode: GroupOrderMode =
      body.mode === "tree" ||
      body.mode === "spectral" ||
      body.mode === "minimal" ||
      body.mode === "stable" ||
      body.mode === "gather"
        ? body.mode
        : "chain";
    const jobs = (body.jobs ?? []).map((j) => ({
      id: j.id,
      filenames: j.orderedImageFilenames ?? [],
    }));
    return sseResponse(async (send) => {
      const results = await orderImagesBatch(
        targetDir,
        jobs,
        body.weights ?? {},
        {
          mode,
          minimalLocality: body.minimalLocality,
          stableClusters: body.stableClusters,
          gatherMinGain: body.gatherMinGain,
        },
        (msg) => send("progress", { message: msg }),
      );
      send("result", { results });
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
    const rowMap = new Map<string, { refGroupId: string; similar: MergeSuggestionSimilar[] }>();

    // Rejected pairs are already filtered inside computeMergeSuggestions.
    for (const d of entries) {
      const gA = groupMap.get(d.groupA);
      const gB = groupMap.get(d.groupB);
      if (!gA || !gB) continue;

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
