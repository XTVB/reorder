// /api/merge-suggestions — pairwise group similarity via DINOv3 patch matching.

import type { MergeSuggestionSimilar } from "../../client/types.ts";
import { computeMergeSuggestions, loadConstraints, mergePairKey } from "../../cluster/index.ts";
import { loadGroups } from "../../fs/index.ts";
import { sseResponse } from "../middleware/sse.ts";
import type { RouteHandler } from "../types.ts";

export const mergeRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;
  if (path !== "/api/merge-suggestions" || req.method !== "POST") return null;

  const body = (await req.json()) as {
    threshold?: number;
    maxPerGroup?: number;
    fullResolution?: boolean;
    maxCombinedSize?: number;
  };
  const threshold = body.threshold ?? 0.65;
  const maxPerGroup = body.maxPerGroup ?? 8;
  const fullResolution = body.fullResolution ?? false;
  const maxCombinedSize = Math.max(0, Math.floor(body.maxCombinedSize ?? 0));

  return sseResponse(async (send) => {
    const startTime = performance.now();

    const entries = await computeMergeSuggestions(targetDir, threshold, {
      fullResolution,
      maxCombinedSize,
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
