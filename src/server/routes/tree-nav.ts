// Tree-navigation operations: metrics, merge candidates, split, expand candidates.

import {
  computeClusterMetrics,
  expandCandidates,
  rankMergeCandidates,
  splitClusterByTree,
} from "../../cluster/tree-nav.ts";
import type { WeightConfig } from "../../shared/types.ts";
import { json } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

export const treeNavRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;

  if (path === "/api/cluster/tree-nav/metrics" && req.method === "POST") {
    const body = (await req.json()) as {
      clusters?: { id: string; images: string[] }[];
      weights?: WeightConfig;
    };
    if (!Array.isArray(body.clusters)) {
      return json({ error: "Body must include `clusters` array" }, 400);
    }
    const metrics = computeClusterMetrics(targetDir, body.clusters, body.weights);
    const out: Record<string, { cohesion: number; isolation: number; stability: number }> = {};
    for (const [id, m] of metrics) {
      out[id] = {
        cohesion: m.cohesion,
        isolation: Number.isFinite(m.isolation) ? m.isolation : -1,
        stability: m.stability,
      };
    }
    return json({ metrics: out });
  }

  if (path === "/api/cluster/tree-nav/merge-candidates" && req.method === "POST") {
    const body = (await req.json()) as {
      sourceImages?: string[];
      candidates?: { id: string; images: string[] }[];
      weights?: WeightConfig;
    };
    if (!Array.isArray(body.sourceImages) || body.sourceImages.length === 0) {
      return json({ error: "sourceImages must be non-empty" }, 400);
    }
    if (!Array.isArray(body.candidates)) {
      return json({ error: "candidates must be an array" }, 400);
    }
    const scores = rankMergeCandidates(targetDir, body.sourceImages, body.candidates, body.weights);
    return json({ scores });
  }

  if (path === "/api/cluster/tree-nav/split" && req.method === "POST") {
    const body = (await req.json()) as { images?: string[] };
    if (!Array.isArray(body.images) || body.images.length < 2) {
      return json({ error: "images must have at least 2 entries" }, 400);
    }
    const result = splitClusterByTree(targetDir, body.images);
    if (!result) return json({ error: "Cannot split — no valid binary partition" }, 422);
    return json(result);
  }

  if (path === "/api/cluster/tree-nav/expand-candidates" && req.method === "POST") {
    const body = (await req.json()) as {
      sourceImages?: string[];
      weights?: WeightConfig;
    };
    if (!Array.isArray(body.sourceImages) || body.sourceImages.length === 0) {
      return json({ error: "sourceImages must be non-empty" }, 400);
    }
    const result = expandCandidates(targetDir, body.sourceImages, body.weights);
    return json(result);
  }

  return null;
};
