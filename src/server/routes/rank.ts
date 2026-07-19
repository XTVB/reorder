// GET /api/rank-scores — load persisted Rank engine state (null if none)
// PUT /api/rank-scores — persist engine state to .reorder-cache/
//
// ?target=images ranks the ungrouped images and stores to image_rank_scores.json,
// keyed by content hash so a rename can't scramble it (see fs/rank-scores.ts —
// it does the filename↔hash translation, under the rename lock so no scan sees
// a half-applied rename). The default (groups) target is keyed by stable group
// ids and needs no translation.

import { mkdir } from "node:fs/promises";
import type { RankScoresFile } from "../../client/utils/rankEngine.ts";
import {
  cacheDir,
  loadImageRankScores,
  type RankSession,
  rankScoresPath,
  readJsonTolerant,
  saveImageRankScores,
  saveRankSession,
  withRenameLock,
  writeJsonAtomic,
} from "../../fs/index.ts";
import { json } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

export const rankRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;

  // PUT /api/rank-judgements — upsert one session's raw judgements. Separate
  // from the scores because it's evidence, not state: see fs/rank-judgements.ts.
  if (path === "/api/rank-judgements" && req.method === "PUT") {
    const session = (await req.json()) as RankSession;
    if (!session?.id || !Array.isArray(session.events)) return json({ success: false }, 400);
    await withRenameLock(() => saveRankSession(targetDir, session));
    return json({ success: true });
  }

  if (path !== "/api/rank-scores") return null;

  const isImages = new URL(req.url).searchParams.get("target") === "images";

  if (req.method === "GET") {
    if (isImages) {
      return json(await withRenameLock(() => loadImageRankScores(targetDir)));
    }
    return json(await readJsonTolerant<unknown>(rankScoresPath(targetDir), null));
  }

  if (req.method === "PUT") {
    const body = await req.json();
    if (isImages) {
      await withRenameLock(() => saveImageRankScores(targetDir, body as RankScoresFile));
      return json({ success: true });
    }
    await mkdir(cacheDir(targetDir), { recursive: true });
    await writeJsonAtomic(rankScoresPath(targetDir), body, { atomic: true });
    return json({ success: true });
  }

  return null;
};
