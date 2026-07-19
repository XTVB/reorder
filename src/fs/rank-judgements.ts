// `rank_judgements.json` — the append-only record of what the user actually
// told the Rank modal, in the order they told it.
//
// The scores file holds *beliefs*, which are derived and lossy: you cannot look
// at a posterior and ask whether the model that produced it was any good. This
// file holds the *observations* — every rating and every comparison answer — so
// the engine's predictions can be scored against them after the fact
// (scripts/rank-calibration.ts). Without it there is no way to tell whether the
// modelling assumptions (draw margin, beta, tier anchors) describe this user,
// and every session's judgements are gone the moment the modal closes.
//
// Sessions are upserted by id, so a session that re-answers an earlier question
// overwrites its own record rather than logging the same judgement twice; the
// events within a session stay in presentation order, which is what makes the
// replay prequential (each answer scored by a model that hasn't seen it yet).
//
// Image ids are content hashes, for the same reason the scores are (see
// rank-scores.ts): a rename must not silently re-point a judgement at a
// different photo, or the log slowly becomes fiction.

import { mkdir } from "node:fs/promises";
import type { RankEvent } from "../client/utils/rankEngine.ts";
import { readJsonTolerant, writeJsonAtomic } from "./atomic-json.ts";
import { imageContentHashes } from "./content-hashes.ts";
import { cacheDir, rankJudgementsPath } from "./paths.ts";

export type RankTarget = "groups" | "images";

export interface RankSession {
  id: string;
  target: RankTarget;
  at: string;
  events: RankEvent[];
}

export interface RankJudgementsFile {
  version: 1;
  sessions: RankSession[];
}

const EMPTY: RankJudgementsFile = { version: 1, sessions: [] };

export async function loadRankJudgements(targetDir: string): Promise<RankJudgementsFile> {
  const data = await readJsonTolerant<RankJudgementsFile | null>(
    rankJudgementsPath(targetDir),
    null,
  );
  if (!data || data.version !== 1 || !Array.isArray(data.sessions)) return EMPTY;
  return data;
}

/** Re-key an event's ids through a filename→hash map, dropping ids that don't
 * resolve (the photo is gone; a judgement we can't attribute is worse than none). */
function mapEvent(ev: RankEvent, hashFor: Map<string, string>): RankEvent | null {
  if (ev.kind === "rate") {
    const id = hashFor.get(ev.id);
    return id ? { ...ev, id } : null;
  }
  const ids = ev.ids.map((id) => hashFor.get(id));
  if (ids.some((id) => !id)) return null;
  const outcome = ev.outcome;
  let mapped = outcome;
  if (outcome?.kind === "win") {
    const winnerId = hashFor.get(outcome.winnerId);
    if (!winnerId) return null;
    mapped = { ...outcome, winnerId };
  } else if (outcome?.kind === "top") {
    const topIds = outcome.ids.map((id) => hashFor.get(id));
    if (topIds.some((id) => !id)) return null;
    mapped = { ...outcome, ids: topIds as string[] };
  } else if (outcome?.kind === "pin") {
    const id = hashFor.get(outcome.id);
    if (!id) return null;
    mapped = { kind: "pin", id };
  }
  return { kind: "compare", ids: ids as string[], outcome: mapped ?? null };
}

/**
 * Record (or re-record) one ranking session. Unanswered comparisons are dropped
 * — a question that was shown but never answered carries no judgement.
 */
export async function saveRankSession(targetDir: string, session: RankSession): Promise<void> {
  let events = session.events.filter((ev) => ev.kind === "rate" || ev.outcome !== null);

  if (session.target === "images") {
    const hashFor = await imageContentHashes(targetDir);
    events = events.map((ev) => mapEvent(ev, hashFor)).filter((ev): ev is RankEvent => ev !== null);
  }
  if (events.length === 0) return;

  const file = await loadRankJudgements(targetDir);
  const sessions = file.sessions.filter((s) => s.id !== session.id);
  sessions.push({ ...session, events });

  await mkdir(cacheDir(targetDir), { recursive: true });
  await writeJsonAtomic(
    rankJudgementsPath(targetDir),
    { version: 1, sessions } satisfies RankJudgementsFile,
    { atomic: true },
  );
}
