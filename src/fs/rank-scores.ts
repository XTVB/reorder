// `image_rank_scores.json` — the Rank modal's ungrouped-images engine state.
//
// The client's engine is keyed by filename, but filenames are the one thing this
// app churns: every Apply/Save renumbers files by position, so a photo rated as
// `0007 sunset.jpg` becomes `0003 sunset.jpg`. Scores keyed by filename would
// then read back onto whichever photo inherited the name. So the file on disk is
// keyed by content hash instead — the same rename-surviving identity the
// embeddings cache uses — and the filename↔hash translation happens here, at the
// persistence boundary. The client stays filename-keyed and unchanged.
//
// `group_rank_scores.json` needs none of this: group ids are stable uuids.

import { mkdir } from "node:fs/promises";
import {
  pairKey,
  parsePairKey,
  type RankEntry,
  type RankScoresFile,
} from "../client/utils/rankEngine.ts";
import { readJsonTolerant, writeJsonAtomic } from "./atomic-json.ts";
import { imageContentHashes } from "./content-hashes.ts";
import { cacheDir, rankScoresPath } from "./paths.ts";

/** On-disk shape. Version 1 was filename-keyed; a rename could already have
 * scrambled which photo each of its entries described, and nothing on disk says
 * what the original names were — so a v1 file is discarded rather than migrated
 * into confidently-wrong attributions. */
const STORED_VERSION = 2;

interface StoredScores {
  version: number;
  keyedBy: "contentHash";
  updatedAt: string;
  entries: Record<string, RankEntry>;
  pairs: Record<string, number>;
  /** Tie-propensity evidence (see RankEngine.lambdaHat); item-independent, so
   * no hash translation. Absent in older files. */
  compareCount?: number;
  tieCount?: number;
}

async function readStored(targetDir: string): Promise<StoredScores | null> {
  const data = await readJsonTolerant<StoredScores | null>(
    rankScoresPath(targetDir, "images"),
    null,
  );
  if (!data || typeof data !== "object" || data.version !== STORED_VERSION) return null;
  return data;
}

/**
 * Load the scores as the client wants them: keyed by the filename each hash
 * currently lives under. Entries whose photo isn't in the directory right now
 * are simply absent — a deleted photo self-prunes, and a restored one comes back
 * with its rank. (Byte-identical duplicates share a hash and so collapse onto
 * one filename; de-duping them is the czkawka page's job.)
 */
export async function loadImageRankScores(targetDir: string): Promise<RankScoresFile | null> {
  const stored = await readStored(targetDir);
  if (!stored) return null;

  const nameFor = new Map<string, string>();
  for (const [filename, hash] of await imageContentHashes(targetDir)) nameFor.set(hash, filename);

  const entries: Record<string, RankEntry> = {};
  for (const [hash, entry] of Object.entries(stored.entries ?? {})) {
    const filename = nameFor.get(hash);
    if (filename) entries[filename] = entry;
  }
  const pairs: Record<string, number> = {};
  for (const [key, count] of Object.entries(stored.pairs ?? {})) {
    const pair = parsePairKey(key);
    const a = pair && nameFor.get(pair[0]);
    const b = pair && nameFor.get(pair[1]);
    if (a && b) pairs[pairKey(a, b)] = count;
  }
  return {
    version: 1,
    updatedAt: stored.updatedAt,
    entries,
    pairs,
    compareCount: stored.compareCount,
    tieCount: stored.tieCount,
  };
}

/**
 * Persist the client's filename-keyed engine state, re-keyed by content hash.
 *
 * The payload is authoritative for every photo in the directory: a Rank session
 * covers a scope (the ungrouped images, a selection, one group's contents) but
 * always sends back the beliefs it holds about the whole gallery, so an entry
 * that's present in the directory and *absent* from the payload was deliberately
 * dropped (Start over) and falls away here. A stored entry whose photo isn't in
 * the directory at all (deleted, or moved into a subfolder) is something the
 * client never saw, and a rank is irreplaceable human input — those carry over
 * untouched.
 */
export async function saveImageRankScores(
  targetDir: string,
  incoming: RankScoresFile,
): Promise<void> {
  const hashFor = await imageContentHashes(targetDir);
  const present = new Set(hashFor.values());
  const prev = await readStored(targetDir);

  const entries: Record<string, RankEntry> = {};
  for (const [hash, entry] of Object.entries(prev?.entries ?? {})) {
    if (!present.has(hash)) entries[hash] = entry;
  }
  for (const [filename, entry] of Object.entries(incoming.entries ?? {})) {
    const hash = hashFor.get(filename);
    if (hash) entries[hash] = entry;
  }

  const pairs: Record<string, number> = {};
  for (const [key, count] of Object.entries(prev?.pairs ?? {})) {
    const pair = parsePairKey(key);
    if (pair && (!present.has(pair[0]) || !present.has(pair[1]))) pairs[key] = count;
  }
  for (const [key, count] of Object.entries(incoming.pairs ?? {})) {
    const pair = parsePairKey(key);
    const a = pair && hashFor.get(pair[0]);
    const b = pair && hashFor.get(pair[1]);
    if (a && b) pairs[pairKey(a, b)] = count;
  }

  await mkdir(cacheDir(targetDir), { recursive: true });
  await writeJsonAtomic(
    rankScoresPath(targetDir, "images"),
    {
      version: STORED_VERSION,
      keyedBy: "contentHash",
      updatedAt: new Date().toISOString(),
      entries,
      pairs,
      compareCount: incoming.compareCount ?? prev?.compareCount,
      tieCount: incoming.tieCount ?? prev?.tieCount,
    } satisfies StoredScores,
    { atomic: true },
  );
}
