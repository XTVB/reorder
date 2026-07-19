import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtemp, rename, rm, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pairKey, RankEngine } from "../client/utils/rankEngine.ts";
import { loadImageRankScores, saveImageRankScores } from "./rank-scores.ts";

// Filenames carry a title after the number (computeRenames keeps it), so they
// contain spaces — pair keys have to survive that too.
const A = "0001 sunset.jpg";
const B = "0002 sunset.jpg";
const C = "0003 harbour.jpg";

const BYTES: Record<string, string> = {
  [A]: "photo-a",
  [B]: "photo-b-longer",
  [C]: "photo-c-longer-still",
};

let dir: string;

/** Distinct bytes per photo → distinct content hashes. */
function writePhotos() {
  return Promise.all([A, B, C].map((fn) => writeFile(join(dir, fn), BYTES[fn]!)));
}

/** A session: A rated best, B middling, C worst, with one A-vs-B comparison. */
function session(): RankEngine {
  const engine = new RankEngine();
  engine.observeRating(A, 0, false);
  engine.observeRating(B, 1, false);
  engine.observeRating(C, 4, false);
  engine.observeWin(A, [B]);
  return engine;
}

/** What the client sends: its state, pruned to the images it has in scope. */
function save(engine: RankEngine, scope: string[] = [A, B, C]) {
  return saveImageRankScores(dir, engine.toJSON(new Set(scope)));
}

/** Swap two photos' filenames, as renumbering-by-position does. */
async function swapNames(x: string, y: string) {
  const tmp = join(dir, "swap.tmp");
  await rename(join(dir, x), tmp);
  await rename(join(dir, y), join(dir, x));
  await rename(tmp, join(dir, y));
}

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "rank-scores-"));
  await writePhotos();
});

afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

describe("image rank scores", () => {
  test("round-trips a session through the engine", async () => {
    const before = session();
    await save(before);

    const after = RankEngine.fromJSON(await loadImageRankScores(dir));
    expect(after.get(A)).toEqual(before.get(A)!);
    expect(after.get(B)).toEqual(before.get(B)!);
    expect(after.get(C)).toEqual(before.get(C)!);
    expect(after.pairCounts.get(pairKey(A, B))).toBe(1);
    expect(after.ranking([A, B, C])).toEqual([A, B, C]);
  });

  test("scores follow the photo when Apply renumbers the files", async () => {
    const before = session();
    await save(before);
    // The photo rated best (A) lands in the last slot and takes C's name.
    await swapNames(A, C);

    const after = RankEngine.fromJSON(await loadImageRankScores(dir));
    // Same photos, same ranks — just read back under their new names.
    expect(after.get(C)).toEqual(before.get(A)!); // best photo, now named C
    expect(after.get(A)).toEqual(before.get(C)!); // worst photo, now named A
    expect(after.get(B)).toEqual(before.get(B)!);
    expect(after.observedIds([A, B, C])).toHaveLength(3); // nothing lost
    // The A-vs-B comparison is now C-vs-B: the count follows, so the engine
    // doesn't forget it already asked.
    expect(after.pairCounts.get(pairKey(C, B))).toBe(1);
  });

  test("repeated rename → reopen → autosave rounds keep every score", async () => {
    await save(session());

    // The reported bug: each round shed a few more ratings.
    for (let round = 0; round < 3; round++) {
      await swapNames(A, B);
      const engine = RankEngine.fromJSON(await loadImageRankScores(dir));
      expect(engine.observedIds([A, B, C])).toHaveLength(3);
      await save(engine);
    }
  });

  test("a photo the client can't see is neither returned nor dropped", async () => {
    const before = session();
    await save(before);

    // C leaves the top level (organized into a subfolder, or trashed).
    await unlink(join(dir, C));
    const engine = RankEngine.fromJSON(await loadImageRankScores(dir));
    expect(engine.has(C)).toBe(false);
    // The client re-saves what it can see — C's rank must not be collateral.
    await save(engine, [A, B]);

    await writeFile(join(dir, C), BYTES[C]!); // C comes back
    const restored = RankEngine.fromJSON(await loadImageRankScores(dir));
    expect(restored.get(C)).toEqual(before.get(C)!);
  });

  test("an empty save (Start over) clears the scores of visible photos", async () => {
    await save(session());
    await save(new RankEngine());

    const engine = RankEngine.fromJSON(await loadImageRankScores(dir));
    expect(engine.observedIds([A, B, C])).toHaveLength(0);
  });

  test("no file yet reads as null", async () => {
    expect(await loadImageRankScores(dir)).toBeNull();
  });
});
