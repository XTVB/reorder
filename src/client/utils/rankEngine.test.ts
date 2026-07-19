import { describe, expect, test } from "bun:test";
import {
  applyEvent,
  CONSIDERING_SLOT,
  DELETE_SLOT,
  nearestTier,
  pairKey,
  RankEngine,
  type RankEvent,
} from "./rankEngine.ts";

// outcomeProbs is what makes the model falsifiable — the calibration harness
// scores real judgements against it — so it has to be an honest distribution.
describe("RankEngine.outcomeProbs", () => {
  test("is a proper distribution over wins + draw, for a pair and a best-of-N", () => {
    const e = new RankEngine();
    e.observeRating("a", 0, false);
    e.observeRating("b", 2, false);
    e.observeRating("c", 3, false);
    e.observeRating("d", 4, false);

    for (const ids of [
      ["a", "b"],
      ["a", "b", "c"],
      ["a", "b", "c", "d"],
    ]) {
      const { win, draw } = e.outcomeProbs(ids);
      expect(win).toHaveLength(ids.length);
      for (const p of [...win, draw]) {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
      expect([...win, draw].reduce((s, p) => s + p, 0)).toBeCloseTo(1, 2);
    }
  });

  test("favours the higher-rated item, and calls an even matchup near 50/50", () => {
    const e = new RankEngine();
    e.observeRating("best", 0, true);
    e.observeRating("worst", 4, true);
    const lopsided = e.outcomeProbs(["best", "worst"]);
    // Win mass is capped at 1 - lambdaHat(): the tie hurdle comes off the top.
    expect(lopsided.win[0]!).toBeGreaterThan(0.9 * (1 - e.lambdaHat()));
    expect(lopsided.win[1]!).toBeLessThan(0.05);

    const f = new RankEngine();
    f.observeRating("x", 2, false);
    f.observeRating("y", 2, false);
    const even = f.outcomeProbs(["x", "y"]);
    expect(Math.abs(even.win[0]! - even.win[1]!)).toBeLessThan(0.05);
  });

  test("draw probability is the learned tie propensity, not a margin", () => {
    const e = new RankEngine();
    e.observeRating("x", 2, false);
    e.observeRating("y", 2, false);
    // Before any comparisons: the Beta prior's mean.
    const prior = e.t.tiePriorA / (e.t.tiePriorA + e.t.tiePriorB);
    expect(e.outcomeProbs(["x", "y"]).draw).toBeCloseTo(prior, 10);
    // A user who keeps tying is predicted to keep tying...
    e.observeTie(["x", "y"]);
    const afterTie = e.outcomeProbs(["x", "y"]).draw;
    expect(afterTie).toBeGreaterThan(prior);
    // ...and decisive answers pull the tie propensity back down.
    e.observeWin("x", ["y"]);
    expect(e.outcomeProbs(["x", "y"]).draw).toBeLessThan(afterTie);
  });

  test("tie propensity survives serialization and snapshot undo", () => {
    const e = new RankEngine();
    e.observeRating("x", 2, false);
    e.observeRating("y", 2, false);
    e.observeTie(["x", "y"]);
    e.observeWin("x", ["y"]);

    const back = RankEngine.fromJSON(e.toJSON(new Set(["x", "y"])));
    expect(back.lambdaHat()).toBeCloseTo(e.lambdaHat(), 12);

    const snap = e.snapshot();
    e.observeTie(["x", "y"]);
    expect(e.lambdaHat()).toBeGreaterThan(back.lambdaHat());
    e.restore(snap);
    expect(e.lambdaHat()).toBeCloseTo(back.lambdaHat(), 12);
  });

  test("predictions are reproducible across replays (seeded sampling)", () => {
    const build = () => {
      const e = new RankEngine();
      e.observeRating("a", 1, false);
      e.observeRating("b", 2, false);
      e.observeRating("c", 2, false);
      return e;
    };
    expect(build().outcomeProbs(["a", "b", "c"])).toEqual(build().outcomeProbs(["a", "b", "c"]));
  });
});

describe("applyEvent", () => {
  test("replaying a logged timeline reproduces the live engine exactly", () => {
    const events: RankEvent[] = [
      { kind: "rate", id: "a", slot: 0, confident: false },
      { kind: "rate", id: "b", slot: 1, confident: true },
      { kind: "rate", id: "c", slot: 1, confident: false },
      { kind: "compare", ids: ["a", "b"], outcome: { kind: "win", winnerId: "b" } },
      { kind: "compare", ids: ["b", "c"], outcome: { kind: "tie" } },
      { kind: "compare", ids: ["a", "c"], outcome: { kind: "skip" } },
      { kind: "compare", ids: ["a", "b", "c"], outcome: { kind: "tie" } },
    ];

    const live = new RankEngine();
    live.observeRating("a", 0, false);
    live.observeRating("b", 1, true);
    live.observeRating("c", 1, false);
    live.observeWin("b", ["a"]);
    live.observeTie(["b", "c"]);
    live.skip(["a", "c"]);
    live.observeTie(["a", "b", "c"]);

    const replayed = new RankEngine();
    for (const ev of events) applyEvent(replayed, ev);

    for (const id of ["a", "b", "c"]) {
      expect(replayed.get(id)).toEqual(live.get(id)!);
    }
    expect([...replayed.pairCounts].sort()).toEqual([...live.pairCounts].sort());
  });

  test("an unanswered comparison changes nothing", () => {
    const e = new RankEngine();
    e.observeRating("a", 1, false);
    const before = { ...e.get("a")! };
    applyEvent(e, { kind: "compare", ids: ["a", "b"], outcome: null });
    expect(e.get("a")).toEqual(before);
    expect(e.has("b")).toBe(false);
  });
});

describe("RankEngine", () => {
  test("rating lands near the tier anchor and repeat ratings are no-ops", () => {
    const e = new RankEngine();
    e.observeRating("a", 0, false);
    const first = { ...e.get("a")! };
    expect(first.mu).toBeGreaterThan(1.5);
    e.observeRating("a", 0, false);
    expect(e.get("a")!.mu).toBe(first.mu);
    expect(e.get("a")!.sigma).toBe(first.sigma);
  });

  test("confident rating tightens sigma well below a quick rating", () => {
    const e = new RankEngine();
    e.observeRating("quick", 2, false);
    e.observeRating("sure", 2, true);
    expect(e.get("sure")!.sigma).toBeLessThan(e.get("quick")!.sigma * 0.5);
  });

  test("wins raise the winner and lower the loser, shrinking sigma", () => {
    const e = new RankEngine();
    e.observeRating("a", 2, false);
    e.observeRating("b", 2, false);
    const beforeA = { ...e.get("a")! };
    e.observeWin("a", ["b"]);
    expect(e.get("a")!.mu).toBeGreaterThan(beforeA.mu);
    expect(e.get("b")!.mu).toBeLessThan(beforeA.mu);
    expect(e.get("a")!.sigma).toBeLessThan(beforeA.sigma);
  });

  test("repeated wins converge to the correct order", () => {
    const e = new RankEngine();
    e.observeRating("worse", 1, false);
    e.observeRating("better", 2, false);
    for (let i = 0; i < 6; i++) e.observeWin("better", ["worse"]);
    expect(e.get("better")!.mu).toBeGreaterThan(e.get("worse")!.mu);
  });

  test("ties pull tangled groups together", () => {
    const e = new RankEngine();
    e.observeRating("a", 1, false);
    e.observeRating("b", 3, false);
    const gapBefore = e.get("a")!.mu - e.get("b")!.mu;
    e.observeTie(["a", "b"]);
    const gapAfter = e.get("a")!.mu - e.get("b")!.mu;
    expect(Math.abs(gapAfter)).toBeLessThan(Math.abs(gapBefore));
  });

  test("a best-of-N tie draws every pair on the screen", () => {
    const e = new RankEngine();
    const ids = ["a", "b", "c", "d"];
    e.observeRating("a", 0, false);
    e.observeRating("b", 1, false);
    e.observeRating("c", 2, false);
    e.observeRating("d", 3, false);
    const spreadBefore = e.get("a")!.mu - e.get("d")!.mu;
    const sigmaBefore = e.get("a")!.sigma;

    e.observeTie(ids);

    // Every pair is pulled together and marked asked, the extremes included —
    // not just the neighbours.
    expect(e.get("a")!.mu - e.get("d")!.mu).toBeLessThan(spreadBefore);
    expect(e.get("a")!.sigma).toBeLessThan(sigmaBefore);
    for (const [x, y] of [
      ["a", "b"],
      ["a", "c"],
      ["a", "d"],
      ["b", "c"],
      ["b", "d"],
      ["c", "d"],
    ]) {
      expect(e.pairCounts.get(pairKey(x!, y!))).toBe(1);
    }
    // Order is preserved — a tie says "close", not "identical".
    expect(e.ranking(ids)).toEqual(ids);
  });

  test("comparisons need no ratings: unrated items are asked at the prior", () => {
    const e = new RankEngine();
    // Nothing rated at all — a compare-first session must still get questions.
    const q = e.nextComparison(["a", "b", "c"]);
    expect(q).not.toBeNull();
    for (const id of q!.ids) expect(["a", "b", "c"]).toContain(id);

    // Answering creates entries and orders winner above loser; the untouched
    // item stays unobserved (ranking keeps it in incoming order between
    // normals and floors).
    e.observeWin(q!.ids[0]!, [q!.ids[1]!]);
    expect(e.get(q!.ids[0]!)!.mu).toBeGreaterThan(e.get(q!.ids[1]!)!.mu);
    expect(e.get(q!.ids[0]!)!.tier).toBeUndefined();

    // A mixed pool works too: rated and unrated items can be paired.
    const f = new RankEngine();
    f.observeRating("rated", 2, false);
    const mixed = f.nextComparison(["rated", "fresh"]);
    expect(mixed).not.toBeNull();
    expect(mixed!.ids.sort()).toEqual(["fresh", "rated"]);
  });

  test("unrated items respect the pin and repeat-cap exclusions", () => {
    const e = new RankEngine();
    e.pin("a");
    expect(e.nextComparison(["a", "b"])).toBeNull(); // pinned leaves only one

    const f = new RankEngine();
    for (let i = 0; i < 3; i++) f.skip(["x", "y"]);
    expect(f.nextComparison(["x", "y"])).toBeNull(); // repeat cap holds unrated too
  });

  test("policy never asks non-overlapping pairs and respects repeat cap", () => {
    const e = new RankEngine();
    e.observeRating("top", 0, true);
    e.observeRating("bottom", 4, true);
    // Only two groups, far apart with tight posteriors: nothing worth asking.
    expect(e.nextComparison(["top", "bottom"])).toBeNull();

    const f = new RankEngine();
    f.observeRating("a", 2, false);
    f.observeRating("b", 2, false);
    const q = f.nextComparison(["a", "b"]);
    expect(q).not.toBeNull();
    expect(q!.ids.sort()).toEqual(["a", "b"]);
    for (let i = 0; i < 3; i++) f.skip(["a", "b"]);
    expect(f.nextComparison(["a", "b"])).toBeNull();
  });

  test("pinned and delete-floor groups leave the question pool", () => {
    const e = new RankEngine();
    e.observeRating("a", 2, false);
    e.observeRating("b", 2, false);
    e.pin("a");
    expect(e.nextComparison(["a", "b"])).toBeNull();

    const f = new RankEngine();
    f.observeRating("a", DELETE_SLOT, false);
    f.observeRating("b", DELETE_SLOT, false);
    expect(f.nextComparison(["a", "b"])).toBeNull();
  });

  test("ranking sinks floors below normals and unobserved between", () => {
    const e = new RankEngine();
    e.observeRating("good", 0, false);
    e.observeRating("considering", CONSIDERING_SLOT, false);
    e.observeRating("deleting", DELETE_SLOT, false);
    const order = e.ranking(["deleting", "unseen", "considering", "good"]);
    expect(order).toEqual(["good", "unseen", "considering", "deleting"]);
  });

  test("uncertainty meter drops as comparisons resolve a tangle", () => {
    const e = new RankEngine();
    const scope = ["a", "b", "c"];
    for (const id of scope) e.observeRating(id, 2, false);
    const before = e.expectedUncertainPairs(scope);
    e.observeWin("a", ["b"]);
    e.observeWin("b", ["c"]);
    e.observeWin("a", ["c"]);
    expect(e.expectedUncertainPairs(scope)).toBeLessThan(before);
  });

  test("sortedFraction climbs with coverage and with comparison resolution", () => {
    const scope = ["a", "b", "c", "d"];
    const e = new RankEngine();
    // Nothing rated: 0% sorted.
    expect(e.sortedFraction(scope)).toBe(0);

    // Rate all into distinct tiers → partly sorted (quick ratings overlap).
    e.observeRating("a", 0, false);
    e.observeRating("b", 1, false);
    e.observeRating("c", 2, false);
    e.observeRating("d", 3, false);
    const afterRating = e.sortedFraction(scope);
    expect(afterRating).toBeGreaterThan(0);
    expect(afterRating).toBeLessThan(1);

    // Confirming order with comparisons tightens the boundaries → higher %.
    for (let i = 0; i < 4; i++) {
      e.observeWin("a", ["b"]);
      e.observeWin("b", ["c"]);
      e.observeWin("c", ["d"]);
    }
    expect(e.sortedFraction(scope)).toBeGreaterThan(afterRating);
  });

  test("sortedFraction: an unrated group drags the percentage down", () => {
    const scope = ["a", "b", "c"];
    const e = new RankEngine();
    e.observeRating("a", 0, true);
    e.observeRating("b", 4, true);
    const withGap = e.sortedFraction(scope); // "c" unrated
    e.observeRating("c", 2, true);
    expect(e.sortedFraction(scope)).toBeGreaterThan(withGap);
  });

  test("serialization round-trips and prunes stale ids", () => {
    const e = new RankEngine();
    e.observeRating("keep", 1, true);
    e.observeRating("stale", 3, false);
    e.observeWin("keep", ["stale"]);
    const json = e.toJSON(new Set(["keep"]));
    expect(Object.keys(json.entries)).toEqual(["keep"]);
    expect(Object.keys(json.pairs)).toEqual([]);
    const back = RankEngine.fromJSON(json);
    expect(back.get("keep")!.mu).toBeCloseTo(e.get("keep")!.mu, 10);
    expect(back.get("keep")!.tier).toBe(1);
  });

  test("snapshot/restore rolls a comparison back exactly (undo)", () => {
    const e = new RankEngine();
    e.observeRating("a", 2, false);
    e.observeRating("b", 2, false);
    const snap = e.snapshot();
    const beforeA = { ...e.get("a")! };
    e.observeWin("a", ["b"]);
    expect(e.get("a")!.mu).not.toBeCloseTo(beforeA.mu, 6); // the win moved it
    e.restore(snap);
    expect(e.get("a")!.mu).toBeCloseTo(beforeA.mu, 12);
    expect(e.get("a")!.sigma).toBeCloseTo(beforeA.sigma, 12);
    // Pair count is rolled back too, so the same question is offered again.
    const q = e.nextComparison(["a", "b"]);
    expect(q!.ids.sort()).toEqual(["a", "b"]);
    // Restore is a deep copy: mutating the engine must not touch the snapshot.
    e.observeWin("a", ["b"]);
    expect(snap.entries.get("a")!.mu).toBeCloseTo(beforeA.mu, 12);
  });

  test("nearestTier maps posterior means back to tier slots", () => {
    expect(nearestTier(3.0)).toBe(0);
    expect(nearestTier(0.2)).toBe(2);
    expect(nearestTier(-5)).toBe(4);
  });
});
