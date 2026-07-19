#!/usr/bin/env bun
/**
 * Score the Rank engine's model against what the user actually did.
 *
 *   bun run scripts/rank-calibration.ts <dir>            # is the model any good?
 *   bun run scripts/rank-calibration.ts <dir> --sweep    # which settings predict best?
 *
 * The engine can only ever be checked from OUTSIDE itself. Its update maths can
 * be verified against its own assumptions (and is — see the Monte Carlo note on
 * observeTie), but that says nothing about whether those assumptions describe
 * this person's taste. The only evidence that can is the judgements themselves.
 *
 * The method is prequential: replay the judgement log in the order it happened
 * and, before applying each comparison, ask the engine for its predictive
 * distribution over the outcomes (RankEngine.outcomeProbs). The engine has never
 * seen that answer, so the user's actual answer is a held-out label for a
 * probability the model already committed to. No hand-labelling: the answers
 * *are* the labels.
 *
 * What to read:
 *   - log loss vs. the uniform baseline. Below it = the model has real
 *     predictive power. At or above it = its beliefs are noise.
 *   - the reliability table. The engine deliberately asks near-coin-flips, so
 *     accuracy is a bad metric — but when it *does* claim 80%, it should be
 *     right about 80% of the time. If "predicted" consistently exceeds
 *     "observed", the model is overconfident: sigmas shrink too fast (which is
 *     exactly what an over-weighted tie update would look like).
 *   - --sweep replays the same log under alternative settings and ranks them by
 *     log loss. That is how the 4-way-tie question gets settled by data rather
 *     than by argument.
 */

import { resolve } from "node:path";
import {
  applyEvent,
  DEFAULT_TUNING,
  RankEngine,
  type RankTuning,
} from "../src/client/utils/rankEngine.ts";
import { loadRankJudgements, type RankSession } from "../src/fs/index.ts";

interface Scored {
  /** Probability the model gave to the outcome that actually happened. */
  p: number;
  /** Probability it gave to its own favourite, and whether that favourite won. */
  pFavourite: number;
  favouriteWon: boolean;
  n: number;
  isTie: boolean;
}

/** Replay one target's sessions in order, scoring each comparison before it's applied. */
function replay(sessions: RankSession[], tuning: RankTuning): Scored[] {
  const engine = new RankEngine(tuning);
  const scored: Scored[] = [];

  for (const session of sessions) {
    for (const ev of session.events) {
      if (ev.kind === "compare" && ev.outcome) {
        const o = ev.outcome;
        // "top" answers are applied but not scored: the priced outcome space is
        // N outright winners + a full draw, and a joint-top is a coarser label
        // that fits neither category. Their belief updates still shape every
        // later prediction, so margin/top tunings remain sweepable.
        if (o.kind === "win" || o.kind === "tie") {
          const probs = engine.outcomeProbs(ev.ids);
          const outcomes = [...probs.win, probs.draw];
          const actual =
            o.kind === "tie" ? ev.ids.length : ev.ids.indexOf((o as { winnerId: string }).winnerId);
          if (actual >= 0) {
            let favI = 0;
            for (let i = 1; i < outcomes.length; i++) {
              if (outcomes[i]! > outcomes[favI]!) favI = i;
            }
            scored.push({
              p: outcomes[actual]!,
              pFavourite: outcomes[favI]!,
              favouriteWon: favI === actual,
              n: ev.ids.length,
              isTie: o.kind === "tie",
            });
          }
        }
      }
      applyEvent(engine, ev); // only now does the engine get to see it
    }
  }
  return scored;
}

const EPS = 1e-6;
const logLoss = (s: Scored[]) =>
  s.reduce((sum, x) => sum - Math.log(Math.max(x.p, EPS)), 0) / (s.length || 1);
/** Uniform over the N winners + a draw — what a model that knows nothing scores. */
const baselineLoss = (s: Scored[]) =>
  s.reduce((sum, x) => sum + Math.log(x.n + 1), 0) / (s.length || 1);

function reliability(scored: Scored[]): string {
  const buckets = [
    [0.0, 0.4],
    [0.4, 0.5],
    [0.5, 0.6],
    [0.6, 0.7],
    [0.7, 0.8],
    [0.8, 0.9],
    [0.9, 1.01],
  ];
  const rows = buckets.map(([lo, hi]) => {
    const inB = scored.filter((s) => s.pFavourite >= lo! && s.pFavourite < hi!);
    if (inB.length === 0) return null;
    const predicted = inB.reduce((a, b) => a + b.pFavourite, 0) / inB.length;
    const observed = inB.filter((s) => s.favouriteWon).length / inB.length;
    const gap = predicted - observed;
    const flag = inB.length < 10 ? "(thin)" : Math.abs(gap) > 0.15 ? "  <-- off" : "";
    return `    ${(lo! * 100).toFixed(0).padStart(3)}-${(hi! * 100).toFixed(0).padStart(3)}%  n=${String(inB.length).padStart(4)}  predicted=${(predicted * 100).toFixed(0).padStart(3)}%  observed=${(observed * 100).toFixed(0).padStart(3)}%  ${flag}`;
  });
  return rows.filter(Boolean).join("\n");
}

// --- Tunings to try in --sweep. Each is "the same log, a different model". ---
function sweepGrid(): [string, RankTuning][] {
  const out: [string, RankTuning][] = [["default (shipped)", DEFAULT_TUNING]];
  for (const beta of [0.3, 0.5, 0.8, 1.2]) out.push([`beta=${beta}`, { ...DEFAULT_TUNING, beta }]);
  // The hurdle's Beta prior over tie propensity: mean a/(a+b), weight a+b.
  for (const [a, b] of [
    [1, 3],
    [1, 7],
    [1, 15],
    [2, 14],
  ] as const)
    out.push([`tiePrior(${a},${b})`, { ...DEFAULT_TUNING, tiePriorA: a, tiePriorB: b }]);
  // drawMargin no longer enters prediction; it only shapes the tie update.
  for (const drawMargin of [0.1, 0.25, 0.5])
    out.push([`drawMargin=${drawMargin}`, { ...DEFAULT_TUNING, drawMargin }]);
  for (const ratingSigma of [0.5, 0.9, 1.4])
    out.push([`ratingSigma=${ratingSigma}`, { ...DEFAULT_TUNING, ratingSigma }]);
  for (const betaMulti of [0.5, 0.7, 1.0])
    out.push([`betaMulti=${betaMulti}`, { ...DEFAULT_TUNING, betaMulti }]);
  out.push(["tieMode=chain", { ...DEFAULT_TUNING, tieMode: "chain" }]);
  // Win-margin widths only differentiate once the log contains margin-annotated
  // answers ("barely"/"clearly" wins, joint-top picks) — until then they score
  // identically to the default.
  for (const slimWinMargin of [0.3, 0.5, 0.8])
    out.push([`slimWin=${slimWinMargin}`, { ...DEFAULT_TUNING, slimWinMargin }]);
  for (const clearWinMargin of [0.7, 1.0, 1.5])
    out.push([`clearWin=${clearWinMargin}`, { ...DEFAULT_TUNING, clearWinMargin }]);
  // Tier spacing: is one tier really 1.7 apart on the latent scale? (x0.6
  // recovers the old 1.0 spacing; the corpus put the optimum at 1.7-2.0.)
  for (const k of [0.6, 0.85, 1.2, 1.4]) {
    out.push([
      `anchors x${k}`,
      { ...DEFAULT_TUNING, anchors: DEFAULT_TUNING.anchors.map((a) => a * k) },
    ]);
  }
  return out;
}

async function main() {
  const args = process.argv.slice(2);
  const sweep = args.includes("--sweep");
  const dir = resolve(args.find((a) => !a.startsWith("--")) ?? ".");

  const file = await loadRankJudgements(dir);
  const byTarget = new Map<string, RankSession[]>();
  for (const s of file.sessions) {
    if (!byTarget.has(s.target)) byTarget.set(s.target, []);
    byTarget.get(s.target)!.push(s);
  }
  for (const list of byTarget.values()) list.sort((a, b) => a.at.localeCompare(b.at));

  if (byTarget.size === 0) {
    console.log(`No judgements logged yet in ${dir}/.reorder-cache/rank_judgements.json`);
    console.log("Rank some images or groups first — the log is written as you go.");
    return;
  }

  for (const [target, sessions] of byTarget) {
    const events = sessions.reduce((n, s) => n + s.events.length, 0);
    console.log(`\n=== ${target} — ${sessions.length} session(s), ${events} judgement(s) ===`);

    const scored = replay(sessions, DEFAULT_TUNING);
    if (scored.length < 10) {
      console.log(
        `  Only ${scored.length} comparison(s) — too few to say anything. Keep ranking; ` +
          `the log accumulates across sessions.`,
      );
      continue;
    }

    const loss = logLoss(scored);
    const base = baselineLoss(scored);
    const ties = scored.filter((s) => s.isTie).length;
    console.log(`  comparisons scored : ${scored.length} (${ties} ties)`);
    console.log(`  log loss           : ${loss.toFixed(4)}`);
    console.log(`  uniform baseline   : ${base.toFixed(4)}`);
    console.log(
      `  verdict            : ${
        loss < base - 0.02
          ? `model beats chance by ${(base - loss).toFixed(3)} nats — its beliefs predict you`
          : "NO better than chance — the beliefs are not tracking this user"
      }`,
    );
    console.log("\n  reliability (when it says X%, does X% happen?)");
    console.log(reliability(scored));

    if (sweep) {
      console.log("\n  --- sweep: same judgements, different model ---");
      const results = sweepGrid()
        .map(([name, tuning]) => ({ name, loss: logLoss(replay(sessions, tuning)) }))
        .sort((a, b) => a.loss - b.loss);
      const best = results[0]!.loss;
      for (const r of results) {
        const delta = r.loss - best;
        console.log(
          `    ${r.name.padEnd(20)} logloss=${r.loss.toFixed(4)}  ${
            delta === 0 ? "<-- best" : `(+${delta.toFixed(4)})`
          }`,
        );
      }
      console.log(
        "\n  A setting that beats the shipped default by a meaningful margin is a\n" +
          "  real finding — change the constant. Differences under ~0.01 nats on a\n" +
          "  few hundred judgements are noise; get more data before believing them.",
      );
    }
  }
}

main();
