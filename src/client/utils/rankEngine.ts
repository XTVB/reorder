// Ranking engine for the Rank Groups modal.
//
// Every group carries a Gaussian belief (mu, sigma) over a latent "liking"
// score. Two kinds of observations refine it:
//
//   - Ratings (tier 1-5 keypresses, plus optional Considering/Delete floor
//     slots in cull sessions) are treated as *noisy* absolute measurements:
//     a precision-weighted update toward the tier's anchor with noise wide
//     enough to overlap adjacent tiers. A confident rating (shift) uses a
//     much tighter noise, effectively pinning the group and dropping it out
//     of the comparison pool.
//   - Pairwise / best-of-N outcomes use TrueSkill-style updates, the precise
//     instrument for resolving order between nearby groups.
//
// The question policy picks the comparison with the highest expected
// information: match quality (how close to 50/50) weighted by how much
// posterior variance the pair still carries, discounted for repeats. Pairs
// whose posteriors barely overlap (a "1" vs a "5") never surface.

export type RankFloor = "considering" | "delete";

export interface RankEntry {
  mu: number;
  sigma: number;
  /** Last explicit rating slot (0-4 tiers, 5 considering, 6 delete). */
  tier?: number;
  /** Last rating was a confident (shift) rating. */
  confident?: boolean;
  /** Manually locked: excluded from all future questions. */
  pinned?: boolean;
  floor?: RankFloor;
}

export interface RankQuestion {
  /** 2 for a pair, 3-4 for a best-of-N screen. */
  ids: string[];
}

/** Optional decisiveness annotation on a win: "clear" = the pick was obvious,
 * "slim" = it could have gone either way. Absent = the plain win (and every
 * pre-margin log replays unchanged). */
export type CompareMargin = "slim" | "clear";

/** The answer recorded for one shown comparison. */
export type CompareOutcome =
  | { kind: "win"; winnerId: string; margin?: CompareMargin }
  /** Best-of-N answered with a joint top: `ids` (a subset of the screen) each
   * beat the rest, and are tied among themselves. */
  | { kind: "top"; ids: string[]; margin?: CompareMargin }
  | { kind: "tie" }
  | { kind: "skip" }
  | { kind: "pin"; id: string };

/**
 * One thing the user was shown, in presentation order: a tier rating or a
 * comparison (outcome null = shown but not yet answered). The modal's session
 * timeline is a list of these; persisted to the judgement log, the same list is
 * the raw data the calibration harness replays — beliefs are derived, these are
 * the observations. Keep it serializable.
 */
export type RankEvent =
  | { kind: "rate"; id: string; slot: number; confident: boolean }
  | { kind: "compare"; ids: string[]; outcome: CompareOutcome | null };

/** Apply one logged event to an engine — the single definition of what replay
 * means, shared by the modal and the harness so they can't drift. */
export function applyEvent(engine: RankEngine, ev: RankEvent): void {
  if (ev.kind === "rate") {
    engine.observeRating(ev.id, ev.slot, ev.confident);
    return;
  }
  const o = ev.outcome;
  if (!o) return;
  if (o.kind === "win") {
    engine.observeWin(
      o.winnerId,
      ev.ids.filter((id) => id !== o.winnerId),
      o.margin,
    );
  } else if (o.kind === "top") {
    engine.observeTop(o.ids, ev.ids, o.margin);
  } else if (o.kind === "tie") {
    engine.observeTie(ev.ids);
  } else if (o.kind === "skip") {
    engine.skip(ev.ids);
  } else if (o.kind === "pin") {
    engine.pin(o.id);
  }
}

export interface RankScoresFile {
  version: 1;
  updatedAt: string;
  entries: Record<string, RankEntry>;
  /** Times each pair has been shown, keyed by the two ids sorted and
   * NUL-joined (see pairKey). */
  pairs: Record<string, number>;
  /** Tie-propensity evidence (see outcomeProbs). Optional: files written
   * before these existed load as "no evidence yet" (prior only). */
  compareCount?: number;
  tieCount?: number;
}

/** Full in-memory engine state, captured before a comparison so it can be
 * restored to let the user re-answer (see snapshot/restore). */
export interface RankSnapshot {
  entries: Map<string, RankEntry>;
  pairCounts: Map<string, number>;
  questionsServed: number;
  compareCount: number;
  tieCount: number;
}

// ---- Tuning constants (one place) -----------------------------------------
// The latent scale is anchored to tier spacing = 1.7. Only the spacing/noise
// *ratio* means anything (scaling anchors and every sigma together is a no-op);
// 1.7 per tier at beta 0.5 is what the calibration corpus supports — pooled
// sweeps put the optimum at 1.7-2.0x the old 1.0 spacing, i.e. a tier keypress
// separates two photos more decisively than the noise dials assumed.

export const TIER_SLOTS = 5;
export const CONSIDERING_SLOT = 5;
export const DELETE_SLOT = 6;

/**
 * The generative model's parameters — everything that decides what an
 * observation *means*, as opposed to which question to ask next (those stay
 * module constants below; they don't enter the likelihood).
 *
 * Grouped into an object so the calibration harness can replay a real judgement
 * log under alternative settings and see which predicts the user best:
 * `bun run scripts/rank-calibration.ts --sweep`. Guessing at these numbers is
 * how you end up confidently wrong; measure them.
 */
export interface RankTuning {
  /** Anchor means for slots 0..6 (tier 1 best → tier 5 least, then floors). */
  anchors: number[];
  /** Prior for an item with no observations. */
  priorMu: number;
  priorSigma: number;
  /** Observation noise of a quick 1-5 keypress (overlaps adjacent tiers). */
  ratingSigma: number;
  /** Observation noise of a confident (shift) rating. */
  confidentSigma: number;
  /** Sigma assigned when an item is manually pinned during comparison. */
  pinnedSigma: number;
  /** TrueSkill performance noise for a single comparison. */
  beta: number;
  /** Inflated noise for best-of-N outcomes (they're correlated, not independent). */
  betaMulti: number;
  /** Draw margin on the latent scale, used by tie *updates* (how hard a tie
   * pulls the items together). Prediction doesn't use it — see outcomeProbs. */
  drawMargin: number;
  /** Performance-gap bounds (latent scale) behind the optional win margins: a
   * "slim" win means the gap fell inside (0, slimWinMargin); a "clear" win
   * means it exceeded clearWinMargin. Provisional guesses — no logged margins
   * existed to calibrate against when these shipped; sweep them once
   * margin-annotated judgements accumulate. */
  slimWinMargin: number;
  clearWinMargin: number;
  /** How a best-of-N tie updates beliefs — see observeTie. */
  tieMode: "all-pairs" | "chain";
  /** Beta prior over the user's tie propensity (see outcomeProbs). Mean
   * a/(a+b); a+b is the prior's weight in pseudo-comparisons. */
  tiePriorA: number;
  tiePriorB: number;
}

export const DEFAULT_TUNING: RankTuning = {
  anchors: [3.4, 1.7, 0, -1.7, -3.4, -5.4, -7.5],
  priorMu: 0,
  priorSigma: 2.5,
  ratingSigma: 0.9,
  confidentSigma: 0.3,
  pinnedSigma: 0.2,
  beta: 0.5,
  betaMulti: 0.7,
  drawMargin: 0.25,
  slimWinMargin: 0.5,
  clearWinMargin: 1.0,
  tieMode: "all-pairs",
  tiePriorA: 1,
  tiePriorB: 7,
};

/** Only pairs within this many rank positions are candidate questions. */
const CANDIDATE_WINDOW = 4;
/** Stop offering a pair after this many showings. */
const MAX_PAIR_REPEATS = 3;
/** Below this expected-information score no comparison is worth asking. */
const MIN_INFO = 0.12;
/** Serve a best-of-N (when one is available) every Nth question, starting at
 * every 12th and widening by 3 every 12 questions into the session. Measured on
 * the completed itscayyay log (multi-value.ts harness): a 4-panel delivers
 * ~one pair's worth of TRUE ordering information at ~2.5x the effort, and
 * fatigue hits multis hardest — 18% of them were full ties in the first
 * session quartile, 53% by the third. */
const MULTI_EVERY_START = 15;
const MULTI_FATIGUE_QUESTIONS = 15;
const MULTI_FATIGUE_STEP = 3;
/** Minimum match quality for a third/fourth group to join a best-of-N. */
const MULTI_JOIN_QUALITY = 0.3;

// ---- Gaussian helpers ------------------------------------------------------

function pdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/** Abramowitz-Stegun 7.1.26 erf approximation (|error| < 1.5e-7). */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly =
    t *
    (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return sign * (1 - poly * Math.exp(-ax * ax));
}

function normCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

/** Mean-shift factor v(t) = φ(t)/Φ(t) for a win observation, guarded. */
function vWin(t: number): number {
  const denom = normCdf(t);
  if (denom < 1e-10) return -t; // deep tail: v(t) → -t
  return pdf(t) / denom;
}

function wWin(t: number): number {
  const v = vWin(t);
  return Math.min(0.9999, Math.max(1e-6, v * (v + t)));
}

/** Mean-shift factor for a "slim" win — the performance gap fell inside
 * (0, eps): the winner did win, but barely. Same truncation family as vDraw
 * with the lower boundary at zero instead of -eps. */
function vSlim(t: number, eps: number): number {
  const denom = normCdf(eps - t) - normCdf(-t);
  if (denom < 1e-10) return (t > eps / 2 ? eps : 0) - t; // deep tail: nearest boundary
  return (pdf(t) - pdf(eps - t)) / denom;
}

function wSlim(t: number, eps: number): number {
  const denom = normCdf(eps - t) - normCdf(-t);
  if (denom < 1e-10) return 0.9999;
  const v = vSlim(t, eps);
  const w = v * v + ((eps - t) * pdf(eps - t) + t * pdf(t)) / denom;
  return Math.min(0.9999, Math.max(1e-6, w));
}

/** Mean-shift factor for a draw with margin eps (symmetric in the pair). */
function vDraw(t: number, eps: number): number {
  const denom = normCdf(eps - t) - normCdf(-eps - t);
  if (denom < 1e-10) return t < 0 ? eps : -eps;
  return (pdf(-eps - t) - pdf(eps - t)) / denom;
}

function wDraw(t: number, eps: number): number {
  const denom = normCdf(eps - t) - normCdf(-eps - t);
  if (denom < 1e-10) return 0.9999;
  const v = vDraw(t, eps);
  const w = v * v + ((eps - t) * pdf(eps - t) + (eps + t) * pdf(eps + t)) / denom;
  return Math.min(0.9999, Math.max(1e-6, w));
}

/** Pair keys join the two ids with NUL — it can't occur in a filename (image
 * ids) or a uuid (group ids), so ids containing spaces still round-trip. */
const PAIR_SEP = "\u0000";

/** Key for a pair's showing-count, order-independent. */
export function pairKey(a: string, b: string): string {
  return a < b ? `${a}${PAIR_SEP}${b}` : `${b}${PAIR_SEP}${a}`;
}

// Seeded RNG, so outcomeProbs' sampling is reproducible across replays.
function hashSeed(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller, one draw per call (the spare is cheap to discard here). */
function gauss(rand: () => number): number {
  const u = Math.max(rand(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/** Inverse of pairKey; null when the key isn't a well-formed pair. */
export function parsePairKey(key: string): [string, string] | null {
  const [a, b, ...rest] = key.split(PAIR_SEP);
  return a && b && rest.length === 0 ? [a, b] : null;
}

export function anchorForSlot(slot: number, tuning: RankTuning = DEFAULT_TUNING): number {
  const a = tuning.anchors;
  return a[Math.min(Math.max(slot, 0), a.length - 1)]!;
}

/** Nearest tier slot (0-4) for a posterior mean — used for Apply's tier tags. */
export function nearestTier(mu: number, tuning: RankTuning = DEFAULT_TUNING): number {
  let best = 0;
  let bestDist = Infinity;
  for (let s = 0; s < TIER_SLOTS; s++) {
    const d = Math.abs(mu - tuning.anchors[s]!);
    if (d < bestDist) {
      bestDist = d;
      best = s;
    }
  }
  return best;
}

// ---- Engine ----------------------------------------------------------------

export class RankEngine {
  entries = new Map<string, RankEntry>();
  pairCounts = new Map<string, number>();
  private questionsServed = 0;
  /** Answered comparisons (wins + ties) and how many were ties — the evidence
   * behind lambdaHat. Persisted per kind, so images and groups each learn
   * their own tie propensity. */
  compareCount = 0;
  tieCount = 0;
  readonly t: RankTuning;

  constructor(tuning: RankTuning = DEFAULT_TUNING) {
    this.t = tuning;
  }

  static fromJSON(data: RankScoresFile | null, tuning: RankTuning = DEFAULT_TUNING): RankEngine {
    const engine = new RankEngine(tuning);
    if (!data || data.version !== 1) return engine;
    for (const [id, e] of Object.entries(data.entries)) {
      if (typeof e?.mu === "number" && typeof e?.sigma === "number") {
        engine.entries.set(id, { ...e });
      }
    }
    for (const [key, count] of Object.entries(data.pairs ?? {})) {
      if (typeof count === "number") engine.pairCounts.set(key, count);
    }
    if (typeof data.compareCount === "number") engine.compareCount = data.compareCount;
    if (typeof data.tieCount === "number") engine.tieCount = data.tieCount;
    return engine;
  }

  /** Serialize, pruned to ids that still exist. */
  toJSON(validIds: Set<string>): RankScoresFile {
    const entries: Record<string, RankEntry> = {};
    for (const [id, e] of this.entries) {
      if (validIds.has(id)) entries[id] = { ...e };
    }
    const pairs: Record<string, number> = {};
    for (const [key, count] of this.pairCounts) {
      const pair = parsePairKey(key);
      if (pair && validIds.has(pair[0]) && validIds.has(pair[1])) pairs[key] = count;
    }
    return {
      version: 1,
      updatedAt: new Date().toISOString(),
      entries,
      pairs,
      compareCount: this.compareCount,
      tieCount: this.tieCount,
    };
  }

  /** Deep-copy the whole state so a comparison answer can be rolled back. */
  snapshot(): RankSnapshot {
    return {
      entries: new Map(Array.from(this.entries, ([k, v]) => [k, { ...v }])),
      pairCounts: new Map(this.pairCounts),
      questionsServed: this.questionsServed,
      compareCount: this.compareCount,
      tieCount: this.tieCount,
    };
  }

  /** Restore a prior snapshot (used by the modal's comparison undo). */
  restore(s: RankSnapshot): void {
    this.entries = new Map(Array.from(s.entries, ([k, v]) => [k, { ...v }]));
    this.pairCounts = new Map(s.pairCounts);
    this.questionsServed = s.questionsServed;
    this.compareCount = s.compareCount;
    this.tieCount = s.tieCount;
  }

  has(id: string): boolean {
    return this.entries.has(id);
  }

  /**
   * Forget everything about `ids` — their beliefs and every pair count touching
   * them — leaving the rest of the state intact. A Rank session covers a scope
   * (all groups, one selection, one group's contents) but the engine it loads
   * holds beliefs about everything in the file, so "Start over" has to erase the
   * scope, not the file.
   */
  clearScope(ids: Iterable<string>): void {
    const set = new Set(ids);
    for (const id of set) this.entries.delete(id);
    for (const key of [...this.pairCounts.keys()]) {
      const pair = parsePairKey(key);
      if (!pair || set.has(pair[0]) || set.has(pair[1])) this.pairCounts.delete(key);
    }
  }

  get(id: string): RankEntry | undefined {
    return this.entries.get(id);
  }

  private ensure(id: string): RankEntry {
    let e = this.entries.get(id);
    if (!e) {
      e = { mu: this.t.priorMu, sigma: this.t.priorSigma };
      this.entries.set(id, e);
    }
    return e;
  }

  /** Absolute rating observation for slot 0-6. Repeating the identical rating
   * is a no-op so re-confirming while navigating never stacks precision. */
  observeRating(id: string, slot: number, confident: boolean): void {
    const existing = this.entries.get(id);
    if (existing && existing.tier === slot && Boolean(existing.confident) === confident) return;
    const e = this.ensure(id);
    const anchor = anchorForSlot(slot, this.t);
    const noise = confident ? this.t.confidentSigma : this.t.ratingSigma;
    const prec = 1 / (e.sigma * e.sigma) + 1 / (noise * noise);
    e.mu = (e.mu / (e.sigma * e.sigma) + anchor / (noise * noise)) / prec;
    e.sigma = Math.sqrt(1 / prec);
    e.tier = slot;
    e.confident = confident;
    e.floor =
      slot === CONSIDERING_SLOT ? "considering" : slot === DELETE_SLOT ? "delete" : undefined;
  }

  /** Winner beats each loser. Pass one loser for a pair, 2-3 for best-of-N
   * (updates use an inflated beta since the outcomes are correlated). The
   * optional margin says how decisive the pick felt: "clear" conditions the
   * performance gap above clearWinMargin (separates the pair harder), "slim"
   * inside (0, slimWinMargin) — the order is asserted but the pair is pulled
   * together, which is what "better, but barely" means. */
  observeWin(winnerId: string, loserIds: string[], margin?: CompareMargin): void {
    const beta = loserIds.length > 1 ? this.t.betaMulti : this.t.beta;
    for (const loserId of loserIds) this.winPair(winnerId, loserId, beta, margin);
    this.questionsServed++;
    this.compareCount++;
  }

  /** One TrueSkill win update (winner over loser) with the optional margin. */
  private winPair(winnerId: string, loserId: string, beta: number, margin?: CompareMargin): void {
    const w = this.ensure(winnerId);
    const l = this.ensure(loserId);
    const c2 = 2 * beta * beta + w.sigma * w.sigma + l.sigma * l.sigma;
    const c = Math.sqrt(c2);
    const t = (w.mu - l.mu) / c;
    let v: number;
    let wf: number;
    if (margin === "clear") {
      // Gap > clearWinMargin: the standard win truncation, boundary shifted.
      const s = t - this.t.clearWinMargin / c;
      v = vWin(s);
      wf = wWin(s);
    } else if (margin === "slim") {
      const eps = this.t.slimWinMargin / c;
      v = vSlim(t, eps);
      wf = wSlim(t, eps);
    } else {
      v = vWin(t);
      wf = wWin(t);
    }
    w.mu += ((w.sigma * w.sigma) / c) * v;
    l.mu -= ((l.sigma * l.sigma) / c) * v;
    w.sigma = Math.sqrt(w.sigma * w.sigma * (1 - ((w.sigma * w.sigma) / c2) * wf));
    l.sigma = Math.sqrt(l.sigma * l.sigma * (1 - ((l.sigma * l.sigma) / c2) * wf));
    this.bumpPair(winnerId, loserId);
  }

  /**
   * Everything on the screen was indistinguishable: a draw on *every* pair,
   * with the inflated beta (BETA_MULTI) for a best-of-N.
   *
   * It looks like double counting — "all four are equal" spending 6 pairwise
   * updates, 3 of them landing on each image — but it isn't, and it's worth
   * saying why, because the obvious "fix" is much worse. A tie is not merely 6
   * redundant restatements of one fact; it pins each image to the *group's*
   * level, and a level pooled from 4 mutually-tied estimates is genuinely far
   * better determined than any one of them. The large sigma drop is real
   * information, not an artifact.
   *
   * Checked against the exact posterior of this very model (rejection-sampled:
   * latent + BETA performance noise, conditioned on every pairwise performance
   * gap falling inside DRAW_MARGIN), over spread/tight/mixed-sigma priors at
   * N=3 and N=4. All-pairs at BETA_MULTI tracks it closely (max |Δmu| ≈ 0.12,
   * |Δsigma| ≈ 0.04). Two tempting alternatives are both clearly worse:
   *   - the same all-pairs draws at plain BETA over-shrink sigma (|Δsigma| ≈
   *     0.12) — BETA_MULTI is exactly what corrects the naive sequential
   *     updates' over-counting, so it's calibrated, not arbitrary;
   *   - chaining draws between adjacent ranks only (TrueSkill's multi-team
   *     factorization) badly under-updates the means (|Δmu| up to 0.5): it
   *     never asserts that the best and worst of the tied set are close, which
   *     is precisely what the user just said.
   */
  observeTie(ids: string[]): void {
    if (ids.length < 2) return;
    const beta = ids.length > 2 ? this.t.betaMulti : this.t.beta;
    this.tieAmong(ids, beta);
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) this.bumpPair(ids[i]!, ids[j]!);
    }
    this.questionsServed++;
    this.compareCount++;
    this.tieCount++;
  }

  /** The draw updates for a mutually-tied set (no bookkeeping) — shared by
   * observeTie and observeTop's within-top tie. */
  private tieAmong(ids: string[], beta: number): void {
    if (this.t.tieMode === "chain") {
      // The rejected alternative, kept so the harness can re-test it on real
      // judgements rather than on my say-so: draws between adjacent ranks only.
      const ordered = [...ids].sort(
        (a, b) =>
          (this.entries.get(b)?.mu ?? this.t.priorMu) - (this.entries.get(a)?.mu ?? this.t.priorMu),
      );
      for (let i = 0; i + 1 < ordered.length; i++)
        this.drawPair(ordered[i]!, ordered[i + 1]!, beta);
    } else {
      for (let i = 0; i < ids.length; i++) {
        for (let j = i + 1; j < ids.length; j++) this.drawPair(ids[i]!, ids[j]!, beta);
      }
    }
  }

  /**
   * A best-of-N answered with a joint top — "these are the best, can't
   * separate them": every id in topIds beats every other id on the screen
   * (win updates at the inflated beta, margin optional), and the top ids draw
   * against each other. Degenerate shapes fold into the plain outcomes: a
   * single top id is just a win, the whole screen is just a tie. Counts as a
   * decisive answer for the tie-propensity record — lambdaHat prices "the user
   * calls the whole screen a tie", which this is not.
   */
  observeTop(topIds: string[], screenIds: string[], margin?: CompareMargin): void {
    const top = new Set(topIds);
    const rest = screenIds.filter((id) => !top.has(id));
    if (topIds.length === 0 || topIds.length + rest.length < 2) return;
    if (rest.length === 0) {
      this.observeTie(topIds);
      return;
    }
    if (topIds.length === 1) {
      this.observeWin(topIds[0]!, rest, margin);
      return;
    }
    for (const t of topIds) {
      for (const r of rest) this.winPair(t, r, this.t.betaMulti, margin);
    }
    this.tieAmong(topIds, this.t.betaMulti);
    for (let i = 0; i < topIds.length; i++) {
      for (let j = i + 1; j < topIds.length; j++) this.bumpPair(topIds[i]!, topIds[j]!);
    }
    this.questionsServed++;
    this.compareCount++;
  }

  /** Current belief about an id, without creating an entry for it. */
  private belief(id: string): RankEntry {
    return this.entries.get(id) ?? { mu: this.t.priorMu, sigma: this.t.priorSigma };
  }

  /** Posterior mean of the user's tie propensity: Beta prior + the observed
   * tie/win record. This is what prices a draw in outcomeProbs. */
  lambdaHat(): number {
    return (
      (this.tieCount + this.t.tiePriorA) / (this.compareCount + this.t.tiePriorA + this.t.tiePriorB)
    );
  }

  /**
   * The model's predictive distribution over a question's outcomes, stated
   * *before* it is answered: the probability each id wins outright, plus the
   * probability of a draw. Summing to 1.
   *
   * The draw is a hurdle, not a margin: with probability lambdaHat() the user
   * calls the screen a tie, otherwise they pick a winner per plain (margin-less)
   * TrueSkill. Calibration on real judgement logs forced this shape — the
   * empirical tie rate is roughly *flat* in the model's predicted gap (~25% for
   * images at every gap, ~4% for groups), where any "draw = |performance gap|
   * inside a window" geometry decays with the gap and priced those logs at
   * chance level or worse. Tying is a per-user, per-kind propensity, so it is
   * learned online (Beta prior tiePriorA/tiePriorB, counters persisted per
   * kind). The margin survives only in observeTie's belief update.
   *
   * This is the whole basis for checking the model against reality. The answer
   * the user gives next is a held-out label for the probability the engine just
   * committed to, so scoring these against a log of real judgements says whether
   * the modelling assumptions (the tie prior, beta, the tier anchors) describe
   * this user — something no amount of internal consistency can establish.
   * See scripts/rank-calibration.ts.
   */
  outcomeProbs(ids: string[]): { win: number[]; draw: number } {
    if (ids.length < 2) return { win: ids.map(() => 0), draw: 1 };
    const beta = ids.length > 2 ? this.t.betaMulti : this.t.beta;
    const lam = this.lambdaHat();

    if (ids.length === 2) {
      const a = this.belief(ids[0]!);
      const b = this.belief(ids[1]!);
      const c = Math.sqrt(2 * beta * beta + a.sigma * a.sigma + b.sigma * b.sigma);
      const winA = normCdf((a.mu - b.mu) / c);
      return { win: [(1 - lam) * winA, (1 - lam) * (1 - winA)], draw: lam };
    }

    // Best-of-N has no clean closed form: sample the performances instead. The
    // RNG is seeded from the ids so a replay is bit-for-bit reproducible.
    const SAMPLES = 20000;
    const spread = ids.map((id) => {
      const e = this.belief(id);
      return { mu: e.mu, sd: Math.sqrt(e.sigma * e.sigma + beta * beta) };
    });
    const rand = mulberry32(hashSeed(ids.join(" ")));
    const wins = new Array<number>(ids.length).fill(0);
    for (let s = 0; s < SAMPLES; s++) {
      let best = -Infinity;
      let bestI = 0;
      for (let i = 0; i < spread.length; i++) {
        const p = spread[i]!.mu + spread[i]!.sd * gauss(rand);
        if (p > best) {
          best = p;
          bestI = i;
        }
      }
      wins[bestI]!++;
    }
    return { win: wins.map((w) => ((1 - lam) * w) / SAMPLES), draw: lam };
  }

  /** One TrueSkill draw update: pulls the two means toward each other by less
   * the further apart they are, and shrinks both sigmas. */
  private drawPair(aId: string, bId: string, beta: number): void {
    const a = this.ensure(aId);
    const b = this.ensure(bId);
    const c2 = 2 * beta * beta + a.sigma * a.sigma + b.sigma * b.sigma;
    const c = Math.sqrt(c2);
    const t = (a.mu - b.mu) / c;
    const eps = this.t.drawMargin / c;
    const v = vDraw(t, eps);
    const wf = wDraw(t, eps);
    a.mu += ((a.sigma * a.sigma) / c) * v;
    b.mu -= ((b.sigma * b.sigma) / c) * v;
    a.sigma = Math.sqrt(a.sigma * a.sigma * (1 - ((a.sigma * a.sigma) / c2) * wf));
    b.sigma = Math.sqrt(b.sigma * b.sigma * (1 - ((b.sigma * b.sigma) / c2) * wf));
  }

  /** Record that a question was shown but not answered, so it isn't re-asked
   * immediately. */
  skip(ids: string[]): void {
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) this.bumpPair(ids[i]!, ids[j]!);
    }
    this.questionsServed++;
  }

  pin(id: string): void {
    const e = this.ensure(id);
    e.pinned = true;
    e.sigma = Math.min(e.sigma, this.t.pinnedSigma);
  }

  private bumpPair(a: string, b: string): void {
    const key = pairKey(a, b);
    this.pairCounts.set(key, (this.pairCounts.get(key) ?? 0) + 1);
  }

  /** Ids in scope that have any observation. */
  observedIds(scope: string[]): string[] {
    return scope.filter((id) => this.entries.has(id));
  }

  /** Final order: explicit floors sink below all normal groups (a categorical
   * statement), everything else by posterior mean, unobserved groups last in
   * their incoming relative order. */
  ranking(scope: string[]): string[] {
    const floorRank = (e: RankEntry | undefined): number => {
      if (!e) return 1; // unobserved: between normal groups and floors
      if (e.floor === "considering") return 2;
      if (e.floor === "delete") return 3;
      return 0;
    };
    return scope
      .map((id, idx) => ({ id, idx, e: this.entries.get(id) }))
      .sort((a, b) => {
        const fr = floorRank(a.e) - floorRank(b.e);
        if (fr !== 0) return fr;
        if (a.e && b.e && a.e.mu !== b.e.mu) return b.e.mu - a.e.mu;
        return a.idx - b.idx;
      })
      .map((x) => x.id);
  }

  /**
   * Pick the most informative next comparison, or null when nothing left is
   * worth asking. Candidates are near-rank pairs among unpinned, non-delete
   * items; score = match quality × remaining variance, discounted for repeats.
   * Every few questions the best pair is widened to a best-of-N when nearby
   * groups are also tangled with it.
   *
   * Unrated items participate at their prior belief — ratings are not a
   * prerequisite for comparing. Their maximal sigma makes them the
   * highest-information questions, so a compare-first session naturally
   * drains the unrated pool, and the update path (ensure) creates their
   * entries on first contact. This exists because the old observed-only gate
   * pushed users into mass "rank 1" passes just to unlock Compare mode, and
   * the calibration corpus showed those fake ratings actively hurt: they
   * tighten sigma with zero ordering information.
   */
  nextComparison(scope: string[]): RankQuestion | null {
    const pool = scope.filter((id) => {
      const e = this.entries.get(id);
      return !e || (!e.pinned && e.floor !== "delete");
    });
    if (pool.length < 2) return null;
    const sorted = pool.map((id) => ({ id, e: this.belief(id) })).sort((a, b) => b.e.mu - a.e.mu);

    let best: { i: number; j: number; info: number } | null = null;
    for (let i = 0; i < sorted.length; i++) {
      for (let j = i + 1; j < Math.min(sorted.length, i + 1 + CANDIDATE_WINDOW); j++) {
        const count = this.pairCounts.get(pairKey(sorted[i]!.id, sorted[j]!.id)) ?? 0;
        if (count >= MAX_PAIR_REPEATS) continue;
        const info = this.pairInfo(sorted[i]!.e, sorted[j]!.e) / (1 + 2 * count);
        if (info > (best?.info ?? MIN_INFO)) best = { i, j, info };
      }
    }
    if (!best) return null;

    const ids = [sorted[best.i]!.id, sorted[best.j]!.id];
    const multiEvery =
      MULTI_EVERY_START +
      MULTI_FATIGUE_STEP * Math.floor(this.questionsServed / MULTI_FATIGUE_QUESTIONS);
    if ((this.questionsServed + 1) % multiEvery === 0) {
      // Widen with rank-neighbours tangled with both current members.
      for (const k of [best.j + 1, best.i - 1, best.j + 2]) {
        if (ids.length >= 4) break;
        const cand = sorted[k];
        if (!cand || ids.includes(cand.id)) continue;
        const qualityOk = ids.every((id) => {
          if ((this.pairCounts.get(pairKey(id, cand.id)) ?? 0) >= MAX_PAIR_REPEATS) return false;
          return this.matchQuality(this.belief(id), cand.e) >= MULTI_JOIN_QUALITY;
        });
        if (qualityOk) ids.push(cand.id);
      }
    }
    return { ids };
  }

  private matchQuality(a: RankEntry, b: RankEntry): number {
    const c2 = 2 * this.t.beta * this.t.beta + a.sigma * a.sigma + b.sigma * b.sigma;
    const d = a.mu - b.mu;
    return Math.exp(-(d * d) / (2 * c2));
  }

  private pairInfo(a: RankEntry, b: RankEntry): number {
    return this.matchQuality(a, b) * (a.sigma * a.sigma + b.sigma * b.sigma);
  }

  /** Expected number of adjacent pairs in the current ranking whose true
   * order is uncertain — the "how settled is this" meter. */
  expectedUncertainPairs(scope: string[]): number {
    const pool = this.observedIds(scope).filter((id) => this.entries.get(id)!.floor !== "delete");
    const sorted = pool.map((id) => this.entries.get(id)!).sort((a, b) => b.mu - a.mu);
    let total = 0;
    for (let i = 0; i + 1 < sorted.length; i++) {
      const a = sorted[i]!;
      const b = sorted[i + 1]!;
      const spread = Math.sqrt(a.sigma * a.sigma + b.sigma * b.sigma);
      total += normCdf(-(a.mu - b.mu) / spread);
    }
    return total;
  }

  /**
   * Fraction in [0,1] of "how sorted" the whole scope is — the % the user
   * watches to decide when to stop. Walks every adjacent boundary in the full
   * ranking (all N groups, not just rated ones) and scores each:
   *   - unrated on either side → 0 (that boundary isn't sorted at all yet)
   *   - a categorical floor separation (delete vs not, considering vs not)
   *     → 1 (decisively placed)
   *   - two rated groups → 1 − 2·p(swap): a coin-flip ordering scores 0, a
   *     confident one scores ~1 — UNLESS the question policy will never revisit
   *     that pair (compared to the repeat cap, or its information value has
   *     fallen below MIN_INFO). Such a boundary is as sorted as it will ever
   *     get — an exhausted near-tie between two equally-liked groups is settled,
   *     not half-unsorted — so it scores a full 1. This keeps the meter in step
   *     with nextComparison: when no informative comparison remains it reads
   *     ~100%, rather than being permanently capped by unresolvable ties.
   * So it climbs both as you rate more groups (coverage) and as comparisons
   * pin down within-tier order (resolution).
   */
  sortedFraction(scope: string[]): number {
    const order = this.ranking(scope);
    if (order.length < 2) return 1;
    let resolved = 0;
    for (let i = 0; i + 1 < order.length; i++) {
      const aId = order[i]!;
      const bId = order[i + 1]!;
      const a = this.entries.get(aId);
      const b = this.entries.get(bId);
      if (!a || !b) continue; // unrated boundary contributes 0
      const floorA = a.floor ?? null;
      const floorB = b.floor ?? null;
      if (floorA !== floorB) {
        resolved += 1;
        continue;
      }
      const count = this.pairCounts.get(pairKey(aId, bId)) ?? 0;
      const settled =
        count >= MAX_PAIR_REPEATS || this.pairInfo(a, b) / (1 + 2 * count) <= MIN_INFO;
      if (settled) {
        resolved += 1;
        continue;
      }
      const spread = Math.sqrt(a.sigma * a.sigma + b.sigma * b.sigma);
      const pSwap = normCdf(-(a.mu - b.mu) / spread); // 0..0.5 (order is mu-desc)
      resolved += 1 - 2 * pSwap;
    }
    return resolved / (order.length - 1);
  }

  /**
   * Items whose comparison record disagrees with their rated tier: the
   * posterior mean has drifted into another tier's territory with enough
   * confidence (p = probability the score lies past the midpoint toward the
   * suggested tier) to be worth a glance. Ratings are the strong instrument in
   * this model — a corrected anchor moves an item globally where a comparison
   * only nudges it locally — so surfacing these for a one-keypress re-rate is
   * cheap for the user and dense in information. Worst-first.
   */
  misfits(
    scope: string[],
    minP = 0.75,
  ): { id: string; tier: number; suggested: number; p: number }[] {
    const out: { id: string; tier: number; suggested: number; p: number }[] = [];
    for (const id of scope) {
      const e = this.entries.get(id);
      if (!e || e.tier === undefined || e.tier >= TIER_SLOTS || e.floor) continue;
      const suggested = nearestTier(e.mu, this.t);
      if (suggested === e.tier) continue;
      const rated = anchorForSlot(e.tier, this.t);
      const toward = anchorForSlot(suggested, this.t);
      // mu is past the midpoint by construction; p asks how confidently.
      const boundary = (rated + toward) / 2;
      const p = normCdf(((e.mu - boundary) * Math.sign(toward - rated)) / Math.max(e.sigma, 1e-6));
      if (p >= minP) out.push({ id, tier: e.tier, suggested, p });
    }
    return out.sort((a, b) => b.p - a.p);
  }

  /**
   * Positional view of the posterior — the meter the user actually cares
   * about. Samples plausible "true" orders from the current beliefs (score_i ~
   * N(mu_i, sigma_i), seeded so repeated calls are stable) and measures how far
   * each item lands from its position in the current ranking.
   *
   * meanDisplacement is the headline: "each item is typically within ±N places
   * of where it would end up". Unlike sortedFraction — which scores every
   * adjacent boundary and therefore reads ~0 on a large scope until the whole
   * within-tier order is ground out — this falls smoothly from the first
   * comparison and directly expresses the "good enough" framing: a user who
   * doesn't care about #200 vs #201 stops when ±N reaches their tolerance.
   * Delete-floor items are excluded (their order is categorical, not ranked).
   *
   * tiers breaks the same displacement down by rated tier, because the user's
   * real deliverable is region-shaped ("right rank, right part of the rank"):
   * a global mean can hide one rank that is still fuzzy, and small tiers —
   * where finer order matters most — barely move the global number at all.
   */
  positionalSummary(
    scope: string[],
    samples = 48,
  ): { meanDisplacement: number; tiers: { tier: number; n: number; disp: number }[] } {
    const pool = this.ranking(scope).filter((id) => this.entries.get(id)?.floor !== "delete");
    const n = pool.length;
    if (n < 2) return { meanDisplacement: 0, tiers: [] };
    const beliefs = pool.map((id) => this.belief(id));
    const rand = mulberry32(0xc0ffee);
    const idx = new Array<number>(n);
    const scores = new Array<number>(n);
    const perItem = new Array<number>(n).fill(0);
    for (let s = 0; s < samples; s++) {
      for (let i = 0; i < n; i++) {
        scores[i] = beliefs[i]!.mu + beliefs[i]!.sigma * gauss(rand);
        idx[i] = i;
      }
      idx.sort((a, b) => scores[b]! - scores[a]!);
      // idx[pos] = item currently at rank idx[pos]; displacement = |pos - idx[pos]|
      for (let pos = 0; pos < n; pos++) perItem[idx[pos]!]! += Math.abs(pos - idx[pos]!);
    }
    let total = 0;
    const byTier = new Map<number, { n: number; disp: number }>();
    for (let i = 0; i < n; i++) {
      const d = perItem[i]! / samples;
      total += d;
      const tier = this.entries.get(pool[i]!)?.tier;
      if (tier === undefined || tier >= TIER_SLOTS) continue;
      const agg = byTier.get(tier) ?? { n: 0, disp: 0 };
      agg.n++;
      agg.disp += d;
      byTier.set(tier, agg);
    }
    const tiers = [...byTier.entries()]
      .map(([tier, a]) => ({ tier, n: a.n, disp: a.disp / a.n }))
      .sort((a, b) => a.tier - b.tier);
    return { meanDisplacement: total / n, tiers };
  }

  /** 1-based position of a group in the current ranking, for the subtitle. */
  estimatedRank(id: string, scope: string[]): number | null {
    if (!this.entries.has(id)) return null;
    return this.ranking(scope).indexOf(id) + 1;
  }
}
