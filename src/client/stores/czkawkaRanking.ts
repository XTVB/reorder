// Ranking engine + store for the czkawka duplicate-compare page.
//
// Two independent priority chains of conditions, evaluated top-down:
//   Content — which image's *bytes* to keep (drives Y and W)
//   Target  — which *filename* survives (drives W copy-replace)
// Each condition can carry a guard: a threshold gate that must pass for the
// condition to apply; a failed guard skips to the next condition.
//
// Conditions live in a Zustand store (seeded from / mirrored to localStorage)
// so the comparison view re-renders the Y/W buttons the moment a rule is
// edited in the sidebar. The chain evaluators are pure functions.

import { create } from "zustand";
import type { CzkawkaImage } from "../types.ts";

// ── Types ──────────────────────────────────────────────────────────────

export type GuardMode = "absolute" | "relative";
export type GuardAttribute = "long_side" | "width" | "height" | "size";
export type GuardOp = ">=" | "<=" | ">" | "<" | "=" | "!=";

export interface GuardConfig {
  enabled: boolean;
  mode: GuardMode;
  attribute: GuardAttribute;
  op: GuardOp;
  value: string;
}

export type ConditionType =
  | "path_prefix"
  | "path_regex"
  | "resolution"
  | "file_size"
  | "filename_order";

export interface RankingCondition {
  id: string;
  type: ConditionType;
  label: string;
  enabled: boolean;
  config: {
    pattern?: string;
    prefer?: "higher" | "lower" | "bigger" | "smaller";
  };
  guard: GuardConfig;
}

// ── Registry (single source for the sidebar UI) ────────────────────────

export const GUARD_ATTRS: [GuardAttribute, string][] = [
  ["long_side", "longest side (px)"],
  ["width", "width (px)"],
  ["height", "height (px)"],
  ["size", "size (MB)"],
];
export const GUARD_OPS: [GuardOp, string][] = [
  [">=", "≥"],
  ["<=", "≤"],
  [">", ">"],
  ["<", "<"],
  ["=", "="],
  ["!=", "≠"],
];
export const GUARD_MODES: [GuardMode, string][] = [
  ["absolute", "value of"],
  ["relative", "difference in"],
];
export const CONDITION_TYPES: [ConditionType, string][] = [
  ["path_prefix", "Path prefix"],
  ["path_regex", "Path regex"],
  ["resolution", "Resolution"],
  ["file_size", "File size"],
  ["filename_order", "Filename order"],
];

function defaultGuard(): GuardConfig {
  return { enabled: false, mode: "absolute", attribute: "long_side", op: ">=", value: "" };
}

function condition(
  id: string,
  type: ConditionType,
  enabled: boolean,
  config: RankingCondition["config"],
): RankingCondition {
  const label = CONDITION_TYPES.find(([t]) => t === type)?.[1] ?? type;
  return { id, type, label, enabled, config, guard: defaultGuard() };
}

const DEFAULT_CONTENT_CONDITIONS = (): RankingCondition[] => [
  condition("path_prefix", "path_prefix", false, { pattern: "" }),
  condition("path_regex", "path_regex", false, { pattern: "" }),
  condition("resolution", "resolution", true, { prefer: "higher" }),
  condition("file_size", "file_size", true, { prefer: "smaller" }),
  condition("filename_order", "filename_order", true, { prefer: "lower" }),
];

const DEFAULT_TARGET_CONDITIONS = (): RankingCondition[] => [
  condition("target_path_prefix", "path_prefix", false, { pattern: "" }),
  condition("target_path_regex", "path_regex", false, { pattern: "" }),
  condition("target_filename_order", "filename_order", true, { prefer: "lower" }),
];

let conditionSeq = 0;
export function makeCondition(type: ConditionType): RankingCondition {
  conditionSeq += 1;
  const config: RankingCondition["config"] =
    type === "path_prefix" || type === "path_regex"
      ? { pattern: "" }
      : type === "resolution"
        ? { prefer: "higher" }
        : type === "file_size"
          ? { prefer: "smaller" }
          : { prefer: "lower" };
  return condition(`${type}_${Date.now()}_${conditionSeq}`, type, true, config);
}

// ── Persistence ────────────────────────────────────────────────────────

const CONTENT_KEY = "czkawka_content_conditions";
const TARGET_KEY = "czkawka_target_conditions";

function loadConditions(key: string, fallback: () => RankingCondition[]): RankingCondition[] {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback();
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return fallback();
    // Backfill complete guards — older saves may predate guard fields.
    for (const c of parsed) {
      c.guard = { ...defaultGuard(), ...(c.guard ?? {}) };
    }
    return parsed as RankingCondition[];
  } catch {
    return fallback();
  }
}

function saveConditions(key: string, conditions: RankingCondition[]): void {
  try {
    localStorage.setItem(key, JSON.stringify(conditions));
  } catch {
    /* ignore */
  }
}

// ── Store ──────────────────────────────────────────────────────────────

interface CzkawkaRankingState {
  contentConditions: RankingCondition[];
  targetConditions: RankingCondition[];
  setContentConditions: (v: RankingCondition[]) => void;
  setTargetConditions: (v: RankingCondition[]) => void;
}

export const useCzkawkaRankingStore = create<CzkawkaRankingState>((set) => ({
  contentConditions: loadConditions(CONTENT_KEY, DEFAULT_CONTENT_CONDITIONS),
  targetConditions: loadConditions(TARGET_KEY, DEFAULT_TARGET_CONDITIONS),

  setContentConditions: (v) => {
    saveConditions(CONTENT_KEY, v);
    set({ contentConditions: v });
  },
  setTargetConditions: (v) => {
    saveConditions(TARGET_KEY, v);
    set({ targetConditions: v });
  },
}));

// ── Scoring (pure) ─────────────────────────────────────────────────────

function scoreImage(img: CzkawkaImage, cond: RankingCondition): number {
  // Prefix/regex match the full absolute path so rules can target whole
  // directories in multi-dir comparisons; filename_order uses the basename.
  const fullPath = `${img.dir}/${img.filename}`;
  switch (cond.type) {
    case "path_prefix":
      return cond.config.pattern && fullPath.startsWith(cond.config.pattern) ? 1 : 0;
    case "path_regex": {
      if (!cond.config.pattern) return 0;
      try {
        return new RegExp(cond.config.pattern).test(fullPath) ? 1 : 0;
      } catch {
        return 0;
      }
    }
    case "resolution": {
      const px = (img.width || 0) * (img.height || 0);
      return cond.config.prefer === "higher" ? px : -px;
    }
    case "file_size": {
      const sz = img.size ?? 0;
      return cond.config.prefer === "bigger" ? sz : -sz;
    }
    case "filename_order": {
      // Last numeric run in the stem, so "IMG_0042 (1).jpg" ranks by 1.
      const stem = img.filename.replace(/\.[^.]+$/, "");
      const matches = stem.match(/\d+/g);
      const num = matches ? Number.parseInt(matches[matches.length - 1]!, 10) : 0;
      return cond.config.prefer === "lower" ? -num : num;
    }
    default:
      return 0;
  }
}

function guardMetric(img: CzkawkaImage, attribute: GuardAttribute): number | null {
  switch (attribute) {
    case "long_side":
      return img.width && img.height ? Math.max(img.width, img.height) : null;
    case "width":
      return img.width || null;
    case "height":
      return img.height || null;
    case "size":
      return (img.size ?? 0) / 1_048_576;
    default:
      return null;
  }
}

function compareOp(actual: number, op: GuardOp, value: number): boolean {
  switch (op) {
    case ">=":
      return actual >= value;
    case "<=":
      return actual <= value;
    case ">":
      return actual > value;
    case "<":
      return actual < value;
    // Epsilon so MB-valued metrics survive float representation error.
    case "=":
      return Math.abs(actual - value) < 1e-9;
    case "!=":
      return Math.abs(actual - value) >= 1e-9;
    default:
      return true;
  }
}

/** absolute: every prospective winner clears the threshold itself.
 * relative: the spread (max−min) across the whole pool clears it — "≤ 0"
 * means all candidates must share the metric. Blank value → no guard. */
function passesGuard(guard: GuardConfig, winners: CzkawkaImage[], pool: CzkawkaImage[]): boolean {
  if (!guard.enabled) return true;
  const value = Number.parseFloat(guard.value);
  if (!Number.isFinite(value)) return true;

  if (guard.mode === "relative") {
    const metrics = pool.map((img) => guardMetric(img, guard.attribute));
    if (metrics.some((m) => m === null)) return false;
    const nums = metrics as number[];
    return compareOp(Math.max(...nums) - Math.min(...nums), guard.op, value);
  }

  return winners.every((img) => {
    const actual = guardMetric(img, guard.attribute);
    return actual !== null && compareOp(actual, guard.op, value);
  });
}

// ── Chain evaluation (pure) ────────────────────────────────────────────

export interface ChainResult {
  /** Indices into the candidates array passed in. */
  winners: number[];
  total: number;
}

/** Walk the priority chain, narrowing candidates at each enabled condition
 * (guard failures skip the condition). Returns null when no conditions are
 * enabled or fewer than 2 candidates exist. */
export function computeChain(
  conditions: RankingCondition[],
  candidates: CzkawkaImage[],
): ChainResult | null {
  const active = conditions.filter((c) => c.enabled);
  if (active.length === 0 || candidates.length < 2) return null;

  let remaining = candidates.map((img, index) => ({ img, index }));

  for (const cond of active) {
    const scored = remaining.map((c) => ({ ...c, score: scoreImage(c.img, cond) }));
    const maxScore = Math.max(...scored.map((s) => s.score));
    const next = scored.filter((s) => s.score === maxScore);

    if (
      cond.guard.enabled &&
      !passesGuard(
        cond.guard,
        next.map((s) => s.img),
        remaining.map((s) => s.img),
      )
    ) {
      continue;
    }

    remaining = next.map(({ img, index }) => ({ img, index }));
    if (remaining.length === 1) break;
  }

  return { winners: remaining.map((c) => c.index), total: candidates.length };
}
