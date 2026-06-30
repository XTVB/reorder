// Naming-rule presets for the Naming Rules modal. A rule is a template string
// composed of literal text and `<token>` placeholders that resolve against a
// group's generated metadata (name/title/subtitle/short_sub). A rule *set*
// bundles several templates so they can be picked per-group via number keys.
//
// Persistence mirrors reviewConfigs' makeConfigStore: a localStorage-backed
// preset library with a read-only built-in seeded from the examples in the
// feature request.

import type { ImageGroup } from "../types.ts";

export interface NamingRule {
  id: string;
  template: string;
}

export interface NamingRuleSet {
  id: string;
  name: string;
  rules: NamingRule[];
  builtIn?: boolean;
}

// Tokens a template may reference. Each maps onto an ImageGroup field; missing
// fields resolve to an empty string. Kept in display order for the editor.
export const NAMING_TOKENS = ["name", "title", "subtitle", "short_sub"] as const;
export type NamingToken = (typeof NAMING_TOKENS)[number];

const TOKEN_RE = /<(name|title|subtitle|short_sub)>/g;

/** Substitute `<token>` placeholders against a group, then tidy whitespace. */
export function renderTemplate(template: string, group: ImageGroup): string {
  const out = template.replace(TOKEN_RE, (_, token: NamingToken) => {
    const value = group[token];
    return typeof value === "string" ? value : "";
  });
  // Collapse runs of internal whitespace introduced by an empty token and trim
  // dangling separators left when a leading/trailing token resolved to nothing.
  return out
    .replace(/\s+/g, " ")
    .replace(/^[\s:\-–—|/,]+|[\s:\-–—|/,]+$/g, "")
    .trim();
}

export const NAMING_BUILTIN_ID = "__builtin_naming_rules";

export const NAMING_BUILTIN: NamingRuleSet = {
  id: NAMING_BUILTIN_ID,
  name: "Starter",
  builtIn: true,
  rules: [
    { id: "starter_1", template: "<subtitle> : <title>" },
    { id: "starter_2", template: "<short_sub> - <name>" },
  ],
};

const CONFIGS_KEY = "reorder.namingRuleSets";
const LAST_ID_KEY = "reorder.lastNamingRuleSetId";

export function makeRuleId(): string {
  return `rule_${crypto.randomUUID()}`;
}

export function makeRuleSetId(): string {
  return `nrs_${crypto.randomUUID()}`;
}

function isValidRuleSet(c: unknown): c is NamingRuleSet {
  if (!c || typeof c !== "object") return false;
  const rs = c as NamingRuleSet;
  if (typeof rs.id !== "string" || !rs.id) return false;
  if (typeof rs.name !== "string") return false;
  if (!Array.isArray(rs.rules)) return false;
  return rs.rules.every(
    (r) => r && typeof r.id === "string" && r.id !== "" && typeof r.template === "string",
  );
}

function loadCustom(): NamingRuleSet[] {
  try {
    const raw = localStorage.getItem(CONFIGS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isValidRuleSet).filter((c) => c.id !== NAMING_BUILTIN_ID);
  } catch {
    return [];
  }
}

function saveCustom(sets: NamingRuleSet[]) {
  try {
    const filtered = sets.filter((c) => !c.builtIn && c.id !== NAMING_BUILTIN_ID);
    localStorage.setItem(CONFIGS_KEY, JSON.stringify(filtered));
  } catch {
    // localStorage might be full / disabled; the user just loses persistence.
  }
}

export function loadAllRuleSets(): NamingRuleSet[] {
  return [NAMING_BUILTIN, ...loadCustom()];
}

export function loadLastRuleSetId(sets: NamingRuleSet[]): string {
  let stored: string | null = null;
  try {
    stored = localStorage.getItem(LAST_ID_KEY);
  } catch {
    stored = null;
  }
  if (stored && sets.some((c) => c.id === stored)) return stored;
  return NAMING_BUILTIN_ID;
}

export function saveLastRuleSetId(id: string) {
  try {
    localStorage.setItem(LAST_ID_KEY, id);
  } catch {
    // ignore
  }
}

export function upsertRuleSet(sets: NamingRuleSet[], next: NamingRuleSet): NamingRuleSet[] {
  const idx = sets.findIndex((c) => c.id === next.id);
  const out = sets.slice();
  if (idx === -1) out.push(next);
  else out[idx] = next;
  saveCustom(out);
  return out;
}

export function deleteRuleSetById(sets: NamingRuleSet[], id: string): NamingRuleSet[] {
  if (id === NAMING_BUILTIN_ID) return sets;
  const out = sets.filter((c) => c.id !== id);
  saveCustom(out);
  return out;
}

export function emptyRuleSetDraft(): NamingRuleSet {
  return {
    id: makeRuleSetId(),
    name: "New rule set",
    rules: [{ id: makeRuleId(), template: "<title>" }],
  };
}

/** Editable copy of a (possibly built-in) set, with fresh ids and a new name. */
export function duplicateRuleSet(src: NamingRuleSet): NamingRuleSet {
  return {
    id: makeRuleSetId(),
    name: `${src.name} (copy)`,
    rules: src.rules.map((r) => ({ id: makeRuleId(), template: r.template })),
  };
}

/** Deep clone preserving ids — for editing a custom set in place. */
export function cloneRuleSet(src: NamingRuleSet): NamingRuleSet {
  return { ...src, rules: src.rules.map((r) => ({ ...r })) };
}
