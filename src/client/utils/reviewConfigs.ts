// Grouping configurations shared by the Review Groups and Create Groups modals.
//
// A ReviewConfig describes the categories (top-level buckets) and optional
// subcategories a GroupingSortModal exposes. `makeConfigStore` namespaces the
// localStorage persistence so each modal keeps an independent preset library
// with its own read-only built-in.

export interface ReviewSubcategory {
  id: string;
  label: string;
}

export interface ReviewCategory {
  id: string;
  label: string;
  subcategories: ReviewSubcategory[];
  defaultSubcategoryId?: string;
}

export interface ReviewConfig {
  id: string;
  name: string;
  categories: ReviewCategory[];
  defaultCategoryId: string;
  builtIn?: boolean;
}

import type { CSSProperties } from "react";

export const BUILTIN_CONFIG_ID = "__builtin_keep_maybe_delete";

// Sentinel <option> value for the "create a config" entry in the picker.
export const NEW_CONFIG_OPTION = "__new__";

export function reviewColorVar(color: string): CSSProperties {
  return { "--review-color": color } as CSSProperties;
}

const COMMON_SUBS: ReviewSubcategory[] = [
  { id: "top", label: "Top" },
  { id: "middle", label: "Middle" },
  { id: "bottom", label: "Bottom" },
];

export const BUILTIN_CONFIG: ReviewConfig = {
  id: BUILTIN_CONFIG_ID,
  name: "Keep / Maybe / Delete",
  builtIn: true,
  defaultCategoryId: "maybe",
  categories: [
    { id: "keep", label: "Keep", subcategories: COMMON_SUBS, defaultSubcategoryId: "middle" },
    { id: "maybe", label: "Maybe", subcategories: COMMON_SUBS, defaultSubcategoryId: "middle" },
    { id: "delete", label: "Delete", subcategories: COMMON_SUBS, defaultSubcategoryId: "middle" },
  ],
};

// Built-in starter for the Create Groups modal. Unlike the review built-in,
// these "categories" describe groups to be *created* from ungrouped photos;
// `defaultCategoryId` is unused by the create flow (unassigned photos are
// left ungrouped) but is required by the shared ReviewConfig shape/editor.
export const CREATE_GROUPS_BUILTIN_CONFIG_ID = "__builtin_create_groups_starter";

export const CREATE_GROUPS_BUILTIN_CONFIG: ReviewConfig = {
  id: CREATE_GROUPS_BUILTIN_CONFIG_ID,
  name: "Starter (Group 1–3)",
  builtIn: true,
  defaultCategoryId: "g1",
  categories: [
    { id: "g1", label: "Group 1", subcategories: [] },
    { id: "g2", label: "Group 2", subcategories: [] },
    { id: "g3", label: "Group 3", subcategories: [] },
  ],
};

// Hard-coded colours for the built-in vocabulary so the existing visual
// language is preserved. Custom configs fall through to the palette below.
const KNOWN_COLORS: Record<string, string> = {
  keep: "#22c55e",
  maybe: "#fbbf24",
  delete: "#ff5757",
  top: "#63b3ed",
  middle: "#9ca3af",
  bottom: "#b482ff",
};

const PALETTE = [
  "#22c55e",
  "#fbbf24",
  "#ff5757",
  "#63b3ed",
  "#b482ff",
  "#22d3ee",
  "#f97316",
  "#a78bfa",
  "#10b981",
  "#ec4899",
];

export function colorForId(id: string, index: number): string {
  return KNOWN_COLORS[id] ?? PALETTE[index % PALETTE.length]!;
}

// Number-key shortcuts. Two bands of ten:
//   slots 0..9  → 1..9 then 0
//   slots 10..19 → ⇧1..⇧9 then ⇧0
// Slots beyond 20 have no key.
const SLOT_DIGITS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"];

export function shortcutForSlot(slot: number): string | null {
  if (slot < 0 || slot >= 20) return null;
  const digit = SLOT_DIGITS[slot % 10]!;
  return slot < 10 ? digit : `⇧${digit}`;
}

// Maps a digit-row code ("Digit1".."Digit9","Digit0") to its 0-based position.
// Using `code` (not `key`) keeps this layout-robust and survives Shift turning
// "1" into "!".
const DIGIT_CODE_SLOT: Record<string, number> = {
  Digit1: 0,
  Digit2: 1,
  Digit3: 2,
  Digit4: 3,
  Digit5: 4,
  Digit6: 5,
  Digit7: 6,
  Digit8: 7,
  Digit9: 8,
  Digit0: 9,
};

/** Resolve a keydown to a slot index, or null if it isn't a slot shortcut. */
export function slotForKeyEvent(e: { code: string; shiftKey: boolean }): number | null {
  const base = DIGIT_CODE_SLOT[e.code];
  if (base === undefined) return null;
  return e.shiftKey ? base + 10 : base;
}

export function isValidConfig(c: unknown): c is ReviewConfig {
  if (!c || typeof c !== "object") return false;
  const cfg = c as ReviewConfig;
  if (typeof cfg.id !== "string" || !cfg.id) return false;
  if (typeof cfg.name !== "string") return false;
  if (!Array.isArray(cfg.categories) || cfg.categories.length === 0) return false;
  for (const cat of cfg.categories) {
    if (!cat || typeof cat.id !== "string" || !cat.id) return false;
    if (typeof cat.label !== "string") return false;
    if (!Array.isArray(cat.subcategories)) return false;
    for (const sub of cat.subcategories) {
      if (!sub || typeof sub.id !== "string" || !sub.id) return false;
      if (typeof sub.label !== "string") return false;
    }
  }
  if (typeof cfg.defaultCategoryId !== "string") return false;
  if (!cfg.categories.some((c) => c.id === cfg.defaultCategoryId)) return false;
  return true;
}

// A ConfigStore namespaces persistence (localStorage keys + a read-only
// built-in) so independent modals keep independent preset libraries while
// sharing the same ReviewConfig shape, editor, and pure helpers.
export interface ConfigStore {
  builtin: ReviewConfig;
  builtinId: string;
  loadAllConfigs: () => ReviewConfig[];
  loadInitialConfigs: () => { configs: ReviewConfig[]; lastId: string };
  loadLastConfigId: (configs: ReviewConfig[]) => string;
  saveLastConfigId: (id: string) => void;
  upsertConfig: (configs: ReviewConfig[], next: ReviewConfig) => ReviewConfig[];
  deleteConfigById: (configs: ReviewConfig[], id: string) => ReviewConfig[];
}

export function makeConfigStore(opts: {
  configsKey: string;
  lastIdKey: string;
  builtin: ReviewConfig;
}): ConfigStore {
  const { configsKey, lastIdKey, builtin } = opts;
  const builtinId = builtin.id;

  function loadCustomConfigs(): ReviewConfig[] {
    try {
      const raw = localStorage.getItem(configsKey);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(isValidConfig).filter((c) => c.id !== builtinId);
    } catch {
      return [];
    }
  }

  function saveCustomConfigs(configs: ReviewConfig[]) {
    try {
      const filtered = configs.filter((c) => !c.builtIn && c.id !== builtinId);
      localStorage.setItem(configsKey, JSON.stringify(filtered));
    } catch {
      // localStorage might be full / disabled; the user just loses persistence.
    }
  }

  function loadAllConfigs(): ReviewConfig[] {
    return [builtin, ...loadCustomConfigs()];
  }

  function loadLastConfigId(configs: ReviewConfig[]): string {
    const stored = (() => {
      try {
        return localStorage.getItem(lastIdKey);
      } catch {
        return null;
      }
    })();
    if (stored && configs.some((c) => c.id === stored)) return stored;
    return builtinId;
  }

  function loadInitialConfigs(): { configs: ReviewConfig[]; lastId: string } {
    const configs = loadAllConfigs();
    return { configs, lastId: loadLastConfigId(configs) };
  }

  function saveLastConfigId(id: string) {
    try {
      localStorage.setItem(lastIdKey, id);
    } catch {
      // ignore
    }
  }

  function upsertConfig(configs: ReviewConfig[], next: ReviewConfig): ReviewConfig[] {
    const idx = configs.findIndex((c) => c.id === next.id);
    const out = configs.slice();
    if (idx === -1) out.push(next);
    else out[idx] = next;
    saveCustomConfigs(out);
    return out;
  }

  function deleteConfigById(configs: ReviewConfig[], id: string): ReviewConfig[] {
    if (id === builtinId) return configs;
    const out = configs.filter((c) => c.id !== id);
    saveCustomConfigs(out);
    return out;
  }

  return {
    builtin,
    builtinId,
    loadAllConfigs,
    loadInitialConfigs,
    loadLastConfigId,
    saveLastConfigId,
    upsertConfig,
    deleteConfigById,
  };
}

// Review Groups modal: original storage keys preserved for back-compat.
export const reviewConfigStore = makeConfigStore({
  configsKey: "reorder.reviewConfigs",
  lastIdKey: "reorder.lastReviewConfigId",
  builtin: BUILTIN_CONFIG,
});

// Create Groups modal: independent preset library (separate keys).
export const createGroupsConfigStore = makeConfigStore({
  configsKey: "reorder.createGroupsConfigs",
  lastIdKey: "reorder.lastCreateGroupsConfigId",
  builtin: CREATE_GROUPS_BUILTIN_CONFIG,
});

// Tags to persist on a group from its *explicit* bucket assignment in a
// grouping sort. Untouched groups (no explicit category/sub — even if Apply
// sorts them into the default category) yield no tags, so a partial
// categorisation can be resumed by finding the still-untagged groups.
//
//   category only       → ["Keep"]
//   category + sub       → ["Keep", "Keep - Top"]
//   nothing assigned     → []
//
// A group may carry a sub assignment without an explicit category status (it
// was refined while sitting in a default-category bucket); the sub's own
// categoryId supplies the category in that case.
export function explicitGroupTags(
  config: ReviewConfig,
  statuses: Map<string, string>,
  subs: Map<string, { categoryId: string; subId: string }>,
  groupId: string,
): string[] {
  const sub = subs.get(groupId);
  const categoryId = statuses.get(groupId) ?? sub?.categoryId;
  if (!categoryId) return [];
  const category = config.categories.find((c) => c.id === categoryId);
  if (!category) return [];
  const tags = [category.label];
  if (sub && sub.categoryId === category.id) {
    const subcategory = category.subcategories.find((s) => s.id === sub.subId);
    if (subcategory) tags.push(`${category.label} - ${subcategory.label}`);
  }
  return tags;
}

// Every tag string a config can emit: each category label, plus "Category -
// Sub" for each subcategory. On Apply a config replaces *its own* slice of a
// group's tags (recomputed from the current assignment) while leaving tags
// owned by other configs intact — so changing a previously-matching tag
// replaces it, but categorisations from different schemes still accumulate.
export function configOwnedTags(config: ReviewConfig): Set<string> {
  const out = new Set<string>();
  for (const c of config.categories) {
    out.add(c.label);
    for (const s of c.subcategories) out.add(`${c.label} - ${s.label}`);
  }
  return out;
}

// Reverse of explicitGroupTags: recover the (category, optional sub) a group
// was bucketed into from its persisted tags, matched against `config` by
// label. Returns the first category whose label appears in the tags, or null
// if none do. Used to pre-seed a grouping sort from existing tags.
export function assignmentFromTags(
  config: ReviewConfig,
  tags: string[] | undefined,
): { categoryId: string; sub?: { categoryId: string; subId: string } } | null {
  if (!tags || tags.length === 0) return null;
  const tagSet = new Set(tags);
  for (const cat of config.categories) {
    if (!tagSet.has(cat.label)) continue;
    let sub: { categoryId: string; subId: string } | undefined;
    for (const s of cat.subcategories) {
      if (tagSet.has(`${cat.label} - ${s.label}`)) {
        sub = { categoryId: cat.id, subId: s.id };
        break;
      }
    }
    return { categoryId: cat.id, sub };
  }
  return null;
}

export function makeId(prefix: string): string {
  return `${prefix}_${crypto.randomUUID()}`;
}

// Deep-copies categories/subcategories so edits never mutate shared state.
export function cloneConfig(src: ReviewConfig): ReviewConfig {
  return {
    ...src,
    categories: src.categories.map((c) => ({
      ...c,
      subcategories: c.subcategories.map((s) => ({ ...s })),
    })),
  };
}

export function emptyConfigDraft(): ReviewConfig {
  const catId = makeId("cat");
  return {
    id: makeId("cfg"),
    name: "New grouping",
    defaultCategoryId: catId,
    categories: [{ id: catId, label: "Category 1", subcategories: [] }],
  };
}

export function duplicateConfig(src: ReviewConfig): ReviewConfig {
  return { ...cloneConfig(src), id: makeId("cfg"), name: `${src.name} (copy)`, builtIn: false };
}
