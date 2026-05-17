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

// Number-key shortcuts: 1..9 then 0 for slot 10. Slots beyond 10 have no key.
export function shortcutForSlot(slot: number): string | null {
  if (slot < 0) return null;
  if (slot < 9) return String(slot + 1);
  if (slot === 9) return "0";
  return null;
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
