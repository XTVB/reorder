// Review-mode grouping configurations.
//
// A ReviewConfig describes what categories (top-level buckets) and optional
// subcategories the ReviewModal exposes. The built-in `Keep / Maybe / Delete`
// config preserves the original behaviour; users can author additional configs
// that are persisted to localStorage and selectable via the modal's dropdown.

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

const CONFIGS_KEY = "reorder.reviewConfigs";
const LAST_ID_KEY = "reorder.lastReviewConfigId";

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

function loadCustomConfigs(): ReviewConfig[] {
  try {
    const raw = localStorage.getItem(CONFIGS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isValidConfig).filter((c) => c.id !== BUILTIN_CONFIG_ID);
  } catch {
    return [];
  }
}

function saveCustomConfigs(configs: ReviewConfig[]) {
  try {
    const filtered = configs.filter((c) => !c.builtIn && c.id !== BUILTIN_CONFIG_ID);
    localStorage.setItem(CONFIGS_KEY, JSON.stringify(filtered));
  } catch {
    // localStorage might be full / disabled; the user just loses persistence.
  }
}

export function loadAllConfigs(): ReviewConfig[] {
  return [BUILTIN_CONFIG, ...loadCustomConfigs()];
}

export function loadInitialConfigs(): { configs: ReviewConfig[]; lastId: string } {
  const configs = loadAllConfigs();
  return { configs, lastId: loadLastConfigId(configs) };
}

export function loadLastConfigId(configs: ReviewConfig[]): string {
  const stored = (() => {
    try {
      return localStorage.getItem(LAST_ID_KEY);
    } catch {
      return null;
    }
  })();
  if (stored && configs.some((c) => c.id === stored)) return stored;
  return BUILTIN_CONFIG_ID;
}

export function saveLastConfigId(id: string) {
  try {
    localStorage.setItem(LAST_ID_KEY, id);
  } catch {
    // ignore
  }
}

export function upsertConfig(configs: ReviewConfig[], next: ReviewConfig): ReviewConfig[] {
  const idx = configs.findIndex((c) => c.id === next.id);
  const out = configs.slice();
  if (idx === -1) out.push(next);
  else out[idx] = next;
  saveCustomConfigs(out);
  return out;
}

export function deleteConfigById(configs: ReviewConfig[], id: string): ReviewConfig[] {
  if (id === BUILTIN_CONFIG_ID) return configs;
  const out = configs.filter((c) => c.id !== id);
  saveCustomConfigs(out);
  return out;
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
