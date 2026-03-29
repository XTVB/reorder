import type { ActiveFilter, ImageTagData } from "../types.ts";

export const CATEGORY_ORDER = [
  "colors", "clothing_type", "clothing_style", "clothing_piece", "clothing_color",
  "setting", "location", "lighting", "pose", "framing", "hair",
  "accessories", "mood", "theme", "photo_type", "background", "notable",
];

export function categoryOrderIndex(cat: string): number {
  const i = CATEGORY_ORDER.indexOf(cat);
  return i === -1 ? 99 : i;
}

/** category → value → Set<filename> */
export type InvertedIndex = Map<string, Map<string, Set<string>>>;

export function buildInvertedIndex(images: ImageTagData[]): InvertedIndex {
  const index: InvertedIndex = new Map();

  function add(category: string, value: string, filename: string) {
    let catMap = index.get(category);
    if (!catMap) { catMap = new Map(); index.set(category, catMap); }
    let fileSet = catMap.get(value);
    if (!fileSet) { fileSet = new Set(); catMap.set(value, fileSet); }
    fileSet.add(filename);
  }

  for (const img of images) {
    // Flat tag fields
    for (const [cat, values] of Object.entries(img.tags)) {
      for (const v of values) add(cat, v, img.filename);
    }
    // Denormalize clothing items into searchable flat categories
    for (const item of img.clothing) {
      add("clothing_piece", item.piece, img.filename);
      for (const c of item.colors) add("clothing_color", c, img.filename);
      for (const s of item.styles) add("clothing_style", s, img.filename);
    }
  }

  return index;
}

function intersect(a: Set<string>, b: Set<string>): Set<string> {
  const [small, big] = a.size <= b.size ? [a, b] : [b, a];
  const out = new Set<string>();
  for (const x of small) { if (big.has(x)) out.add(x); }
  return out;
}

/**
 * Apply active filters against the inverted index.
 * Returns filenames from allFilenames that match, preserving order.
 */
export function applyFilters(
  allFilenames: string[],
  index: InvertedIndex,
  filters: ActiveFilter[],
  groupedFilenames: Set<string> | null,
  scope: "all" | "ungrouped",
): string[] {
  if (filters.length === 0 && scope === "all") return allFilenames;
  if (filters.length === 0) {
    return allFilenames.filter((fn) => !groupedFilenames?.has(fn));
  }

  const andFilters = filters.filter((f) => f.mode === "AND" && !f.category.startsWith("__"));
  const orFilters = filters.filter((f) => f.mode === "OR" && !f.category.startsWith("__"));
  const notFilters = filters.filter((f) => f.mode === "NOT" && !f.category.startsWith("__"));

  let result: Set<string> | null = null; // null = universe

  // AND: intersect each filter's matching set
  for (const f of andFilters) {
    const matches = index.get(f.category)?.get(f.value) ?? new Set<string>();
    result = result === null ? new Set(matches) : intersect(result, matches);
  }

  // OR: union of all OR values, then intersect with result
  if (orFilters.length > 0) {
    const orUnion = new Set<string>();
    for (const f of orFilters) {
      const matches = index.get(f.category)?.get(f.value);
      if (matches) for (const fn of matches) orUnion.add(fn);
    }
    result = result === null ? orUnion : intersect(result, orUnion);
  }

  if (result === null) result = new Set(allFilenames);

  // NOT: subtract
  for (const f of notFilters) {
    const matches = index.get(f.category)?.get(f.value);
    if (matches) for (const fn of matches) result.delete(fn);
  }

  // Scope filter
  if (scope === "ungrouped" && groupedFilenames) {
    for (const fn of groupedFilenames) result.delete(fn);
  }

  // Preserve file order
  return allFilenames.filter((fn) => result!.has(fn));
}

/**
 * Compute tag value counts within a result set for the tag browser panel.
 * Returns category → value → count.
 */
export function computeTagCounts(
  index: InvertedIndex,
  resultSet: Set<string>,
): Map<string, Map<string, number>> {
  const counts = new Map<string, Map<string, number>>();

  for (const [category, valueMap] of index) {
    const catCounts = new Map<string, number>();
    for (const [value, filenames] of valueMap) {
      let count = 0;
      // Iterate the smaller set
      if (filenames.size <= resultSet.size) {
        for (const fn of filenames) { if (resultSet.has(fn)) count++; }
      } else {
        for (const fn of resultSet) { if (filenames.has(fn)) count++; }
      }
      if (count > 0) catCounts.set(value, count);
    }
    if (catCounts.size > 0) counts.set(category, catCounts);
  }

  return counts;
}
