import type { ImageGroup, ImageTagData } from "../types.ts";

export interface GroupTagProfile {
  groupId: string;
  tags: Set<string>; // "category:value" tokens
}

export interface SimilarityPair {
  groupA: string;
  groupB: string;
  score: number;
  sharedTags: string[]; // "category:value" tokens they share
}

export function computeGroupTagProfiles(
  groups: ImageGroup[],
  tagData: Map<string, ImageTagData>,
): GroupTagProfile[] {
  return groups.map((g) => {
    const tags = new Set<string>();
    for (const fn of g.images) {
      const data = tagData.get(fn);
      if (!data) continue;
      for (const [cat, values] of Object.entries(data.tags)) {
        for (const v of values) tags.add(`${cat}:${v}`);
      }
      for (const item of data.clothing) {
        tags.add(`clothing_piece:${item.piece}`);
        for (const c of item.colors) tags.add(`clothing_color:${c}`);
        for (const s of item.styles) tags.add(`clothing_style:${s}`);
      }
    }
    return { groupId: g.id, tags };
  });
}

export function computeSimilarityMatrix(
  profiles: GroupTagProfile[],
  threshold = 0.15,
): SimilarityPair[] {
  const pairs: SimilarityPair[] = [];

  for (let i = 0; i < profiles.length; i++) {
    for (let j = i + 1; j < profiles.length; j++) {
      const a = profiles[i];
      const b = profiles[j];

      // Jaccard similarity
      const shared: string[] = [];
      const [small, big] = a.tags.size <= b.tags.size ? [a.tags, b.tags] : [b.tags, a.tags];
      for (const tag of small) {
        if (big.has(tag)) shared.push(tag);
      }

      const unionSize = a.tags.size + b.tags.size - shared.length;
      if (unionSize === 0) continue;

      const score = shared.length / unionSize;
      if (score >= threshold) {
        pairs.push({
          groupA: a.groupId,
          groupB: b.groupId,
          score,
          sharedTags: shared,
        });
      }
    }
  }

  pairs.sort((a, b) => b.score - a.score);
  return pairs;
}
