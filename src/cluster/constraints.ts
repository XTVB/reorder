// Cannot-link / group-lock constraints for clustering. Persisted in
// `.reorder-constraints.json` (camelCase shape, written atomically with a
// .bak backup). Resolved-into-Rust input files live inside `.reorder-cache/`
// and use camelCase too (matched by `#[serde(rename_all = "camelCase")]` in
// the Rust deserializer).

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { loadContentHashes } from "../cache-utils.ts";
import { writeJsonAtomic } from "../fs/atomic-json.ts";
import { loadGroups } from "../fs/groups.ts";
import {
  CANNOT_LINK_RESOLVED_FILE,
  cacheDir,
  constraintsPath,
  LOCKED_GROUPS_RESOLVED_FILE,
  REJECTED_MERGE_PAIRS_RESOLVED_FILE,
} from "../fs/paths.ts";
import { cachedHashMapping } from "./embeddings.ts";

export interface CannotLinkEntry {
  imageHash: string;
  groupId: string;
}

export interface RejectedMergePair {
  groupA: string;
  groupB: string;
}

export interface Constraints {
  version: 1;
  imageGroupCannotLink: CannotLinkEntry[];
  lockedGroupIds: string[];
  rejectedMergePairs: RejectedMergePair[];
}

const EMPTY_CONSTRAINTS: Constraints = {
  version: 1,
  imageGroupCannotLink: [],
  lockedGroupIds: [],
  rejectedMergePairs: [],
};

// Pairs are stored with groupA < groupB so each unordered pair has one
// canonical representation. Use this everywhere a pair is written or compared.
export function normalizeMergePair(a: string, b: string): RejectedMergePair {
  return a <= b ? { groupA: a, groupB: b } : { groupA: b, groupB: a };
}

export function mergePairKey(a: string, b: string): string {
  const p = normalizeMergePair(a, b);
  return `${p.groupA}\t${p.groupB}`;
}

// Single-writer (this server), so an in-memory copy is authoritative once
// loaded. mtime keying isn't safe — same-millisecond write/read can collide.
let _constraintsCache: { targetDir: string; value: Constraints } | null = null;

export function loadConstraints(targetDir: string): Constraints {
  if (_constraintsCache && _constraintsCache.targetDir === targetDir) {
    return _constraintsCache.value;
  }
  const path = constraintsPath(targetDir);
  let value: Constraints;
  if (!existsSync(path)) {
    value = { ...EMPTY_CONSTRAINTS };
  } else {
    try {
      const raw = JSON.parse(readFileSync(path, "utf-8")) as Partial<Constraints>;
      const rawRejected = Array.isArray(raw.rejectedMergePairs)
        ? raw.rejectedMergePairs.filter(
            (p) => p && typeof p.groupA === "string" && typeof p.groupB === "string",
          )
        : [];
      // Normalize and dedupe on load — a malformed file is silently repaired
      // next time anything writes.
      const seen = new Set<string>();
      const rejected: RejectedMergePair[] = [];
      for (const p of rawRejected) {
        const n = normalizeMergePair(p.groupA, p.groupB);
        if (n.groupA === n.groupB) continue;
        const k = `${n.groupA}\t${n.groupB}`;
        if (seen.has(k)) continue;
        seen.add(k);
        rejected.push(n);
      }
      value = {
        version: 1,
        imageGroupCannotLink: Array.isArray(raw.imageGroupCannotLink)
          ? raw.imageGroupCannotLink.filter(
              (c) => c && typeof c.imageHash === "string" && typeof c.groupId === "string",
            )
          : [],
        lockedGroupIds: Array.isArray(raw.lockedGroupIds)
          ? raw.lockedGroupIds.filter((id) => typeof id === "string")
          : [],
        rejectedMergePairs: rejected,
      };
    } catch {
      value = { ...EMPTY_CONSTRAINTS };
    }
  }
  _constraintsCache = { targetDir, value };
  return value;
}

export async function writeConstraintsFile(targetDir: string, c: Constraints): Promise<void> {
  await writeJsonAtomic(constraintsPath(targetDir), c, { backup: true });
  _constraintsCache = { targetDir, value: c };
}

/**
 * Apply a pure transform to the on-disk constraints. If the function returns
 * a different object reference, write it; otherwise leave the file alone.
 */
export async function mutateConstraints(
  targetDir: string,
  fn: (c: Constraints) => Constraints,
): Promise<{ changed: boolean; next: Constraints }> {
  const c = loadConstraints(targetDir);
  const next = fn(c);
  const changed = next !== c;
  if (changed) await writeConstraintsFile(targetDir, next);
  return { changed, next };
}

export interface ResolvedConstraintFiles {
  cannotLinkPath: string | null;
  lockedGroupsPath: string | null;
}

/**
 * Build the JSON payloads that the Rust cluster-tool consumes via --cannot-link
 * and --locked-groups, resolving content-hashes back to filenames and dropping
 * entries whose group or image is no longer available. Returns null for either
 * field when there's nothing to write.
 *
 * `scope` (optional) restricts both kinds of constraint to a filename subset:
 *   - cannot-link is dropped if the image isn't in the subset
 *   - group-lock is dropped if any member of the group is missing from the
 *     subset (the lock is meaningless when the group is partially represented)
 */
export async function writeResolvedConstraintFiles(
  targetDir: string,
  scope?: { allowedFilenames: Set<string> },
): Promise<ResolvedConstraintFiles> {
  const constraints = loadConstraints(targetDir);
  if (constraints.imageGroupCannotLink.length === 0 && constraints.lockedGroupIds.length === 0) {
    return { cannotLinkPath: null, lockedGroupsPath: null };
  }

  const groups = loadGroups(targetDir);
  const groupById = new Map(groups.map((g) => [g.id, g]));
  const cache = cacheDir(targetDir);
  // Pre-extraction state: content_hashes.json may not exist yet — fall back to
  // an empty map so constraint resolution drops every cannot-link entry.
  let hashToFilename: Map<string, string>;
  try {
    hashToFilename = cachedHashMapping(targetDir).hashToFilename;
  } catch {
    hashToFilename = new Map();
  }

  const cl: { imageFilename: string; groupId: string }[] = [];
  for (const c of constraints.imageGroupCannotLink) {
    const fname = hashToFilename.get(c.imageHash);
    if (!fname) continue;
    if (!groupById.has(c.groupId)) continue;
    if (scope && !scope.allowedFilenames.has(fname)) continue;
    cl.push({ imageFilename: fname, groupId: c.groupId });
  }

  const lockedExisting: string[] = [];
  for (const id of constraints.lockedGroupIds) {
    const g = groupById.get(id);
    if (!g) continue;
    if (scope) {
      const allMembersInScope = g.images.every((f) => scope.allowedFilenames.has(f));
      if (!allMembersInScope) continue;
    }
    lockedExisting.push(id);
  }

  const cannotLinkPath = cl.length > 0 ? join(cache, CANNOT_LINK_RESOLVED_FILE) : null;
  const lockedGroupsPath =
    lockedExisting.length > 0 ? join(cache, LOCKED_GROUPS_RESOLVED_FILE) : null;
  await Promise.all([
    cannotLinkPath ? writeJsonAtomic(cannotLinkPath, cl, { pretty: false }) : Promise.resolve(),
    lockedGroupsPath
      ? writeJsonAtomic(
          lockedGroupsPath,
          lockedExisting.map((id) => ({ groupId: id })),
          { pretty: false },
        )
      : Promise.resolve(),
  ]);

  return { cannotLinkPath, lockedGroupsPath };
}

/**
 * Write the JSON payload that the Rust group-similarity binary consumes via
 * --rejected-pairs. Returns the path written, or null when there's nothing
 * to skip. Rejected pairs whose groups no longer exist are dropped.
 */
export async function writeResolvedRejectedPairsFile(targetDir: string): Promise<string | null> {
  const constraints = loadConstraints(targetDir);
  if (constraints.rejectedMergePairs.length === 0) return null;

  const groups = loadGroups(targetDir);
  const groupIds = new Set(groups.map((g) => g.id));
  const live = constraints.rejectedMergePairs.filter(
    (p) => groupIds.has(p.groupA) && groupIds.has(p.groupB),
  );
  if (live.length === 0) return null;

  const out = join(cacheDir(targetDir), REJECTED_MERGE_PAIRS_RESOLVED_FILE);
  await writeJsonAtomic(out, live, { pretty: false });
  return out;
}

/**
 * Drop constraints whose imageHash has no current filename (file gone) or
 * whose groupId no longer exists. Called from remapAfterRename.
 */
export async function pruneDanglingConstraints(targetDir: string): Promise<void> {
  const c = loadConstraints(targetDir);
  if (
    c.imageGroupCannotLink.length === 0 &&
    c.lockedGroupIds.length === 0 &&
    c.rejectedMergePairs.length === 0
  ) {
    return;
  }

  const groups = loadGroups(targetDir);
  const groupIds = new Set(groups.map((g) => g.id));

  const contentHashes = loadContentHashes(cacheDir(targetDir));
  const hashKnown = Object.keys(contentHashes).length > 0;
  const validHashes = hashKnown ? new Set(Object.values(contentHashes)) : null;

  const filteredCL = c.imageGroupCannotLink.filter(
    (e) => groupIds.has(e.groupId) && (validHashes === null || validHashes.has(e.imageHash)),
  );
  const filteredLocked = c.lockedGroupIds.filter((id) => groupIds.has(id));
  const filteredRejected = c.rejectedMergePairs.filter(
    (p) => groupIds.has(p.groupA) && groupIds.has(p.groupB),
  );

  if (
    filteredCL.length === c.imageGroupCannotLink.length &&
    filteredLocked.length === c.lockedGroupIds.length &&
    filteredRejected.length === c.rejectedMergePairs.length
  ) {
    return;
  }
  await writeConstraintsFile(targetDir, {
    version: 1,
    imageGroupCannotLink: filteredCL,
    lockedGroupIds: filteredLocked,
    rejectedMergePairs: filteredRejected,
  });
}
