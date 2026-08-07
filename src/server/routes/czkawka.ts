// /api/czkawka/* — duplicate-image discovery and resolution.
//
// `run` is an SSE compute job (shares the cluster job mutex): content-hash
// every image across the configured directories, perceptually hash unique
// contents via rust/hash-tool (original + flop/mirror, so mirrored duplicates
// match), then group by Hamming distance. Hashes are cached per directory and
// per (alg, filter, size) config, keyed by content hash, so renames never
// invalidate the cache and exact duplicates hash once.
//
// Multiple directories can participate (the launch target dir always does).
// One dir may be the czkawka-style *reference*: comparison then becomes
// "which images match something in the reference dir" — reference images are
// only compared against non-reference images (no intra-reference and no
// intra-working-dir groups). Reference files are ordinary group members and
// can be trashed/overwritten like any other.
//
// `action` applies batched trash / copy-replace operations under the rename
// lock, journalling every mutation to .reorder-cache/czkawka_session.json so
// `undo` can rename files back out of ~/.Trash even after a server restart.

import { copyFile, mkdir, readdir, stat, unlink } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { HASH_TOOL_BINARY, spawnJSON } from "../../cluster/index.ts";
import { computeContentHash, pruneContentHashes } from "../../fs/content-hashes.ts";
import {
  buildGroupsPatch,
  CZKAWKA_HASH_CACHE_PREFIX,
  type CzkawkaSessionData,
  cacheDir,
  czkawkaHashCachePath,
  groupsBeforeEntry,
  imageKey,
  isImageFile,
  listImages,
  listImagesRecursive,
  listSubdirectories,
  loadCzkawkaSession,
  readJsonTolerant,
  restoreFromTrash,
  saveCzkawkaSession,
  trashFilesRestorable,
  withRenameLock,
  writeJsonAtomic,
} from "../../fs/index.ts";
import { log } from "../../log.ts";
import type {
  CzkawkaDirEntry,
  CzkawkaGroup,
  CzkawkaImage,
  CzkawkaOperation,
  CzkawkaRunResult,
  CzkawkaStateResponse,
} from "../../shared/types.ts";
import { cleanupAfterDelete, invalidateDerivedCaches } from "../cleanup.ts";
import { runClusterJobSSE } from "../middleware/cluster-job.ts";
import { json, serveFileWithCache } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

// ── Run config ──────────────────────────────────────────────────────────

const HASH_ALGS = new Set([
  "DoubleGradient",
  "Gradient",
  "Mean",
  "VertGradient",
  "Blockhash",
  "Median",
]);
const IMAGE_FILTERS = new Set(["Lanczos3", "Nearest", "Triangle", "Gaussian", "CatmullRom"]);
const HASH_SIZES = new Set([8, 16, 32, 64]);

interface RunConfig {
  hashAlg: string;
  imageFilter: string;
  hashSize: number;
  similarity: number;
  /** Normalized: absolute, deduped, targetDir included, ≤1 reference. */
  dirs: CzkawkaDirEntry[];
}

function expandHome(p: string): string {
  return p.startsWith("~/") || p === "~" ? join(homedir(), p.slice(1)) : p;
}

/** Dedupe + resolve the requested dirs. The launch dir is no longer forced in
 * — two unrelated dirs can be compared — but an empty request falls back to
 * it so a fresh session still has something to scan. */
function normalizeDirs(targetDir: string, raw: unknown): CzkawkaDirEntry[] | string {
  const list: CzkawkaDirEntry[] = [];
  if (Array.isArray(raw)) {
    for (const item of raw as {
      path?: unknown;
      reference?: unknown;
      recursive?: unknown;
    }[]) {
      if (!item || typeof item.path !== "string" || item.path.trim() === "") continue;
      const path = resolve(expandHome(item.path.trim()));
      const existing = list.find((d) => d.path === path);
      if (existing) {
        existing.reference = existing.reference || item.reference === true;
        existing.recursive = existing.recursive || item.recursive === true;
      } else {
        list.push({ path, reference: item.reference === true, recursive: item.recursive === true });
      }
    }
  }
  if (list.length === 0) {
    list.push({ path: resolve(targetDir), reference: false, recursive: false });
  }
  if (list.filter((d) => d.reference).length > 1) {
    return "At most one directory can be marked as the reference";
  }
  return list;
}

function parseRunConfig(targetDir: string, body: Record<string, unknown>): RunConfig | string {
  const hashAlg = (body.hashAlg as string) ?? "DoubleGradient";
  const imageFilter = (body.imageFilter as string) ?? "Lanczos3";
  const hashSize = (body.hashSize as number) ?? 16;
  const similarity = (body.similarity as number) ?? 15;
  if (!HASH_ALGS.has(hashAlg)) return `Unknown hash algorithm: ${hashAlg}`;
  if (!IMAGE_FILTERS.has(imageFilter)) return `Unknown resize filter: ${imageFilter}`;
  if (!HASH_SIZES.has(hashSize)) return `Unsupported hash size: ${hashSize}`;
  if (!Number.isFinite(similarity) || similarity < 0) return `Invalid similarity: ${similarity}`;
  const dirs = normalizeDirs(targetDir, body.dirs);
  if (typeof dirs === "string") return dirs;
  return { hashAlg, imageFilter, hashSize, similarity, dirs };
}

// ── Perceptual-hash cache (per dir + config, keyed by content hash) ─────

interface CachedHashEntry {
  /** base64 image_hasher bytes for the original / horizontal mirror (flop). */
  hash: string;
  flopHash: string;
  width: number;
  height: number;
}

type HashCache = Record<string, CachedHashEntry>;

interface HashToolResult {
  results: {
    key: string;
    hash: string;
    flopHash: string;
    width: number;
    height: number;
  }[];
  errors: { key: string; path: string; error: string }[];
}

// ── Hamming comparison + grouping ──────────────────────────────────────

function popcount32(v: number): number {
  let x = v - ((v >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return (x * 0x01010101) >>> 24;
}

/** Hashes are compared word-wise (they're stored as Uint32Array); the compare
 * phase is O(n²) pairs × 5 orientations, so this inner loop matters. */
function hamming(a: Uint32Array, b: Uint32Array): number {
  let dist = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dist += popcount32(a[i]! ^ b[i]!);
  }
  return dist;
}

interface FileRef {
  /** Directory the file actually lives in (a sub-dir when a root recurses). */
  dir: string;
  filename: string;
  /** The configured root dir this file was scanned under — used for ref/rank
   * membership and for keying the per-root hash cache. */
  root: string;
}

interface UniqueEntry {
  /** Every file sharing this content hash, across all dirs. */
  files: FileRef[];
  size: number;
  width: number;
  height: number;
  hash: Uint32Array;
  flop: Uint32Array;
}

/** Orientation-aware distance: original plus the horizontal mirror (flop) —
 * mirrored duplicates happen in practice, upside-down ones don't, so no
 * vertical-flip hash. Comparing the flop in both directions guards against
 * resize-rounding asymmetries; flop-vs-flop is redundant (mirroring both
 * images preserves Hamming distance). */
function orientedDistance(a: UniqueEntry, b: UniqueEntry): number {
  const d = hamming(a.hash, b.hash);
  return Math.min(d, hamming(a.hash, b.flop), hamming(a.flop, b.hash));
}

interface GroupMember {
  entry: UniqueEntry;
  difference: number;
}

function expandGroup(
  members: GroupMember[],
  dirRank: (root: string) => number,
  collator: Intl.Collator,
): CzkawkaImage[] {
  const rows = members.flatMap(({ entry, difference }) =>
    entry.files.map((f) => ({
      root: f.root,
      img: {
        dir: f.dir,
        filename: f.filename,
        size: entry.size,
        width: entry.width,
        height: entry.height,
        difference,
      } satisfies CzkawkaImage,
    })),
  );
  // Reference images lead, then the launch dir, then extras.
  rows.sort(
    (a, b) =>
      dirRank(a.root) - dirRank(b.root) ||
      collator.compare(join(a.img.dir, a.img.filename), join(b.img.dir, b.img.filename)),
  );
  return rows.map((r) => r.img);
}

/** Grouping over unique contents, then expansion to files. A content hash
 * carrying 2+ files forms a group even when no other content is similar —
 * that's the exact-duplicate case.
 *
 * Czkawka-style best-anchor assignment: collect every in-tolerance pair,
 * walk them tightest-first, and attach each entry to its most-similar anchor
 * ("parent"), re-parenting when a closer anchor turns up. Groups stay
 * star-shaped — every member is within maxDistance of its group's anchor, so
 * no transitive chaining — but unlike the old file-order greedy the outcome
 * doesn't depend on filename order, and a tight pair can never be broken up
 * by a looser neighbor that happened to sort first. An entry can still go
 * unreported when its only in-range neighbor is bound tighter to a different
 * anchor it can't reach — inherent to disjoint star groups (czkawka too).
 *
 * With a reference dir the question becomes "which images match something in
 * the reference dir": only reference→non-reference edges exist (czkawka does
 * the same — reference hashes query a tree of working-dir hashes), so anchors
 * are reference entries and each working-dir entry joins its closest
 * reference. Intra-reference and intra-working-dir duplicates are not
 * reported. (An entry whose identical content exists on both sides is a
 * ref↔non-ref exact match by itself.) */
function buildGroups(
  entries: UniqueEntry[],
  maxDistance: number,
  refDir: string | null,
  dirRank: (root: string) => number,
): CzkawkaGroup[] {
  const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
  const n = entries.length;
  const isRef = refDir === null ? null : entries.map((e) => e.files.some((f) => f.root === refDir));

  // Candidate edges: a = anchor candidate, b = member candidate. Entries are
  // pre-sorted (reference dir first, then launch dir, then name), so ties
  // deterministically prefer the earlier entry as anchor.
  const edges: { a: number; b: number; d: number }[] = [];
  for (let i = 0; i < n; i++) {
    if (isRef !== null && !isRef[i]) continue;
    for (let j = isRef !== null ? 0 : i + 1; j < n; j++) {
      if (isRef !== null && (j === i || isRef[j])) continue;
      const d = orientedDistance(entries[i]!, entries[j]!);
      if (d <= maxDistance) edges.push({ a: i, b: j, d });
    }
  }
  edges.sort((x, y) => x.d - y.d || x.a - y.a || x.b - y.b);

  // Best-anchor assignment, tightest edges first.
  const parent = new Int32Array(n).fill(-1);
  const parentDist = new Int32Array(n);
  const childCount = new Uint32Array(n);
  const detach = (c: number) => {
    childCount[parent[c]!]!--;
    parent[c] = -1;
  };
  /** Try to attach b as a member of a's group. */
  const attach = (a: number, b: number, d: number): boolean => {
    if (childCount[b]! > 0) return false; // b anchors its own group
    if (parent[b] !== -1 && parentDist[b]! <= d) return false; // b bound tighter elsewhere
    if (parent[a] !== -1) {
      if (parentDist[a]! <= d) return false; // a bound tighter elsewhere
      detach(a); // a's tightest relation is this edge — promote it to anchor
    }
    if (parent[b] !== -1) detach(b);
    parent[b] = a;
    parentDist[b] = d;
    childCount[a]!++;
    return true;
  };
  for (const { a, b, d } of edges) {
    // Without a reference dir roles are symmetric — try the reverse too.
    if (!attach(a, b, d) && refDir === null) attach(b, a, d);
  }

  // Emit groups in entry order: anchors with their members, plus unattached
  // exact-duplicate contents (2+ files sharing a content hash; in reference
  // mode only when those files span the reference boundary).
  const childrenOf = new Map<number, GroupMember[]>();
  for (let j = 0; j < n; j++) {
    const p = parent[j]!;
    if (p === -1) continue;
    const list = childrenOf.get(p) ?? [];
    list.push({ entry: entries[j]!, difference: parentDist[j]! });
    childrenOf.set(p, list);
  }
  const groups: CzkawkaGroup[] = [];
  for (let i = 0; i < n; i++) {
    const e = entries[i]!;
    if (childCount[i]! > 0) {
      const members = [{ entry: e, difference: 0 }, ...childrenOf.get(i)!];
      groups.push({ images: expandGroup(members, dirRank, collator) });
    } else if (parent[i] === -1 && e.files.length >= 2) {
      const spansBoundary =
        refDir === null ||
        (e.files.some((f) => f.root === refDir) && e.files.some((f) => f.root !== refDir));
      if (spansBoundary) {
        groups.push({ images: expandGroup([{ entry: e, difference: 0 }], dirRank, collator) });
      }
    }
  }
  return groups;
}

// ── The run job (inside the SSE stream + cluster job mutex) ────────────

async function runComparison(
  targetDir: string,
  config: RunConfig,
  onProgress: (line: string) => void,
): Promise<CzkawkaRunResult> {
  const startTime = performance.now();
  const refDir = config.dirs.find((d) => d.reference)?.path ?? null;
  const dirRank = (dir: string) => (dir === refDir ? 0 : dir === targetDir ? 1 : 2);

  for (const d of config.dirs) {
    const s = await stat(d.path).catch(() => null);
    if (!s?.isDirectory()) throw new Error(`Not a directory: ${d.path}`);
  }

  // Phase 1 — content-hash every image in every dir, under the rename lock
  // so we never fingerprint the target dir mid-rename. Fast: one 16KB read
  // per file.
  onProgress("Scanning directories...");
  const byContent = new Map<string, { files: FileRef[]; size: number }>();
  await withRenameLock(async () => {
    const allFiles: FileRef[] = [];
    for (const d of config.dirs) {
      if (d.recursive) {
        for (const r of await listImagesRecursive(d.path)) {
          allFiles.push({ dir: r.dir, filename: r.filename, root: d.path });
        }
      } else {
        for (const filename of await listImages(d.path)) {
          allFiles.push({ dir: d.path, filename, root: d.path });
        }
      }
    }
    const BATCH = 128;
    for (let i = 0; i < allFiles.length; i += BATCH) {
      const batch = allFiles.slice(i, i + BATCH);
      await Promise.all(
        batch.map(async (f) => {
          try {
            const { hash, size } = await computeContentHash(join(f.dir, f.filename));
            const entry = byContent.get(hash);
            if (entry) entry.files.push(f);
            else byContent.set(hash, { files: [f], size });
          } catch {
            // unreadable file — skip
          }
        }),
      );
      onProgress(
        `Fingerprinting files ${Math.min(i + BATCH, allFiles.length)}/${allFiles.length}...`,
      );
    }
  });

  // Phase 2 — perceptual-hash contents missing from the per-dir caches.
  const cachePathFor = (dir: string) =>
    czkawkaHashCachePath(dir, config.hashAlg, config.imageFilter, config.hashSize);
  const cache: HashCache = {};
  for (const d of config.dirs) {
    Object.assign(cache, await readJsonTolerant<HashCache>(cachePathFor(d.path), {}));
  }

  const missing: { key: string; path: string }[] = [];
  for (const [ch, { files }] of byContent) {
    if (!cache[ch]) missing.push({ key: ch, path: join(files[0]!.dir, files[0]!.filename) });
  }

  let failed = 0;
  if (missing.length > 0) {
    onProgress(`Hashing ${missing.length} new image(s)...`);
    const jobsPath = join(cacheDir(targetDir), "czkawka_hash_jobs.json");
    await mkdir(cacheDir(targetDir), { recursive: true });
    await writeJsonAtomic(
      jobsPath,
      {
        hashAlg: config.hashAlg,
        imageFilter: config.imageFilter,
        hashSize: config.hashSize,
        images: missing,
      },
      { pretty: false },
    );
    const toolOut = await spawnJSON<HashToolResult>([HASH_TOOL_BINARY, "--jobs", jobsPath], {
      label: "czkawka-hash",
      onProgress,
    });
    await unlink(jobsPath).catch(() => {});
    for (const r of toolOut.results) {
      cache[r.key] = {
        hash: r.hash,
        flopHash: r.flopHash,
        width: r.width,
        height: r.height,
      };
    }
    failed = toolOut.errors.length;
    for (const e of toolOut.errors.slice(0, 5)) {
      log("czkawka", `Hash failed for ${e.path}: ${e.error}`);
    }
  }

  // Persist per-dir caches: each dir stores hashes for the contents it holds,
  // pruned to its current files. Failures (e.g. read-only dir) just mean a
  // recompute next run.
  for (const d of config.dirs) {
    const out: HashCache = {};
    for (const [ch, { files }] of byContent) {
      if (cache[ch] && files.some((f) => f.root === d.path)) out[ch] = cache[ch];
    }
    try {
      await mkdir(cacheDir(d.path), { recursive: true });
      await writeJsonAtomic(cachePathFor(d.path), out, { pretty: false, atomic: true });
      // Caches from older pipeline versions are dead weight — prune them.
      for (const f of await readdir(cacheDir(d.path))) {
        if (f.startsWith("czkawka_hashes_") && !f.startsWith(CZKAWKA_HASH_CACHE_PREFIX)) {
          await unlink(join(cacheDir(d.path), f)).catch(() => {});
        }
      }
    } catch {
      log("czkawka", `Could not write hash cache in ${d.path} (read-only?)`);
    }
  }

  // Phase 3 — compare. Reference-dir entries sort first so group seeds tend
  // to be reference images and `difference` reads as distance-to-reference.
  onProgress("Comparing hashes...");
  const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
  // Copy into a fresh (aligned, zero-padded to 4 bytes) buffer so a u32 view
  // is always valid — Buffer.from may return an unaligned pool slice.
  const decode = (b64: string): Uint32Array => {
    const raw = Buffer.from(b64, "base64");
    const padded = new Uint8Array(Math.ceil(raw.length / 4) * 4);
    padded.set(raw);
    return new Uint32Array(padded.buffer);
  };
  const entries: UniqueEntry[] = [];
  for (const [ch, { files, size }] of byContent) {
    const cached = cache[ch];
    if (!cached) continue;
    const sorted = [...files].sort(
      (a, b) =>
        dirRank(a.root) - dirRank(b.root) ||
        collator.compare(join(a.dir, a.filename), join(b.dir, b.filename)),
    );
    entries.push({
      files: sorted,
      size,
      width: cached.width,
      height: cached.height,
      hash: decode(cached.hash),
      flop: decode(cached.flopHash),
    });
  }
  entries.sort((a, b) => {
    const ra = Math.min(...a.files.map((f) => dirRank(f.root)));
    const rb = Math.min(...b.files.map((f) => dirRank(f.root)));
    if (ra !== rb) return ra - rb;
    return collator.compare(
      join(a.files[0]!.dir, a.files[0]!.filename),
      join(b.files[0]!.dir, b.files[0]!.filename),
    );
  });

  const groups = buildGroups(entries, config.similarity, refDir, dirRank);
  const computeTimeMs = Math.round(performance.now() - startTime);

  const session: CzkawkaSessionData = {
    groups,
    computeTimeMs,
    undoStack: [],
    dirs: config.dirs,
  };
  sessions.set(targetDir, session);
  await saveCzkawkaSession(targetDir, session);

  log(
    "czkawka",
    `Found ${groups.length} duplicate group(s) across ${config.dirs.length} dir(s) in ` +
      `${computeTimeMs}ms (${entries.length - missing.length} cached, ${missing.length} hashed, ` +
      `${failed} failed)`,
  );

  return {
    groups,
    computeTimeMs,
    cached: entries.length - missing.length,
    computed: missing.length,
    failed,
  };
}

// ── Session state (in-memory mirror of the on-disk journal) ─────────────

const sessions = new Map<string, CzkawkaSessionData>();

async function getSession(targetDir: string): Promise<CzkawkaSessionData> {
  let s = sessions.get(targetDir);
  if (!s) {
    s = await loadCzkawkaSession(targetDir);
    sessions.set(targetDir, s);
  }
  return s;
}

function stateResponse(targetDir: string, session: CzkawkaSessionData): CzkawkaStateResponse {
  return {
    groups: session.groups,
    undoDepth: session.undoStack.length,
    targetDir,
    dirs: session.dirs,
  };
}

/** True when `path` names an image inside one of the session dirs — directly
 * for a plain dir, or anywhere in the tree for a recursive one. */
function isPathInDirs(path: string, dirs: CzkawkaDirEntry[]): boolean {
  if (!isImageFile(basename(path))) return false;
  const dir = dirname(path);
  return dirs.some((d) =>
    d.recursive ? dir === d.path || dir.startsWith(`${d.path}/`) : dir === d.path,
  );
}

// ── Actions ─────────────────────────────────────────────────────────────

function validateOperations(ops: CzkawkaOperation[], session: CzkawkaSessionData): string | null {
  const allPaths = new Set(session.groups.flatMap((g) => g.images.map(imageKey)));
  const checkPath = (p: string): string | null =>
    allPaths.has(p) ? null : `"${p}" is not in the current groups`;

  for (const op of ops) {
    if (op.type === "trash") {
      if (op.paths.length === 0) return "trash operation with no paths";
      if (op.keep && op.paths.includes(op.keep)) return "keep cannot also be trashed";
      for (const p of [...op.paths, ...(op.keep ? [op.keep] : [])]) {
        const err = checkPath(p);
        if (err) return err;
      }
    } else if (op.type === "copy_replace") {
      if (op.others.includes(op.source) || op.others.includes(op.target)) {
        return "source/target cannot also be in others";
      }
      for (const p of [op.source, op.target, ...op.others]) {
        const err = checkPath(p);
        if (err) return err;
      }
    } else {
      return `Unknown operation type: ${(op as { type: string }).type}`;
    }
  }
  return null;
}

async function applyOperations(targetDir: string, ops: CzkawkaOperation[]): Promise<Response> {
  const session = await getSession(targetDir);
  const err = validateOperations(ops, session);
  if (err) return json({ error: err }, 400);

  // Safe to alias: the rebuild below replaces the array rather than mutating it.
  const groupsBefore = session.groups;
  const trashed: Awaited<ReturnType<typeof trashFilesRestorable>> = [];
  const copiedTargets: string[] = [];
  /** Paths physically gone (or content-replaced) → removed from groups. */
  const removedPaths = new Set<string>();
  /** Paths whose whole group is resolved (kept winners). */
  const resolvedPaths = new Set<string>();

  for (const op of ops) {
    if (op.type === "trash") {
      trashed.push(...(await trashFilesRestorable(op.paths)));
      for (const p of op.paths) removedPaths.add(p);
      if (op.keep) resolvedPaths.add(op.keep);
    } else {
      const { source, target, others } = op;
      if (source !== target) {
        // Reference semantics: the target's original bytes go to the Trash
        // BEFORE the copy lands, so undo can restore them.
        trashed.push(...(await trashFilesRestorable([target])));
        await copyFile(source, target);
        copiedTargets.push(target);
        trashed.push(...(await trashFilesRestorable([source])));
        removedPaths.add(source);
      }
      trashed.push(...(await trashFilesRestorable(others)));
      for (const p of others) removedPaths.add(p);
      resolvedPaths.add(target);
    }
  }

  session.groups = session.groups
    .filter((g) => !g.images.some((i) => resolvedPaths.has(imageKey(i))))
    .map((g) => ({ images: g.images.filter((i) => !removedPaths.has(imageKey(i))) }))
    .filter((g) => g.images.length >= 2);

  session.undoStack.push({
    patch: buildGroupsPatch(groupsBefore, session.groups),
    trashed,
    copiedTargets,
    deletedCount: trashed.length,
  });
  await saveCzkawkaSession(targetDir, session);

  // Target-dir files that were trashed get their content hashes pruned and the
  // position-indexed cluster artifacts dropped; a copy-replace target still
  // exists but with new bytes, so only its (now stale) content-hash entry goes.
  // Files in other dirs don't touch this project's caches. A copy-replace
  // trashes the target's ORIGINAL bytes (for undo) so it appears in `trashed`,
  // but the filename is still live on disk — exclude those paths or the file
  // gets yanked from its caches even though it never left.
  const copiedSet = new Set(copiedTargets);
  const trashedLocal = trashed
    .filter((t) => dirname(t.path) === targetDir && !copiedSet.has(t.path))
    .map((t) => basename(t.path));
  const warnings = await cleanupAfterDelete(targetDir, trashedLocal);
  const copiedLocal = copiedTargets.filter((p) => dirname(p) === targetDir).map((p) => basename(p));
  if (copiedLocal.length > 0) {
    await pruneContentHashes(targetDir, new Set(copiedLocal));
  }

  return json({
    ...stateResponse(targetDir, session),
    deletedCount: trashed.length,
    warnings,
  });
}

async function undoLast(targetDir: string): Promise<Response> {
  const session = await getSession(targetDir);
  // `sessions` hands every caller the same mutable object, so pop before the
  // first await: two undos admitted together must never observe the same top
  // of stack and restore it twice.
  const last = session.undoStack.pop();
  if (!last) return json({ error: "Nothing to undo" }, 400);

  // Reconstruct the pre-action groups before touching disk, so a malformed
  // patch fails with the entry still on the stack and nothing moved.
  let restoredGroups: CzkawkaGroup[];
  try {
    restoredGroups = groupsBeforeEntry(session.groups, last);
  } catch (err) {
    session.undoStack.push(last);
    throw err;
  }

  // Order matters for copy-replace: delete the copy sitting at the target
  // first, then restore in reverse execution order so the source comes back
  // before the target's original bytes land at the vacated path.
  for (const p of last.copiedTargets) {
    await unlink(p).catch(() => {});
  }
  await restoreFromTrash([...last.trashed].reverse());
  // No compensating push past this point: the disk is already reverted, and
  // re-running the entry would unlink files that were just restored.
  session.groups = restoredGroups;
  await saveCzkawkaSession(targetDir, session);

  // Files reappeared — position-indexed cluster artifacts are stale, but the
  // restored files must NOT be pruned from anything, so skip the delete path.
  const touchedDisk = last.trashed.length > 0 || last.copiedTargets.length > 0;
  const warnings = touchedDisk ? await invalidateDerivedCaches(targetDir) : [];

  return json({ ...stateResponse(targetDir, session), warnings });
}

// ── Route handler ───────────────────────────────────────────────────────

export const czkawkaRoutes: RouteHandler = async (req, ctx) => {
  const { path } = ctx;
  // Canonical absolute target dir — dir configs and session keys compare
  // paths as strings, so a relative launch path must never leak in here.
  const targetDir = resolve(ctx.targetDir);

  // POST /api/czkawka/run — SSE compute job (409 if a cluster job is running)
  if (path === "/api/czkawka/run" && req.method === "POST") {
    const body = (await req.json()) as Record<string, unknown>;
    const config = parseRunConfig(targetDir, body);
    if (typeof config === "string") return json({ error: config }, 400);
    return runClusterJobSSE(
      (_send, onProgress) => runComparison(targetDir, config, onProgress),
      "Another compute job is already running",
    );
  }

  // GET /api/czkawka/groups — current session state (survives restarts)
  if (path === "/api/czkawka/groups" && req.method === "GET") {
    return json(stateResponse(targetDir, await getSession(targetDir)));
  }

  // GET /api/czkawka/file?path=… — serve an image from any configured dir
  if (path === "/api/czkawka/file" && req.method === "GET") {
    const p = new URL(req.url).searchParams.get("path") ?? "";
    const session = await getSession(targetDir);
    if (!p || !isPathInDirs(p, session.dirs)) {
      return json({ error: "Path is not in a configured directory" }, 403);
    }
    return serveFileWithCache(req, p, "private, max-age=300");
  }

  // POST /api/czkawka/check-dir — validate a directory before adding it.
  // Relative paths resolve against the launch dir.
  if (path === "/api/czkawka/check-dir" && req.method === "POST") {
    const body = (await req.json()) as { path?: string; recursive?: boolean };
    const raw = (body.path ?? "").trim();
    if (!raw) return json({ error: "Path is required" }, 400);
    const resolved = resolve(targetDir, expandHome(raw));
    const s = await stat(resolved).catch(() => null);
    if (!s?.isDirectory()) return json({ error: `Not a directory: ${resolved}` }, 400);
    try {
      const imageCount = body.recursive
        ? (await listImagesRecursive(resolved)).length
        : (await listImages(resolved)).length;
      return json({ ok: true, path: resolved, imageCount });
    } catch {
      return json({ error: `Cannot read directory: ${resolved}` }, 400);
    }
  }

  // GET /api/czkawka/browse?path=… — list sub-directories for the picker.
  // A relative or ~-path resolves against the launch dir; blank means the
  // launch dir itself.
  if (path === "/api/czkawka/browse" && req.method === "GET") {
    const raw = (new URL(req.url).searchParams.get("path") ?? "").trim();
    const base = raw ? resolve(targetDir, expandHome(raw)) : targetDir;
    const s = await stat(base).catch(() => null);
    if (!s?.isDirectory()) return json({ error: `Not a directory: ${base}` }, 400);
    try {
      const dirs = (await listSubdirectories(base)).map((name) => ({
        name,
        path: join(base, name),
      }));
      const imageCount = (await listImages(base)).length;
      const parent = dirname(base);
      return json({ path: base, parent: parent === base ? null : parent, imageCount, dirs });
    } catch {
      return json({ error: `Cannot read directory: ${base}` }, 400);
    }
  }

  // POST /api/czkawka/action — batched trash / copy-replace operations
  if (path === "/api/czkawka/action" && req.method === "POST") {
    const body = (await req.json()) as { operations?: CzkawkaOperation[] };
    const ops = Array.isArray(body.operations) ? body.operations : [];
    if (ops.length === 0) return json({ error: "operations must be a non-empty array" }, 400);
    return withRenameLock(() => applyOperations(targetDir, ops));
  }

  // POST /api/czkawka/undo — restore the last action's files + group state
  if (path === "/api/czkawka/undo" && req.method === "POST") {
    return withRenameLock(() => undoLast(targetDir));
  }

  // POST /api/czkawka/reveal — select the file in Finder
  if (path === "/api/czkawka/reveal" && req.method === "POST") {
    const body = (await req.json()) as { path?: string };
    const p = body.path ?? "";
    const session = await getSession(targetDir);
    if (!p || !isPathInDirs(p, session.dirs)) {
      return json({ error: "Path is not in a configured directory" }, 403);
    }
    Bun.spawn(["open", "-R", p]);
    return json({ ok: true });
  }

  return null;
};
