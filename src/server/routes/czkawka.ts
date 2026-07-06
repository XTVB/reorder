// /api/czkawka/* — duplicate-image discovery and resolution.
//
// `run` is an SSE compute job (shares the cluster job mutex): content-hash
// every image across the configured directories, perceptually hash unique
// contents via rust/hash-tool (original + flip + flop, so mirrored duplicates
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

import { createHash } from "node:crypto";
import { copyFile, mkdir, open, stat, unlink } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { HASH_TOOL_BINARY, spawnJSON } from "../../cluster/index.ts";
import { pruneContentHashes } from "../../fs/content-hashes.ts";
import {
  type CzkawkaSessionData,
  cacheDir,
  czkawkaHashCachePath,
  isImageFile,
  listImages,
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

/** Dedupe + resolve the requested dirs, forcing targetDir into the list (a
 * request entry may flip its reference flag). */
function normalizeDirs(targetDir: string, raw: unknown): CzkawkaDirEntry[] | string {
  const list: CzkawkaDirEntry[] = [{ path: resolve(targetDir), reference: false }];
  if (Array.isArray(raw)) {
    for (const item of raw as { path?: unknown; reference?: unknown }[]) {
      if (!item || typeof item.path !== "string" || item.path.trim() === "") continue;
      const path = resolve(expandHome(item.path.trim()));
      const existing = list.find((d) => d.path === path);
      if (existing) {
        existing.reference = existing.reference || item.reference === true;
      } else {
        list.push({ path, reference: item.reference === true });
      }
    }
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

// ── Content hashing (blake2b of first 16KB + filesize, matching the
//    extraction pipeline's rename-surviving convention) ─────────────────

async function computeContentHash(filePath: string): Promise<{ hash: string; size: number }> {
  const file = await open(filePath, "r");
  try {
    const buf = Buffer.alloc(16384);
    const { bytesRead } = await file.read(buf, 0, 16384);
    const stat = await file.stat();
    const h = createHash("blake2b256");
    h.update(buf.subarray(0, bytesRead));
    h.update(String(stat.size));
    return { hash: h.digest("hex"), size: stat.size };
  } finally {
    await file.close();
  }
}

// ── Perceptual-hash cache (per dir + config, keyed by content hash) ─────

interface CachedHashEntry {
  /** base64 image_hasher bytes for the original / vertical flip / mirror. */
  hash: string;
  flipHash: string;
  flopHash: string;
  width: number;
  height: number;
}

type HashCache = Record<string, CachedHashEntry>;

interface HashToolResult {
  results: {
    key: string;
    hash: string;
    flipHash: string;
    flopHash: string;
    width: number;
    height: number;
  }[];
  errors: { key: string; path: string; error: string }[];
}

// ── Hamming comparison + grouping ──────────────────────────────────────

const POPCOUNT = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  POPCOUNT[i] = (i & 1) + POPCOUNT[i >>> 1]!;
}

function hamming(a: Uint8Array, b: Uint8Array): number {
  let dist = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dist += POPCOUNT[a[i]! ^ b[i]!]!;
  }
  return dist;
}

interface FileRef {
  dir: string;
  filename: string;
}

interface UniqueEntry {
  /** Every file sharing this content hash, across all dirs. */
  files: FileRef[];
  size: number;
  width: number;
  height: number;
  hash: Uint8Array;
  flip: Uint8Array;
  flop: Uint8Array;
}

/** Orientation-aware distance. Comparing both flips against both originals
 * guards against resize-rounding asymmetries; flip-vs-flip combos are
 * redundant (flipping both images preserves Hamming distance). */
function orientedDistance(a: UniqueEntry, b: UniqueEntry): number {
  let d = hamming(a.hash, b.hash);
  d = Math.min(d, hamming(a.hash, b.flip), hamming(a.flip, b.hash));
  d = Math.min(d, hamming(a.hash, b.flop), hamming(a.flop, b.hash));
  return d;
}

interface GroupMember {
  entry: UniqueEntry;
  difference: number;
}

function expandGroup(
  members: GroupMember[],
  dirRank: (dir: string) => number,
  collator: Intl.Collator,
): CzkawkaImage[] {
  const images: CzkawkaImage[] = members.flatMap(({ entry, difference }) =>
    entry.files.map(({ dir, filename }) => ({
      dir,
      filename,
      size: entry.size,
      width: entry.width,
      height: entry.height,
      difference,
    })),
  );
  // Reference images lead, then the launch dir, then extras.
  images.sort(
    (a, b) =>
      dirRank(a.dir) - dirRank(b.dir) ||
      collator.compare(join(a.dir, a.filename), join(b.dir, b.filename)),
  );
  return images;
}

/** Grouping over unique contents, then expansion to files. A content hash
 * carrying 2+ files forms a group even when no other content is similar —
 * that's the exact-duplicate case.
 *
 * Without a reference dir: greedy single-link over all entries.
 *
 * With a reference dir the question becomes "which images match something in
 * the reference dir": each reference entry seeds a group and absorbs matching
 * non-reference entries. Reference entries are never compared to each other,
 * and neither are non-reference entries, so intra-reference and
 * intra-working-dir duplicates are not reported. (An entry whose identical
 * content exists on both sides is a ref↔non-ref exact match by itself.) */
function buildGroups(
  entries: UniqueEntry[],
  maxDistance: number,
  refDir: string | null,
  dirRank: (dir: string) => number,
): CzkawkaGroup[] {
  const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
  const groups: CzkawkaGroup[] = [];

  if (refDir !== null) {
    const isRefEntry = (e: UniqueEntry) => e.files.some((f) => f.dir === refDir);
    const refEntries = entries.filter(isRefEntry);
    const nonRefEntries = entries.filter((e) => !isRefEntry(e));
    const claimed = new Array<boolean>(nonRefEntries.length).fill(false);

    for (const ref of refEntries) {
      const members: GroupMember[] = [{ entry: ref, difference: 0 }];
      for (let j = 0; j < nonRefEntries.length; j++) {
        if (claimed[j]) continue;
        const dist = orientedDistance(ref, nonRefEntries[j]!);
        if (dist <= maxDistance) {
          claimed[j] = true;
          members.push({ entry: nonRefEntries[j]!, difference: dist });
        }
      }
      const images = expandGroup(members, dirRank, collator);
      // Needs at least one image on each side of the reference boundary.
      if (images.some((i) => i.dir === refDir) && images.some((i) => i.dir !== refDir)) {
        groups.push({ images });
      }
    }
    return groups;
  }

  const n = entries.length;
  const visited = new Array<boolean>(n).fill(false);
  for (let i = 0; i < n; i++) {
    if (visited[i]) continue;
    visited[i] = true;
    const members: GroupMember[] = [{ entry: entries[i]!, difference: 0 }];

    for (let j = i + 1; j < n; j++) {
      if (visited[j]) continue;
      const dist = orientedDistance(entries[i]!, entries[j]!);
      if (dist <= maxDistance) {
        visited[j] = true;
        members.push({ entry: entries[j]!, difference: dist });
      }
    }

    const images = expandGroup(members, dirRank, collator);
    if (images.length >= 2) groups.push({ images });
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
      const fns = await listImages(d.path);
      for (const filename of fns) allFiles.push({ dir: d.path, filename });
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
        flipHash: r.flipHash,
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
      if (cache[ch] && files.some((f) => f.dir === d.path)) out[ch] = cache[ch];
    }
    try {
      await mkdir(cacheDir(d.path), { recursive: true });
      await writeJsonAtomic(cachePathFor(d.path), out, { pretty: false, atomic: true });
    } catch {
      log("czkawka", `Could not write hash cache in ${d.path} (read-only?)`);
    }
  }

  // Phase 3 — compare. Reference-dir entries sort first so group seeds tend
  // to be reference images and `difference` reads as distance-to-reference.
  onProgress("Comparing hashes...");
  const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
  const decode = (b64: string) => Uint8Array.from(Buffer.from(b64, "base64"));
  const entries: UniqueEntry[] = [];
  for (const [ch, { files, size }] of byContent) {
    const cached = cache[ch];
    if (!cached) continue;
    const sorted = [...files].sort(
      (a, b) =>
        dirRank(a.dir) - dirRank(b.dir) ||
        collator.compare(join(a.dir, a.filename), join(b.dir, b.filename)),
    );
    entries.push({
      files: sorted,
      size,
      width: cached.width,
      height: cached.height,
      hash: decode(cached.hash),
      flip: decode(cached.flipHash),
      flop: decode(cached.flopHash),
    });
  }
  entries.sort((a, b) => {
    const ra = Math.min(...a.files.map((f) => dirRank(f.dir)));
    const rb = Math.min(...b.files.map((f) => dirRank(f.dir)));
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

const imgPath = (img: CzkawkaImage) => join(img.dir, img.filename);

/** True when `path` names an image directly inside one of the session dirs. */
function isPathInDirs(path: string, dirs: CzkawkaDirEntry[]): boolean {
  return dirs.some((d) => dirname(path) === d.path) && isImageFile(basename(path));
}

// ── Actions ─────────────────────────────────────────────────────────────

function validateOperations(ops: CzkawkaOperation[], session: CzkawkaSessionData): string | null {
  const allPaths = new Set(session.groups.flatMap((g) => g.images.map(imgPath)));
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

  const snapshotBefore = structuredClone(session.groups);
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
    .filter((g) => !g.images.some((i) => resolvedPaths.has(imgPath(i))))
    .map((g) => ({ images: g.images.filter((i) => !removedPaths.has(imgPath(i))) }))
    .filter((g) => g.images.length >= 2);

  session.undoStack.push({
    snapshotBefore,
    trashed,
    copiedTargets,
    deletedCount: trashed.length,
  });
  await saveCzkawkaSession(targetDir, session);

  // Target-dir files that were trashed get pruned from reorder groups /
  // content hashes; a copy-replace target still exists but with new bytes,
  // so only its (now stale) content-hash entry goes. Files in other dirs
  // don't touch this project's caches.
  const trashedLocal = trashed
    .filter((t) => dirname(t.path) === targetDir)
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
  const last = session.undoStack.pop();
  if (!last) return json({ error: "Nothing to undo" }, 400);

  // Remove copies before restoring so the original bytes win at the target.
  for (const p of last.copiedTargets) {
    await unlink(p).catch(() => {});
  }
  await restoreFromTrash([...last.trashed].reverse());
  session.groups = last.snapshotBefore;
  await saveCzkawkaSession(targetDir, session);

  // Files reappeared — position-indexed cluster artifacts are stale, but the
  // restored files must NOT be pruned from anything, so skip the delete path.
  const warnings = await invalidateDerivedCaches(targetDir);

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

  // POST /api/czkawka/check-dir — validate a directory before adding it
  if (path === "/api/czkawka/check-dir" && req.method === "POST") {
    const body = (await req.json()) as { path?: string };
    const raw = (body.path ?? "").trim();
    if (!raw) return json({ error: "Path is required" }, 400);
    const resolved = resolve(expandHome(raw));
    const s = await stat(resolved).catch(() => null);
    if (!s?.isDirectory()) return json({ error: `Not a directory: ${resolved}` }, 400);
    try {
      const imageCount = (await listImages(resolved)).length;
      return json({ ok: true, path: resolved, imageCount });
    } catch {
      return json({ error: `Cannot read directory: ${resolved}` }, 400);
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
