/**
 * Shared utilities for reading hash-keyed embedding caches (.npz format)
 * and reindexing from hash order to filename order.
 *
 * Used by src/cluster.ts and scripts/{diagnose_distances,test_scoring}.ts.
 */

import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

interface NpzEntry {
  name: string;
  compMethod: number;
  compSize: number;
  uncompSize: number;
  dataStart: number;
}

/** Iterate local file headers in an in-memory .npz (ZIP) buffer. */
function* iterNpzEntries(npzBuf: Buffer): IterableIterator<NpzEntry> {
  let offset = 0;
  while (offset < npzBuf.length - 4) {
    const sig = npzBuf.readUInt32LE(offset);
    if (sig !== 0x04034b50) break; // PK\x03\x04
    const compMethod = npzBuf.readUInt16LE(offset + 8);
    const compSize = npzBuf.readUInt32LE(offset + 18);
    const uncompSize = npzBuf.readUInt32LE(offset + 22);
    const fnLen = npzBuf.readUInt16LE(offset + 26);
    const extraLen = npzBuf.readUInt16LE(offset + 28);
    const name = npzBuf.subarray(offset + 30, offset + 30 + fnLen).toString("utf-8");
    const dataStart = offset + 30 + fnLen + extraLen;
    yield { name, compMethod, compSize, uncompSize, dataStart };
    offset = dataStart + compSize;
  }
}

/** Extract a raw .npy buffer from an in-memory .npz (ZIP) file by entry name. */
function extractNpyEntry(npzBuf: Buffer, entryName: string): Buffer {
  for (const e of iterNpzEntries(npzBuf)) {
    if (e.name !== entryName) continue;
    if (e.compMethod === 0) {
      return npzBuf.subarray(e.dataStart, e.dataStart + e.uncompSize);
    }
    return Buffer.from(
      Bun.inflateSync(
        npzBuf.subarray(e.dataStart, e.dataStart + e.compSize) as Uint8Array<ArrayBuffer>,
      ),
    );
  }
  throw new Error(`Entry ${entryName} not found in npz`);
}

/** Extract a single .npy entry from an in-memory .npz (ZIP) buffer. */
export function parseNpyFromNpz(npzBuf: Buffer, entryName: string): Float32Array {
  const data = extractNpyEntry(npzBuf, entryName);
  const headerLen = data.readUInt16LE(8);
  const arrayData = data.subarray(10 + headerLen);
  return new Float32Array(arrayData.buffer, arrayData.byteOffset, arrayData.byteLength / 4);
}

export interface HashMapping {
  contentHashes: Record<string, string>;
  filenames: string[];
  nImages: number;
  hashOrder: string[];
  hashToRow: Map<string, number>;
}

/** Extract a numpy Unicode string array (<U*) from an .npz entry as string[]. */
export function parseNpyStringsFromNpz(npzBuf: Buffer, entryName: string): string[] {
  const data = extractNpyEntry(npzBuf, entryName);
  const headerLen = data.readUInt16LE(8);
  const headerStr = data.subarray(10, 10 + headerLen).toString("utf-8");
  const arrayData = data.subarray(10 + headerLen);

  const dtypeMatch = headerStr.match(/<U(\d+)/);
  if (!dtypeMatch) throw new Error(`Expected <U* dtype in ${entryName}, got: ${headerStr}`);
  const maxChars = Number.parseInt(dtypeMatch[1]!, 10);
  const bytesPerItem = maxChars * 4; // UTF-32LE: 4 bytes per codepoint

  const count = arrayData.byteLength / bytesPerItem;
  const view = new DataView(arrayData.buffer, arrayData.byteOffset, arrayData.byteLength);
  const result: string[] = [];
  for (let i = 0; i < count; i++) {
    const itemOffset = i * bytesPerItem;
    let str = "";
    for (let j = 0; j < maxChars; j++) {
      const cp = view.getUint32(itemOffset + j * 4, true);
      if (cp === 0) break;
      str += String.fromCodePoint(cp);
    }
    result.push(str);
  }
  return result;
}

/**
 * Ensure hash_cache_order.json exists. If missing, regenerate it from the
 * `hashes` array stored inside clip_hash_cache.npz (the Rust cluster-tool
 * can't read numpy string arrays, so it needs this JSON sidecar).
 */
export function ensureHashOrderJson(cachePath: string): void {
  const orderPath = join(cachePath, "hash_cache_order.json");
  if (existsSync(orderPath)) return;
  const npzPath = join(cachePath, "clip_hash_cache.npz");
  if (!existsSync(npzPath)) return; // nothing to regenerate from
  const npzBuf = readFileSync(npzPath) as Buffer;
  const hashes = parseNpyStringsFromNpz(npzBuf, "hashes.npy");
  writeFileSync(orderPath, JSON.stringify(hashes));
}

/** Load content_hashes.json + hash_cache_order.json and build the hash→row mapping. */
export function loadHashMapping(cacheDir: string): HashMapping {
  ensureHashOrderJson(cacheDir);
  const contentHashes: Record<string, string> = JSON.parse(
    readFileSync(join(cacheDir, "content_hashes.json"), "utf-8"),
  );
  const filenames = Object.keys(contentHashes).sort();
  const hashOrder: string[] = JSON.parse(
    readFileSync(join(cacheDir, "hash_cache_order.json"), "utf-8"),
  );
  const hashToRow = new Map(hashOrder.map((h, i) => [h, i]));
  return { contentHashes, filenames, nImages: filenames.length, hashOrder, hashToRow };
}

/**
 * Read per-model version strings (`_v_<key>`) from clip_hash_cache.npz.
 * Returns `{ key: version }` for every model whose version key exists in the
 * NPZ. These are the same strings extract_features.py writes via MODEL_VERSIONS.
 * Used for cache invalidation: if a model gets re-extracted with a new
 * version, its version string changes and downstream caches that depend on
 * those features are stale.
 *
 * Memoized by NPZ (size, mtime). The NPZ can be hundreds of MB; computing
 * a signature for both rerank- and patch-dist caches in a single request
 * would otherwise read it twice.
 */
const _modelVersionsCache = new Map<
  string,
  { size: number; mtime: number; versions: Record<string, string> }
>();
export function readModelVersions(cacheDir: string): Record<string, string> {
  const npzPath = join(cacheDir, "clip_hash_cache.npz");
  if (!existsSync(npzPath)) return {};
  const stat = statSync(npzPath);
  const cached = _modelVersionsCache.get(npzPath);
  if (cached && cached.size === stat.size && cached.mtime === stat.mtimeMs) {
    return cached.versions;
  }
  const npzBuf = readFileSync(npzPath) as Buffer;
  const versions: Record<string, string> = {};
  for (const e of iterNpzEntries(npzBuf)) {
    if (!e.name.startsWith("_v_") || !e.name.endsWith(".npy")) continue;
    const modelKey = e.name.slice(3, -4);
    try {
      const strs = parseNpyStringsFromNpz(npzBuf, e.name);
      if (strs.length > 0) versions[modelKey] = strs[0]!;
    } catch {
      // ignore unparseable entry
    }
  }
  _modelVersionsCache.set(npzPath, { size: stat.size, mtime: stat.mtimeMs, versions });
  return versions;
}

/**
 * Compute the content-derived signature for a downstream cache (e.g. the
 * rerank distance matrix). Captures everything that, if changed, should
 * invalidate the cache:
 *  - the set + ordering of (filename, content-hash) pairs (image set + sort
 *    order, since the matrix is indexed by sorted filenames),
 *  - the version strings of each model whose weight is non-zero (so
 *    re-extraction of an active model invalidates),
 *  - the explicit (weights, paramsHash) tuple supplied by the caller.
 *
 * Uses content-derived inputs only — never mtime — so it's robust to
 * extract_features.py rewriting content_hashes.json on every run regardless
 * of actual change.
 */
export function computeCacheSignature(input: {
  cacheDir: string;
  weights: Record<string, number | undefined>;
  extra?: Record<string, unknown>;
}): string {
  const ch: Record<string, string> = JSON.parse(
    readFileSync(join(input.cacheDir, "content_hashes.json"), "utf-8"),
  );
  // Sorted (filename, hash) — sort order matters because cluster-tool indexes
  // the distance matrix by sorted-filename position.
  const sortedPairs = Object.keys(ch)
    .sort()
    .map((fname) => [fname, ch[fname]!] as const);

  const allVersions = readModelVersions(input.cacheDir);
  const activeVersions: Record<string, string> = {};
  for (const [k, w] of Object.entries(input.weights)) {
    if (w != null && w > 0 && allVersions[k]) {
      activeVersions[k] = allVersions[k]!;
    }
  }

  // Canonical, deterministic JSON. Object keys are sorted by Object.keys/sort.
  const payload = {
    pairs: sortedPairs,
    versions: activeVersions,
    weights: Object.fromEntries(
      Object.entries(input.weights)
        .filter(([, v]) => v != null && v > 0)
        .sort(([a], [b]) => a.localeCompare(b)),
    ),
    extra: input.extra ?? {},
  };
  return createHash("sha1").update(JSON.stringify(payload)).digest("hex");
}

/** Read a `<artifact>.meta.json` sidecar's signature, or null if missing/unparseable. */
export function readSidecarSignature(sidecarPath: string): string | null {
  if (!existsSync(sidecarPath)) return null;
  try {
    const meta = JSON.parse(readFileSync(sidecarPath, "utf-8")) as { signature?: string };
    return typeof meta.signature === "string" ? meta.signature : null;
  } catch {
    return null;
  }
}

/** Write a `<artifact>.meta.json` sidecar with just the cache signature. */
export function writeSidecarSignature(sidecarPath: string, signature: string): void {
  writeFileSync(sidecarPath, JSON.stringify({ signature }, null, 2));
}

/** Reindex a flat Float32Array from hash-cache row order to sorted-filename order. */
export function reindexToFilenameOrder(
  hashOrdered: Float32Array,
  dim: number,
  mapping: HashMapping,
): Float32Array {
  const { contentHashes, filenames, nImages, hashToRow } = mapping;
  const out = new Float32Array(nImages * dim);
  for (let i = 0; i < nImages; i++) {
    const hash = contentHashes[filenames[i]!]!;
    const cacheRow = hashToRow.get(hash)!;
    out.set(hashOrdered.subarray(cacheRow * dim, (cacheRow + 1) * dim), i * dim);
  }
  return out;
}
