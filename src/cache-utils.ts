/**
 * Shared utilities for reading hash-keyed embedding caches (.npz format)
 * and reindexing from hash order to filename order.
 *
 * Used by src/cluster.ts and scripts/{diagnose_distances,test_scoring}.ts.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/** Extract a raw .npy buffer from an in-memory .npz (ZIP) file by entry name. */
function extractNpyEntry(npzBuf: Buffer, entryName: string): Buffer {
  let offset = 0;
  while (offset < npzBuf.length - 4) {
    const sig = npzBuf.readUInt32LE(offset);
    if (sig !== 0x04034b50) break; // PK\x03\x04
    const compMethod = npzBuf.readUInt16LE(offset + 8);
    const compSize = npzBuf.readUInt32LE(offset + 18);
    const uncompSize = npzBuf.readUInt32LE(offset + 22);
    const fnLen = npzBuf.readUInt16LE(offset + 26);
    const extraLen = npzBuf.readUInt16LE(offset + 28);
    const fn = npzBuf.subarray(offset + 30, offset + 30 + fnLen).toString("utf-8");
    const dataStart = offset + 30 + fnLen + extraLen;

    if (fn === entryName) {
      if (compMethod === 0) {
        return npzBuf.subarray(dataStart, dataStart + uncompSize);
      }
      return Buffer.from(
        Bun.inflateSync(
          npzBuf.subarray(dataStart, dataStart + compSize) as Uint8Array<ArrayBuffer>,
        ),
      );
    }

    offset = dataStart + compSize;
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
function parseNpyStringsFromNpz(npzBuf: Buffer, entryName: string): string[] {
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
