/**
 * Shared utilities for reading hash-keyed embedding caches (.npz format)
 * and reindexing from hash order to filename order.
 *
 * Used by src/cluster.ts and scripts/{diagnose_distances,test_scoring}.ts.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

/** Extract a single .npy entry from an in-memory .npz (ZIP) buffer. */
export function parseNpyFromNpz(npzBuf: Buffer, entryName: string): Float32Array {
  // npz is a ZIP file. Find the entry by scanning local file headers.
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
      let data: Buffer;
      if (compMethod === 0) {
        data = npzBuf.subarray(dataStart, dataStart + uncompSize);
      } else {
        // Deflate
        data = Buffer.from(
          Bun.inflateSync(
            npzBuf.subarray(dataStart, dataStart + compSize) as Uint8Array<ArrayBuffer>,
          ),
        );
      }
      // Parse .npy header
      // Magic: \x93NUMPY, version, header_len, then header string, then data
      const headerLen = data.readUInt16LE(8);
      const arrayData = data.subarray(10 + headerLen);
      return new Float32Array(arrayData.buffer, arrayData.byteOffset, arrayData.byteLength / 4);
    }

    offset = dataStart + compSize;
  }
  throw new Error(`Entry ${entryName} not found in npz`);
}

export interface HashMapping {
  contentHashes: Record<string, string>;
  filenames: string[];
  nImages: number;
  hashOrder: string[];
  hashToRow: Map<string, number>;
}

/** Load content_hashes.json + hash_cache_order.json and build the hash→row mapping. */
export function loadHashMapping(cacheDir: string): HashMapping {
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
