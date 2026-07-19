// Listing image files in a directory. Sorted naturally (numeric-aware) so
// "10.jpg" comes after "9.jpg" rather than between "1.jpg" and "2.jpg".

import { readdir } from "node:fs/promises";
import { extname, join } from "node:path";

const IMAGE_EXTENSIONS = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".gif",
  ".webp",
  ".svg",
  ".avif",
  ".bmp",
  ".tiff",
  ".tif",
]);

export function isImageFile(filename: string): boolean {
  if (filename.startsWith(".")) return false;
  return IMAGE_EXTENSIONS.has(extname(filename).toLowerCase());
}

export async function listImages(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  return entries
    .filter((e) => (e.isFile() || e.isSymbolicLink()) && isImageFile(e.name))
    .map((e) => e.name)
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

/** A single image file located somewhere under a scanned directory tree. */
export interface ImageFileRef {
  /** The directory the file actually lives in (a sub-dir when recursing). */
  dir: string;
  filename: string;
}

/** Walk `root` and every sub-directory, collecting image files. Hidden
 * directories (names starting with ".", which includes ".reorder-cache") are
 * skipped, as are directory symlinks (avoids cycles). To scan an otherwise
 * skipped directory, configure it as a root of its own. */
export async function listImagesRecursive(root: string): Promise<ImageFileRef[]> {
  async function walk(dir: string): Promise<ImageFileRef[]> {
    const entries = await readdir(dir, { withFileTypes: true }).catch(() => null);
    if (!entries) return []; // unreadable dir — skip
    const out: ImageFileRef[] = [];
    const subdirs: string[] = [];
    for (const e of entries) {
      if (e.isDirectory()) {
        if (!e.name.startsWith(".")) subdirs.push(join(dir, e.name));
      } else if ((e.isFile() || e.isSymbolicLink()) && isImageFile(e.name)) {
        out.push({ dir, filename: e.name });
      }
    }
    // Sibling subtrees are independent — walk them concurrently, then splice
    // the results back in directory order so the listing stays deterministic.
    for (const sub of await Promise.all(subdirs.map(walk))) out.push(...sub);
    return out;
  }
  return walk(root);
}
