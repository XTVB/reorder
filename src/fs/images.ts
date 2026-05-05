// Listing image files in a directory. Sorted naturally (numeric-aware) so
// "10.jpg" comes after "9.jpg" rather than between "1.jpg" and "2.jpg".

import { readdir } from "node:fs/promises";
import { extname } from "node:path";

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
