import { useImageStore } from "../stores/imageStore.ts";
import type { ImageGroup, ImageInfo } from "../types.ts";
export const GROUP_PREFIX = "group:";
export const FOLDER_PREFIX = "folder:";

export function toGroupSortId(groupId: string): string {
  return GROUP_PREFIX + groupId;
}

export function isGroupSortId(id: string): boolean {
  return id.startsWith(GROUP_PREFIX);
}

export function fromGroupSortId(id: string): string {
  return id.slice(GROUP_PREFIX.length);
}

export function toFolderSortId(folderName: string): string {
  return FOLDER_PREFIX + folderName;
}

export function isFolderSortId(id: string): boolean {
  return id.startsWith(FOLDER_PREFIX);
}

export function fromFolderSortId(id: string): string {
  return id.slice(FOLDER_PREFIX.length);
}

export function selectedImageFilenames(selectedIds: Iterable<string>): string[] {
  const out: string[] = [];
  for (const id of selectedIds) {
    if (!isGroupSortId(id) && !isFolderSortId(id)) out.push(id);
  }
  return out;
}

export function stripFolderNumber(name: string): string {
  return name.replace(/^\d+\s*-\s*/, "").trim();
}

export function cn(...classes: (string | false | undefined | null)[]): string {
  return classes.filter(Boolean).join(" ");
}

export function getErrorMessage(err: unknown, fallback: string): string {
  return err instanceof Error ? err.message : fallback;
}

export function imageUrl(filename: string): string {
  const v = useImageStore.getState().imageVersion;
  return `/api/thumbnails/${encodeURIComponent(filename)}?v=${v}`;
}

export function fullImageUrl(filename: string): string {
  const v = useImageStore.getState().imageVersion;
  return `/api/images/${encodeURIComponent(filename)}?v=${v}`;
}

// Drag-end timing — shared across card components to suppress click after drag
let dragEndTimeMs = 0;
export function setDragEndTime() {
  dragEndTimeMs = Date.now();
}
export function wasJustDragged(): boolean {
  return Date.now() - dragEndTimeMs < 100;
}

/** Shorthand for JSON POST fetch calls. */
export function postJson(url: string, body: unknown): Promise<Response> {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/**
 * Copy a JPEG contact sheet (served from /api/contact-sheet/:filename) alongside
 * some text via the async Clipboard API. Transcodes JPEG→PNG because the
 * Clipboard API rejects image/jpeg.
 */
export async function copyContactSheetToClipboard(sheetFilename: string, text: string) {
  const imgRes = await fetch(`/api/contact-sheet/${encodeURIComponent(sheetFilename)}`);
  const jpegBlob = await imgRes.blob();
  const bitmap = await createImageBitmap(jpegBlob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(bitmap, 0, 0);
  const pngBlob = await canvas.convertToBlob({ type: "image/png" });
  await navigator.clipboard.write([
    new ClipboardItem({
      "text/plain": new Blob([text], { type: "text/plain" }),
      "image/png": pngBlob,
    }),
  ]);
}

export interface ContactSheetRequest {
  filenames: string[];
  clusterName: string;
  withLabels?: boolean;
}

export interface ContactSheetResult {
  path: string;
  filename: string;
}

/** Kick off N contact-sheet jobs in parallel against /api/cluster/contact-sheet. */
export async function generateContactSheetsBatch(
  requests: ContactSheetRequest[],
): Promise<ContactSheetResult[]> {
  return Promise.all(
    requests.map(async (req, i) => {
      const res = await postJson("/api/cluster/contact-sheet", req);
      const data = (await res.json()) as { path?: string; filename?: string; error?: string };
      if (!res.ok || !data.path || !data.filename) {
        throw new Error(data.error ?? `Failed batch ${i + 1}`);
      }
      return { path: data.path, filename: data.filename };
    }),
  );
}

/**
 * Rebuild a flat images array by placing grouped images (in the given group order)
 * before ungrouped images (preserving their original relative order). The save flow
 * reads image order from imageStore.images, so this is what drives the rename.
 */
export function reorderImagesByGroups(
  images: ImageInfo[],
  imageMap: Map<string, ImageInfo>,
  groups: ImageGroup[],
): ImageInfo[] {
  const grouped = new Set<string>();
  const result: ImageInfo[] = [];
  for (const g of groups) {
    for (const fn of g.images) {
      grouped.add(fn);
      const img = imageMap.get(fn);
      if (img) result.push(img);
    }
  }
  for (const img of images) {
    if (!grouped.has(img.filename)) result.push(img);
  }
  return result;
}

/** Pick up to 4 evenly-spaced sample thumbnails from a group's images. */
export function pickThumbSamples(images: string[]): string[] {
  const n = images.length;
  if (n === 0) return [];
  if (n <= 4) return images.slice();
  return [images[0]!, images[Math.floor(n / 3)]!, images[Math.floor((n * 2) / 3)]!, images[n - 1]!];
}
