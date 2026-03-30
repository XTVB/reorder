import { useImageStore } from "../stores/imageStore.ts";
import type { FilterMode } from "../types.ts";

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

export function formatCategoryName(cat: string): string {
  return cat.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatClothingValue(val: string): string {
  const [piece, color] = val.split("|");
  const p = piece.replace(/_/g, " ");
  if (!color) return p;
  return `${color.replace(/_/g, " ")} ${p}`;
}

export function resolveFilterMode(e: { shiftKey: boolean; type: string }): FilterMode {
  if (e.shiftKey) return "OR";
  if (e.type === "contextmenu") return "NOT";
  return "AND";
}

/** Shorthand for JSON POST fetch calls. */
export function postJson(url: string, body: unknown): Promise<Response> {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
