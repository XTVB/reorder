// Export the current duplicate-comparison results so they can leave the app
// (the group list otherwise lives only in the session). Two shapes: JSON with
// full per-file metadata, and a human-readable text listing (czkawka-style
// groups of absolute paths).

import type { CzkawkaDirEntry, CzkawkaImage } from "../../types.ts";

export interface ExportContext {
  groups: CzkawkaImage[][];
  dirs: CzkawkaDirEntry[];
  config: { hashAlg: string; imageFilter: string; hashSize: number; similarity: number };
}

const path = (img: CzkawkaImage) => `${img.dir}/${img.filename}`;

function formatSize(bytes: number): string {
  const units = ["B", "KB", "MB", "GB"];
  let s = bytes;
  for (const u of units) {
    if (s < 1024) return `${s.toFixed(1)} ${u}`;
    s /= 1024;
  }
  return `${s.toFixed(1)} TB`;
}

export function buildExportJson(ctx: ExportContext): string {
  return JSON.stringify(
    {
      exportedAt: new Date().toISOString(),
      config: ctx.config,
      dirs: ctx.dirs,
      groupCount: ctx.groups.length,
      fileCount: ctx.groups.reduce((n, g) => n + g.length, 0),
      groups: ctx.groups.map((g) => ({
        images: g.map((img) => ({
          path: path(img),
          size: img.size,
          width: img.width,
          height: img.height,
          difference: img.difference,
        })),
      })),
    },
    null,
    2,
  );
}

export function buildExportText(ctx: ExportContext): string {
  const { groups, dirs, config } = ctx;
  const fileCount = groups.reduce((n, g) => n + g.length, 0);
  const lines: string[] = [
    `Duplicate comparison — ${groups.length} group(s), ${fileCount} file(s)`,
    `Exported: ${new Date().toISOString()}`,
    `Config: ${config.hashAlg} · ${config.imageFilter} · hash size ${config.hashSize} · similarity ${config.similarity}`,
    "Directories:",
    ...dirs.map(
      (d) => `  ${d.path}${d.reference ? " (reference)" : ""}${d.recursive ? " (recursive)" : ""}`,
    ),
  ];
  groups.forEach((g, i) => {
    lines.push("", `Group ${i + 1} of ${groups.length} (${g.length} files)`);
    for (const img of g) {
      const delta = img.difference > 0 ? `, Δ${img.difference}` : "";
      lines.push(`  ${path(img)}  (${formatSize(img.size)}, ${img.width}×${img.height}${delta})`);
    }
  });
  return `${lines.join("\n")}\n`;
}

export function downloadFile(filename: string, mimeType: string, content: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mimeType }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** duplicates_2026-07-07_14-30-05 — local time, filename-safe. */
export function exportBasename(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  const date = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const time = `${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
  return `duplicates_${date}_${time}`;
}
