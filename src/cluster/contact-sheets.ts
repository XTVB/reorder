// Contact-sheet (collage) generation using Sharp. Justified-row layout on a
// 2000px-wide canvas, optional filename label overlays.

import { mkdirSync } from "node:fs";
import { join } from "node:path";
import sharp from "sharp";
import { contactSheetsDir } from "../fs/paths.ts";

function escapeSvgText(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function truncateMiddle(s: string, max: number): string {
  if (s.length <= max) return s;
  const keep = max - 1;
  const left = Math.ceil(keep * 0.55);
  const right = keep - left;
  return `${s.slice(0, left)}…${s.slice(s.length - right)}`;
}

interface LabelCell {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
}

function buildLabelOverlay(cells: LabelCell[], width: number, height: number): Buffer {
  // Fixed sizes — labels just need to be legible, not scale with the thumb.
  const labelH = 34;
  const fontSize = 16;
  const padX = 10;
  const parts: string[] = [];
  for (const c of cells) {
    const maxChars = Math.max(12, Math.floor((c.width - padX * 2) / 9));
    const text = escapeSvgText(truncateMiddle(c.label, maxChars));
    parts.push(
      `<rect x="${c.x}" y="${c.y + c.height - labelH}" width="${c.width}" height="${labelH}" fill="black" fill-opacity="0.72"/>`,
    );
    parts.push(
      `<text x="${c.x + padX}" y="${c.y + c.height - 11}" font-family="Arial, sans-serif" font-size="${fontSize}" font-weight="600" fill="white">${text}</text>`,
    );
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${parts.join("")}</svg>`;
  return Buffer.from(svg);
}

// Grid for N images (index = N-1). Minimises empty cells while keeping the
// layout roughly square; N=10 keeps 4×3 with 2 blanks for aspect consistency.
const CONTACT_SHEET_GRID: [cols: number, rows: number][] = [
  [1, 1], // 1
  [2, 1], // 2
  [3, 1], // 3
  [2, 2], // 4
  [3, 2], // 5 (1 blank)
  [3, 2], // 6
  [4, 2], // 7 (1 blank)
  [4, 2], // 8
  [3, 3], // 9
  [4, 3], // 10 (2 blanks)
  [4, 3], // 11 (1 blank)
  [4, 3], // 12
];

export async function generateContactSheet(
  targetDir: string,
  filenames: string[],
  clusterName: string,
  withLabels = false,
): Promise<string> {
  const outDir = contactSheetsDir(targetDir);
  mkdirSync(outDir, { recursive: true });

  // Pick up to 12 evenly spaced images
  const maxImages = 12;
  let selected: string[];
  if (filenames.length <= maxImages) {
    selected = filenames;
  } else {
    selected = [];
    for (let i = 0; i < maxImages; i++) {
      const idx = Math.floor((i * filenames.length) / maxImages);
      selected.push(filenames[idx]!);
    }
  }

  const [cols, rows] = CONTACT_SHEET_GRID[selected.length - 1] ?? [4, 3];

  // Justified-row layout: per-image widths sum exactly to canvasW so no letterbox
  // bars appear around real images; rows with blanks pad the width budget with
  // avgAspect so non-full rows scale the same as full ones.
  const paths = selected.map((f) => join(targetDir, f));
  const aspects = await Promise.all(
    paths.map(async (p) => {
      const m = await sharp(p).metadata();
      return (m.width ?? 1) / (m.height ?? 1);
    }),
  );
  const avgAspect = aspects.reduce((s, a) => s + a, 0) / aspects.length;

  const canvasW = 2000;
  interface Cell {
    x: number;
    y: number;
    width: number;
    height: number;
  }
  const cells: Cell[] = [];
  let yCursor = 0;
  for (let r = 0; r < rows; r++) {
    const start = r * cols;
    const end = Math.min(start + cols, selected.length);
    const rowAspects = aspects.slice(start, end);
    const blanks = cols - rowAspects.length;
    const effectiveSum = rowAspects.reduce((s, a) => s + a, 0) + blanks * avgAspect;
    const rowH = Math.round(canvasW / effectiveSum);

    let x = 0;
    for (let i = 0; i < rowAspects.length; i++) {
      const isLastInFullRow = blanks === 0 && i === rowAspects.length - 1;
      const right = isLastInFullRow ? canvasW : Math.round(x + rowH * rowAspects[i]!);
      cells.push({ x, y: yCursor, width: right - x, height: rowH });
      x = right;
    }
    yCursor += rowH;
  }
  const canvasH = yCursor;

  const thumbnails = await Promise.all(
    cells.map((c, i) =>
      sharp(paths[i]!).resize(c.width, c.height, { fit: "cover" }).jpeg({ quality: 85 }).toBuffer(),
    ),
  );

  // Label overlay composited last so text sits above thumbnails.
  const composites: { input: Buffer; left: number; top: number }[] = cells.map((c, i) => ({
    input: thumbnails[i]!,
    left: c.x,
    top: c.y,
  }));
  if (withLabels) {
    const labelCells = cells.map((c, i) => ({ ...c, label: selected[i]! }));
    composites.push({
      input: buildLabelOverlay(labelCells, canvasW, canvasH),
      left: 0,
      top: 0,
    });
  }

  const safeName = clusterName.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 80);
  const outPath = join(outDir, `${safeName}.jpg`);

  await sharp({
    create: {
      width: canvasW,
      height: canvasH,
      channels: 3,
      background: { r: 26, g: 26, b: 46 },
    },
  })
    .composite(composites)
    .jpeg({ quality: 85 })
    .toFile(outPath);

  return outPath;
}
