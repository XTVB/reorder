import { Database } from "bun:sqlite";
import { join } from "node:path";
import { existsSync } from "node:fs";
import type { RenameMapping } from "./rename.ts";
import type { ClothingItemData, ImageTagData, ClothingOption } from "./client/types.ts";

export type { ClothingItemData, ImageTagData, ClothingOption };

const DB_FILE = ".reorder-tags.db";

// Cached DB connection per target directory
const dbCache = new Map<string, Database>();

const SCHEMA_SQL = `
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS images (
  filename TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS tags (
  filename TEXT NOT NULL REFERENCES images(filename) ON UPDATE CASCADE ON DELETE CASCADE,
  category TEXT NOT NULL,
  value    TEXT NOT NULL,
  PRIMARY KEY (filename, category, value)
);
CREATE INDEX IF NOT EXISTS idx_tags_cat_val ON tags(category, value);
CREATE INDEX IF NOT EXISTS idx_tags_filename ON tags(filename);

CREATE TABLE IF NOT EXISTS clothing_items (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT NOT NULL REFERENCES images(filename) ON UPDATE CASCADE ON DELETE CASCADE,
  piece    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clothing_filename ON clothing_items(filename);

CREATE TABLE IF NOT EXISTS clothing_colors (
  item_id INTEGER NOT NULL REFERENCES clothing_items(id) ON DELETE CASCADE,
  color   TEXT NOT NULL,
  PRIMARY KEY (item_id, color)
);

CREATE TABLE IF NOT EXISTS clothing_styles (
  item_id INTEGER NOT NULL REFERENCES clothing_items(id) ON DELETE CASCADE,
  style   TEXT NOT NULL,
  PRIMARY KEY (item_id, style)
);
`;

// Flat tag field names from the schema (everything except clothing_items)
const FLAT_FIELDS = new Set([
  "clothing_type", "clothing_style", "colors", "setting", "location",
  "lighting", "pose", "framing", "hair", "accessories", "mood",
  "theme", "photo_type", "background", "notable",
]);

function dbPath(targetDir: string): string {
  return join(targetDir, DB_FILE);
}

export function openTagsDb(targetDir: string, create = false): Database | null {
  const path = dbPath(targetDir);
  if (!create && !existsSync(path)) return null;

  const cached = dbCache.get(path);
  if (cached) return cached;

  const db = new Database(path);
  db.exec(SCHEMA_SQL);
  dbCache.set(path, db);
  return db;
}

export function closeTagsDb(targetDir: string): void {
  const path = dbPath(targetDir);
  const db = dbCache.get(path);
  if (db) {
    db.close();
    dbCache.delete(path);
  }
}

export interface IngestResult {
  success: boolean;
  ingested: number;
  skipped: number;
}

export function ingestTags(targetDir: string, data: unknown): IngestResult {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("Invalid tags data: expected an object keyed by filename");
  }

  // Ingest does a full replace — close cached handle first, then reopen fresh
  closeTagsDb(targetDir);
  const db = openTagsDb(targetDir, true)!;
  let ingested = 0;
  let skipped = 0;

  try {
    db.exec("PRAGMA synchronous = OFF");

    const insertImage = db.prepare("INSERT INTO images (filename) VALUES (?)");
    const insertTag = db.prepare("INSERT OR IGNORE INTO tags (filename, category, value) VALUES (?, ?, ?)");
    const insertClothing = db.prepare("INSERT INTO clothing_items (filename, piece) VALUES (?, ?)");
    const insertColor = db.prepare("INSERT OR IGNORE INTO clothing_colors (item_id, color) VALUES (?, ?)");
    const insertStyle = db.prepare("INSERT OR IGNORE INTO clothing_styles (item_id, style) VALUES (?, ?)");

    const tx = db.transaction(() => {
      // Full replace
      db.exec("DELETE FROM images");

      for (const [filename, entry] of Object.entries(data as Record<string, unknown>)) {
        if (!entry || typeof entry !== "object") {
          skipped++;
          continue;
        }

        const tags = entry as Record<string, unknown>;
        insertImage.run(filename);

        // Insert flat tag fields
        for (const field of FLAT_FIELDS) {
          const values = tags[field];
          if (Array.isArray(values)) {
            for (const v of values) {
              if (typeof v === "string") {
                insertTag.run(filename, field, v);
              }
            }
          }
        }

        // Insert structured clothing items
        const clothingItems = tags.clothing_items;
        if (Array.isArray(clothingItems)) {
          for (const item of clothingItems) {
            if (!item || typeof item !== "object") continue;
            const ci = item as Record<string, unknown>;
            const piece = ci.piece;
            if (typeof piece !== "string") continue;

            const result = insertClothing.run(filename, piece);
            const itemId = Number(result.lastInsertRowid);

            if (Array.isArray(ci.color)) {
              for (const c of ci.color) {
                if (typeof c === "string") insertColor.run(itemId, c);
              }
            }
            if (Array.isArray(ci.style)) {
              for (const s of ci.style) {
                if (typeof s === "string") insertStyle.run(itemId, s);
              }
            }
          }
        }

        ingested++;
      }
    });

    tx();
  } finally {
    db.exec("PRAGMA synchronous = FULL");
  }

  return { success: true, ingested, skipped };
}

export function getAllTags(targetDir: string): { images: ImageTagData[] } | null {
  const db = openTagsDb(targetDir);
  if (!db) return null;

  const allImages = db.prepare("SELECT filename FROM images ORDER BY filename").all() as { filename: string }[];
  const allTags = db.prepare("SELECT filename, category, value FROM tags").all() as { filename: string; category: string; value: string }[];
  const allClothing = db.prepare(`
    SELECT ci.id, ci.filename, ci.piece,
      GROUP_CONCAT(DISTINCT cc.color) AS colors,
      GROUP_CONCAT(DISTINCT cs.style) AS styles
    FROM clothing_items ci
    LEFT JOIN clothing_colors cc ON cc.item_id = ci.id
    LEFT JOIN clothing_styles cs ON cs.item_id = ci.id
    GROUP BY ci.id
  `).all() as { id: number; filename: string; piece: string; colors: string | null; styles: string | null }[];

  // Build lookup maps
  const tagsByFile = new Map<string, Record<string, string[]>>();
  for (const { filename, category, value } of allTags) {
    let record = tagsByFile.get(filename);
    if (!record) { record = {}; tagsByFile.set(filename, record); }
    (record[category] ??= []).push(value);
  }

  const clothingByFile = new Map<string, ClothingItemData[]>();
  for (const row of allClothing) {
    let items = clothingByFile.get(row.filename);
    if (!items) { items = []; clothingByFile.set(row.filename, items); }
    items.push({
      piece: row.piece,
      colors: row.colors ? row.colors.split(",") : [],
      styles: row.styles ? row.styles.split(",") : [],
    });
  }

  const images: ImageTagData[] = allImages.map(({ filename }) => ({
    filename,
    tags: tagsByFile.get(filename) ?? {},
    clothing: clothingByFile.get(filename) ?? [],
  }));

  return { images };
}

export function getClothingStructured(targetDir: string): ClothingOption[] | null {
  const db = openTagsDb(targetDir);
  if (!db) return null;

  const rows = db.prepare(`
      SELECT ci.piece,
        GROUP_CONCAT(DISTINCT cc.color) AS colors,
        GROUP_CONCAT(DISTINCT cs.style) AS styles
      FROM clothing_items ci
      LEFT JOIN clothing_colors cc ON cc.item_id = ci.id
      LEFT JOIN clothing_styles cs ON cs.item_id = ci.id
      GROUP BY ci.piece
      ORDER BY ci.piece
    `).all() as { piece: string; colors: string | null; styles: string | null }[];

    return rows.map((r) => ({
      piece: r.piece,
      colors: r.colors ? [...new Set(r.colors.split(","))].sort() : [],
      styles: r.styles ? [...new Set(r.styles.split(","))].sort() : [],
    }));
}

export function getDbStatus(targetDir: string): { hasDb: boolean; imageCount: number } {
  const db = openTagsDb(targetDir);
  if (!db) return { hasDb: false, imageCount: 0 };

  const row = db.prepare("SELECT COUNT(*) AS cnt FROM images").get() as { cnt: number };
  return { hasDb: true, imageCount: row.cnt };
}

export function remapTagsDb(targetDir: string, renames: RenameMapping[]): void {
  const db = openTagsDb(targetDir);
  if (!db) return;

  const update = db.prepare("UPDATE images SET filename = ? WHERE filename = ?");
  const tx = db.transaction((mappings: RenameMapping[]) => {
    for (const { from, to } of mappings) {
      if (from !== to) update.run(to, from);
    }
  });
  tx(renames);
}
