import { Database } from "bun:sqlite";
import { join } from "node:path";
import { existsSync } from "node:fs";
import { stat } from "node:fs/promises";
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
  inode    INTEGER PRIMARY KEY,
  filename TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_filename ON images(filename);

CREATE TABLE IF NOT EXISTS tags (
  inode    INTEGER NOT NULL REFERENCES images(inode) ON DELETE CASCADE,
  category TEXT NOT NULL,
  value    TEXT NOT NULL,
  PRIMARY KEY (inode, category, value)
);
CREATE INDEX IF NOT EXISTS idx_tags_cat_val ON tags(category, value);

CREATE TABLE IF NOT EXISTS clothing_items (
  id    INTEGER PRIMARY KEY AUTOINCREMENT,
  inode INTEGER NOT NULL REFERENCES images(inode) ON DELETE CASCADE,
  piece TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clothing_inode ON clothing_items(inode);

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

export async function ingestTags(targetDir: string, data: unknown): Promise<IngestResult> {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("Invalid tags data: expected an object keyed by filename");
  }

  // Ingest does a full replace — close cached handle first, then reopen fresh
  closeTagsDb(targetDir);
  const db = openTagsDb(targetDir, true)!;
  let ingested = 0;
  let skipped = 0;

  // Resolve inodes for all filenames in the data
  const inodeMap = new Map<string, number>();
  await Promise.all(
    Object.keys(data as Record<string, unknown>).map(async (filename) => {
      try {
        const s = await stat(join(targetDir, filename));
        inodeMap.set(filename, s.ino);
      } catch {
        // File doesn't exist on disk — will be skipped
      }
    })
  );

  try {
    db.exec("PRAGMA synchronous = OFF");

    const insertImage = db.prepare("INSERT INTO images (inode, filename) VALUES (?, ?)");
    const insertTag = db.prepare("INSERT OR IGNORE INTO tags (inode, category, value) VALUES (?, ?, ?)");
    const insertClothing = db.prepare("INSERT INTO clothing_items (inode, piece) VALUES (?, ?)");
    const insertColor = db.prepare("INSERT OR IGNORE INTO clothing_colors (item_id, color) VALUES (?, ?)");
    const insertStyle = db.prepare("INSERT OR IGNORE INTO clothing_styles (item_id, style) VALUES (?, ?)");

    const tx = db.transaction(() => {
      // Full replace
      db.exec("DELETE FROM images");

      for (const [filename, entry] of Object.entries(data as Record<string, unknown>)) {
        const ino = inodeMap.get(filename);
        if (!ino) {
          skipped++;
          continue;
        }

        if (!entry || typeof entry !== "object") {
          skipped++;
          continue;
        }

        const tags = entry as Record<string, unknown>;
        insertImage.run(ino, filename);

        // Insert flat tag fields
        for (const field of FLAT_FIELDS) {
          const values = tags[field];
          if (Array.isArray(values)) {
            for (const v of values) {
              if (typeof v === "string") {
                insertTag.run(ino, field, v);
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

            const result = insertClothing.run(ino, piece);
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

  const allImages = db.prepare("SELECT inode, filename FROM images ORDER BY filename").all() as { inode: number; filename: string }[];
  const allTags = db.prepare("SELECT inode, category, value FROM tags").all() as { inode: number; category: string; value: string }[];
  const allClothing = db.prepare(`
    SELECT ci.inode, ci.piece,
      GROUP_CONCAT(DISTINCT cc.color) AS colors,
      GROUP_CONCAT(DISTINCT cs.style) AS styles
    FROM clothing_items ci
    LEFT JOIN clothing_colors cc ON cc.item_id = ci.id
    LEFT JOIN clothing_styles cs ON cs.item_id = ci.id
    GROUP BY ci.id
  `).all() as { inode: number; piece: string; colors: string | null; styles: string | null }[];

  // Build lookup maps by inode
  const tagsByInode = new Map<number, Record<string, string[]>>();
  for (const { inode, category, value } of allTags) {
    let record = tagsByInode.get(inode);
    if (!record) { record = {}; tagsByInode.set(inode, record); }
    (record[category] ??= []).push(value);
  }

  const clothingByInode = new Map<number, ClothingItemData[]>();
  for (const row of allClothing) {
    let items = clothingByInode.get(row.inode);
    if (!items) { items = []; clothingByInode.set(row.inode, items); }
    items.push({
      piece: row.piece,
      colors: row.colors ? row.colors.split(",") : [],
      styles: row.styles ? row.styles.split(",") : [],
    });
  }

  const images: ImageTagData[] = allImages.map(({ inode, filename }) => ({
    filename,
    tags: tagsByInode.get(inode) ?? {},
    clothing: clothingByInode.get(inode) ?? [],
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

/**
 * Update filenames in the tags DB after a rename operation.
 * Since inode is the primary key and renames preserve inodes,
 * we look up the row by the old filename and update it to the new one.
 */
export function remapTagsDb(targetDir: string, renames: RenameMapping[]): void {
  const db = openTagsDb(targetDir);
  if (!db) return;

  const actual = renames.filter(({ from, to }) => from !== to);
  if (actual.length === 0) return;

  const update = db.prepare("UPDATE images SET filename = ? WHERE filename = ?");
  const tx = db.transaction((mappings: RenameMapping[]) => {
    for (const { from, to } of mappings) {
      update.run(to, from);
    }
  });
  tx(actual);
}

/**
 * Idempotent sync: stat every image on disk, update the DB filename for each
 * known inode. Fixes any desync regardless of how it happened.
 * Returns the number of filenames corrected.
 */
export async function syncTagsDbFilenames(targetDir: string): Promise<number> {
  const db = openTagsDb(targetDir);
  if (!db) return 0;

  const dbRows = db.prepare("SELECT inode, filename FROM images").all() as { inode: number; filename: string }[];
  const dbByInode = new Map(dbRows.map(r => [r.inode, r.filename]));

  const { readdir } = await import("node:fs/promises");
  const { isImageFile } = await import("./rename.ts");
  const entries = await readdir(targetDir, { withFileTypes: true });
  const imageFiles = entries.filter(e => e.isFile() && isImageFile(e.name));

  const fixes: { ino: number; filename: string }[] = [];
  await Promise.all(
    imageFiles.map(async (e) => {
      const s = await stat(join(targetDir, e.name));
      const dbFilename = dbByInode.get(s.ino);
      if (dbFilename !== undefined && dbFilename !== e.name) {
        fixes.push({ ino: s.ino, filename: e.name });
      }
    })
  );

  if (fixes.length > 0) {
    const update = db.prepare("UPDATE images SET filename = ? WHERE inode = ?");
    const tx = db.transaction(() => {
      for (const { filename, ino } of fixes) {
        update.run(filename, ino);
      }
    });
    tx();
  }

  return fixes.length;
}
