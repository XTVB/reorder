import { appendFile, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

const MAX_LINES = 50000;

let logFilePath: string | null = null;

export async function initLog(targetDir: string): Promise<void> {
  logFilePath = join(targetDir, ".reorder-log");

  // Truncate to last MAX_LINES lines on startup
  try {
    const raw = await readFile(logFilePath, "utf8");
    const lines = raw.split("\n").filter((l) => l.length > 0);
    if (lines.length > MAX_LINES) {
      const trimmed = `${lines.slice(lines.length - MAX_LINES).join("\n")}\n`;
      await writeFile(logFilePath, trimmed, "utf8");
    }
  } catch {
    // File doesn't exist yet — that's fine
  }
}

function ts(): string {
  return new Date().toISOString();
}

async function writeLine(line: string): Promise<void> {
  if (!logFilePath) return;
  await appendFile(logFilePath, line, "utf8").catch(() => {});
}

/** Log a single line to console and file. */
export function log(tag: string, message: string): void {
  const line = `[${ts()}] [${tag}] ${message}\n`;
  process.stdout.write(line);
  writeLine(line);
}

/** Log an error to stderr and file. */
export function logError(tag: string, message: string, err?: unknown): void {
  const detail = err instanceof Error ? `: ${err.message}` : err != null ? `: ${String(err)}` : "";
  const line = `[${ts()}] [${tag}] ERROR ${message}${detail}\n`;
  process.stderr.write(line);
  writeLine(line);
}

/**
 * Log a block of structured data to the log file only (not console).
 * Used for full audit trails (rename mappings, group state) that would
 * flood the terminal but are essential for post-mortem.
 */
export function logData(tag: string, label: string, data: string): void {
  const block = `[${ts()}] [${tag}] ${label}:\n${data}\n`;
  writeLine(block);
}

/**
 * Log a block to both console and file.
 * Use sparingly — for critical data you always want visible.
 */
export function logBlock(tag: string, label: string, data: string): void {
  const block = `[${ts()}] [${tag}] ${label}:\n${data}\n`;
  process.stdout.write(block);
  writeLine(block);
}
