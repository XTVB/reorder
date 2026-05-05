// Unified subprocess driver for Rust binaries and Python scripts.
// progressPrefix: "progress:" (default) | "" (every non-empty line) | null.
// signal: AbortSignal → SIGINT (graceful, lets Python checkpoint).

import { log } from "../log.ts";

export interface SpawnOpts {
  /** Label used in log() output and error messages. Default: "subprocess". */
  label?: string;
  /** Forwarded with each progress line. */
  onProgress?: (line: string) => void;
  /** Stderr-line prefix that marks a progress line. Defaults to "progress:".
   * Pass `""` to forward every non-empty stderr line as progress (Python
   * scripts that emit free-form stderr). Pass `null` to disable progress
   * parsing entirely. */
  progressPrefix?: string | null;
  /** Aborting fires SIGINT to the child (graceful shutdown, lets Python save
   * checkpoints). */
  signal?: AbortSignal;
  /** Override the SIGINT signal (default "SIGINT"). Use "SIGTERM" for harder
   * kills if a child doesn't respond to interrupt. */
  abortSignal?: "SIGINT" | "SIGTERM" | "SIGKILL";
}

export interface SpawnResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

/**
 * Spawn a subprocess, drain stdout/stderr concurrently with awaiting exit
 * (avoids pipe-buffer deadlock), and forward stderr lines to `onProgress`
 * matching `progressPrefix`. Throws on non-zero exit.
 */
export async function spawn(args: string[], opts: SpawnOpts = {}): Promise<SpawnResult> {
  const label = opts.label ?? "subprocess";
  const progressPrefix = opts.progressPrefix === undefined ? "progress:" : opts.progressPrefix;

  const proc = Bun.spawn(args, { stdout: "pipe", stderr: "pipe" });

  const onAbort = () => {
    log(label, `Sending ${opts.abortSignal ?? "SIGINT"} to subprocess for graceful shutdown...`);
    proc.kill(opts.abortSignal ?? "SIGINT");
  };
  opts.signal?.addEventListener("abort", onAbort, { once: true });

  // Drain stderr line-by-line so we can stream progress without buffering up
  // the whole stream. We also accumulate the full text to attach to errors.
  const stderrPromise = (async () => {
    const reader = proc.stderr.getReader();
    const decoder = new TextDecoder();
    const lines: string[] = [];
    let buf = "";

    const handleLine = (line: string) => {
      lines.push(line);
      if (!opts.onProgress || progressPrefix === null) return;
      if (progressPrefix === "") {
        const trimmed = line.trim();
        if (trimmed) opts.onProgress(trimmed);
      } else if (line.startsWith(progressPrefix)) {
        opts.onProgress(line.slice(progressPrefix.length).trim());
      }
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const split = buf.split("\n");
        buf = split.pop() ?? "";
        for (const line of split) handleLine(line);
      }
      if (buf) handleLine(buf);
    } finally {
      reader.releaseLock();
    }
    return lines.join("\n");
  })();

  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    stderrPromise,
    proc.exited,
  ]);

  opts.signal?.removeEventListener("abort", onAbort);

  if (stderr.trim()) log(label, stderr.trim());

  if (exitCode !== 0) {
    throw new Error(`${label} failed (exit ${exitCode}): ${stderr}`);
  }

  return { stdout, stderr, exitCode };
}

/**
 * Spawn and parse stdout as JSON. Throws on non-zero exit or parse failure.
 */
export async function spawnJSON<T>(args: string[], opts: SpawnOpts = {}): Promise<T> {
  const { stdout } = await spawn(args, opts);
  return JSON.parse(stdout) as T;
}
