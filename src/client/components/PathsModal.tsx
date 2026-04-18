import { useEffect, useMemo, useState } from "react";
import { useUIStore } from "../stores/uiStore.ts";
import {
  copyContactSheetToClipboard,
  generateContactSheetsBatch,
  getErrorMessage,
} from "../utils/helpers.ts";
import { Modal } from "./Modal.tsx";

interface PathsModalProps {
  filenames: string[];
  targetDir: string;
  onClose: () => void;
}

const PROMPT_STORAGE_KEY = "reorder.pathsModal.batchPrompt";
const SIZE_STORAGE_KEY = "reorder.pathsModal.batchSize";
const AT_PREFIX_STORAGE_KEY = "reorder.pathsModal.atPrefix";
const DEFAULT_PROMPT = "Describe each image above in one sentence.";
const DEFAULT_SIZE = 10;

interface SheetInfo {
  filename: string;
  path: string;
}

async function hashShort(filenames: string[]): Promise<string> {
  const data = new TextEncoder().encode(filenames.slice().sort().join("\n"));
  const buf = await crypto.subtle.digest("SHA-1", data);
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 8);
}

export function PathsModal({ filenames, targetDir, onClose }: PathsModalProps) {
  const showToast = useUIStore((s) => s.showToast);
  const [batchMode, setBatchMode] = useState(false);
  const [batchSize, setBatchSize] = useState<number>(() => {
    const stored = localStorage.getItem(SIZE_STORAGE_KEY);
    const n = stored ? parseInt(stored, 10) : NaN;
    return Number.isFinite(n) && n > 0 ? n : DEFAULT_SIZE;
  });
  const [prompt, setPrompt] = useState<string>(
    () => localStorage.getItem(PROMPT_STORAGE_KEY) ?? DEFAULT_PROMPT,
  );
  const [atPrefix, setAtPrefix] = useState<boolean>(
    () => localStorage.getItem(AT_PREFIX_STORAGE_KEY) === "1",
  );
  const [batchIndex, setBatchIndex] = useState(0);
  const [sheets, setSheets] = useState<Map<number, SheetInfo>>(() => new Map());
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    localStorage.setItem(PROMPT_STORAGE_KEY, prompt);
  }, [prompt]);

  useEffect(() => {
    if (batchSize > 0) localStorage.setItem(SIZE_STORAGE_KEY, String(batchSize));
  }, [batchSize]);

  useEffect(() => {
    localStorage.setItem(AT_PREFIX_STORAGE_KEY, atPrefix ? "1" : "0");
  }, [atPrefix]);

  const filenamesKey = filenames.join("\n");
  // biome-ignore lint/correctness/useExhaustiveDependencies: filenamesKey is the intentional trigger to reset generated sheets when the selection content changes
  useEffect(() => {
    setSheets(new Map());
    setBatchIndex(0);
  }, [filenamesKey]);

  function formatPath(fn: string) {
    const quoted = `"${targetDir}/${fn}"`;
    return atPrefix ? `@${quoted}` : quoted;
  }

  const allText = filenames.map(formatPath).join("\n");

  const batches = useMemo(() => {
    const size = Math.max(1, Math.floor(batchSize || 1));
    const result: string[][] = [];
    for (let i = 0; i < filenames.length; i += size) {
      result.push(filenames.slice(i, i + size));
    }
    return result;
  }, [filenames, batchSize]);

  function changeBatchSize(n: number) {
    setBatchSize(n);
    setBatchIndex(0);
    setSheets(new Map());
  }

  const clampedIndex = Math.min(batchIndex, Math.max(0, batches.length - 1));
  const hasBatches = batches.length > 0;
  const isLast = clampedIndex >= batches.length - 1;
  const currentBatchFilenames = batches[clampedIndex] ?? [];
  const currentBatchText = batchMode
    ? currentBatchFilenames.map(formatPath).join("\n") + (prompt.trim() ? `\n\n${prompt}` : "")
    : allText;

  const currentSheet = sheets.get(clampedIndex);

  async function copyTextOnly(text: string) {
    await navigator.clipboard.writeText(text);
    showToast("Copied to clipboard", "success");
  }

  async function copyBatchWithSheet(sheet: SheetInfo, text: string) {
    try {
      await copyContactSheetToClipboard(sheet.filename, text);
      showToast("Copied paths + contact sheet", "success");
    } catch (err) {
      showToast(
        `Image copy failed, copied text only: ${getErrorMessage(err, "clipboard error")}`,
        "error",
      );
      await navigator.clipboard.writeText(text);
    }
  }

  async function handleCopyCurrentBatch() {
    if (currentSheet) await copyBatchWithSheet(currentSheet, currentBatchText);
    else await copyTextOnly(currentBatchText);
    if (!isLast) setBatchIndex(clampedIndex + 1);
  }

  async function handleGenerateSheets() {
    setGenerating(true);
    try {
      const stamp = Date.now().toString(36);
      const requests = await Promise.all(
        batches.map(async (batch, i) => ({
          filenames: batch,
          clusterName: `paths-batch-${String(i + 1).padStart(3, "0")}-${stamp}-${await hashShort(batch)}`,
          withLabels: true,
        })),
      );
      const results = await generateContactSheetsBatch(requests);
      const next = new Map<number, SheetInfo>();
      results.forEach((r, i) => {
        next.set(i, r);
      });
      setSheets(next);
      await navigator.clipboard.writeText(results.map((r) => r.path).join("\n"));
      showToast(
        `Generated ${results.length} contact sheet${results.length === 1 ? "" : "s"} · paths copied`,
        "success",
      );
    } catch (err) {
      showToast(getErrorMessage(err, "Failed to generate contact sheets"), "error");
    } finally {
      setGenerating(false);
    }
  }

  const footer = batchMode ? (
    <>
      <button className="btn btn-secondary modal-footer-spacer" onClick={() => setBatchMode(false)}>
        ← Single
      </button>
      <button className="btn btn-secondary" onClick={onClose}>
        Close
      </button>
      <button
        className="btn btn-secondary"
        disabled={clampedIndex === 0}
        onClick={() => setBatchIndex(Math.max(0, clampedIndex - 1))}
      >
        Prev
      </button>
      <button
        className="btn btn-primary"
        disabled={!hasBatches}
        onClick={handleCopyCurrentBatch}
        title={currentSheet ? "Copy batch text + contact sheet image" : "Copy batch text"}
      >
        {isLast ? "Copy" : "Copy & Next"}
        {currentSheet ? " ＋ Sheet" : ""}
      </button>
    </>
  ) : (
    <>
      <button className="btn btn-secondary modal-footer-spacer" onClick={() => setBatchMode(true)}>
        Batch…
      </button>
      <button className="btn btn-secondary" onClick={onClose}>
        Close
      </button>
      <button
        className="btn btn-primary"
        onClick={async () => {
          await copyTextOnly(allText);
          onClose();
        }}
      >
        Copy
      </button>
    </>
  );

  return (
    <Modal
      title={batchMode ? "Copy Paths — Batch" : "Copy Paths"}
      onClose={onClose}
      footer={footer}
    >
      <div className="paths-atprefix-row">
        <label className="paths-atprefix-label">
          <input
            type="checkbox"
            checked={atPrefix}
            onChange={(e) => setAtPrefix(e.target.checked)}
          />
          <span>
            Prefix each path with <code>@</code>
          </span>
        </label>
      </div>
      {batchMode && (
        <>
          <div className="paths-batch-controls">
            <label className="paths-batch-field">
              <span>Batch size</span>
              <input
                type="number"
                min={1}
                value={batchSize}
                onChange={(e) => {
                  const n = parseInt(e.target.value, 10);
                  changeBatchSize(Number.isFinite(n) && n > 0 ? n : 1);
                }}
              />
            </label>
            <label className="paths-batch-field">
              <span>Prompt</span>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={3}
                placeholder="Prompt appended after each batch"
              />
            </label>
          </div>
          <div className="paths-batch-meta">
            {hasBatches ? (
              <>
                Batch {clampedIndex + 1} / {batches.length} · {currentBatchFilenames.length} path
                {currentBatchFilenames.length === 1 ? "" : "s"} · {filenames.length} total
                {sheets.size > 0 && (
                  <>
                    {" "}
                    · {sheets.size} sheet{sheets.size === 1 ? "" : "s"} ready
                  </>
                )}
              </>
            ) : (
              <>No paths</>
            )}
          </div>
          <div className="paths-batch-actions">
            <button
              className="btn btn-secondary"
              onClick={handleGenerateSheets}
              disabled={!hasBatches || generating}
              title="Generate a contact sheet for each batch; copies all sheet paths to the clipboard"
            >
              {generating ? "Generating…" : "Generate Contact Sheets"}
            </button>
            {currentSheet && (
              <span className="paths-batch-sheet-hint">
                Sheet ready for this batch — Copy will include it
              </span>
            )}
          </div>
        </>
      )}
      <pre className="paths-list">{currentBatchText}</pre>
    </Modal>
  );
}
