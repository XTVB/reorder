import { useEffect, useMemo, useState } from "react";
import { Modal } from "./Modal.tsx";

interface PathsModalProps {
  paths: string[];
  onClose: () => void;
  onCopyText: (text: string) => void;
}

const PROMPT_STORAGE_KEY = "reorder.pathsModal.batchPrompt";
const SIZE_STORAGE_KEY = "reorder.pathsModal.batchSize";
const DEFAULT_PROMPT = "Describe each image above in one sentence.";
const DEFAULT_SIZE = 10;

export function PathsModal({ paths, onClose, onCopyText }: PathsModalProps) {
  const [batchMode, setBatchMode] = useState(false);
  const [batchSize, setBatchSize] = useState<number>(() => {
    const stored = localStorage.getItem(SIZE_STORAGE_KEY);
    const n = stored ? parseInt(stored, 10) : NaN;
    return Number.isFinite(n) && n > 0 ? n : DEFAULT_SIZE;
  });
  const [prompt, setPrompt] = useState<string>(
    () => localStorage.getItem(PROMPT_STORAGE_KEY) ?? DEFAULT_PROMPT,
  );
  const [batchIndex, setBatchIndex] = useState(0);

  useEffect(() => {
    localStorage.setItem(PROMPT_STORAGE_KEY, prompt);
  }, [prompt]);

  useEffect(() => {
    if (batchSize > 0) localStorage.setItem(SIZE_STORAGE_KEY, String(batchSize));
  }, [batchSize]);

  const allText = paths.join("\n");

  const batches = useMemo(() => {
    const size = Math.max(1, Math.floor(batchSize || 1));
    const result: string[][] = [];
    for (let i = 0; i < paths.length; i += size) {
      result.push(paths.slice(i, i + size));
    }
    return result;
  }, [paths, batchSize]);

  const clampedIndex = Math.min(batchIndex, Math.max(0, batches.length - 1));
  const hasBatches = batches.length > 0;
  const isLast = clampedIndex >= batches.length - 1;
  const currentBatch = batches[clampedIndex] ?? [];
  const currentBatchText = batchMode
    ? currentBatch.join("\n") + (prompt.trim() ? `\n\n${prompt}` : "")
    : allText;

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
        onClick={() => {
          onCopyText(currentBatchText);
          if (!isLast) setBatchIndex(clampedIndex + 1);
        }}
      >
        {isLast ? "Copy" : "Copy & Next"}
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
        onClick={() => {
          onCopyText(allText);
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
                  setBatchSize(Number.isFinite(n) && n > 0 ? n : 1);
                  setBatchIndex(0);
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
                Batch {clampedIndex + 1} / {batches.length} · {currentBatch.length} path
                {currentBatch.length === 1 ? "" : "s"} · {paths.length} total
              </>
            ) : (
              <>No paths</>
            )}
          </div>
        </>
      )}
      <pre className="paths-list">{currentBatchText}</pre>
    </Modal>
  );
}
