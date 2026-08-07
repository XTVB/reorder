import { useRef, useState } from "react";
import { useDismissOnOutside } from "../../hooks/useDismissOnOutside.ts";
import { useToastStore } from "../../stores/core/toastStore.ts";
import {
  HASH_ALGS,
  HASH_SIZES,
  type HashSize,
  IMAGE_FILTERS,
  useCzkawkaStore,
} from "../../stores/czkawkaStore.ts";
import { cn } from "../../utils/helpers.ts";
import { DirsPanel } from "./DirsPanel.tsx";
import { buildExportJson, buildExportText, downloadFile, exportBasename } from "./export.ts";

/**
 * The two-pass workflow: a strict pass first (high-confidence matches only),
 * then a loose pass to surface the tenuous ones worth eyeballing.
 */
const PRESETS = [
  { label: "Strict", hashSize: 32 as HashSize, similarity: 40 },
  { label: "Loose", hashSize: 16 as HashSize, similarity: 15 },
];

export function CzkawkaPrimary() {
  const loading = useCzkawkaStore((s) => s.loading);
  const runComparison = useCzkawkaStore((s) => s.runComparison);
  return (
    <button className="btn btn-primary" onClick={runComparison} disabled={loading}>
      {loading ? "Running…" : "Run Comparison"}
    </button>
  );
}

/** Take the results out of the app: download the group list as JSON or a
 * czkawka-style text listing, or copy the text to the clipboard. */
function ExportMenu() {
  const groups = useCzkawkaStore((s) => s.groups);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const showToast = useToastStore((s) => s.showToast);

  useDismissOnOutside(ref, open, () => setOpen(false));

  const context = () => {
    const s = useCzkawkaStore.getState();
    return {
      groups: s.groups,
      dirs: s.dirs,
      config: {
        hashAlg: s.hashAlg,
        imageFilter: s.imageFilter,
        hashSize: s.hashSize,
        similarity: s.similarity,
      },
    };
  };

  const exportJson = () => {
    downloadFile(`${exportBasename()}.json`, "application/json", buildExportJson(context()));
    setOpen(false);
  };
  const exportText = () => {
    downloadFile(`${exportBasename()}.txt`, "text/plain", buildExportText(context()));
    setOpen(false);
  };
  const copyText = () => {
    navigator.clipboard.writeText(buildExportText(context())).then(
      () => showToast("Comparison list copied to clipboard", "success"),
      () => showToast("Could not copy to clipboard", "error"),
    );
    setOpen(false);
  };

  return (
    <div className="cluster-weights-control" ref={ref}>
      <button
        className="btn btn-secondary"
        onClick={() => setOpen((v) => !v)}
        disabled={groups.length === 0}
        aria-expanded={open}
        title={
          groups.length === 0
            ? "Run a comparison first"
            : "Export the current duplicate groups (JSON, text, or clipboard)"
        }
      >
        Export
      </button>
      {open && (
        <div className="cluster-weights-dropdown czkawka-export-dropdown">
          <button className="btn btn-secondary" onClick={exportJson}>
            Download JSON
          </button>
          <button className="btn btn-secondary" onClick={exportText}>
            Download text
          </button>
          <button className="btn btn-secondary" onClick={copyText}>
            Copy to clipboard
          </button>
        </div>
      )}
    </div>
  );
}

function PruneGroupsButton() {
  // trashedCount is in-memory (resets on reload); undoDepth comes from the
  // persisted server session. Either being non-zero means this directory may
  // hold group members whose files are gone.
  const trashedCount = useCzkawkaStore((s) => s.trashedCount);
  const undoDepth = useCzkawkaStore((s) => s.undoDepth);
  const pruneDeletedFromGroups = useCzkawkaStore((s) => s.pruneDeletedFromGroups);
  const enabled = trashedCount > 0 || undoDepth > 0;
  return (
    <button
      className="btn btn-secondary"
      onClick={() => void pruneDeletedFromGroups()}
      disabled={!enabled}
      title={
        enabled
          ? "Remove deleted images from the reorder groups, and drop any group left empty. Deletes stay undoable until you do this."
          : "Nothing trashed yet — deleted images are removed from reorder groups here once you've resolved some duplicates"
      }
    >
      Prune groups
    </button>
  );
}

export function CzkawkaTools() {
  const hashAlg = useCzkawkaStore((s) => s.hashAlg);
  const imageFilter = useCzkawkaStore((s) => s.imageFilter);
  const similarity = useCzkawkaStore((s) => s.similarity);
  const hashSize = useCzkawkaStore((s) => s.hashSize);
  const loading = useCzkawkaStore((s) => s.loading);
  const dirCount = useCzkawkaStore((s) => s.dirs.length);
  const hasReference = useCzkawkaStore((s) => s.dirs.some((d) => d.reference));
  const dirsDirty = useCzkawkaStore((s) => s.dirsDirty);
  const setHashAlg = useCzkawkaStore((s) => s.setHashAlg);
  const setImageFilter = useCzkawkaStore((s) => s.setImageFilter);
  const setSimilarity = useCzkawkaStore((s) => s.setSimilarity);
  const setHashSize = useCzkawkaStore((s) => s.setHashSize);

  const [dirsOpen, setDirsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const settingsRef = useRef<HTMLDivElement>(null);

  useDismissOnOutside(settingsRef, settingsOpen, () => setSettingsOpen(false));

  return (
    <>
      <div className="toolbar-group">
        {PRESETS.map((p) => {
          const active = hashSize === p.hashSize && similarity === p.similarity;
          return (
            <button
              key={p.label}
              className={cn("btn btn-secondary", active && "btn-toggle-active")}
              onClick={() => {
                setHashSize(p.hashSize);
                setSimilarity(p.similarity);
              }}
              disabled={loading}
              title={
                p.label === "Strict"
                  ? "High-confidence matches only (size 32, similarity 40)"
                  : "Looser matching — more finds, more false positives (size 16, similarity 15)"
              }
            >
              {p.label}
            </button>
          );
        })}
        <div className="toolbar-divider" />
        <label className="toolbar-field">
          Size
          <select
            value={String(hashSize)}
            onChange={(e) => setHashSize(Number(e.target.value) as HashSize)}
            disabled={loading}
          >
            {HASH_SIZES.map((opt) => (
              <option key={String(opt)} value={String(opt)}>
                {opt}
              </option>
            ))}
          </select>
        </label>
        <label className="toolbar-field">
          Similarity
          <input
            type="number"
            min={0}
            max={500}
            value={similarity}
            onChange={(e) => setSimilarity(e.target.valueAsNumber || 0)}
            className="czkawka-input--narrow"
            disabled={loading}
          />
        </label>
        <div className="toolbar-divider" />

        <div className="cluster-weights-control" ref={settingsRef}>
          <button
            className="btn btn-secondary"
            onClick={() => setSettingsOpen((v) => !v)}
            aria-expanded={settingsOpen}
            title="Hash algorithm and resize filter (rarely need changing)"
          >
            {hashAlg} · {imageFilter}
          </button>
          {settingsOpen && (
            <div className="cluster-weights-dropdown czkawka-settings-dropdown">
              <label className="czkawka-toolbar-field">
                Hash
                <select
                  value={hashAlg}
                  onChange={(e) => setHashAlg(e.target.value as typeof hashAlg)}
                  disabled={loading}
                >
                  {HASH_ALGS.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              </label>
              <label className="czkawka-toolbar-field">
                Filter
                <select
                  value={imageFilter}
                  onChange={(e) => setImageFilter(e.target.value as typeof imageFilter)}
                  disabled={loading}
                >
                  {IMAGE_FILTERS.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}
        </div>
        <div className="toolbar-divider" />

        <button
          className={cn("btn", "czkawka-dirs-btn", dirsDirty && "is-dirty")}
          onClick={() => setDirsOpen((v) => !v)}
          title="Directories in the comparison"
        >
          Folders{dirCount > 1 ? ` · ${dirCount}` : ""}
          {hasReference && <span className="czkawka-dirs-btn-ref">REF</span>}
        </button>
        <div className="toolbar-divider" />
        <ExportMenu />
        <PruneGroupsButton />
      </div>
      <DirsPanel open={dirsOpen} onClose={() => setDirsOpen(false)} />
    </>
  );
}
