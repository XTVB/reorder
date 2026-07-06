import { useState } from "react";
import { useCzkawkaStore } from "../../stores/czkawkaStore.ts";
import { cn } from "../../utils/helpers.ts";

/** Last two path segments, prefixed with … when the path is deeper. */
function shortPath(path: string): string {
  const segs = path.split("/").filter(Boolean);
  if (segs.length <= 2) return path;
  return `…/${segs.slice(-2).join("/")}`;
}

/** Directory list for multi-dir comparison: add/remove dirs and mark at most
 * one as the czkawka-style reference ("which images match this dir"). */
export function DirsPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dirs = useCzkawkaStore((s) => s.dirs);
  const targetDir = useCzkawkaStore((s) => s.targetDir);
  const dirsDirty = useCzkawkaStore((s) => s.dirsDirty);
  const addDir = useCzkawkaStore((s) => s.addDir);
  const removeDir = useCzkawkaStore((s) => s.removeDir);
  const setReferenceDir = useCzkawkaStore((s) => s.setReferenceDir);

  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  if (!open) return null;

  const handleAdd = async () => {
    const path = input.trim();
    if (!path || busy) return;
    setBusy(true);
    const ok = await addDir(path);
    setBusy(false);
    if (ok) setInput("");
  };

  return (
    <>
      <div className="czkawka-dirs-overlay" onClick={onClose} />
      <div className="czkawka-dirs-panel">
        <div className="czkawka-dirs-header">
          <span className="czkawka-dirs-title">Compare directories</span>
          <button className="czkawka-dirs-close" onClick={onClose} title="Close">
            ✕
          </button>
        </div>
        <p className="czkawka-dirs-help">
          Duplicates are searched across every listed directory. Mark one as <strong>REF</strong> to
          only find images that match something in it (its own internal duplicates are skipped).
        </p>
        <div className="czkawka-dirs-list">
          {dirs.map((d) => (
            <div key={d.path} className="czkawka-dirs-row">
              <button
                type="button"
                className={cn("czkawka-dirs-ref", d.reference && "is-active")}
                title={
                  d.reference
                    ? "Unmark as reference"
                    : "Only report images matching something in this directory"
                }
                onClick={() => setReferenceDir(d.reference ? null : d.path)}
              >
                REF
              </button>
              <span className="czkawka-dirs-path" title={d.path}>
                {shortPath(d.path)}
                {d.path === targetDir && <span className="czkawka-dirs-this"> (this folder)</span>}
              </span>
              {d.path !== targetDir && (
                <button
                  type="button"
                  className="czkawka-dirs-remove"
                  onClick={() => removeDir(d.path)}
                  title="Remove from comparison"
                >
                  ✕
                </button>
              )}
            </div>
          ))}
        </div>
        <div className="czkawka-dirs-add">
          <input
            type="text"
            value={input}
            placeholder="/path/to/other/directory"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleAdd();
            }}
          />
          <button className="btn btn-sm" onClick={() => void handleAdd()} disabled={busy}>
            {busy ? "Checking…" : "Add"}
          </button>
        </div>
        {dirsDirty && (
          <p className="czkawka-dirs-dirty">Changes apply on the next Run Comparison.</p>
        )}
      </div>
    </>
  );
}
