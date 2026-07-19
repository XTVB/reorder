import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getJson } from "../../api/client.ts";
import { useCzkawkaStore } from "../../stores/czkawkaStore.ts";
import { cn, getErrorMessage } from "../../utils/helpers.ts";

/** Last two path segments, prefixed with … when the path is deeper. */
function shortPath(path: string): string {
  const segs = path.split("/").filter(Boolean);
  if (segs.length <= 2) return path;
  return `…/${segs.slice(-2).join("/")}`;
}

/** Split a typed path into the directory to browse and the trailing fragment to
 * fuzzy-filter its listing by. A trailing slash (or a plain, slash-less string)
 * means "no fragment" → the directory's children show unfiltered. Everything
 * after the last slash is the fragment, so typing `/a/b/re` browses `/a/b/` and
 * filters by `re`, while clicking a folder (which appends `/`) descends into it. */
function splitQuery(q: string): { base: string; fragment: string } {
  const slash = q.lastIndexOf("/");
  if (slash === -1) return { base: "", fragment: q };
  return { base: q.slice(0, slash + 1), fragment: q.slice(slash + 1) };
}

/** Case-insensitive subsequence fuzzy match. Returns a score (higher = better)
 * and the matched character indices for highlighting, or null when the query
 * isn't a subsequence of the name. Rewards contiguous runs and matches at word
 * boundaries; lightly prefers shorter names. */
function fuzzyMatch(text: string, query: string): { score: number; hits: number[] } | null {
  if (!query) return { score: 0, hits: [] };
  const t = text.toLowerCase();
  const q = query.toLowerCase();
  const hits: number[] = [];
  let ti = 0;
  let score = 0;
  let prev = -2;
  for (const c of q) {
    const found = t.indexOf(c, ti);
    if (found === -1) return null;
    if (found === prev + 1) score += 3; // contiguous run
    if (found === 0 || /[\s\-_./]/.test(t[found - 1] ?? "")) score += 2; // word boundary
    score += 1;
    hits.push(found);
    prev = found;
    ti = found + 1;
  }
  return { score: score - t.length * 0.01, hits };
}

/** Folder name with the fuzzy-matched characters emphasised. */
function HighlightedName({ name, hits }: { name: string; hits: number[] }) {
  if (hits.length === 0) return <>{name}</>;
  const set = new Set(hits);
  return (
    <>
      {[...name].map((ch, i) =>
        set.has(i) ? (
          // biome-ignore lint/suspicious/noArrayIndexKey: characters are positional
          <span key={i} className="czkawka-browser-match">
            {ch}
          </span>
        ) : (
          ch
        ),
      )}
    </>
  );
}

interface BrowseResponse {
  path: string;
  parent: string | null;
  imageCount: number;
  dirs: { name: string; path: string }[];
}

/** Inline directory browser. It follows the `query` (whatever is typed in the
 * add field) so typing a path and navigating the tree stay in sync: clicking a
 * folder writes it back to the field via `onPick`. Relative paths resolve
 * server-side against the launch dir. */
function DirBrowser({
  query,
  onPick,
  onAdd,
}: {
  query: string;
  onPick: (path: string) => void;
  onAdd: (path: string) => void;
}) {
  const [browse, setBrowse] = useState<BrowseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastBase = useRef<string | null>(null);

  const { base, fragment } = splitQuery(query.trim());

  const navigate = useCallback(async (path: string) => {
    setLoading(true);
    try {
      const res = await getJson<BrowseResponse>(
        `/api/czkawka/browse?path=${encodeURIComponent(path)}`,
      );
      setBrowse(res);
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err, "Could not read directory"));
    } finally {
      setLoading(false);
    }
  }, []);

  // Follow the typed path's *directory* portion (debounced). The fragment after
  // the last slash only filters the listing client-side, so it never triggers a
  // re-fetch — typing feels instant and clicking a folder (which appends `/`)
  // descends into it.
  useEffect(() => {
    const t = setTimeout(() => {
      if (base === lastBase.current) return;
      lastBase.current = base;
      void navigate(base); // empty → server defaults to the launch dir
    }, 250);
    return () => clearTimeout(t);
  }, [base, navigate]);

  // Fuzzy-filter the current listing by the trailing fragment, best match first.
  const filtered = useMemo(() => {
    if (!browse) return [];
    return browse.dirs
      .map((d) => ({ d, m: fuzzyMatch(d.name, fragment) }))
      .filter((x): x is { d: (typeof browse.dirs)[number]; m: { score: number; hits: number[] } } =>
        Boolean(x.m),
      )
      .sort((a, b) => b.m.score - a.m.score)
      .map((x) => ({ ...x.d, hits: x.m.hits }));
  }, [browse, fragment]);

  return (
    <div className={cn("czkawka-browser", loading && "is-loading")}>
      <div className="czkawka-browser-bar">
        <button
          type="button"
          className="czkawka-browser-up"
          disabled={!browse?.parent}
          onClick={() => browse?.parent && onPick(`${browse.parent}/`)}
          title="Parent directory"
        >
          ↑
        </button>
        <span className="czkawka-browser-path" title={browse?.path}>
          {browse ? shortPath(browse.path) : "…"}
          {browse && (
            <span className="czkawka-browser-count">
              {" "}
              · {browse.imageCount} image{browse.imageCount === 1 ? "" : "s"}
            </span>
          )}
        </span>
        <button
          type="button"
          className="btn btn-sm"
          disabled={!browse}
          onClick={() => browse && onAdd(browse.path)}
          title="Add this folder to the comparison"
        >
          Add this
        </button>
      </div>
      <div className="czkawka-browser-list">
        {error && <div className="czkawka-browser-empty">{error}</div>}
        {!error && browse && browse.dirs.length === 0 && (
          <div className="czkawka-browser-empty">No sub-folders</div>
        )}
        {!error && browse && browse.dirs.length > 0 && filtered.length === 0 && (
          <div className="czkawka-browser-empty">No folders match “{fragment}”</div>
        )}
        {!error &&
          filtered.map((d) => (
            <div key={d.path} className="czkawka-browser-row">
              <button
                type="button"
                className="czkawka-browser-name"
                onClick={() => onPick(`${d.path}/`)}
                title={`Open ${d.path}`}
              >
                <span className="czkawka-browser-folder">📁</span>
                <HighlightedName name={d.name} hits={d.hits} />
              </button>
              <button
                type="button"
                className="czkawka-browser-add"
                onClick={() => onAdd(d.path)}
                title="Add this folder"
              >
                +
              </button>
            </div>
          ))}
      </div>
    </div>
  );
}

/** Directory list for multi-dir comparison: add/remove dirs, mark at most one
 * as the czkawka-style reference ("which images match this dir"), and toggle
 * recursive sub-folder scanning per directory (the SUB button on each row). */
export function DirsPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dirs = useCzkawkaStore((s) => s.dirs);
  const targetDir = useCzkawkaStore((s) => s.targetDir);
  const dirsDirty = useCzkawkaStore((s) => s.dirsDirty);
  const addDir = useCzkawkaStore((s) => s.addDir);
  const removeDir = useCzkawkaStore((s) => s.removeDir);
  const setReferenceDir = useCzkawkaStore((s) => s.setReferenceDir);
  const setRecursiveDir = useCzkawkaStore((s) => s.setRecursiveDir);

  const [input, setInput] = useState("");
  const [browseOpen, setBrowseOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    }
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, onClose]);

  if (!open) return null;

  // New dirs come in non-recursive; recursion is toggled per row afterward.
  const add = async (path: string) => {
    if (!path.trim() || busy) return;
    setBusy(true);
    const ok = await addDir(path.trim(), false);
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
          Toggle <strong>SUB</strong> on a folder to also scan its sub-folders.
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
              <button
                type="button"
                className={cn("czkawka-dirs-sub", d.recursive && "is-active")}
                title={
                  d.recursive
                    ? "Scanning sub-folders — click to scan this folder only"
                    : "Also scan every sub-folder (skips .reorder-cache and hidden folders)"
                }
                onClick={() => setRecursiveDir(d.path, !d.recursive)}
              >
                SUB
              </button>
              <span className="czkawka-dirs-path" title={d.path}>
                {shortPath(d.path)}
                {d.path === targetDir && <span className="czkawka-dirs-this"> (this folder)</span>}
              </span>
              <button
                type="button"
                className="czkawka-dirs-remove"
                onClick={() => removeDir(d.path)}
                title="Remove from comparison"
                disabled={dirs.length <= 1}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
        <div className="czkawka-dirs-add">
          <input
            type="text"
            value={input}
            placeholder="Path (absolute or relative to this folder)"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void add(input);
            }}
          />
          <button className="btn btn-sm" onClick={() => void add(input)} disabled={busy}>
            {busy ? "Checking…" : "Add"}
          </button>
        </div>
        <div className="czkawka-dirs-add-opts">
          <button
            type="button"
            className={cn("czkawka-dirs-browsebtn", browseOpen && "is-active")}
            onClick={() => setBrowseOpen((v) => !v)}
          >
            {browseOpen ? "Hide browser" : "Browse…"}
          </button>
        </div>
        {browseOpen && <DirBrowser query={input} onPick={setInput} onAdd={(p) => void add(p)} />}
        {dirsDirty && (
          <p className="czkawka-dirs-dirty">Changes apply on the next Run Comparison.</p>
        )}
      </div>
    </>
  );
}
