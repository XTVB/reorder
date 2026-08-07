// Store for the czkawka duplicate-compare page. The server owns group state
// (persisted in .reorder-cache/czkawka_session.json); every mutating call
// returns the full updated state and the client replaces its copy wholesale.
// Skip / keep-both are purely client-side navigation, so undo interleaves a
// local history (index moves) with server undo (files restored from Trash).
//
// Comparison can span multiple directories (the launch dir always included);
// one may be marked as the czkawka-style reference, which scopes matching to
// "images that match something in the reference dir".

import { create } from "zustand";
import { getJson, postJson } from "../api/client.ts";
import { consumeSSE, startSSE } from "../api/sse.ts";
import type {
  CzkawkaDirEntry,
  CzkawkaImage,
  CzkawkaOperation,
  CzkawkaRunResult,
  CzkawkaStateResponse,
} from "../types.ts";
import { getErrorMessage } from "../utils/helpers.ts";
import { useToastStore } from "./core/toastStore.ts";

export const HASH_ALGS = [
  "DoubleGradient",
  "Gradient",
  "Mean",
  "VertGradient",
  "Blockhash",
  "Median",
] as const;
export const IMAGE_FILTERS = ["Lanczos3", "Nearest", "Triangle", "Gaussian", "CatmullRom"] as const;
export const HASH_SIZES = [8, 16, 32, 64] as const;

export type HashAlg = (typeof HASH_ALGS)[number];
export type ImageFilter = (typeof IMAGE_FILTERS)[number];
export type HashSize = (typeof HASH_SIZES)[number];

/** Absolute path of a group image — the identity used everywhere client-side. */
export const imgPath = (img: CzkawkaImage): string => `${img.dir}/${img.filename}`;

interface RunConfig {
  hashAlg: HashAlg;
  imageFilter: ImageFilter;
  hashSize: HashSize;
  similarity: number;
}

const CONFIG_KEY = "czkawka_run_config";

function loadConfig(): RunConfig {
  const fallback: RunConfig = {
    hashAlg: "DoubleGradient",
    imageFilter: "Lanczos3",
    hashSize: 16,
    similarity: 15,
  };
  try {
    const raw = localStorage.getItem(CONFIG_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Partial<RunConfig>;
    return {
      hashAlg: HASH_ALGS.includes(parsed.hashAlg as HashAlg)
        ? (parsed.hashAlg as HashAlg)
        : fallback.hashAlg,
      imageFilter: IMAGE_FILTERS.includes(parsed.imageFilter as ImageFilter)
        ? (parsed.imageFilter as ImageFilter)
        : fallback.imageFilter,
      hashSize: HASH_SIZES.includes(parsed.hashSize as HashSize)
        ? (parsed.hashSize as HashSize)
        : fallback.hashSize,
      similarity:
        Number.isFinite(parsed.similarity) && (parsed.similarity as number) >= 0
          ? (parsed.similarity as number)
          : fallback.similarity,
    };
  } catch {
    return fallback;
  }
}

interface HistoryEntry {
  kind: "local" | "server";
  indexBefore: number;
  deletedCount: number;
}

interface RunStats {
  cached: number;
  computed: number;
  failed: number;
}

interface CzkawkaState extends RunConfig {
  groups: CzkawkaImage[][];
  /** Server-side restorable actions available to undo. */
  undoDepth: number;
  undoing: boolean;
  /** Initial GET /groups completed (restores a persisted session). */
  loaded: boolean;
  loading: boolean;
  progress: string | null;
  error: string | null;
  computeTimeMs: number | null;
  runStats: RunStats | null;

  /** Absolute path of the launch dir (known after the first server response). */
  targetDir: string | null;
  /** Directories in the comparison; the target dir is always present. */
  dirs: CzkawkaDirEntry[];
  /** Dirs changed since the displayed results were computed. */
  dirsDirty: boolean;

  /** May equal groups.length — that's the done screen. */
  currentIndex: number;
  history: HistoryEntry[];
  trashedCount: number;

  /** Paths excluded from Y/W ranking (right-click a thumbnail). */
  excludedPaths: Set<string>;

  sliderMode: boolean;

  setHashAlg: (v: HashAlg) => void;
  setImageFilter: (v: ImageFilter) => void;
  setHashSize: (v: HashSize) => void;
  setSimilarity: (v: number) => void;
  setSliderMode: (v: boolean) => void;

  /** Validate via the server, then add to the dir list. Returns success. */
  addDir: (path: string, recursive?: boolean) => Promise<boolean>;
  removeDir: (path: string) => void;
  /** Mark one dir as reference (or none). */
  setReferenceDir: (path: string | null) => void;
  /** Toggle recursive sub-folder scanning for a dir. */
  setRecursiveDir: (path: string, recursive: boolean) => void;

  loadExisting: () => Promise<void>;
  runComparison: () => Promise<void>;

  goToGroup: (idx: number) => void;
  advance: () => void;
  goBack: () => void;

  /** POST operations to the server; replaces groups from the response.
   * Returns true on success. */
  applyOperations: (ops: CzkawkaOperation[], toast?: string) => Promise<boolean>;
  undo: () => Promise<void>;

  /**
   * Drop reorder-group members whose files this page trashed, plus any group
   * left empty. Deletes here stay undoable, so they deliberately leave group
   * membership in place — this is the explicit "done undoing, tidy up" step.
   * Confirms first: the prune is itself not undoable.
   */
  pruneDeletedFromGroups: () => Promise<void>;

  toggleExclude: (path: string) => void;
  clearExclusions: () => void;
}

function fromState(res: CzkawkaStateResponse) {
  return {
    groups: res.groups.map((g) => g.images),
    undoDepth: res.undoDepth,
    targetDir: res.targetDir,
    dirs: res.dirs,
  };
}

function saveConfig(state: CzkawkaState): void {
  const { hashAlg, imageFilter, hashSize, similarity } = state;
  try {
    localStorage.setItem(
      CONFIG_KEY,
      JSON.stringify({ hashAlg, imageFilter, hashSize, similarity }),
    );
  } catch {
    /* ignore */
  }
}

const toast = (message: string, type: "success" | "error" | "warning") =>
  useToastStore.getState().showToast(message, type);

export const useCzkawkaStore = create<CzkawkaState>((set, get) => {
  const setConfig = <K extends keyof RunConfig>(key: K, value: RunConfig[K]) => {
    set({ [key]: value } as Pick<CzkawkaState, K>);
    saveConfig(get());
  };

  return {
    ...loadConfig(),
    groups: [],
    undoDepth: 0,
    undoing: false,
    loaded: false,
    loading: false,
    progress: null,
    error: null,
    computeTimeMs: null,
    runStats: null,
    targetDir: null,
    dirs: [],
    dirsDirty: false,
    currentIndex: 0,
    history: [],
    trashedCount: 0,
    excludedPaths: new Set<string>(),
    sliderMode: false,

    setHashAlg: (v) => setConfig("hashAlg", v),
    setImageFilter: (v) => setConfig("imageFilter", v),
    setHashSize: (v) => setConfig("hashSize", v),
    setSimilarity: (v) => setConfig("similarity", Math.max(0, v)),
    setSliderMode: (v) => set({ sliderMode: v }),

    addDir: async (path, recursive = false) => {
      try {
        const res = await postJson<{ ok: true; path: string; imageCount: number }>(
          "/api/czkawka/check-dir",
          { path, recursive },
        );
        const { dirs } = get();
        if (dirs.some((d) => d.path === res.path)) {
          toast("Directory is already in the comparison", "warning");
          return false;
        }
        set({
          dirs: [...dirs, { path: res.path, reference: false, recursive }],
          dirsDirty: true,
        });
        toast(
          `Added ${res.path} (${res.imageCount} image${res.imageCount === 1 ? "" : "s"}${
            recursive ? ", recursive" : ""
          })`,
          "success",
        );
        return true;
      } catch (err) {
        toast(getErrorMessage(err, "Could not add directory"), "error");
        return false;
      }
    },

    removeDir: (path) => {
      set((s) => {
        // Keep at least one directory — an empty comparison has nothing to run.
        if (s.dirs.length <= 1) {
          toast("At least one directory is required", "warning");
          return {};
        }
        return { dirs: s.dirs.filter((d) => d.path !== path), dirsDirty: true };
      });
    },

    setReferenceDir: (path) => {
      set((s) => ({
        dirs: s.dirs.map((d) => ({ ...d, reference: d.path === path })),
        dirsDirty: true,
      }));
    },

    setRecursiveDir: (path, recursive) => {
      set((s) => ({
        dirs: s.dirs.map((d) => (d.path === path ? { ...d, recursive } : d)),
        dirsDirty: true,
      }));
    },

    loadExisting: async () => {
      if (get().loaded) return;
      try {
        const res = await getJson<CzkawkaStateResponse>("/api/czkawka/groups");
        set({ ...fromState(res), loaded: true });
      } catch {
        set({ loaded: true });
      }
    },

    runComparison: async () => {
      const { hashAlg, imageFilter, similarity, hashSize, dirs, loading } = get();
      if (loading) return;
      set({ loading: true, error: null, progress: "Starting..." });
      try {
        const start = await startSSE("/api/czkawka/run", {
          hashAlg,
          imageFilter,
          similarity,
          hashSize,
          dirs,
        });
        if (start.kind === "conflict") {
          set({ loading: false, progress: null, error: start.message });
          return;
        }
        await consumeSSE<CzkawkaRunResult>(start.response, {
          onProgress: (message) => set({ progress: message }),
          onResult: (data) => {
            set({
              groups: data.groups.map((g) => g.images),
              computeTimeMs: data.computeTimeMs,
              runStats: { cached: data.cached, computed: data.computed, failed: data.failed },
              undoDepth: 0,
              loaded: true,
              loading: false,
              progress: null,
              dirsDirty: false,
              currentIndex: 0,
              history: [],
              trashedCount: 0,
              excludedPaths: new Set<string>(),
            });
            if (data.failed > 0) {
              toast(`${data.failed} file(s) could not be hashed`, "warning");
            }
          },
          onError: (error) => set({ loading: false, error, progress: null }),
        });
        if (get().loading) {
          set({ loading: false, progress: null, error: "Stream ended unexpectedly" });
        }
      } catch (err) {
        set({ loading: false, progress: null, error: getErrorMessage(err, "Comparison failed") });
      }
    },

    goToGroup: (idx) => {
      const { groups } = get();
      if (idx >= 0 && idx <= groups.length) set({ currentIndex: idx });
    },

    advance: () => {
      set((s) => {
        if (s.currentIndex >= s.groups.length) return {};
        return {
          history: [
            ...s.history,
            { kind: "local" as const, indexBefore: s.currentIndex, deletedCount: 0 },
          ],
          currentIndex: s.currentIndex + 1,
        };
      });
    },

    goBack: () => {
      set((s) => (s.currentIndex > 0 ? { currentIndex: s.currentIndex - 1 } : {}));
    },

    applyOperations: async (ops, toastMsg) => {
      try {
        const res = await postJson<CzkawkaStateResponse & { deletedCount: number }>(
          "/api/czkawka/action",
          { operations: ops },
        );
        set((s) => ({
          ...fromState(res),
          // Keep any unapplied dir edits instead of the session's dirs.
          dirs: s.dirsDirty ? s.dirs : res.dirs,
          trashedCount: s.trashedCount + res.deletedCount,
          history: [
            ...s.history,
            {
              kind: "server" as const,
              indexBefore: s.currentIndex,
              deletedCount: res.deletedCount,
            },
          ],
          currentIndex: Math.min(s.currentIndex, res.groups.length),
          error: null,
        }));
        if (toastMsg) toast(toastMsg, "success");
        return true;
      } catch (err) {
        toast(getErrorMessage(err, "Action failed"), "error");
        return false;
      }
    },

    undo: async () => {
      const { history, undoDepth, undoing } = get();
      // Drop the keypress rather than queueing it: a queue would still hand the
      // server the same burst, just later. Holding `u` now walks the stack at
      // one request per round-trip.
      if (undoing) return;
      const last = history[history.length - 1];
      // No local history but the server journal has entries — happens after a
      // reload, since local history is in-memory. Fall through to server undo.
      if (!last && undoDepth === 0) {
        toast("Nothing to undo", "warning");
        return;
      }
      if (last?.kind === "local") {
        set({ history: history.slice(0, -1), currentIndex: last.indexBefore });
        return;
      }
      set({ undoing: true });
      try {
        const res = await postJson<CzkawkaStateResponse>("/api/czkawka/undo", {});
        set((s) => ({
          ...fromState(res),
          dirs: s.dirsDirty ? s.dirs : res.dirs,
          // Re-read history from current state: it may have changed while the
          // request was in flight (a local undo, or an apply).
          history: last ? s.history.filter((h) => h !== last) : s.history,
          trashedCount: Math.max(0, s.trashedCount - (last?.deletedCount ?? 0)),
          currentIndex: Math.min(last ? last.indexBefore : s.currentIndex, res.groups.length),
          error: null,
        }));
        const restored = last?.deletedCount ?? 0;
        toast(restored > 0 ? `Restored ${restored} file(s) from Trash` : "Undone", "success");
      } catch (err) {
        toast(getErrorMessage(err, "Undo failed"), "error");
      } finally {
        set({ undoing: false });
      }
    },

    pruneDeletedFromGroups: async () => {
      const { undoDepth } = get();
      const undoWarning =
        undoDepth > 0
          ? `\n\nYou still have ${undoDepth} undoable action(s). Undo restores files to disk, but once pruned they come back ungrouped.`
          : "";
      if (
        !confirm(
          `Remove deleted images from the reorder groups?\n\nThis drops group members whose files are no longer on disk, and any group left empty. It cannot be undone.${undoWarning}`,
        )
      ) {
        return;
      }
      try {
        const res = await postJson<{ removedImages: number; removedGroups: number }>(
          "/api/groups/prune",
          {},
        );
        if (res.removedImages === 0 && res.removedGroups === 0) {
          toast("Nothing to prune — every group member is on disk", "success");
          return;
        }
        const groupPart =
          res.removedGroups > 0
            ? `, removed ${res.removedGroups} empty group${res.removedGroups === 1 ? "" : "s"}`
            : "";
        toast(
          `Pruned ${res.removedImages} deleted image${res.removedImages === 1 ? "" : "s"} from groups${groupPart}`,
          "success",
        );
      } catch (err) {
        toast(getErrorMessage(err, "Prune failed"), "error");
      }
    },

    toggleExclude: (path) => {
      set((s) => {
        const next = new Set(s.excludedPaths);
        if (next.has(path)) next.delete(path);
        else next.add(path);
        return { excludedPaths: next };
      });
    },

    clearExclusions: () => {
      set((s) => (s.excludedPaths.size > 0 ? { excludedPaths: new Set<string>() } : {}));
    },
  };
});
