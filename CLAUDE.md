Default to Bun, not Node.

- `bun run start.ts <dir>` — launch (builds client, opens browser, pre-generates thumbnails)
- `bun build src/client/index.tsx --outdir dist --minify` — test client compilation
- `bun install`, `bun run typecheck`, `bun run lint` (`bun run lint:fix` to auto-fix)
- Rust: `cargo build --release` from `rust/cluster-tool/` or `rust/group-similarity/` (workspace shares `rust/reorder-common`)

## What This App Does

Local macOS tool for organizing image directories. Browser UI with four modes:

1. **Reorder** — drag-and-drop, group, rename to sequential numbering, organize into subfolders
2. **Cluster** — CLIP/PE-Core/DINOv3 visual clustering to discover photoshoot sets
3. **Cluster Compare** — side-by-side clustering runs for weight tuning
4. **Merge Suggestions** — DINOv3 patch matching finds groups likely to belong together

Workflow is iterative: cluster → accept groups → reorder/rename → re-cluster.

## Architecture

The codebase is module-oriented. Each top-level directory under `src/` has an `index.ts` barrel that defines its public surface — prefer importing from the barrel over reaching into internals. Cross-cutting types live in `src/shared/types.ts`.

### `src/server/` — HTTP layer
- `index.ts` — `Bun.serve()` + a route dispatcher that walks an array of `RouteHandler`s and returns the first non-null response. No framework.
- `routes/` — one file per concern (`cluster`, `cluster-extract`, `cluster-scoped`, `constraints`, `delete`, `folders`, `groups`, `images`, `merge`, `nn`, `organize`, `rename`, `tree-nav`). Each exports a single `RouteHandler`.
- `middleware/` — `response` (json/mimeType helpers), `sse` (`sseResponse`, `subscribeProgressSSE` for re-attach), `cluster-job` (`runClusterJobSSE` wraps the 409-or-stream-progress pattern every long-running cluster route shares).
- Startup runs `recoverPendingRename` inside `withRenameLock`; any GET that depends on rename consistency holds the same lock.

### `src/fs/` — filesystem ops
Two-phase rename with write-ahead manifest at `.reorder-pending.json`. `withRenameLock` (in `lock.ts`) is a single mutex serializing every FS-mutating op AND any read that must not observe a half-applied rename. Split into `atomic-json`, `content-hashes`, `folder-save`, `groups`, `images`, `organize`, `paths`, `recovery`, `rename`, `tags`, `trash`. `paths.ts` is the single source for cache/log/manifest paths.

### `src/cluster/` — clustering orchestration
Drives Python/Rust subprocesses; parses linkage tree; TF-IDF auto-naming; contact sheets; merge suggestions. Notable modules: `pipeline` (extract → linkage → naming), `linkage` (tree cuts: fixed-N / threshold / adaptive), `embeddings` (per-model NPZ load + content-hash mapping), `tfidf`, `constraints` (cannot-link + group-lock), `distance-matrices` (patch + re-rank caches), `imported` (bypass for externally-provided clusters), `scoped` (sub-cluster on a subset), `progress` (broadcast channel for SSE re-attach), `job-mutex` (single in-flight compute slot, abortable via SIGINT). `subprocess.ts` centralises spawning. `binaries.ts` resolves Python + Rust binary paths.

The compute-job mutex (`cluster/job-mutex.ts`) is distinct from the FS lock: it guards CPU/GPU work and serves 409 to concurrent compute requests, while `withRenameLock` guards on-disk consistency.

### `src/client/` — React 19 SPA, built with `Bun.build()` (no Vite)
- Routing: `useRouter` hook (pushState). Paths: `/reorder`, `/cluster`, `/cluster-compare`, `/merge-suggestions`. Server returns `index.html` for any non-API, non-asset path.
- Shell: `AppShell` in `index.tsx` → `components/header/AppShellHeader` + mode-specific view.
- Components grouped by mode/concern: `cluster/`, `cluster-compare/`, `merge-suggestions/`, `reorder/`, `header/`, `shared/`.
- `api/` — `client.ts` (`getJson`/`postJson`/`putJson`/`deleteJson`/`postRaw` fetch wrappers; centralised error extraction) and `sse.ts` (`consumeSSE` parser, `startSSE` POSTs and returns either the stream or a 409-conflict signal). Use these instead of bare `fetch`.
- DnD: `@dnd-kit/core` + `@dnd-kit/sortable`. Virtualization: `@tanstack/react-virtual`.
- CSS: per-concern files in `src/client/styles/`, copied to `dist/` at build time.
- Debug: all Zustand stores attached to `window.__stores`.

### Zustand stores (`src/client/stores/`)
- `core/` — cross-mode primitives: `selectionStore` (multi-context: `reorder`, `cluster:images`, `cluster:merge`, `compare`, `expand`, `nn`, `trash`, plus per-row contexts), `modalStore`, `lightboxStore`, `toastStore`, `sessionStore`.
- `modes/cluster/` — split into `listStore` (current `clusterData`), `compareStore`, `expandStore`, `splitStore`, `metricsStore`, `interactionsStore`. `index.ts` is the barrel and installs a single subscription that clears compare/expand/metrics/selection state when the cluster *shape* (id set or per-cluster sizes) changes.
- Top-level: `imageStore`, `groupStore`, `folderStore`, `dndStore`, `mergeSuggestionsStore`, `constraintsStore`, `nnQueryStore`, `trashStore`. Read the files for shape — they're the source of truth.

### Clustering Pipeline

```
Stage 1: Python (scripts/extract_features.py)
  CLIP ViT-B/32, DINOv2, DINOv3, PE-Core L/G, color histograms — all on MPS GPU.
  Per-model version keys (MODEL_VERSIONS dict): only changed models re-extract.
  Content-hash cache (blake2b of first 16KB + filesize) survives renames.
  --models forces re-extract; --required only extracts listed models if missing.
  → .reorder-cache/{clip_embeddings.npz, clip_hash_cache.npz, *.filenames.json, dinov3_patches.npy}

Stage 2: Rust (rust/cluster-tool/, modules: cli/io/distances/linkage/tree)
  Ward's linkage (NNC) matching scipy exactly. Parallel via rayon.
  Pre-seeds confirmed reorder groups as real clusters (true centroid/size/variance).
  Weighted blend of per-model cosine distances (--clip-weight, --dinov3-weight, etc.)
  Optional: blend in precomputed patch distance matrix (--dist-matrix)
  → .reorder-cache/linkage_tree.bin

Stage 2b: Rust (rust/group-similarity/, modes: merge-suggestions / dist-matrix)
  DINOv3 patch matching: for each image pair, max-pool 7x7 cosine sims
  Shared parsing/types live in the rust/reorder-common crate.

Stage 3: Bun (src/cluster/pipeline.ts + linkage.ts)
  Re-cuts cached linkage tree — three modes: fixed N, distance threshold, HDBSCAN-style adaptive
  TF-IDF auto-naming: CLIP × 334-term vocabulary, z-score ranking
  Contact sheets via Sharp (cluster/contact-sheets.ts): justified-row layout on a 2000px-wide canvas
```

### Key Patterns

- **Two locks, two purposes** — `withRenameLock` (fs/lock.ts) serializes disk state; cluster `job-mutex` serializes compute. Don't conflate them.
- **Content-hash cache** — renames don't invalidate the ~5min extraction
- **Pre-seeded groups** — Rust linkage bootstraps confirmed groups as initialized clusters, not zero-distance hacks
- **Patch-dist cache** — `patch_dist_matrix.bin` reused if newer than patches + filenames
- **SSE** — long-running routes use the `runClusterJobSSE` middleware (returns 409 if busy, otherwise streams `progress`/`result`/`error` events). Clients re-attach to a running job via `/api/cluster/progress` (server replays last progress + tails the broadcast channel).
- **Stale tree detection** — client tracks `treeStale` after group changes, prompts re-run
- **Rename safety** — two-phase with manifest; `recoverPendingRename` runs at startup inside the lock
- **Imported-clusters bypass** — `/api/cluster/import` stores clusters at `.reorder-cache/imported_clusters.json` and takes precedence over linkage-tree re-cuts on load
- **Cluster-shape subscription** — selection / compare / expand / metrics state is auto-cleared when cluster ids or sizes change (wired in `stores/modes/cluster/index.ts`)
- **Mixed response envelopes** — server returns `{success}`, `{ok}`, and raw payloads depending on route. `client.ts` does not normalise; only error extraction is centralised.

## Python Environment

Clustering requires a venv at `~/.venvs/imgcluster-env/` with `torch torchvision open-clip-torch pillow numpy transformers`.
Override path via `CLUSTER_PYTHON` env var. (Note: `/tmp` is avoided because macOS clears old files from it, wiping the venv.)

```sh
uv venv ~/.venvs/imgcluster-env
source ~/.venvs/imgcluster-env/bin/activate
uv pip install torch torchvision open-clip-torch pillow numpy transformers
```

DINOv3 weights are loaded from `$DINOV3_WEIGHTS` (default `~/.cache/dinov3-weights/facebook/dinov3-vitb16-pretrain-lvd1689m`). Avoid `/tmp` — macOS purges it and wipes the config files out from under the safetensors blob.
