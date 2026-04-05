Default to Bun, not Node.

- `bun run start.ts <dir>` — launch (builds client, opens browser, pre-generates thumbnails)
- `bun build src/client/index.tsx --outdir dist --minify` — test client compilation
- `bun install`, `bun test`, `biome check .`
- Rust: `cd rust/cluster-tool && cargo build --release` and `cd rust/group-similarity && cargo build --release`

## What This App Does

Local macOS tool for organizing image directories. Browser UI with four modes:

1. **Reorder** — drag-and-drop, group, rename to sequential numbering, organize into subfolders
2. **Cluster** — CLIP/PE-Core/DINOv3 visual clustering to discover photoshoot sets
3. **Cluster Compare** — side-by-side clustering runs for weight tuning
4. **Merge Suggestions** — DINOv3 patch matching finds groups likely to belong together

Workflow is iterative: cluster → accept groups → reorder/rename → re-cluster.

## Architecture

### Server (`start.ts` → `src/server.ts`)
- `Bun.serve()` with manual routing in `handleAPI()`. No framework.
- `src/rename.ts` — filesystem ops (list, rename, undo, organize, folder save, crash recovery). Two-phase rename with write-ahead manifest in `.reorder-pending.json`. `withRenameLock` serializes all FS-mutating ops.
- `src/thumbnails.ts` — Sharp WebP thumbnails, content-addressable cache keyed by `inode-size`
- `src/cluster.ts` — orchestrates Python/Rust subprocesses, parses linkage tree, TF-IDF auto-naming, contact sheets, merge suggestions
- `src/log.ts` — `log()`, `logError()`, `logData()`, `logBlock()` → `.reorder-log` (truncated to 50k lines on startup)

### Client (`src/client/`)
- React 19 SPA, built with `Bun.build()` (no Vite)
- Routing: `useRouter` hook (pushState-based). Paths: `/reorder`, `/cluster`, `/cluster-compare`, `/merge-suggestions`
- Shell: `AppShell` in `index.tsx` → `AppShellHeader` + mode-specific view
- CSS: per-concern files in `src/client/styles/`, copied to `dist/` at build time
- DnD: `@dnd-kit/core` + `@dnd-kit/sortable`. Virtualization: `@tanstack/react-virtual`
- Debug: all Zustand stores on `window.__stores`

### Zustand Stores (`src/client/stores/`)
`imageStore`, `selectionStore`, `dndStore`, `groupStore`, `folderStore`, `uiStore`, `clusterStore`, `mergeSuggestionsStore`. Read the files for shape — they're the source of truth.

### Clustering Pipeline

```
Stage 1: Python (scripts/extract_features.py)
  CLIP ViT-B/32, DINOv2, DINOv3, PE-Core L/G, color histograms — all on MPS GPU.
  Per-model version keys (MODEL_VERSIONS dict): only changed models re-extract.
  Content-hash cache (blake2b of first 16KB + filesize) survives renames.
  --models forces re-extract; --required only extracts listed models if missing.
  → .reorder-cache/{clip_embeddings.npz, clip_hash_cache.npz, *.filenames.json, dinov3_patches.npy}

Stage 2: Rust (rust/cluster-tool/)
  Ward's linkage (NNC) matching scipy exactly. Parallel via rayon.
  Pre-seeds confirmed reorder groups as real clusters (true centroid/size/variance).
  Weighted blend of per-model cosine distances (--clip-weight, --dinov3-weight, etc.)
  Optional: blend in precomputed patch distance matrix (--dist-matrix)
  → .reorder-cache/linkage_tree.bin

Stage 2b: Rust (rust/group-similarity/)
  Two modes: "merge-suggestions" (pairwise group scoring for Merge mode)
             "dist-matrix" (condensed matrix for patch-weighted clustering)
  DINOv3 patch matching: for each image pair, max-pool 7x7 cosine sims

Stage 3: Bun (src/cluster.ts)
  Re-cuts cached linkage tree — three modes: fixed N, distance threshold, HDBSCAN-style adaptive
  TF-IDF auto-naming: CLIP × 334-term vocabulary, z-score ranking
  Contact sheets via Sharp (4×3 grid, 400px thumbs)
```

### Key Patterns

- **Content-hash cache** — renames don't invalidate the ~5min extraction
- **Pre-seeded groups** — Rust linkage bootstraps confirmed groups as initialized clusters, not zero-distance hacks
- **Sorted merge steps** — NNC emits in execution order; sorted before tree-cutting to match scipy
- **Patch-dist cache** — `patch_dist_matrix.bin` reused if newer than patches + filenames
- **SSE streaming** — `/api/cluster` and `/api/cluster/extract` return `text/event-stream` with `progress`/`result`/`error` events. Reconnect via `/api/cluster/progress`.
- **Stale tree detection** — client tracks `treeStale` after group changes, prompts re-run
- **TF-IDF caching** — globalAvg/globalStd invalidated only when embeddings change
- **Rename safety** — two-phase with manifest; `recoverPendingRename` runs at startup inside the lock
- **Group atomicity** — server remaps groups in the same lock as renames; client reloads after save/undo
- **Debounced group persist** — 300ms; `flushGroupPersist()` for critical paths
- **Folder mode change detection** — disk snapshot + computed `hasChanges` against local state
- **Client-side routing** — server returns `index.html` for any non-API, non-asset path

## Python Environment

Clustering requires a venv at `/tmp/imgcluster-env/` with `torch torchvision open-clip-torch pillow numpy transformers`.
Override path via `CLUSTER_PYTHON` env var.

```sh
uv venv /tmp/imgcluster-env
source /tmp/imgcluster-env/bin/activate
uv pip install torch torchvision open-clip-torch pillow numpy transformers
```

DINOv3 weights are loaded from `$DINOV3_WEIGHTS` (default `/tmp/dinov3-weights/facebook/dinov3-vitb16-pretrain-lvd1689m`).
