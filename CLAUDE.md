Default to using Bun instead of Node.js.

- `bun run start.ts <directory>` to launch
- `bun build src/client/index.tsx --outdir dist --minify` to test client compilation
- `bun install` for dependencies, `bun test` for tests
- Rust cluster tool: `cd rust/cluster-tool && cargo build --release`

## What This App Does

A local macOS tool for reordering images in a directory via drag-and-drop. Opens a browser UI with two modes:

1. **Reorder mode**: drag-and-drop image cards, group them, rename files with sequential numbering, organize into subfolders
2. **Cluster mode**: CLIP-based visual clustering to identify photoshoot sets, review/accept/split/merge clusters, convert to groups

Designed for iterative workflow: cluster → accept groups → reorder/rename → re-cluster to find stragglers → repeat.

## Architecture

### Server (`start.ts` → `src/server.ts`)
- `Bun.serve()` with manual routing in `handleAPI()`. No framework.
- `src/rename.ts` — filesystem operations (list, rename, undo, organize, folder save, crash recovery)
- `src/thumbnails.ts` — Sharp WebP thumbnails with content-addressable cache (`inode-size.webp`)
- `src/cluster.ts` — clustering orchestration: spawns Python/Rust, parses linkage tree, TF-IDF auto-naming, contact sheets
- `src/log.ts` — file+console logging with `log()`, `logError()`, `logData()`, `logBlock()`. Writes to `.reorder-log` in target dir.
- `start.ts` — entry point: validates dir, runs `Bun.build()`, copies CSS from `src/client/styles/`, starts server, opens browser, pre-generates thumbnails

### Client (`src/client/`)
- React 19 SPA, built with `Bun.build()` (no Vite/bundler config)
- Routing: `useRouter` hook (`src/client/hooks/useRouter.ts`) — pushState-based, `/reorder` and `/cluster` paths
- Shell: `AppShell` in `index.tsx` renders `AppShellHeader` + mode-specific content (`<App />` or `<ClusterView />`)
- CSS: split into per-concern files in `src/client/styles/` (base, buttons, card, cluster, grid, group, header, lightbox, loading, modal, mode-toggle, search, toast). Copied to `dist/` at build time.
- Drag-and-drop: `@dnd-kit/core` + `@dnd-kit/sortable`
- Virtualization: `@tanstack/react-virtual` for 13k+ images in reorder mode
- Debug: all stores exposed on `window.__stores` for console access

### Zustand Stores (`src/client/stores/`)

| Store | Purpose |
|-------|---------|
| `imageStore` | Image list, original order, `hasChanges`, `imageVersion` for cache busting |
| `selectionStore` | Multi-select with click/shift-range/select-all in reorder mode |
| `dndStore` | Active drag state, drag-over-group tracking |
| `groupStore` | Named image groups, debounced server persist, groups-enabled toggle |
| `folderStore` | Folder mode: local folder/root state, disk snapshot, change detection, reorder/rename/dissolve/move ops |
| `uiStore` | Lightbox, modals, saving state, toast, undo, target dir, header subtitle |
| `clusterStore` | Cluster data, SSE loading, merge/split/accept, image selection, lightbox, stale tree detection |

### Clustering Pipeline (three stages)

```
Stage 1: Python (scripts/extract_features.py)
  → CLIP ViT-B/32 + color histogram extraction on MPS GPU
  → Cached by content hash (blake2b of first 16KB + filesize) — survives file renames
  → Output: .reorder-cache/clip_embeddings.npz + .filenames.json
  → Requires: /tmp/imgcluster-env/bin/python3 (venv with torch, open-clip-torch, pillow, numpy)

Stage 2: Rust (rust/cluster-tool/src/main.rs)
  → Ward's linkage (NNC algorithm) matching scipy exactly
  → Pre-seeds confirmed reorder groups as real clusters (true centroid/size/variance)
  → Uses cosine distances + scipy's _ward update formula (square-sqrt)
  → Parallel distance computation via rayon (~1.2s for 7000 images)
  → Output: linkage tree binary + cluster assignments JSON

Stage 3: Bun (src/cluster.ts)
  → Re-cuts cached linkage tree at any N (instant, no subprocess)
  → TF-IDF auto-naming: CLIP embeddings × 334-term vocabulary, z-score ranking
  → Contact sheet generation via Sharp (4×3 grid, 400×400 thumbs)
```

### Clustering API Endpoints

All routes are in `handleAPI()` in `src/server.ts`. Reorder/folder routes are standard CRUD — read the file. Cluster-specific routes:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/cluster` | Full pipeline (SSE progress stream). Body: `{nClusters?}` |
| GET | `/api/cluster/status` | `{running: boolean}` — prevents double-triggering |
| GET | `/api/cluster/cache-status` | `{cached: boolean}` — check if linkage tree exists |
| POST | `/api/cluster/recut` | Re-cut cached tree. Body: `{nClusters}`. Instant. |
| POST | `/api/cluster/contact-sheet` | Generate thumbnail grid. Body: `{filenames, clusterName}` |

### Disk Files (in target image directory)

| File | Purpose |
|------|---------|
| `.reorder-groups.json` | Persisted groups (shared between reorder and cluster modes) |
| `.reorder-log` | Server operation log (truncated to 50k lines on startup) |
| `.reorder-cache/clip_embeddings.npz` | CLIP+color feature arrays (filename-ordered, for Rust) |
| `.reorder-cache/clip_hash_cache.npz` | Feature cache keyed by content hash (survives renames) |
| `.reorder-cache/clip_embeddings.filenames.json` | Filename→row mapping for the npz |
| `.reorder-cache/text_embeddings.json` | Precomputed CLIP text embeddings (334 terms) |
| `.reorder-cache/linkage_tree.bin` | Binary linkage tree (header: n_images, n_pre_merges, n_steps; then merge step array) |
| `.reorder-cache/contact_sheets/` | Generated contact sheet JPEGs for "Ask Claude" |

## Key Patterns

- **Content-hash cache**: `blake2b(first 16KB + filesize)` keys embeddings, so renaming files doesn't invalidate the ~5min extraction
- **Pre-seeded groups**: Rust linkage starts with confirmed groups as properly-initialized clusters (real centroid/size/variance), not zero-distance hacks
- **Sorted merge steps**: NNC produces merges in execution order (not distance order); they're sorted before tree-cutting to match scipy's behavior
- **SSE streaming**: `/api/cluster` returns `text/event-stream` with `progress`, `result`, and `error` events
- **Stale tree detection**: Client tracks `treeStale` flag after group changes; shows banner prompting re-run
- **TF-IDF caching**: globalAvg/globalStd arrays cached in `_tfidfStatsCache`, invalidated only when embeddings change
- **Rename safety**: two-phase rename with write-ahead manifest. `withRenameLock` serializes filesystem operations
- **Group atomicity**: Server remaps groups in same lock as renames. Client reloads after save/undo
- **Browser cache busting**: `imageVersion` counter + `?v=N` URL parameter
- **Debounced group persist**: `groupStore` debounces server writes (300ms), with `flushGroupPersist()` for critical paths
- **Folder mode change detection**: `folderStore` keeps a disk snapshot and computes `hasChanges` against local state
- **Client-side routing**: pushState-based via `useRouter` hook — server returns `index.html` for all non-API/non-asset paths

## Python Environment

The clustering Python scripts require a venv at `/tmp/imgcluster-env/` with:
```
torch torchvision open-clip-torch pillow numpy
```
Set `CLUSTER_PYTHON` env var to override the Python path (default: `/tmp/imgcluster-env/bin/python3`).

Create with: `uv venv /tmp/imgcluster-env && source /tmp/imgcluster-env/bin/activate && uv pip install torch torchvision open-clip-torch pillow numpy`
