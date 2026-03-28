Default to using Bun instead of Node.js.

- `bun run start.ts <directory>` to launch
- `bun build src/client/index.tsx --outdir dist --minify` to test client compilation
- `bun install` for dependencies, `bun test` for tests

## What This App Does

A local macOS tool for reordering images in a directory via drag-and-drop. Opens a browser UI, lets the user rearrange image cards, then renames files with sequential numbering to persist the new order. Also supports grouping images, organizing groups into subfolders, and undo.

## Architecture

**Server** (`start.ts` → `src/server.ts`):
- `Bun.serve()` with manual routing in `handleAPI()`. No framework.
- `src/rename.ts` — all filesystem operations (list, rename, undo, organize, crash recovery)
- `src/thumbnails.ts` — Sharp-based WebP thumbnail generation with content-addressable cache
- Background thumbnail pre-generation on startup (8 concurrent workers)
- HTTP caching: `no-cache` + ETag (inode+size) on image and thumbnail endpoints

**Client** (`src/client/`):
- React 19 SPA, built with `Bun.build()` (no Vite, no React compiler)
- State: Zustand stores in `stores/` (imageStore, selectionStore, dndStore, groupStore, uiStore)
- Hooks in `hooks/` — extracted from App.tsx (useGridLayout, useKeyboardShortcuts, useGroupOperations, useDragHandlers)
- Components in `components/` (SortableCard, SortableGroupCard, GroupPopover, Lightbox, SearchBar, Toolbar, modals)
- Utilities in `utils/` (helpers, gridItems, reorder)
- `App.tsx` is a thin shell: store subscriptions, virtualization, DndContext, render loop
- Drag-and-drop: `@dnd-kit/core` + `@dnd-kit/sortable` with `rectSortingStrategy`
- Virtualization: `@tanstack/react-virtual` row virtualizer for 13k+ images
- Grid layout: CSS `auto-fill` drives column count; JS reads it via `getComputedStyle` on a hidden measuring row (callback ref pattern)
- In-app search: Cmd+F intercepted, floating search bar with match highlighting

**CSS** (`src/client/styles.css`):
- Plain CSS with custom properties (dark theme). No framework.
- Grid rows use `repeat(auto-fill, minmax(160px, 1fr))` — CSS owns the column layout
- `.grid-measure-row` is the hidden element JS reads to determine column count and row height

## Key Patterns

- **Thumbnail cache**: content-addressable using `inode-size.webp` keys — renames don't invalidate cache, no remapping needed. Orphans cleaned after pre-generation. Cache cleared on shutdown for directories ≤250 images.
- **Rename safety**: two-phase rename with write-ahead manifest (`.reorder-pending.json`). `withRenameLock` serializes all filesystem-mutating operations. Crash recovery on startup completes interrupted renames (including partial step-1). `computeRenames` rejects duplicate filenames in the order array.
- **Group atomicity**: Server remaps groups in the same `withRenameLock` call as file renames (save, undo, and crash recovery). Client reloads groups from server after save/undo rather than remapping locally. `flattenOrder` deduplicates to prevent expanded-group items appearing twice in the order array.
- **Memoization**: event handlers in App.tsx use ref+useCallback pattern for stable references (see `handleGridItemClick`, `handleDragStart`, `handleDragEnd`). Group operations use `getState()` for store access since they're always called at event time.
- **Browser cache busting**: `imageVersion` counter in imageStore bumps on `fetchImages()` (after save/undo). URL helpers append `?v=N` to bust the browser's in-memory `<img>` cache. HTTP cache correctness is handled by ETag + `no-cache`.
- `postJson(url, body)` in `utils/helpers.ts` for all JSON POST fetches
- `wasJustDragged()` suppresses click events that fire at the end of drag gestures
- `updateGroups(fn)` in groupStore auto-prunes empty groups and skips persist on no-op
- Group persistence is debounced (300ms) to the server via `POST /api/groups`

## File Structure

```
start.ts                        Entry point (CLI, build, server start)
src/
  server.ts                     HTTP server + API routes
  rename.ts                     Filesystem operations + crash recovery
  thumbnails.ts                 Sharp thumbnail generation + content-addressable cache
  client/
    index.html / index.tsx      HTML shell + React mount
    App.tsx                     Thin shell (store wiring, virtualization, render)
    types.ts                    Shared interfaces
    styles.css                  All styles
    stores/                     Zustand stores (5 files)
    hooks/                      Extracted hooks (4 files)
    components/                 UI components (12 files)
    utils/                      Helpers, grid item computation, reorder logic
```
