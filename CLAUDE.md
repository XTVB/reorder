Default to using Bun instead of Node.js.

- `bun run start.ts <directory>` to launch
- `bun build src/client/index.tsx --outdir dist --minify` to test client compilation
- `bun install` for dependencies, `bun test` for tests

## What This App Does

A local macOS tool for reordering images in a directory via drag-and-drop. Opens a browser UI, lets the user rearrange image cards, then renames files with sequential numbering to persist the new order. Also supports grouping images, organizing groups into subfolders, and undo.

## Architecture

**Server** (`start.ts` → `src/server.ts`):
- `Bun.serve()` with manual routing in `handleAPI()`. No framework.
- `src/rename.ts` — all filesystem operations (list, rename, undo, organize)
- `src/thumbnails.ts` — Sharp-based WebP thumbnail generation with `.reorder-cache/` directory
- Background thumbnail pre-generation on startup (8 concurrent workers)
- HTTP caching with ETag/304 on image and thumbnail endpoints

**Client** (`src/client/`):
- React 19 SPA, built with `Bun.build()` (no Vite)
- State: Zustand stores in `stores/` (imageStore, selectionStore, dndStore, groupStore, uiStore)
- Components in `components/` — extracted from what was originally a monolithic App.tsx
- Utilities in `utils/` (helpers, gridItems, reorder)
- `App.tsx` is a thin shell: layout, DndContext, virtualizer, effects
- Drag-and-drop: `@dnd-kit/core` + `@dnd-kit/sortable` with `rectSortingStrategy`
- Virtualization: `@tanstack/react-virtual` row virtualizer for 13k+ images
- Grid layout: CSS `auto-fill` drives column count; JS reads it via `getComputedStyle` on a hidden measuring row (callback ref pattern — important for timing)
- In-app search: Cmd+F intercepted, floating search bar with match highlighting

**CSS** (`src/client/styles.css`):
- Plain CSS with custom properties (dark theme). No framework.
- Grid rows use `repeat(auto-fill, minmax(160px, 1fr))` — CSS owns the column layout
- `.grid-measure-row` is the hidden element JS reads to determine column count and row height

## Key Patterns

- `postJson(url, body)` in `utils/helpers.ts` for all JSON POST fetches
- `wasJustDragged()` suppresses click events that fire at the end of drag gestures
- `updateGroups(fn)` in groupStore auto-prunes empty groups and skips persist on no-op
- Group persistence is debounced (300ms) to the server via `POST /api/groups`
- Thumbnails: grid cards use `/api/thumbnails/:filename` (400px WebP), lightbox uses `/api/images/:filename` (full resolution)

## File Structure

```
start.ts                        Entry point (CLI, build, server start)
src/
  server.ts                     HTTP server + API routes
  rename.ts                     Filesystem operations
  thumbnails.ts                 Sharp thumbnail generation + cache
  client/
    index.html / index.tsx      HTML shell + React mount
    App.tsx                     Thin shell (DndContext, virtualizer, effects)
    types.ts                    Shared interfaces
    styles.css                  All styles
    stores/                     Zustand stores (5 files)
    components/                 Extracted components (8 files)
    utils/                      Helpers, grid item computation, reorder logic
```
