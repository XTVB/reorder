# Reorder App — Performance Overhaul Plan

This plan scales the app to handle 13k+ images smoothly. The changes are ordered so each phase builds on the previous one. **If you are ever unsure about a design decision, UI behavior, or trade-off, use AskUserQuestion to ask for clarity before proceeding.**

---

## Important Context

- This app is only ever run locally on macOS, reading files from the user's filesystem. There are no remote users or network latency concerns.
- Use Bun, not Node.js (see CLAUDE.md for full details).
- The current `App.tsx` is ~1700 lines with all state in one component. All image cards are rendered into the DOM at once with no virtualization. The server loads full-resolution images for grid thumbnails and runs `stat()` on every file on every load.
- The goal is to make all 13k+ images browsable, searchable, and reorderable with no jank.

---

## Phase 1: State Architecture — Zustand Stores

**Do this first.** Every subsequent phase is easier when state is decoupled from the monolithic `App` component.

### Why Zustand
Zustand lets components subscribe to specific slices of state via selectors (`useStore(s => s.selectedIds)`). Only components whose selected slice actually changed re-render. This is critical — when the user selects a card, only that card and the toolbar should re-render, not all 13k card components. React Context re-renders all consumers on any change to the context value, which defeats the purpose. Zustand is ~1KB, zero boilerplate, and works outside of React (useful in callbacks).

### What to do

1. `bun add zustand`

2. Create a `src/client/stores/` directory with the following stores:

   **`imageStore.ts`** — Image data and ordering
   - `images: ImageInfo[]` (the master ordered list)
   - `originalOrder: string[]` (ref for change detection)
   - `hasChanges: boolean` (derived)
   - `imageMap: Map<string, ImageInfo>` (derived)
   - Actions: `fetchImages()`, `setImages()`, `reorderImages()`, `resetOriginalOrder()`

   **`selectionStore.ts`** — Multi-select state
   - `selectedIds: Set<string>`
   - Actions: `select(id)`, `toggleSelect(id)`, `rangeSelect(id, allIds)`, `clearSelection()`, `selectAll(ids)`

   **`dndStore.ts`** — Drag-and-drop state
   - `activeId: string | null`
   - `dragOverGroupId: string | null`
   - `frozenGroupId: string | null`
   - Actions: `setActiveId()`, `setDragOverGroupId()`, `setFrozenGroupId()`, `clearDrag()`

   **`groupStore.ts`** — Group state
   - `groups: ImageGroup[]`
   - `groupsEnabled: boolean`
   - `expandedGroupId: string | null`
   - `groupMap: Map<string, ImageGroup>` (derived)
   - Actions: `fetchGroups()`, `setGroups()`, `toggleGroupsEnabled()`, `expandGroup()`, `collapseGroup()`, `addImagesToGroup()`, `removeImagesFromGroup()`

   **`uiStore.ts`** — UI/modal state
   - `lightboxIndex: number | null`
   - `loading: boolean`, `saving: boolean`, `error: string | null`
   - `showPreview: boolean`, `showOrganize: boolean`, `showPaths: boolean`
   - `toast: Toast | null`
   - `canUndo: boolean`
   - `targetDir: string`
   - Actions: `openLightbox()`, `closeLightbox()`, `setLoading()`, `showToast()`, etc.

3. Refactor `App.tsx`:
   - Remove all `useState` / `useRef` for state that moved to stores.
   - Components call `useXxxStore(selector)` to subscribe to only what they need.
   - Move `SortableCard`, `SortableGroupCard`, `ExpandedGroupItem`, `GroupThumbGrid`, `Lightbox`, and `Modal` into separate files under `src/client/components/`. This is important for maintainability — don't leave them in `App.tsx`.
   - `App.tsx` should become a thin shell: renders the layout, sets up DndContext and the virtual grid (Phase 3), and coordinates top-level effects (initial data fetch, keyboard shortcuts).
   - Move pure utility functions (`computeGridItems`, `gridItemId`, `multiDragReorder`, `consolidateBlock`, `remapGroups`, etc.) into `src/client/utils.ts` or a few focused utility files.

4. Verify everything works identically to before by testing manually. No behavior changes in this phase — pure refactor.

### File structure after Phase 1
```
src/client/
├── index.html
├── index.tsx
├── App.tsx                  (thin shell — layout, DndContext, effects)
├── styles.css
├── stores/
│   ├── imageStore.ts
│   ├── selectionStore.ts
│   ├── dndStore.ts
│   ├── groupStore.ts
│   └── uiStore.ts
├── components/
│   ├── SortableCard.tsx
│   ├── SortableGroupCard.tsx
│   ├── ExpandedGroupItem.tsx
│   ├── GroupThumbGrid.tsx
│   ├── Lightbox.tsx
│   ├── Modal.tsx
│   └── Toolbar.tsx          (extract the top bar into its own component)
└── utils/
    ├── gridItems.ts         (computeGridItems, gridItemId, etc.)
    └── reorder.ts           (multiDragReorder, consolidateBlock, etc.)
```

---

## Phase 2: Server-Side Thumbnails

### 2A: Drop stat from image listing

Modify `/api/images` to return an array of filenames (strings) instead of objects with `size` and `modified`. Keep the `stat` import and the capability for future use, but don't call it on the listing endpoint.

Update the `ImageInfo` interface on the frontend to only require `filename`. Remove `size` and `modified` fields (they were never used in the UI anyway).

### 2B: Thumbnail endpoint with Sharp

1. `bun add sharp`

2. Add a thumbnail generation module at `src/thumbnails.ts`:
   - `generateThumbnail(sourcePath: string, cachePath: string): Promise<void>` — uses Sharp to resize to 400px wide (maintain aspect ratio), output as WebP with quality 80, write to `cachePath`.
   - Cache directory: `.reorder-cache/` inside the target image directory. Create it if it doesn't exist.
   - Cache key: `<original-filename-without-ext>.webp` (e.g., `photo001.jpg` → `.reorder-cache/photo001.webp`).
   - Cache validation: compare source file's `mtime` against cached thumbnail's `mtime`. Regenerate if source is newer.

3. Add `/api/thumbnails/:filename` endpoint in `server.ts`:
   - Check if cached thumbnail exists and is fresh (mtime check).
   - If cached and fresh, serve it directly with `Cache-Control: public, max-age=86400` and an `ETag` header based on mtime.
   - If not cached or stale, generate thumbnail, then serve it.
   - Return WebP content type.
   - If thumbnail generation fails for any reason (corrupt image, unsupported format), fall back to serving the original file.

4. Frontend: Grid cards use `/api/thumbnails/:filename` instead of `/api/images/:filename`. Lightbox continues to use `/api/images/:filename` for full resolution.

### 2C: Background pre-generation

On server start (in `start.ts`), after the server is listening, kick off background thumbnail generation:

- Read the full image list.
- Process images in batches (e.g., 8 concurrent) using a simple semaphore/pool.
- Skip images that already have a fresh cached thumbnail.
- Log progress to the console (e.g., `Thumbnails: 1542/13000 generated...`) but don't spam — log every ~500 images or every 5 seconds.
- This must not block server startup or the initial page load. Use a fire-and-forget async function.

The frontend should work fine before pre-generation completes — individual thumbnails are generated on-demand when requested, the background job just warms the cache proactively.

---

## Phase 3: Virtualized Grid

This is the most complex phase. Take care to preserve all existing drag-and-drop, multi-select, and group behaviors exactly.

### Library

`bun add @tanstack/react-virtual`

### Approach

1. **Grid layout calculation:**
   - The grid currently uses CSS `display: grid` with `grid-template-columns: repeat(auto-fill, minmax(160px, 1fr))`. With virtualization we need to know the column count and row height in JS.
   - Create a `useGridLayout` hook that:
     - Uses a `ResizeObserver` on the grid container to get its width.
     - Calculates column count: `Math.floor(containerWidth / minCardWidth)` (where `minCardWidth` ≈ 160px, matching the CSS).
     - Calculates row height (card height is roughly fixed — image thumbnail + filename label + padding). Measure one card or use a known constant (e.g., 200px). **Ask the user** what the expected card height is if unsure, or measure dynamically.
   - Returns `{ columnCount, rowHeight, containerWidth }`.

2. **Row virtualization with `@tanstack/react-virtual`:**
   - Convert the flat `gridItems` array into rows: `rows[i] = gridItems.slice(i * columnCount, (i + 1) * columnCount)`.
   - Use `useVirtualizer` with:
     - `count: rowCount`
     - `getScrollElement: () => scrollContainerRef`
     - `estimateSize: () => rowHeight`
     - `overscan: 5` (5 extra rows above/below viewport ≈ 25-50 extra cards)
   - Render only the virtual rows. Each virtual row renders its cards.

3. **dnd-kit integration:**
   - `SortableContext` needs ALL item IDs (all 13k) passed to its `items` prop for correct ordering logic, but only visible cards mount `useSortable`.
   - This is a known pattern with dnd-kit + virtualization. The `SortableContext` maintains the order; `useSortable` hooks register/unregister as cards enter/leave the viewport.
   - The `DragOverlay` renders a floating clone already (not part of the grid), so it works naturally with virtualization.
   - **Important:** Test thoroughly that dragging an item from the visible area to a position that requires scrolling works correctly. The virtualizer should handle scroll-based rendering, and dnd-kit's auto-scroll should trigger the virtualizer to render new rows. If auto-scroll doesn't work well out of the box, we may need to hook dnd-kit's `onDragMove` to programmatically scroll the container. **Ask the user to test drag+scroll behavior** after initial implementation.

4. **Group handling in the virtual grid:**
   - Group header cards and expanded group items are part of `gridItems` and get virtualized like everything else.
   - The expanded group popover (inline below the group card) may need special treatment — it's not a grid item, it's an overlay/expansion. Consider rendering it as a full-width row that spans all columns, inserted into the virtual row list at the correct position. **Ask the user** how they want the expanded group to behave if the implementation gets complex.

5. **Scroll container:**
   - The virtualizer needs a scroll container ref. Currently the page body scrolls. Either:
     - (a) Make the grid area a fixed-height scrollable div (easier for virtualizer).
     - (b) Use window-based virtualization (`useWindowVirtualizer`).
   - Option (a) is simpler and recommended. The toolbar stays fixed at top, and the grid area below it scrolls independently.

### What to watch out for
- After reorder (drag-end), the grid items change. The virtualizer needs to be notified — `gridItems` is a dependency of the row computation, so this should happen naturally via React state updates.
- Multi-select visual state (blue border, ghost opacity) must still work — these are props on `SortableCard` driven by `selectionStore`, unaffected by virtualization.
- The "empty state" (no images) should still render normally — skip virtualization when the list is empty.

---

## Phase 4: In-App Search (Ctrl+F Replacement)

Virtualization removes off-screen items from the DOM, breaking browser Ctrl+F. Replace it with a custom search.

### Implementation

1. Create `src/client/components/SearchBar.tsx`:
   - A floating search bar that appears at the top of the grid area (similar to VS Code's Ctrl+F bar).
   - Input field with match count display ("3 of 17 matches"), previous/next buttons (or Shift+Enter / Enter), and Escape to close.
   - Visually styled to be non-intrusive — absolute positioned over the grid, blurred backdrop.

2. **Search logic** (can live in a `searchStore.ts` or within the component):
   - `query: string` — the search input.
   - `matches: number[]` — indices into `gridItems` that match the query.
   - `currentMatchIndex: number` — which match is focused.
   - Matching: case-insensitive substring match against:
     - Image filenames
     - Group names
   - When `currentMatchIndex` changes, programmatically scroll the virtualizer to the row containing that grid item using `virtualizer.scrollToIndex(rowIndex)`.
   - Highlight matching cards visually (e.g., a subtle outline or background color on the card).

3. **Keyboard shortcut:**
   - Intercept `Ctrl+F` / `Cmd+F` with a `keydown` handler on `document`.
   - Call `e.preventDefault()` to suppress the browser's native find.
   - Focus the search input.
   - `Escape` closes the search bar and clears highlights.
   - `Enter` → next match, `Shift+Enter` → previous match.

4. **Edge cases:**
   - Empty query → close/hide match indicators.
   - No matches → show "No matches" in the search bar.
   - If the matched item is inside a collapsed group, **ask the user** whether search should auto-expand that group or just scroll to the group card itself.

---

## Phase 5: Small Fixes & Polish

### 5A: Throttle pointermove during drag

In the `pointermove` handler that calls `document.elementsFromPoint()` for group-dwell detection, gate it behind a `requestAnimationFrame`:

```ts
let rafId: number | null = null;
function onPointerMove(e: PointerEvent) {
  if (rafId !== null) return;
  rafId = requestAnimationFrame(() => {
    rafId = null;
    // existing elementsFromPoint logic here
  });
}
```

Clean up the RAF on drag end (cancel any pending frame).

### 5B: HTTP caching on thumbnail and image endpoints

- Thumbnail responses: `Cache-Control: public, max-age=86400` with `ETag` based on source file mtime + size (e.g., `W/"<mtime>-<size>"`).
- Full image responses: `Cache-Control: public, max-age=3600` with the same ETag scheme.
- Handle `If-None-Match` request header: if it matches the current ETag, return `304 Not Modified` with no body.

### 5C: Remove stale console.log calls

There are `console.log` calls in the group-cleanup `useEffect` (around lines 890-902 of the current `App.tsx`). Remove them — they fire on every `images` state change and add noise.

---

## Implementation Order Summary

```
Phase 1: State architecture (Zustand stores + component extraction)
   ↓
Phase 2: Server thumbnails (2A: drop stat, 2B: Sharp endpoint, 2C: pre-gen)
   ↓
Phase 3: Virtualized grid (@tanstack/react-virtual + dnd-kit integration)
   ↓
Phase 4: In-app search (Ctrl+F replacement)
   ↓
Phase 5: Polish (throttle pointermove, HTTP caching, cleanup)
```

Each phase should be tested manually before proceeding to the next. After each phase, verify:
- All existing functionality works (reorder, multi-select, groups, drag-into-group, lightbox, save/undo, organize).
- No regressions in visual appearance.
- Performance feels snappy with a large image directory.

**When in doubt about any design decision, UI behavior, or edge case, use AskUserQuestion to ask for clarity. Do not guess — the user wants to be consulted on ambiguous decisions.**
