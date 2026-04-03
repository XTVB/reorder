# Remaining Steps & Improvements

## From Original Plan — Not Yet Implemented

### CSS polish for cluster stale banner
The `cluster-stale-banner` class is referenced in ClusterView but has no CSS styles yet. Needs styling as a sticky warning bar (amber/yellow background, centered text).

### Consolidate images on group creation
When a cluster is accepted as a group, `handleAcceptCluster` calls `consolidateBlock` to move images adjacent in the flat order. This was wired up but needs end-to-end testing — verify that after accepting a cluster and switching to reorder mode, the images are actually contiguous and "Save Order" renames them sequentially.

### Remove dead Tags/Merge code
`AppMode` was changed from `"reorder" | "tags" | "merge"` to `"reorder" | "cluster"`. The old `TagExplorer` and `MergeView` components still exist on disk but are no longer imported. They can be deleted, along with tag-related store code, if tags functionality isn't needed going forward.

### Python venv portability
The Python path is hardcoded to `/tmp/imgcluster-env/bin/python3`. The `CLUSTER_PYTHON` env var override exists but the app should auto-detect a suitable Python (check for the venv, fall back to system python3 with the right packages).

## UX Improvements to Consider

### Cluster card thumbnail sizing
Current thumbnails are 120×90px — quite small for reviewing whether images belong together. Consider making them resizable (a slider in the toolbar) or at least larger by default (160×120). The czkawka-compare review UI used 200px wide cards.

### Drag-and-drop between clusters
Currently, moving images between clusters requires select → split → merge. A direct drag-from-one-cluster-to-another would be faster, similar to how reorder mode handles drag-into-group with dwell detection.

### Cluster sorting options
Currently sorted by size (largest first). Add options to sort by: auto-name alphabetically, number of suggested additions (most actionable first), or confirmed/unconfirmed.

### Lightbox navigation across clusters
The lightbox currently navigates within a single cluster's images. Could be useful to navigate across all images in the view (or at least have "next cluster" / "previous cluster" buttons at the boundaries).

### Batch contact sheet generation
Generate contact sheets for all uncollapsed clusters at once (for a Claude session reviewing multiple sets). Output a directory of numbered contact sheets.

### Keyboard shortcuts for cluster mode
- `G` — accept focused cluster as group (matching reorder mode's `G` for grouping)
- `D` — dismiss focused cluster
- `A` — add suggested images to existing group
- Arrow keys — navigate between clusters (focus ring)
- `Enter` on focused cluster — expand/collapse

### Progress bar for initial extraction
The SSE streaming shows text progress. A proper progress bar (percentage, ETA) in the toolbar would be better UX during the ~5min first-run extraction.

### Smart vocabulary expansion
The 334-term vocabulary is generic. Consider:
- Letting the user add custom terms (stored in a config file)
- Auto-expanding vocabulary based on the "Ask Claude" results the user has gotten (learn what terms are useful for their content)
- Language-specific terms (Japanese character names for anime cosplay)

### Re-cut speed optimization
Re-cut takes ~0.9s, mostly auto-naming (7239 images × 334 terms dot products). Could be faster with:
- Pre-multiplying the full `clip × textEmbs.T` matrix once and caching it (7239×334 = 2.4M entries, ~10MB)
- Only recomputing names for clusters that changed between cuts

### Cluster diff view
When re-cutting at a different N, show what changed: which clusters merged, which split, which images moved. This helps the user understand the effect of adjusting granularity.

### Export/import cluster state
Save the current cluster view state (including edits, dismissals, custom names) to a JSON file so the user can resume a review session later without re-running clustering.

### Integration with other tools
The user mentioned wanting to integrate the clustering UI with another tool. The architecture (JSON API, standalone Rust binary, content-hash cache) is designed for this. Potential integration points:
- The Rust binary can be called from any tool
- The `/api/cluster/recut` endpoint is stateless (reads cached tree)
- Contact sheets are standalone JPEG files

## Technical Debt

### npz parser in cluster.ts
The `parseNpyFromNpz` function is a minimal ZIP+NPY parser. It handles uncompressed and deflate-compressed entries but doesn't handle ZIP64 or all NPY dtypes. If the cache format changes, this could break. Consider using a proper npm package or validating the assumptions.

### Rust binary path
Hardcoded to `rust/cluster-tool/target/release/cluster-tool` relative to the project root. Should detect whether the binary exists and give a clear error with build instructions if not.

### Concurrent access to .reorder-cache
The Python extraction and Rust linkage both read/write to `.reorder-cache/`. If somehow triggered simultaneously (e.g., two browser tabs), file corruption is possible. The `clusterJobRunning` flag prevents this at the HTTP level but not at the filesystem level.

### Test coverage
No automated tests for any of the clustering code (Python, Rust, or TypeScript). At minimum:
- Rust: test that pre-seeded groups stay intact at various N values
- TypeScript: test that `recutTree` union-find produces correct labels
- Python: test that content-hash cache survives renames
