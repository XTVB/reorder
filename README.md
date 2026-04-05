# Reorder

A local macOS tool for organizing image directories. Opens a browser UI with drag-and-drop reordering, visual clustering to discover photoshoot sets, and group-merge suggestions.

## Usage

```sh
bun run start.ts /path/to/image/directory
```

## Modes

- **Reorder** — drag-and-drop cards (single or multi-select via Cmd/Shift+click), group with `G`, search with Cmd+F, rename files into sequential order, organize groups into subfolders
- **Cluster** — CLIP + PE-Core + DINOv3 visual clustering to find photoshoot sets; review, split, merge, or accept clusters as groups
- **Compare** — side-by-side clustering runs with different model weights, for tuning
- **Merge** — DINOv3 patch-matching surfaces groups likely to belong together

The intended workflow is iterative: cluster → accept groups → reorder/rename → re-cluster to find stragglers → repeat.

## How It Works

1. Images load from the target directory into a virtualized grid (handles 13k+)
2. Thumbnails (400px WebP) are generated on first access and cached in `.reorder-cache/`
3. Dragging rearranges in-memory order; nothing changes on disk until you click **Save Order**
4. Saving renames files to `001.jpg`, `002.jpg`, … via two-phase rename with crash recovery
5. Groups persist to `.reorder-groups.json` in the target directory

## Requirements

- [Bun](https://bun.sh) runtime
- macOS (uses `open` to launch the browser)
- Clustering additionally requires a Python venv with PyTorch + open-clip + DINOv3 weights, and the Rust binaries built under `rust/` — see `CLAUDE.md`

## Install

```sh
bun install
```

## Tip

```sh
alias reorder="bun run /path/to/reorder/start.ts"
# Usage: reorder /path/to/image/directory
```
