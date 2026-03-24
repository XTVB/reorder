# Reorder

A local macOS tool for reordering images in a directory via a drag-and-drop browser UI. Renames files with sequential numbering to persist the new order.

## Usage

```sh
bun run start.ts /path/to/image/directory
```

Opens a browser window with all images displayed as a draggable grid.

## Features

- **Drag-and-drop reorder** — drag single images or multi-select with Cmd+click / Shift+click
- **Groups** — select images and press `G` to group them; drag images onto groups; rename, dissolve, or reorganize groups into subfolders
- **Search** — Cmd+F to find images by filename or group name
- **Save** — preview renames before committing; undo last save
- **Lightbox** — click an image to view full-resolution with zoom and pan

## How It Works

1. Images are loaded from the target directory and displayed in a virtualized grid
2. Thumbnails (400px WebP) are generated on first access and cached in `.reorder-cache/`
3. Dragging rearranges the in-memory order; nothing changes on disk until you click **Save Order**
4. Saving renames files to `001.jpg`, `002.jpg`, etc. using a two-phase rename to avoid collisions
5. Groups are persisted to `.reorder-groups.json` in the target directory

## Requirements

- [Bun](https://bun.sh) runtime
- macOS (uses `open` command to launch browser)

## Install

```sh
bun install
```

## Tip

Add an alias for quick access:

```sh
alias reorder="bun run /path/to/reorder/start.ts"
# Usage: reorder /path/to/image/directory
```
