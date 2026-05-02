# Clustering UI — Tree-Navigation Operations Spec

This spec covers three new per-cluster operations that let the user manually
refine clustering results: **merge**, **split**, and **expand**. They use the
linkage-tree data already produced by the clustering pipeline.

The goal is to give the user fast, visual tools to fix the cases the global
clustering knobs (N, min cluster size, distance threshold) cannot resolve —
without changing the existing accept/group-creation workflow.

## Per-cluster card additions

Every cluster card in the cluster list gains:

- **Three action buttons**: `merge…`, `split`, `expand…`.
- **Three numeric badges** in the header, formatted as
  `c:<cohesion> · i:<isolation> · s:<stability>`:
  - **cohesion**: max intra-pair cosine distance within the cluster — how
    spread out the cluster is internally.
  - **isolation**: the cluster's death distance in the linkage tree — the
    merge distance at which it would be absorbed into its parent.
  - **stability**: `(death − birth) / death`, where `birth` is the merge
    distance at which the cluster formed and `death` is the merge distance
    at which it would be absorbed. Higher = the cluster persists across
    more distance thresholds.
  Each badge has a hover tooltip naming the metric. Numbers shown raw, no
  pills/colors/bars.

These are present on every cluster card, including child cards inside an
inline split disclosure.

## Operation 1 — Merge (compare-mode view)

### Trigger

Clicking `merge…` on a single source cluster opens compare mode as a
full-panel view (replacing or layered over the current cluster list).

### Layout

- The **source cluster is pinned at the top** of the view, sticky on scroll.
  It shows the source's full contact sheet at the standard image scale.
  Source height is clamped (e.g. 30% of viewport) so candidates aren't
  squeezed out; if the source has too many images for that height its
  contact sheet scrolls horizontally inside the pinned region.
- Below the source, a **vertical stack of candidate clusters**. Each
  candidate is a full-width row containing:
  - A checkbox at the leading edge ("include in merge").
  - The candidate's name, image count, and merge distance to source.
  - The candidate's full contact sheet at the same image scale as the
    source.
- Candidates are ordered by a composite "likely-to-merge" score (no
  ranking labels visible to the user). The composite combines, in
  decreasing influence:
  - **Tree sibling**: the cluster this would merge with under one more
    linkage step (always rank 1 if present and unmerged).
  - **Tree cousins**: descendants of the parent's sibling — clusters that
    are tree-adjacent within a small radius.
  - **Top-K by re-rank distance**: clusters whose centroid (or membership
    signature) is closest under the re-ranked metric.
  - **Top-K by shared kNN overlap**: clusters that share many nearest
    neighbours with the source.
  Sources are deduplicated and merged into a single ranked list; the user
  never sees which source a candidate came from. No more than ~10–15
  candidates shown initially; scrolling past the bottom loads more.

### Filters and pickers

At the top of the candidate stack:

- A single **"include confirmed groups" checkbox**. Default on.
  - When on: confirmed groups appear in the candidate stack alongside
    unconfirmed clusters. Confirmed-group rows are flagged with a small
    lock icon.
  - When off: only unconfirmed clusters appear.
- A **"compare with another cluster" search field** as an escape hatch for
  any cluster the algorithm did not surface. Searching adds a chosen
  cluster to the candidate stack at the top.

### Confirming a merge

- A `merge selected (N)` action button is visible at the top of the view,
  showing the current count of checked candidates.
- Confirming combines the source with **all checked candidates** into a
  single cluster in the underlying cluster list. This is one operation: if
  three candidates are checked, source + three candidates become one
  cluster.
- **Confirmed-group participants**: if any participant in the merge (the
  source itself or any checked candidate) is a confirmed group, the merge
  result is an extension of that confirmed group — its images are added
  to the existing group rather than creating a new unconfirmed cluster.
  When multiple confirmed groups participate, behaviour falls back to the
  existing reorder-mode group-merging flow.
- Cancel/dismiss closes compare mode without changing anything.

### Naming

The merged cluster's auto-name is regenerated from the union of its
images using the existing TF-IDF auto-naming. No prompt for the user.

## Operation 2 — Split (inline disclosure)

### Trigger

Clicking `split` on a cluster card.

### Behavior

- The source card disclosure-expands inline within the cluster list to
  show **two child cards nested below it**.
- Each child is a fully functional cluster card: it has its own contact
  sheet, badges, and `merge…` / `split` / `expand…` buttons, and can be
  promoted to a confirmed group via the existing Accept flow.
- **Nesting is allowed**: clicking `split` on a child card disclosure-
  expands it further, showing its own two children. There is no fixed
  nesting depth.
- Splitting a singleton or 2-image cluster is degenerate; the `split`
  button is disabled in those cases.
- A way to collapse the disclosure (e.g. clicking the disclosure caret
  on the parent card) returns the cluster to its un-split state.

## Operation 3 — Expand (modal staging)

### Trigger

Clicking `expand…` on a cluster card.

### Layout

A modal opens, anchored to the source cluster, containing:

- A header showing the source cluster name and image count.
- A **density-threshold slider** at the top of the modal:
  - Log scale, range 0.5× to 4×.
  - Default position: **1.5×** the cluster's p90 intra-pair distance.
  - Below 1× is stricter than the cluster's own internal spread; above
    2× starts pulling in genuinely outside images.
- A **"include images currently in confirmed groups" checkbox**.
- A **candidate grid** showing images outside the source cluster, ranked
  by distance to the source's centroid (or nearest member). Each
  candidate shows:
  - The image thumbnail.
  - The cluster the image currently belongs to (text label).
  - A checkbox.

### Live updates

Dragging the density slider updates the candidate grid in real time —
candidates fade in/out as the threshold changes. Toggling the
confirmed-groups checkbox likewise updates the candidate set live.

### Confirming an expand

- A `add selected (N)` action button at the top of the modal commits.
- Confirming **moves the checked images into the source cluster**,
  removing them from whatever cluster they currently belong to (visible
  in the cluster list afterwards).
- Cancel/dismiss closes the modal without changes.

## Persistence model

All three operations (merge, split, expand) **modify only the current
cluster view**. They are not "pending edits", they have no draft state,
they require no explicit save, and there is no banner or commit button.

The durable step is unchanged from today: the user uses the existing
"make this a confirmed group" / Accept flow on whatever cluster (merged,
split-out, or expanded) they want to lock in.

**Re-cut behavior**: when the user changes the global clustering knobs (N,
min cluster size, threshold) and re-cuts, the cluster list is regenerated
from the linkage tree and **all in-progress merge/split/expand state is
wiped**. If compare mode or the expand modal is open at the time of
re-cut, **it is dismissed automatically** (the view returns to the
regenerated cluster list). No warning, no save prompt — re-cut is
understood as a fresh take.

There is **no undo**, no per-operation history, and **no visual marker
distinguishing tree-nav-modified clusters from fresh algorithmic
clusters** in the cluster list.

## Out of scope (explicitly decided against)

- Pending-edits draft / save banner.
- "Why is this a candidate?" labels in the merge candidate stack.
- Visual accents/borders on tree-nav-modified clusters.
- Cmd-Z / undo for tree-nav operations.
- Single-pill confidence indicators (we are using three numeric badges
  instead, not one combined score).
- Hierarchical-tree side panel showing the global dendrogram.
- Algorithm-picked sibling-only merge (the user always picks from the
  candidate stack — the algorithm only ranks).
