Default to Bun, not Node.

- `bun run start.ts <dir>` — launch (builds client, opens browser, pre-generates thumbnails)
- `bun build src/client/index.tsx --outdir dist --minify` — test client compilation
- `bun install`, `bun run typecheck`, `bun run lint` (`bun run lint:fix` to auto-fix)
- Rust: `cargo build --release` from `rust/cluster-tool/`, `rust/group-similarity/`, or `rust/order-tool/` (all share `rust/reorder-common`), plus standalone `rust/hash-tool/` (perceptual hashing for the Czkawka page)

## What This App Does

Local macOS tool for organizing image directories. Browser UI with four modes:

1. **Reorder** — drag-and-drop, group, rename to sequential numbering, organize into subfolders
2. **Cluster** — PE-Core-G + DINOv3 + color visual clustering to discover photoshoot sets
3. **Merge Suggestions** — DINOv3 patch matching finds groups likely to belong together
4. **Czkawka** — duplicate finder + step-through resolver: `rust/hash-tool` (image_hasher, same crate as czkawka; luma-only JPEG decode + SIMD pre-shrink) hashes originals + flops/mirrors (catches mirrored dupes; upside-down ones deliberately not covered), hashes cached per dir + config keyed by content hash in `.reorder-cache/czkawka_hashes_v*.json` (version bumps when the pipeline changes); supports comparing multiple directories, optionally with one as czkawka-style "reference" (only ref↔non-ref matches reported, intra-dir dupes skipped); trash/copy-replace actions run under the rename lock with a restorable undo journal (`czkawka_session.json`, rename-based ~/.Trash moves)

Plus **Rank** (a modal on the Reorder page, not a mode): order things by preference — groups or images, scoped to all of them, the grid selection, or the contents of the selected groups. See "Ranking" below.

Workflow is iterative: cluster → accept groups → reorder/rename → re-cluster.

## Architecture

The codebase is module-oriented. Each top-level directory under `src/` has an `index.ts` barrel that defines its public surface — prefer importing from the barrel over reaching into internals. Cross-cutting types live in `src/shared/types.ts`.

### `src/server/` — HTTP layer
- `index.ts` — `Bun.serve()` + a route dispatcher that walks an array of `RouteHandler`s and returns the first non-null response. No framework.
- `routes/` — one file per concern (`cluster`, `cluster-extract`, `constraints`, `delete`, `folders`, `groups`, `images`, `merge`, `nn`, `organize`, `rename`, `tree-nav`). Each exports a single `RouteHandler`.
- `middleware/` — `response` (json/mimeType helpers), `sse` (`sseResponse`, `subscribeProgressSSE` for re-attach), `cluster-job` (`runClusterJobSSE` wraps the 409-or-stream-progress pattern every long-running cluster route shares).
- Startup runs `recoverPendingRename` inside `withRenameLock`; any GET that depends on rename consistency holds the same lock.

### `src/fs/` — filesystem ops
Two-phase rename with write-ahead manifest at `.reorder-pending.json`. `withRenameLock` (in `lock.ts`) is a single mutex serializing every FS-mutating op AND any read that must not observe a half-applied rename. Split into `atomic-json`, `content-hashes`, `folder-save`, `groups`, `images`, `organize`, `paths`, `recovery`, `rename`, `tags`, `trash`. `paths.ts` is the single source for cache/log/manifest paths.

### `src/cluster/` — clustering orchestration
Drives Python/Rust subprocesses; parses linkage tree; contact sheets; merge suggestions. Notable modules: `pipeline` (extract → linkage → name assignment), `linkage` (tree cuts: fixed-N / threshold / adaptive), `embeddings` (per-model NPZ load + content-hash mapping), `constraints` (cannot-link + group-lock), `distance-matrices` (patch + re-rank caches), `imported` (bypass for externally-provided clusters), `progress` (broadcast channel for SSE re-attach), `job-mutex` (single in-flight compute slot, abortable via SIGINT). `subprocess.ts` centralises spawning. `binaries.ts` resolves Python + Rust binary paths.

The compute-job mutex (`cluster/job-mutex.ts`) is distinct from the FS lock: it guards CPU/GPU work and serves 409 to concurrent compute requests, while `withRenameLock` guards on-disk consistency.

### `src/client/` — React 19 SPA, built with `Bun.build()` (no Vite)
- Routing: `useRouter` hook (pushState). Paths: `/reorder`, `/cluster`, `/merge-suggestions`. Server returns `index.html` for any non-API, non-asset path.
- Shell: `AppShell` in `index.tsx` → `components/header/AppShellHeader` + mode-specific view.
- Components grouped by mode/concern: `cluster/`, `merge-suggestions/`, `reorder/`, `header/`, `shared/`.
- `api/` — `client.ts` (`getJson`/`postJson`/`putJson`/`deleteJson`/`postRaw` fetch wrappers; centralised error extraction) and `sse.ts` (`consumeSSE` parser, `startSSE` POSTs and returns either the stream or a 409-conflict signal). Use these instead of bare `fetch`.
- DnD: `@dnd-kit/core` + `@dnd-kit/sortable`. Virtualization: `@tanstack/react-virtual`.
- CSS: per-concern files in `src/client/styles/`, copied to `dist/` at build time.
- Debug: all Zustand stores attached to `window.__stores`.

### Zustand stores (`src/client/stores/`)
- `core/` — cross-mode primitives: `selectionStore` (multi-context: `reorder`, `cluster:images`, `cluster:merge`, `expand`, `nn`, `trash`, plus per-row contexts), `modalStore`, `lightboxStore`, `toastStore`, `sessionStore`.
- `modes/cluster/` — split into `listStore` (current `clusterData`), `expandStore`, `splitStore`, `metricsStore`, `interactionsStore`. `index.ts` is the barrel and installs a single subscription that clears expand/metrics/selection state when the cluster *shape* (id set or per-cluster sizes) changes.
- Top-level: `imageStore`, `groupStore`, `folderStore`, `dndStore`, `mergeSuggestionsStore`, `constraintsStore`, `nnQueryStore`, `trashStore`. Read the files for shape — they're the source of truth.

### Clustering Pipeline

```
Stage 1: Python (scripts/extract_features.py)
  PE-Core-G, DINOv3, color histograms — all on MPS GPU.
  Per-model version keys (MODEL_VERSIONS dict): only changed models re-extract.
  Content-hash cache (blake2b of first 16KB + filesize) survives renames.
  --models forces re-extract; --required only extracts listed models if missing.
  → .reorder-cache/{embeddings_hash_cache.npz, hash_cache_order.json, dinov3_patches_*.npy}

Stage 2: Rust (rust/cluster-tool/, modules: cli/io/distances/linkage/tree)
  Ward's linkage (NNC) matching scipy exactly. Parallel via rayon.
  Pre-seeds confirmed reorder groups as real clusters (true centroid/size/variance).
  Weighted blend of per-model cosine distances (--pecore-g-weight, --dinov3-weight, --color-weight,
  --learned-proj-weight, --learned-proj-peg-weight, --learned-proj-color-weight). The three
  learned-head sliders are each a target fraction of the final cosine signal, converted to raw
  concat weights server-side (rescaleLearnedProjWeight in src/cluster/pipeline.ts); defaults
  .55/.30/.15 joint:peg:color = the winning 3-head blend. A dial whose array is missing from the
  cache contributes nothing (zero-shot absorbs the remainder)
  Optional: blend in precomputed patch distance matrix (--dist-matrix)
  → .reorder-cache/linkage_tree.bin

Stage 2b: Rust (rust/group-similarity/, modes: merge-suggestions / dist-matrix)
  DINOv3 patch matching: for each image pair, max-pool 7x7 cosine sims
  Shared parsing/types live in the rust/reorder-common crate.

Stage 2c: Rust (rust/order-tool/) — reorder page "Sort Similar" image ordering
  Batch seriation: per job (the ungrouped set, or one per selected group) builds
  the per-model linear-weighted cosine distance matrix (rayon-parallel rows) and
  reduces it to a 1D order via six modes (chain/tree/spectral/minimal/stable/
  gather). Jobs run in parallel; embeddings load once. The mode algorithms
  mirror the group-ordering ones (src/cluster/group-ordering.ts). Driven by
  src/cluster/order-batch.ts via /api/groups/similarity-order (target:ungrouped)
  and /api/groups/similarity-order-batch (one job per selected group).

Stage 3: Bun (src/cluster/pipeline.ts + linkage.ts)
  Re-cuts cached linkage tree — three modes: fixed N, distance threshold, HDBSCAN-style adaptive
  Auto-name = confirmed-group name (when one is pre-seeded) or "Cluster N" placeholder
  Contact sheets via Sharp (cluster/contact-sheets.ts): justified-row layout on a 2000px-wide canvas
```

### Ranking (`src/client/utils/rankEngine.ts`, `RankModal.tsx`)

Each item holds a Gaussian belief (mu, sigma) over a latent "liking" score. Two observation types: 1-5 tier ratings (noisy absolute measurements; shift = confident, tighter noise) and TrueSkill pairwise / best-of-N comparisons (the precise instrument for near-ties). A comparison answer can optionally carry a decisiveness margin — ⇧ "clearly" (performance gap conditioned above `clearWinMargin`) or ⌥ "barely" (gap inside `(0, slimWinMargin)`: asserts the order but pulls the pair together) — and a best-of-N can be answered with a joint top (`+` or per-panel Top toggles, ⏎ commits): the staged subset beats the rest and ties among itself. Absent margins replay old logs unchanged; the margin widths are provisional until margin-annotated judgements accumulate for the calibration sweep. The question policy serves the highest-expected-information comparison. Progress is communicated positionally (`positionalSummary`): the meter reads "±N places" — how far each item typically sits from its infinite-comparisons position — because the older boundary-settledness measure (`sortedFraction`, still used in tests) reads ~0 on large scopes until the entire within-tier order is ground out. A target-precision control (Auto/Exact/±2%/±5%; Auto = exact ≤40 items, then `0.05·(n−40)` places) turns "good enough" into a real stop: when mean displacement reaches the target, the engine stops serving comparisons and says so, rather than grinding near-ties the user said don't matter. Simulation on a real 242-group corpus (`rank-sim2` in session scratch, reproducible from the judgement log): target ±12 ≈ 530 comparisons vs ≈ 1,360 to exhaustion. Measured on the *completed* 565-comparison itscayyay log (2026-07): best-of-4s deliver ~the same TRUE accuracy per question as a well-chosen pair (true ± at fixed q identical across pairs-only / every-4th / every-2nd, 8 seeds) — their apparent speedup is meter optimism (betaMulti all-pairs updates over-tighten posteriors, so at the meter's stop true error was ±12.7 vs ±11.2 pairs-only); the calibration sweep's only above-noise finding was that within-window rating differences carry ~no win-prediction signal ("anchors x0" scored best, 0.05 nats), but compressed-anchor constants behave *worse* live (serving dries up early, meter turns pessimistic) so shipped constants stay; fatigue ties (user: late-session tie = "same region, whatever") are harmless to beliefs (skipping every tie update moved win-prediction < 0.01 nats — each tie shifts mu ~0.025). Consequence: the multi cadence starts at every 6th question and widens by 2 every 25 answered (`MULTI_EVERY_START`/`MULTI_FATIGUE_*`; `questionsServed` is deliberately not persisted, so the cadence resets each modal session). The user's stated deliverable is *region-shaped* — right rank, then right top/middle/bottom third of the rank, thirds mattering more in smaller ranks — so `positionalSummary` also returns per-rated-tier displacement, the meter tooltip shows a by-rank breakdown, and Auto's stop additionally requires every rank settled to ~half a third of itself (`disp ≤ max(1, n/6)`; a global mean hides a fuzzy small rank because it barely moves the average). Block-objective policy sims (`block-policy.ts` in session scratch) all *failed* to beat the shipped question scoring: boundary-stakes and width-normalized stakes learn no faster (nearly every near-rank pair already straddles a block boundary), preferring "decisively answerable" gaps is strictly worse (the user's shrug tracks TRUE closeness, which beliefs can't see — 66% of served coin-flips were answered tie/barely and that is not policy-avoidable), and hard "confidently in its block" stops are unreachable (dense-tier positional sd floors at ~14 places ≈ the answer-noise limit; the 9-item tier self-settles to ±1 because density auto-scales granularity). Honest baseline: at equal effort the engine ~ties a careful manual two-pass (5-way binning then 3-way within-rank: 39%/79% exact/within-1 block vs engine 36%/81% at 242 comparisons); it wins by continuing (46%/92% at the Auto stop) and by doing what manual structurally can't — moving pass-1 tier errors across ranks and ordering within blocks. Ratings are *not* a prerequisite for comparing: unrated items join the question pool at the prior (the old observed-only gate pushed users into mass "rank 1" passes, which calibration showed actively hurt). Modal modes: Auto = rate everything then a single handoff to comparisons (no interleaving), Rate / Compare = one kind of question only, switchable any time.

- **Scope ≠ file** — a session ranks a *kind* (groups / images) × a *scope* (all · the grid selection · the selected groups' contents), but there is one scores file per kind, shared by its scopes: a photo's rank is a fact about the photo, not about the session that produced it. So a scoped session persists every id of its kind it knows about, never a payload pruned to its scope (that would delete the rest of the gallery's ranks — `fs/rank-scores.ts` treats the payload as authoritative for every image on disk), and "Start over" erases the scope from the engine (`RankEngine.clearScope`) rather than emptying it. Apply moves the scope within the slots it already occupies — a scoped session never repacks the gallery.
- **Two files, two purposes** — `image_rank_scores.json` / `group_rank_scores.json` hold *beliefs* (derived, lossy, overwritten). `rank_judgements.json` holds the *observations* — every rating and comparison answer, in presentation order. Beliefs can be recomputed from judgements; judgements can never be recovered from beliefs. Treat the log as irreplaceable (it's in `ALWAYS_KEEP_FILES`, so a cache clear spares it).
- **Image scores are content-hash keyed** — filenames are the one thing this app churns (every Apply/Save renumbers by position), so a filename-keyed score file silently re-points ranks at different photos after a rename. `fs/rank-scores.ts` translates filename↔hash at the persistence boundary, under `withRenameLock`; the client stays filename-keyed. Group scores use stable uuids and need none of this. Same rule for the judgement log.
- **Don't guess the model constants — measure them.** Everything that decides what an observation *means* lives in `RankTuning` / `DEFAULT_TUNING` (beta, the tie-propensity prior, tier anchors, tie mode). `bun run scripts/rank-calibration.ts <dir> [--sweep]` replays the real judgement log prequentially (ask `outcomeProbs` *before* applying each answer, so the answer is a held-out label), reports log loss vs. the uniform baseline plus a reliability table, and `--sweep` re-scores the same log under alternative tunings. A setting that beats the default by a meaningful margin is a real finding; under ~0.01 nats on a few hundred judgements is noise.
- **Internal maths vs. modelling** — the update equations can be checked against the model's own assumptions (the `observeTie` comment records a Monte Carlo doing exactly that, which is why a best-of-N tie draws *all* pairs at `betaMulti` rather than chaining adjacent ranks). That proves the arithmetic, *not* that the assumptions describe the user. Only the calibration harness can speak to that.

### Key Patterns

- **Two locks, two purposes** — `withRenameLock` (fs/lock.ts) serializes disk state; cluster `job-mutex` serializes compute. Don't conflate them.
- **Content-hash cache** — renames don't invalidate the ~5min extraction
- **Pre-seeded groups** — Rust linkage bootstraps confirmed groups as initialized clusters, not zero-distance hacks
- **Patch-dist cache** — `patch_dist_matrix.bin` reused if newer than patches + filenames
- **SSE** — long-running routes use the `runClusterJobSSE` middleware (returns 409 if busy, otherwise streams `progress`/`result`/`error` events). Clients re-attach to a running job via `/api/cluster/progress` (server replays last progress + tails the broadcast channel).
- **Stale tree detection** — client tracks `treeStale` after group changes, prompts re-run
- **Rename safety** — two-phase with manifest; `recoverPendingRename` runs at startup inside the lock
- **Imported-clusters bypass** — `/api/cluster/import` stores clusters at `.reorder-cache/imported_clusters.json` and takes precedence over linkage-tree re-cuts on load
- **Cluster-shape subscription** — selection / expand / metrics state is auto-cleared when cluster ids or sizes change (wired in `stores/modes/cluster/index.ts`)
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
