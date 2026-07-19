# Rank calibration corpus

Real judgement logs copied from ranked directories, one subdir per source
gallery, each mirroring the `.reorder-cache/` layout so the calibration
harness runs on them directly:

```sh
bun run scripts/rank-calibration.ts calibration-data/bubble-bunny [--sweep]
```

`rank_judgements.json` is the input that matters — the append-only record of
every rating and comparison answer, in presentation order. The two
`*_rank_scores.json` files are the derived beliefs at copy time, kept for
reference only (they can always be recomputed from the judgements).

To grow the corpus after ranking more images somewhere, copy that directory's
`.reorder-cache/rank_judgements.json` into a subdir here (new gallery → new
subdir; same gallery → overwrite, the log accumulates in place). More
judgements make the sweep's verdicts trustworthy at finer margins — ~0.01
nats on a few hundred judgements is noise.

Data-quality notes (check before trusting a sweep):

- **angeljessie/groups**: every rating is slot 0 — a mass "rank 1" pass done
  only to unlock Compare mode, not real tier judgements. All-same-slot
  ratings make a dataset *insensitive* to the anchor dials (nothing to
  contrast), and replaying it with the rate events dropped actually predicts
  better (0.674 vs 0.722 nats/comparison): fake ratings tighten sigma
  without information. Weight its evidence accordingly.
- **bubble-bunny/images session 1**: near-uniform (34/38 at slot 1) for the
  same reason; session 2's re-rates at tiers 2-3 were genuine, and the
  rating passes still net-help there.

Findings that came out of this corpus so far:

- **The hurdle draw model** (`RankEngine.outcomeProbs`): bubble-bunny showed
  the empirical tie rate is flat in the model's predicted gap (~25% for
  images at every gap, ~4% for groups), which no fixed draw margin can
  price. Ties are a learned per-kind propensity now.
- **Tier spacing 1.7** (`DEFAULT_TUNING.anchors`): with three galleries
  pooled (649 comparisons), scaling the anchors relative to the noise dials
  improved every dataset, optimum ~1.7-2.0x the original spacing; 1.7 chosen
  because 2.0 starts costing bubble-bunny/groups. A tier keypress separates
  photos more decisively than the original constants assumed. Only the
  spacing/noise ratio is identified — scaling anchors and all sigmas
  together is a no-op.
