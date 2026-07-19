// Rank — order things by preference with a mix of quick 1-5 tier ratings (noisy
// priors) and engine-chosen pairwise / best-of-N comparisons (the precise
// instrument). What a session ranks is a *scope*: a kind (confirmed groups /
// images) crossed with a selector (everything / the grid selection / the
// contents of the selected groups) — see RankTarget.
//
// Scores live in one file per kind (groups by uuid, images by content hash),
// shared by that kind's scopes: a photo's rank is a fact about the photo, not
// about the session that produced it. So a scoped session must never write back
// a payload pruned to its scope — it persists every id of its kind it knows
// about (persistIds), and "Start over" erases the scope from the engine rather
// than emptying it.
//
// See utils/rankEngine.ts for the model; this file wires the engine into the
// GroupingSortModal chassis and owns the comparison overlay, autosave to
// /api/rank-scores (?target=images for the image scores), and the Apply
// (reorder; groups also get rank tags — images carry none).

import { useEffect, useMemo, useRef, useState } from "react";
import { getJson, putJson } from "../../api/client.ts";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useToastStore } from "../../stores/core/toastStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import type { ImageGroup, ImageInfo } from "../../types.ts";
import {
  groupedFilenameSet,
  groupsInGalleryOrder,
  withLockedGroupsInPlace,
} from "../../utils/groups.ts";
import {
  cn,
  fromGroupSortId,
  fullImageUrl,
  imageUrl,
  isGroupSortId,
  reorderImagesByGroups,
  selectedImageFilenames,
} from "../../utils/helpers.ts";
import {
  applyEvent,
  CONSIDERING_SLOT,
  type CompareMargin,
  type CompareOutcome,
  DELETE_SLOT,
  nearestTier,
  RankEngine,
  type RankEvent,
  type RankScoresFile,
  type RankSnapshot,
} from "../../utils/rankEngine.ts";
import { reorderImagesWithinSlots, reorderSubsetWithinSlots } from "../../utils/reorder.ts";
import {
  type ConfigStore,
  colorForId,
  configOwnedTags,
  type ReviewCategory,
  type ReviewConfig,
  reviewColorVar,
} from "../../utils/reviewConfigs.ts";
import { beginSortTransition } from "../../utils/sortFlip.ts";
import { GroupingSortModal, type SortState } from "./GroupingSortModal.tsx";
import { Modal } from "./Modal.tsx";

// ---- Fixed tier vocabulary --------------------------------------------------
// Category labels double as the persisted tags, so they are namespaced with
// "Rank" to never collide with review-config vocabularies (configOwnedTags
// matches by label).

const TIER_CATEGORY_IDS = ["rank1", "rank2", "rank3", "rank4", "rank5"] as const;
const TIER_LABELS = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5"] as const;
const CONSIDERING_LABEL = "Rank Considering";
const DELETE_LABEL = "Rank Delete";

const CULL_STORAGE_KEY = "reorder.rankCull";
const KIND_STORAGE_KEY = "reorder.rankKind";
const SCOPE_STORAGE_KEY = "reorder.rankScope";
const TOLERANCE_STORAGE_KEY = "reorder.rankTolerance";

/**
 * How exact the ranking needs to be before the engine stops asking. "Good
 * enough" is positional: a target of ±12 on 242 items means each item is
 * typically within 12 places of where unlimited comparisons would put it —
 * #200 vs #201 never matters at that setting. Exact = keep asking until no
 * informative comparison remains (the old behavior). Auto (default) scales
 * with the scope: a list of ≤40 deserves a full sort, and the acceptable
 * slack grows from there — nobody ranking 242 items cares about #200 vs #201.
 */
type RankToleranceMode = "auto" | "exact" | "fine" | "rough";
const TOLERANCE_MODES: { mode: RankToleranceMode; label: string; hint: string }[] = [
  {
    mode: "auto",
    label: "Auto",
    hint: "Scale with size: full sort up to ~40 items, progressively looser beyond",
  },
  { mode: "exact", label: "Exact", hint: "Keep comparing until nothing informative is left" },
  {
    mode: "fine",
    label: "±2%",
    hint: "Stop once items are typically within ±2% of the list length from their final spot",
  },
  {
    mode: "rough",
    label: "±5%",
    hint: "Stop once items are typically within ±5% of the list length from their final spot",
  },
];

/** Target tolerance in list positions (0 = exact, never auto-stop). */
function tolerancePositions(mode: RankToleranceMode, n: number): number {
  // Auto: exact up to 40 items, then 5% of every item beyond — 80 → ±2,
  // 160 → ±6, 242 → ±10, approaching ±5% of the list for very large scopes.
  if (mode === "auto") return n <= 40 ? 0 : Math.round(0.05 * (n - 40));
  if (mode === "fine") return Math.max(2, Math.round(0.02 * n));
  if (mode === "rough") return Math.max(3, Math.round(0.05 * n));
  return 0;
}

/** Which things the session ranks — decides the scores file and what Apply does. */
type RankKind = "groups" | "images";
/**
 * Which of them: everything of that kind, the grid selection, or (images only)
 * the contents of the selected groups.
 */
type RankScope = "all" | "selected" | "contents";
type RankTarget = { kind: RankKind; scope: RankScope };

/** The grid state a session ranks against, captured once when the modal opens. */
interface RankSourceData {
  groups: ImageGroup[];
  images: ImageInfo[];
  selectedGroupIds: Set<string>;
  selectedFilenames: Set<string>;
}

const SCOPES: Record<RankKind, { scope: RankScope; label: string; hint: string }[]> = {
  groups: [
    { scope: "all", label: "All", hint: "Rank every confirmed group" },
    { scope: "selected", label: "Selected", hint: "Rank only the groups selected in the grid" },
  ],
  images: [
    { scope: "all", label: "Ungrouped", hint: "Rank every image that isn't in a group" },
    {
      scope: "selected",
      label: "Selected",
      hint: "Rank only the images selected in the grid (inside a group or not)",
    },
    {
      scope: "contents",
      label: "In groups",
      hint: "Rank the images inside the selected groups",
    },
  ],
};

/** Ungrouped / selected / in-group images become single-image pseudo-groups so
 * the whole modal (chassis, engine, comparison overlay, media rendering) works
 * unchanged — every consumer only touches id / name / images / tags. */
function imageItems(filenames: string[]): ImageGroup[] {
  return filenames.map((fn) => ({ id: fn, name: fn, images: [fn] }));
}

/** The items a target ranks, in gallery order. */
function itemsFor(src: RankSourceData, { kind, scope }: RankTarget): ImageGroup[] {
  const selectedGroups = src.groups.filter((g) => src.selectedGroupIds.has(g.id));
  if (kind === "groups") return scope === "selected" ? selectedGroups : src.groups;
  if (scope === "selected") {
    return imageItems(
      src.images.filter((i) => src.selectedFilenames.has(i.filename)).map((i) => i.filename),
    );
  }
  if (scope === "contents") {
    const present = new Set(src.images.map((i) => i.filename));
    return imageItems(selectedGroups.flatMap((g) => g.images).filter((fn) => present.has(fn)));
  }
  const grouped = groupedFilenameSet(src.groups);
  return imageItems(src.images.filter((i) => !grouped.has(i.filename)).map((i) => i.filename));
}

/** Noun phrase for this target — titles, empty state, confirm prompts. */
function targetNoun({ kind, scope }: RankTarget): string {
  if (kind === "groups") return scope === "selected" ? "selected groups" : "groups";
  if (scope === "selected") return "selected images";
  if (scope === "contents") return "images in the selected groups";
  return "ungrouped images";
}

function targetTitle(target: RankTarget): string {
  const noun = targetNoun(target);
  return `Rank ${noun.charAt(0).toUpperCase()}${noun.slice(1)}`;
}

function buildRankConfig(cull: boolean): ReviewConfig {
  const categories: ReviewCategory[] = TIER_LABELS.map((label, i) => ({
    id: TIER_CATEGORY_IDS[i]!,
    label,
    subcategories: [],
  }));
  if (cull) {
    categories.push(
      { id: "rankConsidering", label: CONSIDERING_LABEL, subcategories: [] },
      { id: "rankDelete", label: DELETE_LABEL, subcategories: [] },
    );
  }
  return {
    id: cull ? "__rank_cull" : "__rank",
    name: "Rank",
    builtIn: true,
    defaultCategoryId: TIER_CATEGORY_IDS[0],
    categories,
  };
}

function categoryIdForSlot(slot: number): string {
  if (slot === CONSIDERING_SLOT) return "rankConsidering";
  if (slot === DELETE_SLOT) return "rankDelete";
  return TIER_CATEGORY_IDS[slot] ?? TIER_CATEGORY_IDS[0];
}

function slotForCategoryId(categoryId: string): number {
  if (categoryId === "rankConsidering") return CONSIDERING_SLOT;
  if (categoryId === "rankDelete") return DELETE_SLOT;
  const idx = TIER_CATEGORY_IDS.indexOf(categoryId as (typeof TIER_CATEGORY_IDS)[number]);
  return idx === -1 ? 0 : idx;
}

/** Recover a rating slot from persisted rank tags (prior sessions' Apply). */
function slotFromTags(tags: string[] | undefined): number | null {
  if (!tags || tags.length === 0) return null;
  const tagSet = new Set(tags);
  if (tagSet.has(DELETE_LABEL)) return DELETE_SLOT;
  if (tagSet.has(CONSIDERING_LABEL)) return CONSIDERING_SLOT;
  for (let i = 0; i < TIER_LABELS.length; i++) {
    if (tagSet.has(TIER_LABELS[i]!)) return i;
  }
  return null;
}

/** The tier vocabulary is fixed, so the chassis gets a read-only store with
 * no persistence — the config picker/editor is hidden anyway. */
function makeStaticStore(config: ReviewConfig): ConfigStore {
  return {
    builtin: config,
    builtinId: config.id,
    loadAllConfigs: () => [config],
    loadInitialConfigs: () => ({ configs: [config], lastId: config.id }),
    loadLastConfigId: () => config.id,
    saveLastConfigId: () => {},
    upsertConfig: (configs) => configs,
    deleteConfigById: (configs) => configs,
  };
}

type RankMode = "auto" | "rate" | "compare";

interface RankModalProps {
  onClose: () => void;
}

export function RankModal({ onClose }: RankModalProps) {
  // Captured once at open: the confirmed groups, the gallery, and the grid
  // selection, so every target ranks a stable snapshot regardless of later edits.
  const [raw] = useState<RankSourceData>(() => {
    const groups = useGroupStore.getState().groups.map((g) => ({ ...g, images: g.images.slice() }));
    const images = useImageStore.getState().images.slice();
    const selected = useSelectionStore.getState().contexts.reorder;
    const groupIds = new Set(groups.map((g) => g.id));
    const filenames = new Set(images.map((i) => i.filename));
    return {
      groups,
      images,
      selectedGroupIds: new Set(
        [...selected]
          .filter(isGroupSortId)
          .map(fromGroupSortId)
          .filter((id) => groupIds.has(id)),
      ),
      selectedFilenames: new Set(
        selectedImageFilenames(selected).filter((fn) => filenames.has(fn)),
      ),
    };
  });

  // A scope with fewer than two items can't be ranked, so its button is disabled
  // — and a scope remembered from last session that's now empty (nothing
  // selected) silently falls back to the kind's default rather than opening onto
  // a dead end.
  const itemsByTarget = useMemo(() => {
    const out = new Map<string, ImageGroup[]>();
    for (const kind of ["groups", "images"] as const) {
      for (const { scope } of SCOPES[kind]) {
        out.set(`${kind}:${scope}`, itemsFor(raw, { kind, scope }));
      }
    }
    return out;
  }, [raw]);
  const itemsAt = (t: RankTarget) => itemsByTarget.get(`${t.kind}:${t.scope}`) ?? [];
  const isRankable = (t: RankTarget) => itemsAt(t).length >= 2;

  const [target, setTarget] = useState<RankTarget>(() => {
    let kind: RankKind = "groups";
    let scope: RankScope = "all";
    try {
      if (localStorage.getItem(KIND_STORAGE_KEY) === "images") kind = "images";
      const saved = localStorage.getItem(SCOPE_STORAGE_KEY);
      if (saved === "selected" || (saved === "contents" && kind === "images")) scope = saved;
    } catch {}
    const items = itemsByTarget.get(`${kind}:${scope}`) ?? [];
    return { kind, scope: items.length >= 2 ? scope : "all" };
  });
  const isImages = target.kind === "images";

  const snapshot = itemsAt(target);
  const scopeIds = useMemo(() => snapshot.map((g) => g.id), [snapshot]);
  const groupById = useMemo(() => new Map(snapshot.map((g) => [g.id, g])), [snapshot]);

  // Every id of this kind the session could speak for. The engine loads (and
  // must write back) beliefs about all of them, not just the scope — otherwise
  // ranking one group's contents would prune every other photo's score off disk.
  const persistIds = useMemo(
    () =>
      new Set(
        target.kind === "images" ? raw.images.map((i) => i.filename) : raw.groups.map((g) => g.id),
      ),
    [raw, target.kind],
  );

  const [cull, setCull] = useState(() => {
    try {
      return localStorage.getItem(CULL_STORAGE_KEY) === "1";
    } catch {
      return false;
    }
  });
  // undefined = still loading from the server
  const [savedFile, setSavedFile] = useState<RankScoresFile | null | undefined>(undefined);

  const engineRef = useRef<RankEngine | null>(null);
  const [engineTick, setEngineTick] = useState(0);
  const bump = () => setEngineTick((t) => t + 1);

  // ---- Target precision & the positional meter -----------------------------
  const [tolMode, setTolMode] = useState<RankToleranceMode>(() => {
    try {
      const saved = localStorage.getItem(TOLERANCE_STORAGE_KEY);
      if (saved === "auto" || saved === "exact" || saved === "fine" || saved === "rough")
        return saved;
    } catch {}
    return "auto";
  });
  // One "good enough" toast per target/session; picking a new target re-arms it.
  const tolToastShownRef = useRef(false);
  function setTolerance(m: RankToleranceMode) {
    setTolMode(m);
    tolToastShownRef.current = false;
    try {
      localStorage.setItem(TOLERANCE_STORAGE_KEY, m);
    } catch {}
  }
  // Expected placement uncertainty of the current ranking, in list positions —
  // the meter headline and the stop test.
  // biome-ignore lint/correctness/useExhaustiveDependencies: engineTick is the change signal for the mutable engine ref
  const posSummary = useMemo(
    () =>
      engineRef.current?.positionalSummary(scopeIds) ?? {
        meanDisplacement: Infinity,
        tiers: [] as { tier: number; n: number; disp: number }[],
      },
    [engineTick, scopeIds],
  );
  const meanDisp = posSummary.meanDisplacement;
  const tolPlaces = tolerancePositions(tolMode, scopeIds.length);
  // Auto also demands every rank be settled to about half a third of itself —
  // the global mean can look fine while one rank (especially a small one,
  // which barely moves the mean) is still fuzzy. "Right rank, right part of
  // the rank" is the user's actual deliverable; ±half a third is its width.
  const tierGuardOk =
    tolMode !== "auto" || posSummary.tiers.every((t) => t.n < 2 || t.disp <= Math.max(1, t.n / 6));
  const tolReached = tolPlaces > 0 && meanDisp <= tolPlaces && tierGuardOk;

  const [mode, setMode] = useState<RankMode>("auto");

  // ---- Misfit review --------------------------------------------------------
  // Items whose comparisons disagree with their rated tier. A corrected rating
  // moves an item globally where a comparison only nudges it, so stepping
  // through these is the densest re-anchoring the user can do. The review is a
  // Rate-mode overlay on navigation: ←/→ walk the misfits worst-first, the card
  // subtitle shows the rated tier and the direction the evidence points, and
  // re-rating (or ⇧-confirming the original tier) resolves the item.
  // biome-ignore lint/correctness/useExhaustiveDependencies: engineTick is the change signal for the mutable engine ref
  const misfitList = useMemo(
    () => engineRef.current?.misfits(scopeIds) ?? [],
    [engineTick, scopeIds],
  );
  const misfitById = useMemo(() => new Map(misfitList.map((m) => [m.id, m])), [misfitList]);
  const [misfitReview, setMisfitReview] = useState(false);
  const [misfitPos, setMisfitPos] = useState(0);
  // The review lives in Rate mode; leaving it (or running out of misfits)
  // ends the review.
  useEffect(() => {
    if (mode !== "rate") setMisfitReview(false);
  }, [mode]);
  useEffect(() => {
    if (misfitReview && misfitList.length === 0) setMisfitReview(false);
  }, [misfitReview, misfitList]);
  const modeRef = useRef(mode);
  modeRef.current = mode;
  // Auto mode only auto-serves comparisons after the user has done something
  // this session — otherwise resuming a fully-rated session would open the
  // modal straight into a comparison overlay.
  const interactedRef = useRef(false);
  // Bumped by "Start over" to force a fresh chassis mount (clears the chips).
  const [resetNonce, setResetNonce] = useState(0);

  // ---- Session timeline --------------------------------------------------
  // Every action this session — each tier rating and each comparison answer —
  // in the exact order it was shown. The engine's live scores = the session's
  // base snapshot (state when the modal opened) replayed through the timeline,
  // so any past entry can be re-rated / re-answered and the rest replay on top.
  // `cursor` is the view pointer: cursor === timeline.length is the live
  // frontier (rating new items / serving the next comparison); a smaller value
  // is reviewing that entry. Back/forward walk the timeline; in every mode but
  // Auto they skip entries of the other kind, so Auto crosses ratings and
  // comparisons in real order while Compare/Rate stay within their own.
  const timelineRef = useRef<RankEvent[]>([]);
  const cursorRef = useRef(0);
  const baseSnapshotRef = useRef<RankSnapshot | null>(null);
  // In Auto mode the chassis item shown for forward rating (the "live" card).
  const frontierItemRef = useRef(0);
  const [compareTick, setCompareTick] = useState(0);
  const bumpCompare = () => setCompareTick((t) => t + 1);

  function eventInMode(ev: RankEvent, m: RankMode): boolean {
    if (m === "auto") return true;
    return m === "compare" ? ev.kind === "compare" : ev.kind === "rate";
  }

  // Rebuild the engine = session base snapshot + every timeline entry replayed
  // in order. The single source of truth after any edit.
  function replayEngine() {
    const engine = engineRef.current;
    const base = baseSnapshotRef.current;
    if (!engine || !base) return;
    engine.restore(base);
    for (const ev of timelineRef.current) applyEvent(engine, ev);
  }

  function goToFrontier() {
    cursorRef.current = timelineRef.current.length;
    bumpCompare();
  }

  // Append the engine's next most-informative comparison and show it. False
  // when nothing is worth asking.
  function serveComparison(): boolean {
    const engine = engineRef.current;
    if (!engine) return false;
    const q = engine.nextComparison(scopeIds);
    if (!q) return false;
    timelineRef.current.push({ kind: "compare", ids: q.ids, outcome: null });
    cursorRef.current = timelineRef.current.length - 1;
    bumpCompare();
    return true;
  }

  // First item at or after `from` that has no tier yet (skips items rated this
  // or a prior session), so Auto's forward rating never re-walks rated items.
  function firstUnratedFrom(from: number): number {
    const engine = engineRef.current;
    for (let i = Math.max(0, from); i < scopeIds.length; i++) {
      if (engine?.get(scopeIds[i]!)?.tier === undefined) return i;
    }
    return Math.max(0, scopeIds.length - 1); // all rated: sit on the last item
  }

  // Step through the timeline, skipping entries that don't belong to the
  // current mode; forward past the last entry lands on the live frontier.
  function step(delta: number) {
    const tl = timelineRef.current;
    let i = cursorRef.current;
    if (delta < 0) {
      i -= 1;
      while (i >= 0 && !eventInMode(tl[i]!, mode)) i -= 1;
      if (i < 0) return; // nothing earlier in this mode
      cursorRef.current = i;
    } else {
      i += 1;
      while (i < tl.length && !eventInMode(tl[i]!, mode)) i += 1;
      cursorRef.current = Math.min(i, tl.length); // clamp to frontier
    }
    bumpCompare();
  }

  const config = useMemo(() => buildRankConfig(cull), [cull]);
  const store = useMemo(() => makeStaticStore(config), [config]);

  // Groups and images persist to separate files (see rankScoresPath); the scopes
  // within a kind share one file. Reload on every target change — including a
  // scope change inside the same kind, since the engine is rebuilt from this
  // file and a stale copy would replay the session's observations away. The
  // outgoing flush is awaited first so the GET can't overtake it.
  const scoresEndpoint = isImages ? "/api/rank-scores?target=images" : "/api/rank-scores";
  const [reloadNonce, setReloadNonce] = useState(0);
  const pendingFlushRef = useRef<Promise<void>>(Promise.resolve());
  // biome-ignore lint/correctness/useExhaustiveDependencies: reloadNonce is the refetch signal for a scope change within the same kind — the endpoint doesn't change, but the file behind it just did
  useEffect(() => {
    let cancelled = false;
    setSavedFile(undefined);
    pendingFlushRef.current
      .then(() => getJson<RankScoresFile | null>(scoresEndpoint))
      .then((f) => {
        if (!cancelled) setSavedFile(f);
      })
      .catch(() => {
        if (!cancelled) setSavedFile(null);
      });
    return () => {
      cancelled = true;
    };
  }, [scoresEndpoint, reloadNonce]);

  // One id per modal opening: re-saving the session upserts its record rather
  // than logging the same judgements twice when the user edits a past answer.
  const sessionIdRef = useRef(crypto.randomUUID());
  const sessionStartRef = useRef(new Date().toISOString());

  // ---- Autosave (debounced; flushed on unmount and on Apply) ----
  const saveTimerRef = useRef<number | null>(null);
  const persistRef = useRef(async () => {});
  persistRef.current = async () => {
    const engine = engineRef.current;
    if (!engine) return;
    // persistIds, not the scope: the payload is authoritative for everything of
    // this kind that's on disk (see fs/rank-scores.ts), so a scoped session
    // writes its neighbours' beliefs back unchanged instead of pruning them away.
    const saves = [
      putJson(scoresEndpoint, engine.toJSON(persistIds)).catch(() => {
        // Autosave is best-effort; Apply re-saves and surfaces nothing here.
      }),
    ];
    // The scores are derived and lossy; these are the observations behind them,
    // and the only thing that can ever tell us whether the model is any good
    // (scripts/rank-calibration.ts). Losing them is unrecoverable, so they ride
    // along with every autosave.
    if (timelineRef.current.length > 0) {
      saves.push(
        putJson("/api/rank-judgements", {
          id: sessionIdRef.current,
          // The log keys judgements by kind — a comparison between two photos is
          // the same evidence whichever scope served it.
          target: target.kind,
          at: sessionStartRef.current,
          events: timelineRef.current,
        })
          .then(() => {})
          .catch(() => {}),
      );
    }
    await Promise.all(saves);
  };
  function scheduleSave() {
    if (saveTimerRef.current !== null) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = window.setTimeout(() => persistRef.current(), 800);
  }
  useEffect(
    () => () => {
      if (saveTimerRef.current !== null) clearTimeout(saveTimerRef.current);
      persistRef.current();
    },
    [],
  );

  // Build (or rebuild) the engine as soon as the saved scores for the current
  // target finish loading — there's no separate "Start" step; the modal drops
  // straight into ranking. Groups without saved scores seed from their
  // persisted rank tags (a previous Apply) as an ordinary noisy rating;
  // ungrouped images carry no tags, so that only applies to the groups target.
  // savedFile reloads on every target switch, so it's the real rebuild signal;
  // snapshot identity is stable within a target.
  useEffect(() => {
    if (savedFile === undefined) return;
    const engine = RankEngine.fromJSON(savedFile ?? null);
    if (!isImages) {
      for (const g of snapshot) {
        if (engine.has(g.id)) continue;
        const slot = slotFromTags(g.tags);
        if (slot !== null) engine.observeRating(g.id, slot, false);
      }
    }
    engineRef.current = engine;
    interactedRef.current = false;
    tolToastShownRef.current = false;
    // The session base is the seeded engine; the timeline replays on top of it.
    baseSnapshotRef.current = engine.snapshot();
    timelineRef.current = [];
    cursorRef.current = 0;
    // A cleared timeline is a new session for the log — otherwise switching
    // target would upsert the new target's events over the old target's record.
    sessionIdRef.current = crypto.randomUUID();
    sessionStartRef.current = new Date().toISOString();
    // Start Auto's forward rating on the first item that isn't rated yet.
    let f = 0;
    while (f < scopeIds.length && engine.get(scopeIds[f]!)?.tier !== undefined) f++;
    frontierItemRef.current = Math.min(f, Math.max(0, scopeIds.length - 1));
    setCompareTick((t) => t + 1);
    setEngineTick((t) => t + 1);
  }, [savedFile, isImages, snapshot, scopeIds]);

  function flushNow(): Promise<void> {
    if (saveTimerRef.current !== null) {
      clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }
    return persistRef.current();
  }

  // Switching target flushes the current engine first, then drops it so the
  // reload effect refetches and rebuilds for the new scope. The reload waits on
  // the flush, so the session just left can't be read back half-written.
  function switchTarget(t: RankTarget) {
    if (t.kind === target.kind && t.scope === target.scope) return;
    pendingFlushRef.current = flushNow();
    engineRef.current = null;
    setTarget(t);
    setReloadNonce((n) => n + 1);
    try {
      localStorage.setItem(KIND_STORAGE_KEY, t.kind);
      localStorage.setItem(SCOPE_STORAGE_KEY, t.scope);
    } catch {}
  }

  function toggleCull(next: boolean) {
    setCull(next);
    try {
      localStorage.setItem(CULL_STORAGE_KEY, next ? "1" : "0");
    } catch {}
  }

  // Discard the scores for the current scope. Only the scope: the file holds
  // this kind's beliefs about everything (other groups, the photos in other
  // groups), and those are irreplaceable human input — so the engine forgets the
  // scope and is written back, rather than being replaced by a blank one.
  function startOver() {
    const engine = engineRef.current;
    if (!engine) return;
    if (!window.confirm(`Discard all ranking progress for the ${targetNoun(target)}?`)) return;
    engine.clearScope(scopeIds);
    interactedRef.current = false;
    tolToastShownRef.current = false;
    baseSnapshotRef.current = engine.snapshot();
    timelineRef.current = [];
    cursorRef.current = 0;
    // Start over discards the *scores*, but the judgements already logged are
    // evidence about the model and stay — a fresh session id keeps them intact.
    sessionIdRef.current = crypto.randomUUID();
    sessionStartRef.current = new Date().toISOString();
    frontierItemRef.current = 0;
    setCompareTick((t) => t + 1);
    setResetNonce((n) => n + 1);
    putJson(scoresEndpoint, engine.toJSON(persistIds)).catch(() => {});
    setEngineTick((t) => t + 1);
  }

  // Seed the chassis' chip state from explicit ratings (saved or tag-derived).
  function initialStateFor(cfg: ReviewConfig): SortState | null {
    const engine = engineRef.current;
    if (!engine) return null;
    const statuses = new Map<string, string>();
    for (const g of snapshot) {
      const tier = engine.get(g.id)?.tier;
      if (tier === undefined) continue;
      const catId = categoryIdForSlot(tier);
      if (cfg.categories.some((c) => c.id === catId)) statuses.set(g.id, catId);
    }
    return { statuses, subs: new Map() };
  }

  function handleAssign(id: string, categoryId: string | null, meta: { confident: boolean }) {
    const engine = engineRef.current;
    if (!engine || categoryId === null) return;
    interactedRef.current = true;
    const slot = slotForCategoryId(categoryId);
    const tl = timelineRef.current;
    const reviewing = cursorRef.current < tl.length;
    // Keep one rating entry per item, at its place in the timeline: edit if the
    // item already has one (re-rating a past item), otherwise append.
    const existing = tl.findIndex((e) => e.kind === "rate" && e.id === id);
    if (existing >= 0) {
      const ev = tl[existing]!;
      if (ev.kind === "rate") {
        ev.slot = slot;
        ev.confident = meta.confident;
      }
    } else {
      tl.push({ kind: "rate", id, slot, confident: meta.confident });
    }
    replayEngine();
    bump();
    scheduleSave();
    if (mode !== "auto") {
      // Rate mode: the chassis owns its own item navigation; keep cursor live.
      cursorRef.current = tl.length;
      return;
    }
    if (reviewing) {
      step(1); // re-rated a past item under review → advance along the timeline
      return;
    }
    // Forward rating of a new item: advance the frontier to the next unrated
    // item. No comparisons are interleaved — Auto is a single handoff, rate
    // everything then compare (the serve effect below takes over once the last
    // item is rated); anyone who wants to compare sooner switches mode.
    frontierItemRef.current = firstUnratedFrom(frontierItemRef.current + 1);
    goToFrontier();
  }

  // Every item explicitly rated (a comparison-created entry has no tier, so
  // compare-first sessions don't count as rated). This is Auto's handoff test:
  // its rating pass isn't done until every item has a tier.
  function allRated(): boolean {
    const engine = engineRef.current;
    return engine !== null && scopeIds.every((id) => engine.get(id)?.tier !== undefined);
  }

  // At the live frontier, Compare mode (and Auto once everything is rated) keeps
  // serving the next informative comparison — until the target precision is
  // reached, which is the "good enough" stop: finer order exists but the user
  // said it doesn't matter at this scope size.
  // biome-ignore lint/correctness/useExhaustiveDependencies: engineTick/compareTick are the change signals for the mutable engine + timeline refs — new observations must re-evaluate whether a question is worth asking
  useEffect(() => {
    if (cursorRef.current < timelineRef.current.length) return; // reviewing, not live
    if (!engineRef.current) return;
    if (mode !== "compare" && !(mode === "auto" && allRated() && interactedRef.current)) return;
    if (tolReached) {
      if (!tolToastShownRef.current) {
        tolToastShownRef.current = true;
        useToastStore
          .getState()
          .showToast(
            `Good enough — every ${isImages ? "image" : "group"} is typically within ±${tolPlaces} places of its final spot. Switch the target to Exact to keep refining.`,
            "success",
          );
      }
      if (mode === "compare") setMode("auto");
      return;
    }
    if (!serveComparison() && mode === "compare") {
      useToastStore
        .getState()
        .showToast("No informative comparisons left — the ranking is settled", "success");
      setMode("auto");
    }
  }, [mode, scopeIds, engineTick, compareTick, tolReached, tolPlaces, isImages]);

  // Record the answer for the comparison at the cursor, replay, then step
  // forward one entry (to the next timeline card, or the live frontier).
  function recordOutcome(outcome: CompareOutcome) {
    const ev = timelineRef.current[cursorRef.current];
    if (!ev || ev.kind !== "compare") return;
    ev.outcome = outcome;
    replayEngine();
    bump();
    scheduleSave();
    step(1);
  }

  function answerWin(winnerId: string, margin?: CompareMargin) {
    recordOutcome({ kind: "win", winnerId, ...(margin && { margin }) });
  }
  function answerTop(ids: string[], margin?: CompareMargin) {
    recordOutcome({ kind: "top", ids, ...(margin && { margin }) });
  }
  function answerTie() {
    recordOutcome({ kind: "tie" });
  }
  function answerSkip() {
    recordOutcome({ kind: "skip" });
  }
  function pinFromQuestion(id: string) {
    recordOutcome({ kind: "pin", id });
  }

  // Esc / "Back to rating" jumps to the live frontier. In Compare mode (or Auto
  // once all are rated) the serve effect would immediately reopen, so a manual
  // close also drops to Rate mode.
  function dismissQuestion() {
    if (mode === "compare" || (mode === "auto" && allRated())) setMode("rate");
    goToFrontier();
  }

  function groupLightboxItems(groupImages: string[]): string[] {
    const imageMap = useImageStore.getState().imageMap;
    return groupImages.filter((fn) => imageMap.has(fn));
  }

  function openGroupLightbox(groupImages: string[], index: number) {
    const items = groupLightboxItems(groupImages);
    if (items.length === 0) return;
    useLightboxStore.getState().openLightbox(items, index);
  }

  // Reorder the ranked images into ranked order among the slots they already
  // occupy — everything outside the scope (other images, whole groups) stays
  // exactly where it sits. Images carry no persistent tags, so nothing but order
  // changes. A scope that reaches inside a group (selected images, or a group's
  // contents) also rewrites that group's own image list, which is what the
  // popover and the rename order read from — same as Sort Similar's selection
  // scope.
  function applyRankingImages() {
    const engine = engineRef.current;
    if (!engine) return;
    const { images, imageMap, setImages } = useImageStore.getState();
    const movable = scopeIds.filter((fn) => imageMap.has(fn));
    if (movable.length === 0) {
      onClose();
      return;
    }
    const ranked = engine.ranking(movable);
    beginSortTransition();
    setImages(reorderImagesWithinSlots(images, ranked));
    useGroupStore.getState().updateGroups((prev) => {
      let changed = false;
      const next = prev.map((g) => {
        const reordered = reorderSubsetWithinSlots(g.images, ranked);
        if (reordered.every((fn, i) => fn === g.images[i])) return g;
        changed = true;
        return { ...g, images: reordered };
      });
      return changed ? next : prev;
    });
    persistRef.current();
    onClose();
  }

  /** The scope's groups in ranked order, each carrying its refreshed rank tag. */
  function rankedTaggedGroups(engine: RankEngine): ImageGroup[] {
    const ordered = engine
      .ranking(scopeIds)
      .map((id) => groupById.get(id))
      .filter((g): g is ImageGroup => g !== undefined);

    // Own the full vocabulary (floors included) even in a plain session so
    // stale floor tags from an earlier cull session get replaced too.
    const owned = configOwnedTags(buildRankConfig(true));
    return ordered.map((g) => {
      const e = engine.get(g.id);
      const newTag = !e
        ? null
        : e.floor === "delete"
          ? DELETE_LABEL
          : e.floor === "considering"
            ? CONSIDERING_LABEL
            : TIER_LABELS[nearestTier(e.mu)]!;
      const kept = (g.tags ?? []).filter((t) => !owned.has(t));
      const finalTags = newTag ? Array.from(new Set([...kept, newTag])) : kept;
      if (finalTags.length === 0) {
        return g.tags && g.tags.length > 0 ? { ...g, tags: undefined } : g;
      }
      return { ...g, tags: finalTags };
    });
  }

  function applyRankingGroups() {
    const engine = engineRef.current;
    if (!engine) return;
    const tagged = rankedTaggedGroups(engine);
    const allGroups = useGroupStore.getState().groups;
    const { images, imageMap, setImages } = useImageStore.getState();
    beginSortTransition();

    if (target.scope === "selected") {
      // Only the ranked groups move, and only among the gallery slots their own
      // images already occupy: every other group and every ungrouped image keeps
      // its place, so ranking a handful of groups can't repack the whole gallery.
      const scopeSet = new Set(scopeIds);
      const currentOrder = groupsInGalleryOrder(allGroups, images).filter((g) =>
        scopeSet.has(g.id),
      );
      const finalOrder = withLockedGroupsInPlace(currentOrder, tagged);
      setImages(
        reorderImagesWithinSlots(
          images,
          finalOrder.flatMap((g) => g.images),
        ),
      );
      const byId = new Map(finalOrder.map((g) => [g.id, g]));
      useGroupStore.getState().updateGroups((prev) => prev.map((g) => byId.get(g.id) ?? g));
    } else {
      // Gallery-locked groups keep their relative order; the ranking fills in
      // around them, with unlocked groups free to interleave between them.
      const finalOrder = withLockedGroupsInPlace(groupsInGalleryOrder(allGroups, images), tagged);
      setImages(reorderImagesByGroups(images, imageMap, finalOrder));
      useGroupStore.getState().updateGroups(() => finalOrder);
    }
    persistRef.current();
    onClose();
  }

  function applyRanking() {
    if (isImages) return applyRankingImages();
    return applyRankingGroups();
  }

  // Shared scope / cull / start-over controls — live in the header while
  // ranking, and in the not-ready shell below so you can switch scope without
  // a separate setup step. Switching kind keeps the current scope when that
  // scope has something to rank, and falls back to the kind's default otherwise.
  function renderControls() {
    return (
      <span className="rank-controls">
        <span className="rank-target-toggle" role="radiogroup" aria-label="What to rank">
          {(["groups", "images"] as const).map((kind) => (
            <button
              type="button"
              key={kind}
              role="radio"
              aria-checked={target.kind === kind}
              className={cn("rank-target-btn", target.kind === kind && "rank-target-btn-active")}
              onClick={() =>
                switchTarget(
                  isRankable({ kind, scope: target.scope })
                    ? { kind, scope: target.scope }
                    : { kind, scope: "all" },
                )
              }
            >
              {kind === "groups" ? "Groups" : "Images"}
            </button>
          ))}
        </span>
        <span className="rank-target-toggle" role="radiogroup" aria-label="Which of them to rank">
          {SCOPES[target.kind].map(({ scope, label, hint }) => {
            const t: RankTarget = { kind: target.kind, scope };
            const count = itemsAt(t).length;
            const enabled = count >= 2;
            return (
              <button
                type="button"
                key={scope}
                role="radio"
                aria-checked={target.scope === scope}
                disabled={!enabled}
                className={cn(
                  "rank-target-btn",
                  target.scope === scope && "rank-target-btn-active",
                )}
                title={
                  enabled
                    ? `${hint} (${count})`
                    : scope === "contents"
                      ? "Select one or more groups holding at least two images between them"
                      : `Select at least two ${target.kind === "groups" ? "groups" : "images"} in the grid`
                }
                onClick={() => switchTarget(t)}
              >
                {label}
              </button>
            );
          })}
        </span>
        <span className="rank-target-toggle" role="radiogroup" aria-label="Target precision">
          {TOLERANCE_MODES.map(({ mode: m, label, hint }) => {
            const places = tolerancePositions(m, scopeIds.length);
            return (
              <button
                type="button"
                key={m}
                role="radio"
                aria-checked={tolMode === m}
                className={cn("rank-target-btn", tolMode === m && "rank-target-btn-active")}
                title={`${hint} (${places === 0 ? "exact" : `±${places}`} here)`}
                onClick={() => setTolerance(m)}
              >
                {label}
              </button>
            );
          })}
        </span>
        <button
          type="button"
          className={cn("rank-mode-btn rank-toggle-btn", cull && "rank-mode-btn-active")}
          onClick={() => toggleCull(!cull)}
          title={`Cull floor — add ${CONSIDERING_LABEL} (6) and ${DELETE_LABEL} (7) below the tiers`}
        >
          Cull
        </button>
        <button
          type="button"
          className="rank-mode-btn rank-toggle-btn"
          onClick={startOver}
          title={`Discard the scores for the ${targetNoun(target)} and start fresh`}
        >
          Start over
        </button>
      </span>
    );
  }

  // ---- Not-ready / nothing-to-rank shell (no separate setup step) ----
  if (savedFile === undefined || !engineRef.current || snapshot.length < 2) {
    return (
      <Modal
        title={targetTitle(target)}
        onClose={onClose}
        className="rank-setup-modal"
        footer={
          <>
            <span className="modal-footer-spacer" />
            <button type="button" className="btn btn-secondary" onClick={onClose}>
              Close
            </button>
          </>
        }
      >
        <div className="rank-setup-body">
          {renderControls()}
          <p className="rank-setup-status">
            {snapshot.length < 2
              ? `Fewer than two ${targetNoun(target)} — nothing to rank.`
              : "Loading saved scores…"}
          </p>
        </div>
      </Modal>
    );
  }

  // ---- Main phase ----
  const engine = engineRef.current!;
  // compareTick is read so this render tracks timeline/cursor ref mutations.
  void compareTick;
  const timeline = timelineRef.current;
  const cursor = cursorRef.current;
  const currentEvent = cursor < timeline.length ? timeline[cursor]! : null;
  const activeComparison = currentEvent?.kind === "compare" ? currentEvent : null;
  const questionGroups = activeComparison
    ? activeComparison.ids
        .map((id) => groupById.get(id))
        .filter((g): g is ImageGroup => g !== undefined)
    : null;

  // In Auto mode the modal drives which item the rating chassis shows, so one
  // Back/Forward crosses between ratings and comparisons in presentation order.
  // Reviewing a past rating points the chassis at that item; otherwise it sits
  // on the live frontier item. Other modes use the chassis's own navigation.
  const chassisItemIndex =
    currentEvent?.kind === "rate"
      ? Math.max(0, scopeIds.indexOf(currentEvent.id))
      : frontierItemRef.current;
  // Misfit review drives the chassis through the misfit list (worst-first);
  // rating the shown item replays the engine, the list recomputes, and the
  // resolved item drops out — the same position then shows the next one.
  const misfitIdx = Math.min(misfitPos, Math.max(0, misfitList.length - 1));
  const navControl =
    mode === "auto"
      ? {
          index: Math.min(chassisItemIndex, Math.max(0, scopeIds.length - 1)),
          onStep: step,
        }
      : mode === "rate" && misfitReview && misfitList.length > 0
        ? {
            index: Math.max(0, scopeIds.indexOf(misfitList[misfitIdx]!.id)),
            onStep: (d: number) =>
              setMisfitPos(Math.max(0, Math.min(misfitList.length - 1, misfitIdx + d))),
          }
        : undefined;

  // Overlay position + arrows: "N / M" counts comparisons; the arrows walk the
  // timeline (mode-filtered), so in Auto ← from a comparison can step back into
  // the ratings that preceded it.
  const comparisons = timeline.filter((e) => e.kind === "compare");
  const comparePos = activeComparison ? comparisons.indexOf(activeComparison) : -1;
  const canStepPrev = timeline.slice(0, cursor).some((e) => eventInMode(e, mode));

  return (
    <>
      <GroupingSortModal<ImageGroup>
        key={`${target.kind}:${target.scope}:${cull ? "cull" : "plain"}:${resetNonce}`}
        store={store}
        items={snapshot}
        getId={(g) => g.id}
        getName={(g) => g.name}
        initialStateFor={initialStateFor}
        defaultCategoryId={() => null}
        terms={{ group: "tier", sub: "refinement" }}
        titleText={(b) => (b ? `Browsing ${b}` : targetTitle(target))}
        emptyText={(b) =>
          b
            ? `No ${isImages ? "images" : "groups"} in ${b}.`
            : `No ${isImages ? "images" : "groups"} to rank.`
        }
        subBannerLabel={(label) => (
          <>
            Browsing <strong>{label}</strong>
          </>
        )}
        chipTitle={(label, isActive) => (isActive ? `Exit ${label}` : `Browse ${label}`)}
        tailHint="⇧ sure · Esc close"
        hideConfigControls
        shiftConfident
        toggleOffOnRepeat={false}
        suspendKeys={activeComparison !== null}
        navControl={navControl}
        onAssign={handleAssign}
        renderProgressExtra={() => (
          <span className="rank-progress-extra">
            {renderControls()}
            <span
              className="rank-mode-toggle"
              title="Auto rates everything first, then moves to comparisons. Rate and Compare each stick to one kind of question — Compare works fine on unrated items."
            >
              {(["auto", "rate", "compare"] as const).map((m) => (
                <button
                  type="button"
                  key={m}
                  className={cn("rank-mode-btn", mode === m && "rank-mode-btn-active")}
                  onClick={() => setMode(m)}
                >
                  {m === "auto" ? "Auto" : m === "rate" ? "Rate" : "Compare"}
                </button>
              ))}
            </span>
            {misfitList.length > 0 && (
              <button
                type="button"
                className={cn(
                  "rank-mode-btn rank-toggle-btn rank-misfit-btn",
                  misfitReview && "rank-mode-btn-active",
                )}
                onClick={() => {
                  if (misfitReview) {
                    setMisfitReview(false);
                    return;
                  }
                  setMode("rate");
                  setMisfitPos(0);
                  setMisfitReview(true);
                }}
                title="Review items whose comparisons disagree with their rated tier — ←/→ steps through them (worst first); each card shows its current tier and the direction the evidence points. Re-rate to correct, ⇧+same tier to insist, or step past to leave it."
              >
                Misfits ({misfitList.length})
              </button>
            )}
            <RankProgressMeter
              disp={meanDisp}
              tol={tolPlaces}
              unrated={scopeIds.length - engine.observedIds(scopeIds).length}
              total={scopeIds.length}
              tiers={posSummary.tiers}
              guardOk={tierGuardOk}
            />
          </span>
        )}
        renderSubtitle={(group, info) => {
          const estRank = engine.estimatedRank(group.id, scopeIds);
          const entry = engine.get(group.id);
          return (
            <>
              {isImages ? (
                estRank !== null ? (
                  `est. #${estRank} of ${scopeIds.length}`
                ) : (
                  "unrated"
                )
              ) : (
                <>
                  {group.images.length} image{group.images.length === 1 ? "" : "s"}
                  {estRank !== null && (
                    <>
                      {" "}
                      · est. #{estRank} of {scopeIds.length}
                    </>
                  )}
                </>
              )}
              {entry?.pinned && " · pinned"}
              {info.category && (
                <>
                  {" · "}
                  <span
                    className="review-single-tag"
                    style={reviewColorVar(
                      colorForId(info.category.id, info.config.categories.indexOf(info.category)),
                    )}
                  >
                    {info.category.label}
                    {entry?.confident && " ✓"}
                  </span>
                </>
              )}
              {(() => {
                const mis = misfitById.get(group.id);
                if (!mis) return null;
                return (
                  <>
                    {" · "}
                    <span
                      className="rank-misfit-note"
                      title={`Rated ${TIER_LABELS[mis.tier]}, but its comparisons place it ~${Math.round(mis.p * 100)}% likely ${mis.suggested < mis.tier ? "higher" : "lower"} — re-rate, or ⇧+${mis.tier + 1} to insist`}
                    >
                      comparisons say {TIER_LABELS[mis.suggested]}{" "}
                      {mis.suggested < mis.tier ? "↑" : "↓"}
                    </span>
                  </>
                );
              })()}
            </>
          );
        }}
        getLightboxTarget={(group) => {
          const items = groupLightboxItems(group.images);
          return items.length > 0 ? { filenames: items, index: 0 } : null;
        }}
        modalClassName={isImages ? "rank-images-modal" : undefined}
        renderMedia={(group) =>
          isImages ? (
            // One image per item — fill the card with the whole full-size photo
            // (Create Groups' single-photo layout), not a cropped thumb grid.
            <div className="cg-single-image">
              <button
                type="button"
                onClick={() => openGroupLightbox(group.images, 0)}
                aria-label={`Open ${group.images[0]}`}
              >
                <img src={fullImageUrl(group.images[0]!)} alt="" draggable={false} />
              </button>
            </div>
          ) : (
            <div className="review-single-thumbs">
              {group.images.map((fn, i) => (
                <button
                  type="button"
                  key={fn}
                  className="review-single-thumb"
                  onClick={() => openGroupLightbox(group.images, i)}
                  aria-label={`Open ${fn}`}
                >
                  <img src={imageUrl(fn)} alt="" loading="lazy" draggable={false} />
                </button>
              ))}
            </div>
          )
        }
        apply={{
          describe: () => ({
            label: "Apply Ranking",
            disabled: engine.observedIds(scopeIds).length === 0,
            title:
              engine.observedIds(scopeIds).length === 0
                ? `Rate or compare at least one ${isImages ? "image" : "group"} first`
                : isImages
                  ? `Reorder the ${targetNoun(target)} by the current ranking`
                  : `Reorder the ${targetNoun(target)} by the current ranking and write rank tags`,
          }),
          run: applyRanking,
        }}
        onClose={onClose}
      />

      {activeComparison && questionGroups && questionGroups.length >= 2 && (
        <RankCompareOverlay
          groups={questionGroups}
          outcome={activeComparison.outcome}
          index={comparePos}
          count={comparisons.length}
          canPrev={canStepPrev}
          canNext
          imageMode={isImages}
          disp={meanDisp}
          tol={tolPlaces}
          unrated={scopeIds.length - engine.observedIds(scopeIds).length}
          total={scopeIds.length}
          tiers={posSummary.tiers}
          guardOk={tierGuardOk}
          onWin={answerWin}
          onTop={answerTop}
          onTie={answerTie}
          onSkip={answerSkip}
          onPin={pinFromQuestion}
          onPrev={() => step(-1)}
          onNext={() => step(1)}
          onOpenLightbox={openGroupLightbox}
          onClose={dismissQuestion}
        />
      )}
    </>
  );
}

// ---- Shared positional meter (header + comparison overlay) ------------------
// "±N places": each item is typically within N positions of where unlimited
// comparisons would put it (RankEngine.positionalSummary). This replaced the
// old %-of-boundaries-settled meter, which read ~0 on large scopes until the
// entire within-tier order was ground out — honest, but it made real progress
// (a 0.8+ rank correlation) look like the system knew nothing.

function RankProgressMeter({
  disp,
  tol,
  unrated,
  total,
  tiers,
  guardOk = true,
}: {
  /** Mean placement uncertainty in list positions. */
  disp: number;
  /** Target tolerance in positions; 0 = exact (no auto-stop). */
  tol: number;
  unrated: number;
  total: number;
  /** Per-rank displacement breakdown, for the tooltip. */
  tiers?: { tier: number; n: number; disp: number }[];
  /** Auto's per-rank guard: false while some rank is still fuzzier than ~half a third. */
  guardOk?: boolean;
}) {
  const reached = tol > 0 && disp <= tol && guardOk;
  const tierNote = tiers?.length
    ? ` — by rank: ${tiers
        .map(
          (t) =>
            `${TIER_LABELS[t.tier] ?? `tier ${t.tier}`} ±${t.disp < 9.5 ? t.disp.toFixed(1) : Math.round(t.disp)}`,
        )
        .join(" · ")}`
    : "";
  // Bar runs from a shuffled list (mean displacement ≈ N/3) to the target.
  // Rating does most of this distance; comparisons close the rest.
  const maxDisp = Math.max(total / 3, tol + 1);
  const pct = Math.round(100 * Math.max(0, Math.min(1, 1 - (disp - tol) / (maxDisp - tol))));
  const places = Number.isFinite(disp)
    ? disp < 9.5
      ? disp.toFixed(1)
      : String(Math.round(disp))
    : "–";
  return (
    <span
      className="rank-progress-meter"
      title={`Each item is typically within ±${places} places of where unlimited comparisons would put it${
        tol > 0
          ? ` — target ±${tol}${
              reached
                ? " reached: good enough at this precision"
                : guardOk
                  ? ""
                  : " (a rank is still fuzzier than half a third of itself)"
            }`
          : " — Exact: comparisons continue while anything informative remains"
      }${tierNote}${
        unrated > 0 ? ` — ${unrated} of ${total} not yet rated` : ""
      }. Rating does the coarse placement; comparisons tighten it.`}
    >
      <span className="rank-progress-bar">
        <span className="rank-progress-bar-fill" style={{ width: `${pct}%` }} />
      </span>
      <span className="rank-progress-label">
        {reached ? `✓ within ±${tol}` : `±${places} places${tol > 0 ? ` · target ±${tol}` : ""}`}
        {unrated > 0 && ` · ${unrated} unrated`}
      </span>
    </span>
  );
}

// ---- Comparison overlay ------------------------------------------------------

interface RankCompareOverlayProps {
  groups: ImageGroup[]; // 2 = pair (←/→/=), 3-4 = best-of-N (1..N)
  /** The answer already recorded for this comparison (highlights the pick). */
  outcome: CompareOutcome | null;
  /** 0-based position of this comparison within the streak, and the count. */
  index: number;
  count: number;
  canPrev: boolean;
  canNext: boolean;
  /** Ranking single ungrouped images: fill each panel with the whole photo and
   * drop the (always-1) count badge, instead of a thumbnail grid. */
  imageMode: boolean;
  /** Meter inputs — see RankProgressMeter. */
  disp: number;
  tol: number;
  unrated: number;
  total: number;
  tiers?: { tier: number; n: number; disp: number }[];
  guardOk?: boolean;
  /** A win, optionally annotated with how decisive it felt. */
  onWin: (id: string, margin?: CompareMargin) => void;
  /** Best-of-N joint top: these ids beat the rest and tie among themselves. */
  onTop: (ids: string[], margin?: CompareMargin) => void;
  /** All of them are equally good — a draw across every pair on screen. */
  onTie: () => void;
  onSkip: () => void;
  /** Lock a group's position out of future comparisons. */
  onPin: (id: string) => void;
  /** Step to the previous / next comparison already shown this streak. */
  onPrev: () => void;
  onNext: () => void;
  onOpenLightbox: (images: string[], index: number) => void;
  onClose: () => void;
}

function RankCompareOverlay({
  groups,
  outcome,
  index,
  count,
  canPrev,
  canNext,
  imageMode,
  disp,
  tol,
  unrated,
  total,
  tiers,
  guardOk,
  onWin,
  onTop,
  onTie,
  onSkip,
  onPin,
  onPrev,
  onNext,
  onOpenLightbox,
  onClose,
}: RankCompareOverlayProps) {
  const lightboxOpen = useLightboxStore((s) => s.open);
  const isPair = groups.length === 2;

  // "Top picks" staging (best-of-N only): build the joint-best subset, then
  // commit — "these are the best, can't separate them". A pair has no partial
  // top (that's just a win or a tie), so pairs never stage.
  const [staging, setStaging] = useState(false);
  const [topPicks, setTopPicks] = useState<ReadonlySet<string>>(new Set());
  const compareKey = groups.map((g) => g.id).join(" ");
  // The overlay persists across timeline steps — a new comparison starts unstaged.
  // biome-ignore lint/correctness/useExhaustiveDependencies: compareKey identifies which comparison is on screen
  useEffect(() => {
    setStaging(false);
    setTopPicks(new Set());
  }, [compareKey]);

  function togglePick(id: string) {
    setStaging(true);
    setTopPicks((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }
  function startStaging() {
    setStaging(true);
  }
  function clearStaging() {
    setStaging(false);
    setTopPicks(new Set());
  }
  function commitTop(margin?: CompareMargin) {
    // Screen order, not click order, so the logged event is deterministic.
    const ids = groups.filter((g) => topPicks.has(g.id)).map((g) => g.id);
    if (ids.length === 0) return;
    clearStaging();
    onTop(ids, margin);
  }

  const handlersRef = useRef({
    groups,
    staging,
    topPicks,
    onWin,
    onTie,
    onSkip,
    onPin,
    onPrev,
    onNext,
    canPrev,
    canNext,
    onClose,
    togglePick,
    startStaging,
    clearStaging,
    commitTop,
  });
  handlersRef.current = {
    groups,
    staging,
    topPicks,
    onWin,
    onTie,
    onSkip,
    onPin,
    onPrev,
    onNext,
    canPrev,
    canNext,
    onClose,
    togglePick,
    startStaging,
    clearStaging,
    commitTop,
  };

  useEffect(() => {
    if (lightboxOpen) return; // lightbox owns the keyboard while it's up
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLElement) {
        const tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      }
      if (e.metaKey || e.ctrlKey) return; // browser/system chords stay theirs
      const h = handlersRef.current;
      const isPairKeys = h.groups.length === 2;
      const act = (fn: () => void) => {
        // Capture-phase + stopPropagation so the modal's own Escape/key
        // listeners underneath never see overlay keys.
        e.preventDefault();
        e.stopPropagation();
        fn();
      };
      // While staging top picks, Esc backs out of the staging, not the overlay.
      if (e.key === "Escape") return act(h.staging ? h.clearStaging : h.onClose);
      // Arrow keys walk the comparison history (like the rating view's item
      // navigation) so any earlier comparison can be revisited and re-picked.
      if (e.key === "ArrowLeft") return h.canPrev ? act(h.onPrev) : undefined;
      if (e.key === "ArrowRight") return h.canNext ? act(h.onNext) : undefined;
      if (e.key === "s" || e.key === "S") return act(h.onSkip);
      if (e.key === "Enter") {
        if (h.staging && h.topPicks.size > 0)
          return act(() => h.commitTop(e.shiftKey ? "clear" : e.altKey ? "slim" : undefined));
        return;
      }
      if (e.key === "+" && !isPairKeys) return act(h.staging ? h.clearStaging : h.startStaging);
      // Number keys pick a group in every mode (1/2 for a pair, 1..N for
      // best-of-N) — same row you rate with, no left/right context switch.
      // Read the digit from e.code so ⇧/⌥ layers (whose e.key is "!", "¡", …)
      // still land on the right panel.
      const idx = e.code.startsWith("Digit")
        ? Number(e.code.slice(5)) - 1
        : Number.parseInt(e.key, 10) - 1;
      if (idx >= 0 && idx < h.groups.length) {
        const id = h.groups[idx]!.id;
        if (h.staging) return act(() => h.togglePick(id));
        const margin: CompareMargin | undefined = e.shiftKey
          ? "clear"
          : e.altKey
            ? "slim"
            : undefined;
        return act(() => h.onWin(id, margin));
      }
      if (e.key === "=") return act(h.onTie);
      if (isPairKeys) {
        if (e.key === "[") return act(() => h.onPin(h.groups[0]!.id));
        if (e.key === "]") return act(() => h.onPin(h.groups[1]!.id));
      }
    }
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [lightboxOpen]);
  const pickedId =
    outcome?.kind === "win" ? outcome.winnerId : outcome?.kind === "pin" ? outcome.id : null;
  const topIds = outcome?.kind === "top" ? new Set(outcome.ids) : null;
  const winMargin = outcome?.kind === "win" ? outcome.margin : undefined;
  const answerNote =
    outcome?.kind === "tie"
      ? isPair
        ? "tie recorded"
        : "all equal"
      : outcome?.kind === "skip"
        ? "skipped"
        : outcome?.kind === "top"
          ? `top ${outcome.ids.length}${
              outcome.margin === "clear" ? ", clearly" : outcome.margin === "slim" ? ", barely" : ""
            } — equal among them`
          : outcome?.kind === "win" && outcome.margin
            ? outcome.margin === "clear"
              ? "clearly better"
              : "barely better"
            : null;
  // Image mode: the lightbox navigates the full comparison set (each panel's
  // single photo), so a click on any panel can page across all of them.
  const compareImages = imageMode ? groups.map((g) => g.images[0]!) : [];
  return (
    <div className="rank-compare-backdrop">
      <div className="rank-compare">
        <div className="rank-compare-header">
          <span className="rank-compare-header-left">
            <span className="rank-compare-nav">
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={onPrev}
                disabled={!canPrev}
                aria-label="Previous comparison"
                title="Previous comparison (←)"
              >
                ‹
              </button>
              <span className="rank-compare-pos">
                {index + 1} / {count}
              </span>
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={onNext}
                disabled={!canNext}
                aria-label="Next comparison"
                title="Next comparison (→)"
              >
                ›
              </button>
            </span>
            <span className="rank-compare-title">
              {staging
                ? "Pick the joint best — ⏎ when done"
                : isPair
                  ? "Which do you prefer?"
                  : "Pick the best"}
            </span>
            {answerNote && <span className="rank-compare-answer-note">· {answerNote}</span>}
            <RankProgressMeter
              disp={disp}
              tol={tol}
              unrated={unrated}
              total={total}
              tiers={tiers}
              guardOk={guardOk}
            />
          </span>
          <span className="rank-compare-hint">
            {staging
              ? `1–${groups.length} toggle · ⏎ commit · ⇧⏎ clearly · ⌥⏎ barely · + or Esc cancel`
              : isPair
                ? "1 / 2 better · ⇧ clearly · ⌥ barely · = tie · s skip · [ ] pin · ← → history · Esc close"
                : `1–${groups.length} better · ⇧ clearly · ⌥ barely · + top picks · = all equal · s skip · ← → history · Esc close`}
          </span>
        </div>
        <div className={cn("rank-compare-panels", `rank-compare-panels-${groups.length}`)}>
          {groups.map((g, i) => {
            const isTopPick = topIds?.has(g.id) ?? false;
            const isPicked = g.id === pickedId || isTopPick;
            const isPinPick = outcome?.kind === "pin" && isPicked;
            const isStaged = topPicks.has(g.id);
            return (
              <div
                key={g.id}
                className={cn(
                  "rank-compare-panel",
                  isPicked && "rank-compare-panel-picked",
                  isStaged && "rank-compare-panel-staged",
                )}
              >
                <div className="rank-compare-panel-header">
                  <kbd className="rank-compare-key">{i + 1}</kbd>
                  <span className="rank-compare-name" title={g.name}>
                    {g.name}
                  </span>
                  {(isPicked || isStaged) && (
                    <span className="rank-compare-pick-badge">
                      {isPinPick
                        ? "pinned"
                        : isTopPick
                          ? "★ top pick"
                          : isStaged
                            ? "☆ staged"
                            : "✓ your pick"}
                    </span>
                  )}
                  {!imageMode && <span className="rank-compare-count">{g.images.length}</span>}
                </div>
                {imageMode ? (
                  <button
                    type="button"
                    className="rank-compare-photo"
                    // Open the lightbox over the whole comparison set (one photo
                    // per panel) at this panel, so ←/→ steps between the images
                    // being compared instead of trapping on a single one.
                    onClick={() => onOpenLightbox(compareImages, i)}
                    aria-label={`Open ${g.images[0]}`}
                  >
                    <img src={fullImageUrl(g.images[0]!)} alt="" draggable={false} />
                  </button>
                ) : (
                  <div className="rank-compare-thumbs">
                    {g.images.map((fn, idx) => (
                      <button
                        type="button"
                        key={fn}
                        className="review-single-thumb"
                        onClick={() => onOpenLightbox(g.images, idx)}
                        aria-label={`Open ${fn}`}
                      >
                        <img src={imageUrl(fn)} alt="" loading="lazy" draggable={false} />
                      </button>
                    ))}
                  </div>
                )}
                <div className="rank-compare-actions">
                  <button
                    type="button"
                    className={cn(
                      "btn btn-secondary",
                      g.id === pickedId && winMargin === "slim" && "review-status-active",
                    )}
                    onClick={() => onWin(g.id, "slim")}
                    title={`Better, but barely — could have gone either way (⌥${i + 1})`}
                  >
                    Barely
                  </button>
                  <button
                    type="button"
                    className={cn(
                      "btn",
                      g.id === pickedId && !isPinPick && !winMargin
                        ? "btn-primary"
                        : "btn-secondary",
                    )}
                    onClick={() => onWin(g.id)}
                    title={`Better (${i + 1})`}
                  >
                    {g.id === pickedId && !isPinPick && !winMargin ? "✓ Better" : "Better"}
                  </button>
                  <button
                    type="button"
                    className={cn(
                      "btn btn-secondary",
                      g.id === pickedId && winMargin === "clear" && "review-status-active",
                    )}
                    onClick={() => onWin(g.id, "clear")}
                    title={`Clearly better — a decisive win (⇧${i + 1})`}
                  >
                    Clearly
                  </button>
                  {!isPair && (
                    <button
                      type="button"
                      className={cn("btn btn-secondary", isStaged && "review-status-active")}
                      onClick={() => togglePick(g.id)}
                      title="Stage as a joint top pick — the staged few beat the rest and tie among themselves (+ then numbers, ⏎ commits)"
                    >
                      Top
                    </button>
                  )}
                  <button
                    type="button"
                    className={cn("btn btn-secondary", isPinPick && "review-status-active")}
                    onClick={() => onPin(g.id)}
                    title="Lock this group where it stands and stop asking about it"
                  >
                    Pin
                  </button>
                </div>
              </div>
            );
          })}
        </div>
        <div className="rank-compare-footer">
          {staging && (
            <>
              <button
                type="button"
                className="btn btn-secondary"
                disabled={topPicks.size === 0}
                onClick={() => commitTop("slim")}
                title="Commit as barely: the staged picks edge out the rest, tied among themselves (⌥⏎)"
              >
                Barely
              </button>
              <button
                type="button"
                className="btn btn-primary"
                disabled={topPicks.size === 0}
                onClick={(e) => commitTop(e.shiftKey ? "clear" : e.altKey ? "slim" : undefined)}
                title="Commit: the staged picks beat the rest and tie among themselves (⏎)"
              >
                {topPicks.size <= 1
                  ? `✓ Commit top pick${topPicks.size === 0 ? "s" : ""}`
                  : `✓ These ${topPicks.size} are best — equal among them`}
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                disabled={topPicks.size === 0}
                onClick={() => commitTop("clear")}
                title="Commit as clearly: the staged picks decisively beat the rest, tied among themselves (⇧⏎)"
              >
                Clearly
              </button>
            </>
          )}
          <button
            type="button"
            className={cn("btn btn-secondary", outcome?.kind === "tie" && "review-status-active")}
            onClick={onTie}
            title={
              isPair
                ? "Neither is better — record a draw"
                : "None of these is better than another — record a draw across all of them"
            }
          >
            {isPair ? "= Tie" : "= All equal"}
          </button>
          <button
            type="button"
            className={cn("btn btn-secondary", outcome?.kind === "skip" && "review-status-active")}
            onClick={onSkip}
          >
            Skip
          </button>
          <button type="button" className="btn btn-secondary" onClick={onClose}>
            Back to rating
          </button>
        </div>
      </div>
    </div>
  );
}
