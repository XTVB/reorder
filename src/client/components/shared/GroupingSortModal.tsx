import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { useToastStore } from "../../stores/core/toastStore.ts";
import { cn } from "../../utils/helpers.ts";
import {
  type ConfigStore,
  cloneConfig,
  colorForId,
  duplicateConfig,
  emptyConfigDraft,
  makeId,
  NEW_CONFIG_OPTION,
  type ReviewCategory,
  type ReviewConfig,
  reviewColorVar,
  shortcutForSlot,
  slotForKeyEvent,
} from "../../utils/reviewConfigs.ts";
import { Modal } from "./Modal.tsx";
import { ReviewConfigEditor } from "./ReviewConfigEditor.tsx";

export interface SubAssignment {
  categoryId: string;
  subId: string;
}

export interface SortState {
  statuses: Map<string, string>;
  subs: Map<string, SubAssignment>;
}

// Stable empty state so memo deps on `statuses`/`subs` hold until first assign.
const EMPTY_STATE: SortState = { statuses: new Map(), subs: new Map() };

/** State handed to caller hooks (apply / progress / subtitle). */
export interface SortContext<T> {
  items: T[];
  config: ReviewConfig;
  statuses: Map<string, string>;
  subs: Map<string, SubAssignment>;
}

export interface CurrentInfo {
  config: ReviewConfig;
  category: ReviewCategory | undefined;
  sub: SubAssignment | undefined;
  inBucket: boolean;
  bucketCategory: ReviewCategory | undefined;
}

export interface GroupingSortModalProps<T> {
  /** Namespaced config persistence (separate preset libraries per modal). */
  store: ConfigStore;
  /** Snapshot of items to step through (caller captures once via useState). */
  items: T[];
  /** Extra class on the modal shell (e.g. to force a full-height layout). */
  modalClassName?: string;
  getId: (t: T) => string;
  getName: (t: T) => string;
  /**
   * Optional per-config seed: recover in-progress assignments for a config
   * from existing item state (e.g. persisted tags) so items already bucketed
   * by a previous Apply show their category/sub on open. Evaluated once per
   * known config at mount.
   */
  initialStateFor?: (config: ReviewConfig) => SortState | null;
  /** Category an unassigned item falls into when filtering a bucket, or null. */
  defaultCategoryId: (cfg: ReviewConfig) => string | null;
  /** Singular, lowercase nouns used in prompts/hints, e.g. "category"/"group". */
  terms: { group: string; sub: string };
  titleText: (bucketLabel: string | null) => string;
  emptyText: (bucketLabel: string | null) => string;
  subBannerLabel: (label: string) => ReactNode;
  chipTitle: (label: string, isActive: boolean, hasSubs: boolean) => string;
  /** Trailing hint segment shown when not refining a bucket. */
  tailHint: string;
  renderSubtitle: (item: T, info: CurrentInfo) => ReactNode;
  renderMedia: (item: T) => ReactNode;
  renderProgressExtra?: (ctx: SortContext<T>) => ReactNode;
  /**
   * Optional: the lightbox content for an item. While the lightbox is open over
   * the modal the categorisation slot keys stay live (assign + advance), and
   * when the cursor moves to a new item the open lightbox is reopened on this
   * target so it follows along. Return null to close the lightbox for an item
   * with no media.
   */
  getLightboxTarget?: (item: T) => { filenames: string[]; index: number } | null;
  apply: {
    /** Derive the Apply button state from the current assignments (one pass). */
    describe: (ctx: SortContext<T>) => { label: string; disabled: boolean; title?: string };
    run: (ctx: SortContext<T>) => void;
  };
  onClose: () => void;
}

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * Shared "step through items, bucket each into a category/subcategory, apply"
 * engine. Owns every stateful concern (config picker + editor + fork-on-edit,
 * per-config in-progress assignments, bucket navigation, number-key shortcuts,
 * counts). Callers inject only data + label/render/apply hooks.
 */
export function GroupingSortModal<T>({
  store,
  items,
  modalClassName,
  getId,
  getName,
  initialStateFor,
  defaultCategoryId,
  terms,
  titleText,
  emptyText,
  subBannerLabel,
  chipTitle,
  tailHint,
  renderSubtitle,
  renderMedia,
  renderProgressExtra,
  getLightboxTarget,
  apply,
  onClose,
}: GroupingSortModalProps<T>) {
  const [boot] = useState(() => store.loadInitialConfigs());
  const [configs, setConfigs] = useState(boot.configs);
  const [activeConfigId, setActiveConfigId] = useState(boot.lastId);
  const activeConfig =
    configs.find((c) => c.id === activeConfigId) ??
    configs.find((c) => c.id === store.builtinId) ??
    configs[0]!;

  const [editor, setEditor] = useState<{ initial: ReviewConfig; isNew: boolean } | null>(null);

  // Per-config so switching configs and back restores the in-progress sort.
  // Seeded once from existing item state (e.g. persisted tags) so groups that
  // were already bucketed show their category/sub on open.
  const [stateByConfig, setStateByConfig] = useState<Map<string, SortState>>(() => {
    const m = new Map<string, SortState>();
    if (initialStateFor) {
      for (const c of boot.configs) {
        const seed = initialStateFor(c);
        if (seed && (seed.statuses.size > 0 || seed.subs.size > 0)) m.set(c.id, seed);
      }
    }
    return m;
  });
  const stateForConfig = stateByConfig.get(activeConfigId) ?? EMPTY_STATE;
  const { statuses, subs } = stateForConfig;

  const [bucket, setBucket] = useState<string | null>(null);
  const [topIndex, setTopIndex] = useState(0);
  const [subIndex, setSubIndex] = useState(0);

  const lightboxOpen = useLightboxStore((s) => s.open);

  // Adapters pass these as fresh closures each render; mirror them through refs
  // so the derived memos below stay cached across unrelated parent re-renders
  // (same pattern as handlersRef for the keyboard listener).
  const getIdRef = useRef(getId);
  getIdRef.current = getId;
  const defaultCategoryIdRef = useRef(defaultCategoryId);
  defaultCategoryIdRef.current = defaultCategoryId;

  function patchState(fn: (s: SortState) => SortState) {
    setStateByConfig((prev) => {
      const cur = prev.get(activeConfigId) ?? EMPTY_STATE;
      const nextState = fn(cur);
      if (nextState === cur) return prev;
      const next = new Map(prev);
      next.set(activeConfigId, nextState);
      return next;
    });
  }

  function commitConfig(next: ReviewConfig) {
    setConfigs(store.upsertConfig(configs, next));
  }

  function selectConfig(id: string) {
    setActiveConfigId(id);
    store.saveLastConfigId(id);
    setBucket(null);
    setTopIndex(0);
    setSubIndex(0);
  }

  const filtered = useMemo(() => {
    if (!bucket) return items;
    const defId = defaultCategoryIdRef.current(activeConfig);
    return items.filter((t) => (statuses.get(getIdRef.current(t)) ?? defId) === bucket);
  }, [items, statuses, bucket, activeConfig]);

  const total = filtered.length;
  const currentIndex = bucket ? subIndex : topIndex;
  const current = filtered[currentIndex];

  const bucketCategory: ReviewCategory | undefined = useMemo(
    () => (bucket ? activeConfig.categories.find((c) => c.id === bucket) : undefined),
    [bucket, activeConfig],
  );

  function setStatusFor(id: string, categoryId: string) {
    patchState((s) => {
      const nextStatuses = new Map(s.statuses);
      if (nextStatuses.get(id) === categoryId) nextStatuses.delete(id);
      else nextStatuses.set(id, categoryId);
      const existingSub = s.subs.get(id);
      let nextSubs = s.subs;
      if (existingSub && existingSub.categoryId !== categoryId) {
        nextSubs = new Map(s.subs);
        nextSubs.delete(id);
      }
      return { statuses: nextStatuses, subs: nextSubs };
    });
  }

  function setSubStatusFor(id: string, categoryId: string, subId: string) {
    patchState((s) => {
      const nextSubs = new Map(s.subs);
      const existing = nextSubs.get(id);
      if (existing && existing.categoryId === categoryId && existing.subId === subId) {
        nextSubs.delete(id);
      } else {
        nextSubs.set(id, { categoryId, subId });
      }
      return { statuses: s.statuses, subs: nextSubs };
    });
  }

  function advance(delta: number) {
    const setter = bucket ? setSubIndex : setTopIndex;
    setter((i) => Math.min(total - 1, Math.max(0, i + delta)));
  }

  function chooseAndAdvance(slot: number) {
    if (!current) return;
    const id = getId(current);
    if (bucket && bucketCategory) {
      const sub = bucketCategory.subcategories[slot];
      if (!sub) return;
      const cur = subs.get(id);
      const wasSame = cur?.categoryId === bucket && cur?.subId === sub.id;
      setSubStatusFor(id, bucket, sub.id);
      if (!wasSame) setSubIndex((i) => Math.min(total - 1, i + 1));
    } else {
      const cat = activeConfig.categories[slot];
      if (!cat) return;
      const wasSame = statuses.get(id) === cat.id;
      setStatusFor(id, cat.id);
      if (!wasSame) setTopIndex((i) => Math.min(total - 1, i + 1));
    }
  }

  function enterBucket(id: string) {
    if (bucket === id) {
      setBucket(null);
      return;
    }
    if (!activeConfig.categories.some((c) => c.id === id)) return;
    setBucket(id);
    setSubIndex(0);
  }

  function exitBucket() {
    setBucket(null);
  }

  // The built-in is read-only, so it forks to an editable copy. duplicateConfig
  // preserves category/sub ids, so the bucket and in-progress assignments stay
  // valid and carry over to the fork.
  function mutateActiveConfig(mutate: (cfg: ReviewConfig) => ReviewConfig) {
    const src = activeConfig;
    const next = mutate(src.builtIn ? duplicateConfig(src) : cloneConfig(src));
    commitConfig(next);
    if (next.id === activeConfigId) return;

    const oldId = activeConfigId;
    setStateByConfig((prev) => {
      const existing = prev.get(oldId);
      if (!existing) return prev;
      const out = new Map(prev);
      out.set(next.id, {
        statuses: new Map(existing.statuses),
        subs: new Map(existing.subs),
      });
      return out;
    });
    setActiveConfigId(next.id);
    store.saveLastConfigId(next.id);
    useToastStore
      .getState()
      .showToast(`Built-in is read-only — forked to "${next.name}"`, "warning");
  }

  function addCategoryViaShortcut() {
    const cat = bucket ? activeConfig.categories.find((c) => c.id === bucket) : undefined;
    if (bucket && !cat) return;
    const promptMsg = cat ? `New ${terms.sub} in "${cat.label}":` : `New ${terms.group}:`;
    const defaultLabel = cat
      ? `Subgroup ${cat.subcategories.length + 1}`
      : `${cap(terms.group)} ${activeConfig.categories.length + 1}`;
    const trimmed = window.prompt(promptMsg, defaultLabel)?.trim();
    if (!trimmed) return;
    mutateActiveConfig((cfg) =>
      cat
        ? {
            ...cfg,
            categories: cfg.categories.map((c) =>
              c.id === cat.id
                ? {
                    ...c,
                    subcategories: [...c.subcategories, { id: makeId("sub"), label: trimmed }],
                  }
                : c,
            ),
          }
        : {
            ...cfg,
            categories: [
              ...cfg.categories,
              { id: makeId("cat"), label: trimmed, subcategories: [] },
            ],
          },
    );
  }

  // Keyboard handlers read through refs so the listener stays subscribed.
  const handlersRef = useRef<{
    chooseAndAdvance: (slot: number) => void;
    advance: (delta: number) => void;
    onClose: () => void;
    exitBucket: () => void;
    addCategoryViaShortcut: () => void;
    gotoNextUncategorised: () => void;
    bucket: string | null;
  }>(null!);
  handlersRef.current = {
    chooseAndAdvance,
    advance,
    onClose,
    exitBucket,
    addCategoryViaShortcut,
    gotoNextUncategorised,
    bucket,
  };

  const slotCount = bucketCategory
    ? bucketCategory.subcategories.length
    : activeConfig.categories.length;
  const slotCountRef = useRef(slotCount);
  slotCountRef.current = slotCount;

  useEffect(() => {
    if (editor) return;
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLElement) {
        const tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      }
      const h = handlersRef.current;
      // With the lightbox open over the modal, keep only the categorisation slot
      // keys live: the digit assigns the inspected item and advances, and the
      // open lightbox follows the cursor (see the sync effect below). Esc /
      // navigation / add-category stay with the lightbox so they don't fight it.
      if (lightboxOpen) {
        const slot = slotForKeyEvent(e);
        if (slot !== null && slot < slotCountRef.current) {
          e.preventDefault();
          h.chooseAndAdvance(slot);
        }
        return;
      }
      if (e.key === "Escape") {
        if (h.bucket !== null) h.exitBucket();
        else h.onClose();
        return;
      }
      if (e.key === "n" || e.key === "N") {
        e.preventDefault();
        h.addCategoryViaShortcut();
        return;
      }
      if (e.key === "Tab") {
        e.preventDefault();
        h.gotoNextUncategorised();
        return;
      }
      const slot = slotForKeyEvent(e);
      if (slot !== null && slot < slotCountRef.current) {
        e.preventDefault();
        h.chooseAndAdvance(slot);
        return;
      }
      if (e.key === "ArrowLeft" || e.key === "h") {
        e.preventDefault();
        h.advance(-1);
      } else if (e.key === "ArrowRight" || e.key === "l") {
        e.preventDefault();
        h.advance(1);
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [lightboxOpen, editor]);

  // Keep an open lightbox pinned to the item under the cursor. When
  // categorising via slot keys advances to a new item (or the cursor otherwise
  // moves while the lightbox is up), reopen it on that item's media so the
  // inspected image follows along; close it if the new item has none.
  const currentId = current ? getId(current) : null;
  const getLightboxTargetRef = useRef(getLightboxTarget);
  getLightboxTargetRef.current = getLightboxTarget;
  const syncedLightboxIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (!lightboxOpen) {
      syncedLightboxIdRef.current = null;
      return;
    }
    // First observation after opening: adopt whatever item the caller opened it
    // on (it already picked the image) without reopening over their choice.
    if (syncedLightboxIdRef.current === null) {
      syncedLightboxIdRef.current = currentId;
      return;
    }
    if (currentId === syncedLightboxIdRef.current) return;
    syncedLightboxIdRef.current = currentId;
    const getTarget = getLightboxTargetRef.current;
    if (!getTarget) return;
    const target = current ? getTarget(current) : null;
    const lb = useLightboxStore.getState();
    if (target && target.filenames.length > 0) lb.openLightbox(target.filenames, target.index);
    else lb.close();
  }, [lightboxOpen, currentId, current]);

  const topCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const c of activeConfig.categories) counts[c.id] = 0;
    for (const s of statuses.values()) {
      if (counts[s] !== undefined) counts[s]++;
    }
    return counts;
  }, [statuses, activeConfig]);

  const subCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    if (!bucketCategory) return counts;
    for (const s of bucketCategory.subcategories) counts[s.id] = 0;
    for (const t of filtered) {
      const sa = subs.get(getIdRef.current(t));
      if (sa && sa.categoryId === bucketCategory.id && counts[sa.subId] !== undefined) {
        counts[sa.subId] = (counts[sa.subId] ?? 0) + 1;
      }
    }
    return counts;
  }, [filtered, subs, bucketCategory]);

  // Index of the next item after the cursor that hasn't been bucketed yet — at
  // top level that means no category assigned; inside a bucket, no subcategory
  // for that category. -1 when everything ahead is already sorted.
  const nextUncategorisedIndex = useMemo(() => {
    for (let i = currentIndex + 1; i < total; i++) {
      const id = getIdRef.current(filtered[i]!);
      const assigned = bucket
        ? subs.get(id)?.categoryId === bucket
        : statuses.get(id) !== undefined;
      if (!assigned) return i;
    }
    return -1;
  }, [filtered, currentIndex, total, bucket, statuses, subs]);

  function gotoNextUncategorised() {
    if (nextUncategorisedIndex < 0) return;
    (bucket ? setSubIndex : setTopIndex)(nextUncategorisedIndex);
  }

  const ctx: SortContext<T> = { items, config: activeConfig, statuses, subs };
  const applyState = apply.describe(ctx);

  function handleApply() {
    apply.run(ctx);
  }

  function openNewEditor() {
    setEditor({ initial: emptyConfigDraft(), isNew: true });
  }

  function openEditEditor() {
    if (activeConfig.builtIn) {
      setEditor({ initial: duplicateConfig(activeConfig), isNew: true });
    } else {
      setEditor({ initial: activeConfig, isNew: false });
    }
  }

  function handleEditorSave(next: ReviewConfig) {
    commitConfig(next);
    // Editing mid-sort should keep the cursor where it is — category/sub ids are
    // preserved across an edit, so in-progress assignments stay valid. Avoid
    // selectConfig() here: it would reset bucket/topIndex/subIndex to the start.
    const oldId = activeConfigId;
    if (next.id !== oldId) {
      // The editor forked a read-only built-in into a new id. Carry the
      // in-progress assignments over and switch to the fork without rewinding.
      setStateByConfig((prev) => {
        const existing = prev.get(oldId);
        if (!existing) return prev;
        const out = new Map(prev);
        out.set(next.id, {
          statuses: new Map(existing.statuses),
          subs: new Map(existing.subs),
        });
        return out;
      });
      setActiveConfigId(next.id);
      store.saveLastConfigId(next.id);
    }
    // If the category we were refining was deleted in the edit, drop to top level.
    if (bucket && !next.categories.some((c) => c.id === bucket)) setBucket(null);
    setEditor(null);
  }

  function handleEditorDelete() {
    if (!editor) return;
    setConfigs(store.deleteConfigById(configs, editor.initial.id));
    selectConfig(store.builtinId);
    setEditor(null);
  }

  const currentCategoryId = current ? statuses.get(getId(current)) : undefined;
  const currentCategory = currentCategoryId
    ? activeConfig.categories.find((c) => c.id === currentCategoryId)
    : undefined;
  const currentSub = current ? subs.get(getId(current)) : undefined;

  const headerControls = (
    <div className="review-config-picker">
      <select
        className="review-config-select"
        value={activeConfigId}
        onChange={(e) => {
          const v = e.target.value;
          if (v === NEW_CONFIG_OPTION) openNewEditor();
          else selectConfig(v);
        }}
        title="Select grouping configuration"
      >
        {configs.map((c) => (
          <option key={c.id} value={c.id}>
            {c.name}
            {c.builtIn ? " (built-in)" : ""}
          </option>
        ))}
        <option disabled>──────</option>
        <option value={NEW_CONFIG_OPTION}>+ New grouping…</option>
      </select>
      <button
        type="button"
        className="btn-icon"
        onClick={openEditEditor}
        title={activeConfig.builtIn ? "Duplicate this grouping to edit" : "Edit this grouping"}
        aria-label="Edit grouping"
      >
        ⚙
      </button>
    </div>
  );

  const title = (
    <>
      <span className="review-modal-title-text">
        {titleText(bucketCategory ? bucketCategory.label : null)}
      </span>
      {headerControls}
      <span className="review-progress">
        {activeConfig.categories.map((cat, i) => {
          const isActive = bucket === cat.id;
          const hasSubs = cat.subcategories.length > 0;
          return (
            <button
              key={cat.id}
              type="button"
              className={cn(
                "review-progress-chip",
                isActive && "review-progress-chip-active",
                !hasSubs && "review-progress-chip-flat",
              )}
              style={reviewColorVar(colorForId(cat.id, i))}
              onClick={() => enterBucket(cat.id)}
              title={chipTitle(cat.label, isActive, hasSubs)}
            >
              {topCounts[cat.id] ?? 0} {cat.label}
            </button>
          );
        })}
        {renderProgressExtra?.(ctx)}
        <span className="review-progress-total">
          {total === 0 ? "0 / 0" : `${currentIndex + 1} / ${total}`}
        </span>
      </span>
    </>
  );

  const slotSource: { id: string; label: string }[] = bucketCategory
    ? bucketCategory.subcategories
    : activeConfig.categories;
  const slotItems = slotSource.map((item, i) => ({
    id: item.id,
    label: item.label,
    color: colorForId(item.id, i),
    active: bucketCategory
      ? currentSub?.categoryId === bucketCategory.id && currentSub?.subId === item.id
      : currentCategoryId === item.id,
  }));

  const hintText = [
    ...slotSource
      .map((it, i) => {
        const k = shortcutForSlot(i);
        return k ? `${k} ${it.label.toLowerCase()}` : null;
      })
      .filter(Boolean),
    `n add ${bucketCategory ? terms.sub : terms.group}`,
    "← → navigate",
    bucketCategory ? "Esc back" : tailHint,
  ].join(" · ");

  const footer = (
    <>
      <span className="review-footer-hint modal-footer-spacer">{hintText}</span>
      {bucketCategory && (
        <button type="button" className="btn btn-secondary" onClick={exitBucket}>
          ← Back
        </button>
      )}
      <button type="button" className="btn btn-secondary" onClick={onClose}>
        Cancel
      </button>
      <button
        type="button"
        className="btn btn-primary"
        onClick={handleApply}
        disabled={applyState.disabled}
        title={applyState.title}
      >
        {applyState.label}
      </button>
    </>
  );

  const accentSlot = slotItems.find((s) => s.active);
  const currentInfo: CurrentInfo = {
    config: activeConfig,
    category: currentCategory,
    sub: currentSub,
    inBucket: Boolean(bucketCategory),
    bucketCategory,
  };

  return (
    <>
      <Modal
        title={title}
        onClose={onClose}
        footer={footer}
        className={cn("review-modal", modalClassName)}
        headerClassName="review-modal-header"
        bodyClassName="review-modal-body"
      >
        {bucketCategory && (
          <div className="review-sub-banner">
            <span className="review-sub-banner-label">{subBannerLabel(bucketCategory.label)}</span>
            <span className="review-sub-banner-counts">
              {bucketCategory.subcategories.map((s, i) => (
                <span
                  key={s.id}
                  className="review-sub-chip"
                  style={reviewColorVar(colorForId(s.id, i))}
                >
                  {subCounts[s.id] ?? 0} {s.label}
                </span>
              ))}
            </span>
          </div>
        )}

        {!current ? (
          <div className="review-empty">
            {emptyText(bucketCategory ? bucketCategory.label : null)}
          </div>
        ) : (
          <div
            className={cn("review-single", accentSlot && "review-single-accented")}
            style={accentSlot ? reviewColorVar(accentSlot.color) : undefined}
          >
            <div className="review-single-header">
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={() => advance(-1)}
                disabled={currentIndex === 0}
                aria-label="Previous"
              >
                ← Prev
              </button>
              <div className="review-single-title">
                <span className="review-single-name">{getName(current)}</span>
                <span className="review-single-count">{renderSubtitle(current, currentInfo)}</span>
              </div>
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={() => advance(1)}
                disabled={currentIndex === total - 1}
                aria-label="Next"
              >
                Next →
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-small review-single-skip"
                onClick={gotoNextUncategorised}
                disabled={nextUncategorisedIndex < 0}
                title="Jump to the next item you haven't sorted yet"
              >
                Next uncategorised ⇥
              </button>
            </div>

            {renderMedia(current)}

            <div
              className={cn(
                "review-single-actions",
                slotItems.length > 10 && "review-single-actions-dense",
              )}
            >
              {bucketCategory && slotItems.length === 0 && (
                <span className="review-single-no-subs">
                  No {terms.sub}s yet — press <kbd>n</kbd> to add one.
                </span>
              )}
              {slotItems.map((item, idx) => {
                const shortcut = shortcutForSlot(idx);
                return (
                  <button
                    type="button"
                    key={item.id}
                    className={cn(
                      "btn review-single-status-btn",
                      item.active ? "review-status-active" : "btn-secondary",
                    )}
                    style={reviewColorVar(item.color)}
                    onClick={() => chooseAndAdvance(idx)}
                  >
                    {shortcut && <span className="review-single-status-key">{shortcut}</span>}
                    {item.label}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </Modal>

      {editor && (
        <ReviewConfigEditor
          initial={editor.initial}
          isNew={editor.isNew}
          onSave={handleEditorSave}
          onClose={() => setEditor(null)}
          onDelete={editor.isNew ? undefined : handleEditorDelete}
        />
      )}
    </>
  );
}
