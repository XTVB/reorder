import { useEffect, useMemo, useRef, useState } from "react";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { useToastStore } from "../../stores/core/toastStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import type { ImageGroup } from "../../types.ts";
import { cn, imageUrl, reorderImagesByGroups } from "../../utils/helpers.ts";
import {
  BUILTIN_CONFIG_ID,
  cloneConfig,
  colorForId,
  deleteConfigById,
  duplicateConfig,
  emptyConfigDraft,
  loadInitialConfigs,
  makeId,
  NEW_CONFIG_OPTION,
  type ReviewCategory,
  type ReviewConfig,
  reviewColorVar,
  saveLastConfigId,
  shortcutForSlot,
  upsertConfig,
} from "../../utils/reviewConfigs.ts";
import { Modal } from "./Modal.tsx";
import { ReviewConfigEditor } from "./ReviewConfigEditor.tsx";

interface SubAssignment {
  categoryId: string;
  subId: string;
}

interface ReviewState {
  statuses: Map<string, string>;
  subs: Map<string, SubAssignment>;
}

// Stable empty state so memo deps on `statuses`/`subs` hold until first assign.
const EMPTY_REVIEW_STATE: ReviewState = { statuses: new Map(), subs: new Map() };

interface ReviewModalProps {
  onClose: () => void;
}

function chipTitle(label: string, isActive: boolean, hasSubs: boolean): string {
  if (isActive) return `Exit ${label}`;
  if (hasSubs) return `Refine ${label} ordering`;
  return `Open ${label} (press n to add a subcategory)`;
}

export function ReviewModal({ onClose }: ReviewModalProps) {
  const [snapshot] = useState<ImageGroup[]>(() =>
    useGroupStore.getState().groups.map((g) => ({ ...g, images: g.images.slice() })),
  );

  const [boot] = useState(loadInitialConfigs);
  const [configs, setConfigs] = useState(boot.configs);
  const [activeConfigId, setActiveConfigId] = useState(boot.lastId);
  const activeConfig =
    configs.find((c) => c.id === activeConfigId) ??
    configs.find((c) => c.id === BUILTIN_CONFIG_ID) ??
    configs[0]!;

  const [editor, setEditor] = useState<{ initial: ReviewConfig; isNew: boolean } | null>(null);

  // Per-config so switching configs and back restores the in-progress review.
  const [reviewByConfig, setReviewByConfig] = useState<Map<string, ReviewState>>(() => new Map());
  const review = reviewByConfig.get(activeConfigId) ?? EMPTY_REVIEW_STATE;
  const { statuses, subs } = review;

  const [bucket, setBucket] = useState<string | null>(null);
  const [topIndex, setTopIndex] = useState(0);
  const [subIndex, setSubIndex] = useState(0);

  const lightboxOpen = useLightboxStore((s) => s.open);

  function patchReview(fn: (s: ReviewState) => ReviewState) {
    setReviewByConfig((prev) => {
      const cur = prev.get(activeConfigId) ?? EMPTY_REVIEW_STATE;
      const nextState = fn(cur);
      if (nextState === cur) return prev;
      const next = new Map(prev);
      next.set(activeConfigId, nextState);
      return next;
    });
  }

  function commitConfig(next: ReviewConfig) {
    setConfigs(upsertConfig(configs, next));
  }

  function selectConfig(id: string) {
    setActiveConfigId(id);
    saveLastConfigId(id);
    setBucket(null);
    setTopIndex(0);
    setSubIndex(0);
  }

  function effectiveCategoryId(id: string): string {
    return statuses.get(id) ?? activeConfig.defaultCategoryId;
  }

  const filtered = useMemo(() => {
    if (!bucket) return snapshot;
    const defaultId = activeConfig.defaultCategoryId;
    return snapshot.filter((g) => (statuses.get(g.id) ?? defaultId) === bucket);
  }, [snapshot, statuses, bucket, activeConfig.defaultCategoryId]);

  const total = filtered.length;
  const currentIndex = bucket ? subIndex : topIndex;
  const current = filtered[currentIndex];

  const bucketCategory: ReviewCategory | undefined = useMemo(
    () => (bucket ? activeConfig.categories.find((c) => c.id === bucket) : undefined),
    [bucket, activeConfig],
  );

  function setStatusFor(id: string, categoryId: string) {
    patchReview((s) => {
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
    patchReview((s) => {
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
    if (bucket && bucketCategory) {
      const sub = bucketCategory.subcategories[slot];
      if (!sub) return;
      const cur = subs.get(current.id);
      const wasSame = cur?.categoryId === bucket && cur?.subId === sub.id;
      setSubStatusFor(current.id, bucket, sub.id);
      if (!wasSame) setSubIndex((i) => Math.min(total - 1, i + 1));
    } else {
      const cat = activeConfig.categories[slot];
      if (!cat) return;
      const wasSame = statuses.get(current.id) === cat.id;
      setStatusFor(current.id, cat.id);
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
    setReviewByConfig((prev) => {
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
    saveLastConfigId(next.id);
    useToastStore
      .getState()
      .showToast(`Built-in is read-only — forked to "${next.name}"`, "warning");
  }

  function addCategoryViaShortcut() {
    const cat = bucket ? activeConfig.categories.find((c) => c.id === bucket) : undefined;
    if (bucket && !cat) return;
    const promptMsg = cat ? `New subcategory in "${cat.label}":` : "New category:";
    const defaultLabel = cat
      ? `Subgroup ${cat.subcategories.length + 1}`
      : `Category ${activeConfig.categories.length + 1}`;
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
    bucket: string | null;
  }>(null!);
  handlersRef.current = {
    chooseAndAdvance,
    advance,
    onClose,
    exitBucket,
    addCategoryViaShortcut,
    bucket,
  };

  const slotKeyMap = useMemo(() => {
    const map = new Map<string, number>();
    const slots = bucketCategory
      ? bucketCategory.subcategories.length
      : activeConfig.categories.length;
    for (let i = 0; i < slots; i++) {
      const k = shortcutForSlot(i);
      if (k) map.set(k, i);
    }
    return map;
  }, [bucketCategory, activeConfig]);
  const slotKeyMapRef = useRef(slotKeyMap);
  slotKeyMapRef.current = slotKeyMap;

  useEffect(() => {
    if (lightboxOpen) return;
    if (editor) return;
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLElement) {
        const tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      }
      const h = handlersRef.current;
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
      const slot = slotKeyMapRef.current.get(e.key);
      if (slot !== undefined) {
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

  function openGroupLightbox(groupImages: string[], index: number) {
    const imageMap = useImageStore.getState().imageMap;
    const items = groupImages.filter((fn) => imageMap.has(fn));
    if (items.length === 0) return;
    useLightboxStore.getState().openLightbox(items, index);
  }

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
    for (const g of filtered) {
      const sa = subs.get(g.id);
      if (sa && sa.categoryId === bucketCategory.id && counts[sa.subId] !== undefined) {
        counts[sa.subId] = (counts[sa.subId] ?? 0) + 1;
      }
    }
    return counts;
  }, [filtered, subs, bucketCategory]);

  function handleApply() {
    if (snapshot.length === 0) {
      onClose();
      return;
    }

    const buckets = new Map<string, ImageGroup[]>();
    for (const c of activeConfig.categories) buckets.set(c.id, []);
    for (const g of snapshot) {
      buckets.get(effectiveCategoryId(g.id))?.push(g);
    }

    const sortBucket = (cat: ReviewCategory, gs: ImageGroup[]): ImageGroup[] => {
      if (cat.subcategories.length === 0) return gs;
      const subRank = new Map(cat.subcategories.map((s, i) => [s.id, i] as const));
      const fallbackId = cat.defaultSubcategoryId ?? null;
      // Unassigned groups sort to the default sub's rank, else after all subs.
      const fallbackRank =
        fallbackId !== null && subRank.has(fallbackId)
          ? subRank.get(fallbackId)!
          : cat.subcategories.length;
      return gs
        .map((g, originalIdx) => {
          const sa = subs.get(g.id);
          const rank =
            sa && sa.categoryId === cat.id && subRank.has(sa.subId)
              ? subRank.get(sa.subId)!
              : fallbackRank;
          return { g, rank, originalIdx };
        })
        .sort((a, b) => a.rank - b.rank || a.originalIdx - b.originalIdx)
        .map(({ g }) => g);
    };

    const newOrder: ImageGroup[] = [];
    for (const cat of activeConfig.categories) {
      newOrder.push(...sortBucket(cat, buckets.get(cat.id) ?? []));
    }

    const { images, imageMap, setImages } = useImageStore.getState();
    setImages(reorderImagesByGroups(images, imageMap, newOrder));
    useGroupStore.getState().updateGroups(() => newOrder);

    onClose();
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
    selectConfig(next.id);
    setEditor(null);
  }

  function handleEditorDelete() {
    if (!editor) return;
    setConfigs(deleteConfigById(configs, editor.initial.id));
    selectConfig(BUILTIN_CONFIG_ID);
    setEditor(null);
  }

  const currentCategoryId = current ? statuses.get(current.id) : undefined;
  const currentCategory = currentCategoryId
    ? activeConfig.categories.find((c) => c.id === currentCategoryId)
    : undefined;
  const currentSub = current ? subs.get(current.id) : undefined;

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
        {bucketCategory ? `Refine ${bucketCategory.label}` : "Review Groups"}
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
    bucketCategory ? "n add subcategory" : "n add category",
    "← → navigate",
    bucketCategory ? "Esc back" : "click chip to refine",
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
        disabled={snapshot.length === 0}
      >
        Apply Order
      </button>
    </>
  );

  const accentSlot = slotItems.find((s) => s.active);

  return (
    <>
      <Modal
        title={title}
        onClose={onClose}
        footer={footer}
        className="review-modal"
        headerClassName="review-modal-header"
        bodyClassName="review-modal-body"
      >
        {bucketCategory && (
          <div className="review-sub-banner">
            <span className="review-sub-banner-label">
              Refining <strong>{bucketCategory.label}</strong>
            </span>
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
            {bucketCategory ? `No groups in ${bucketCategory.label}.` : "No groups to review."}
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
                aria-label="Previous group"
              >
                ← Prev
              </button>
              <div className="review-single-title">
                <span className="review-single-name">{current.name}</span>
                <span className="review-single-count">
                  {current.images.length} image{current.images.length === 1 ? "" : "s"}
                  {bucketCategory && currentCategory && (
                    <>
                      {" · "}
                      <span
                        className="review-single-tag"
                        style={reviewColorVar(
                          colorForId(
                            currentCategory.id,
                            activeConfig.categories.indexOf(currentCategory),
                          ),
                        )}
                      >
                        {currentCategory.label}
                      </span>
                    </>
                  )}
                </span>
              </div>
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={() => advance(1)}
                disabled={currentIndex === total - 1}
                aria-label="Next group"
              >
                Next →
              </button>
            </div>

            <div className="review-single-thumbs">
              {current.images.map((fn, i) => (
                <button
                  type="button"
                  key={fn}
                  className="review-single-thumb"
                  onClick={() => openGroupLightbox(current.images, i)}
                  aria-label={`Open ${fn}`}
                >
                  <img src={imageUrl(fn)} alt="" loading="lazy" draggable={false} />
                </button>
              ))}
            </div>

            <div className="review-single-actions">
              {bucketCategory && slotItems.length === 0 && (
                <span className="review-single-no-subs">
                  No subcategories yet — press <kbd>n</kbd> to add one.
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
