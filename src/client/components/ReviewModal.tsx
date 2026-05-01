import { useEffect, useMemo, useRef, useState } from "react";
import { useGroupStore } from "../stores/groupStore.ts";
import { useImageStore } from "../stores/imageStore.ts";
import type { ImageGroup, ImageInfo } from "../types.ts";
import { cn, imageUrl, reorderImagesByGroups } from "../utils/helpers.ts";
import { Lightbox } from "./Lightbox.tsx";
import { Modal } from "./Modal.tsx";

type ReviewStatus = "keep" | "maybe" | "delete";
type SubStatus = "top" | "middle" | "bottom";

const STATUS_ORDER: ReviewStatus[] = ["keep", "maybe", "delete"];
const SUB_ORDER: SubStatus[] = ["top", "middle", "bottom"];

const STATUS_LABELS: Record<ReviewStatus, string> = {
  keep: "Keep",
  maybe: "Maybe",
  delete: "Delete",
};

const SUB_LABELS: Record<SubStatus, string> = {
  top: "Top",
  middle: "Middle",
  bottom: "Bottom",
};

const STATUS_ACTIVE_CLASS: Record<ReviewStatus, string> = {
  keep: "review-status-keep-active",
  maybe: "review-status-maybe-active",
  delete: "review-status-delete-active",
};

const SUB_ACTIVE_CLASS: Record<SubStatus, string> = {
  top: "review-sub-top-active",
  middle: "review-sub-middle-active",
  bottom: "review-sub-bottom-active",
};

const SUB_RANK: Record<SubStatus, number> = { top: 0, middle: 1, bottom: 2 };

function effectiveStatus(statuses: Map<string, ReviewStatus>, id: string): ReviewStatus {
  return statuses.get(id) ?? "maybe";
}

interface ReviewModalProps {
  onClose: () => void;
}

export function ReviewModal({ onClose }: ReviewModalProps) {
  const [snapshot] = useState<ImageGroup[]>(() =>
    useGroupStore.getState().groups.map((g) => ({ ...g, images: g.images.slice() })),
  );
  const [statuses, setStatuses] = useState<Map<string, ReviewStatus>>(() => new Map());
  const [subStatuses, setSubStatuses] = useState<Map<string, SubStatus>>(() => new Map());
  const [bucket, setBucket] = useState<ReviewStatus | null>(null);
  const [topIndex, setTopIndex] = useState(0);
  const [subIndex, setSubIndex] = useState(0);
  const [lightbox, setLightbox] = useState<{ images: ImageInfo[]; index: number } | null>(null);

  const filtered = useMemo(() => {
    if (!bucket) return snapshot;
    return snapshot.filter((g) => effectiveStatus(statuses, g.id) === bucket);
  }, [snapshot, statuses, bucket]);

  const total = filtered.length;
  const currentIndex = bucket ? subIndex : topIndex;
  const current = filtered[currentIndex];

  function setStatusFor(id: string, status: ReviewStatus) {
    setStatuses((prev) => {
      const next = new Map(prev);
      if (prev.get(id) === status) next.delete(id);
      else next.set(id, status);
      return next;
    });
  }

  function setSubStatusFor(id: string, status: SubStatus) {
    setSubStatuses((prev) => {
      const next = new Map(prev);
      if (prev.get(id) === status) next.delete(id);
      else next.set(id, status);
      return next;
    });
  }

  function advance(delta: number) {
    const setter = bucket ? setSubIndex : setTopIndex;
    setter((i) => Math.min(total - 1, Math.max(0, i + delta)));
  }

  function chooseAndAdvance(slot: number) {
    if (!current) return;
    if (bucket) {
      const s = SUB_ORDER[slot]!;
      const wasSame = subStatuses.get(current.id) === s;
      setSubStatusFor(current.id, s);
      if (!wasSame) setSubIndex((i) => Math.min(total - 1, i + 1));
    } else {
      const s = STATUS_ORDER[slot]!;
      const wasSame = statuses.get(current.id) === s;
      setStatusFor(current.id, s);
      if (!wasSame) setTopIndex((i) => Math.min(total - 1, i + 1));
    }
  }

  function enterBucket(b: ReviewStatus) {
    if (bucket === b) {
      setBucket(null);
      return;
    }
    setBucket(b);
    setSubIndex(0);
  }

  function exitBucket() {
    setBucket(null);
  }

  // Stash latest handlers so the keyboard effect can stay subscribed across renders.
  const handlersRef = useRef({ chooseAndAdvance, advance, onClose, exitBucket, bucket });
  handlersRef.current = { chooseAndAdvance, advance, onClose, exitBucket, bucket };

  useEffect(() => {
    if (lightbox) return;
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLElement) {
        const tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") return;
      }
      const h = handlersRef.current;
      if (e.key === "Escape") {
        if (h.bucket !== null) h.exitBucket();
        else h.onClose();
        return;
      }
      if (e.key === "1") h.chooseAndAdvance(0);
      else if (e.key === "2") h.chooseAndAdvance(1);
      else if (e.key === "3") h.chooseAndAdvance(2);
      else if (e.key === "ArrowLeft" || e.key === "h") {
        e.preventDefault();
        h.advance(-1);
      } else if (e.key === "ArrowRight" || e.key === "l") {
        e.preventDefault();
        h.advance(1);
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [lightbox]);

  function openGroupLightbox(groupImages: string[], index: number) {
    const imageMap = useImageStore.getState().imageMap;
    const items: ImageInfo[] = [];
    for (const fn of groupImages) {
      const img = imageMap.get(fn);
      if (img) items.push(img);
    }
    if (items.length === 0) return;
    setLightbox({ images: items, index: Math.min(index, items.length - 1) });
  }

  const topCounts = useMemo(() => {
    const counts: Record<ReviewStatus, number> = { keep: 0, maybe: 0, delete: 0 };
    for (const s of statuses.values()) counts[s]++;
    return counts;
  }, [statuses]);

  const subCounts = useMemo(() => {
    const counts: Record<SubStatus, number> = { top: 0, middle: 0, bottom: 0 };
    if (!bucket) return counts;
    for (const g of filtered) {
      const s = subStatuses.get(g.id);
      if (s) counts[s]++;
    }
    return counts;
  }, [filtered, subStatuses, bucket]);

  function handleApply() {
    if (snapshot.length === 0) {
      onClose();
      return;
    }
    // Unreviewed groups default to "maybe" — keeps are explicit opt-ins.
    const buckets: Record<ReviewStatus, ImageGroup[]> = { keep: [], maybe: [], delete: [] };
    for (const g of snapshot) {
      buckets[effectiveStatus(statuses, g.id)].push(g);
    }
    const sortBucket = (gs: ImageGroup[]) =>
      [...gs].sort(
        (a, b) =>
          SUB_RANK[subStatuses.get(a.id) ?? "middle"] - SUB_RANK[subStatuses.get(b.id) ?? "middle"],
      );
    const newOrder = [
      ...sortBucket(buckets.keep),
      ...sortBucket(buckets.maybe),
      ...sortBucket(buckets.delete),
    ];

    const { images, imageMap, setImages } = useImageStore.getState();
    setImages(reorderImagesByGroups(images, imageMap, newOrder));
    useGroupStore.getState().updateGroups(() => newOrder);

    onClose();
  }

  const currentStatus = current ? statuses.get(current.id) : undefined;
  const currentSub = current ? subStatuses.get(current.id) : undefined;

  const title = (
    <>
      <span>{bucket ? `Refine ${STATUS_LABELS[bucket]}` : "Review Groups"}</span>
      <span className="review-progress">
        {STATUS_ORDER.map((s) => (
          <button
            key={s}
            type="button"
            className={cn(
              "review-progress-chip",
              `review-progress-${s}`,
              bucket === s && "review-progress-chip-active",
            )}
            onClick={() => enterBucket(s)}
            title={
              bucket === s
                ? `Exit ${STATUS_LABELS[s]} refinement`
                : `Refine ${STATUS_LABELS[s]} ordering`
            }
          >
            {topCounts[s]} {s}
          </button>
        ))}
        <span className="review-progress-total">
          {total === 0 ? "0 / 0" : `${currentIndex + 1} / ${total}`}
        </span>
      </span>
    </>
  );

  const footerHint = bucket
    ? "1 top · 2 middle · 3 bottom · ← → navigate · Esc back"
    : "1 keep · 2 maybe · 3 delete · ← → navigate · click chip to refine";

  const footer = (
    <>
      <span className="review-footer-hint modal-footer-spacer">{footerHint}</span>
      {bucket && (
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

  const activeSlot = bucket ? currentSub : currentStatus;
  const singleClass = activeSlot && `review-single-${activeSlot}`;
  const actionItems = bucket
    ? SUB_ORDER.map((s) => ({ s, label: SUB_LABELS[s], activeClass: SUB_ACTIVE_CLASS[s] }))
    : STATUS_ORDER.map((s) => ({
        s,
        label: STATUS_LABELS[s],
        activeClass: STATUS_ACTIVE_CLASS[s],
      }));

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
        {bucket && (
          <div className="review-sub-banner">
            <span className="review-sub-banner-label">
              Refining <strong>{STATUS_LABELS[bucket]}</strong>
            </span>
            <span className="review-sub-banner-counts">
              {SUB_ORDER.map((s) => (
                <span key={s} className={cn("review-sub-chip", `review-sub-chip-${s}`)}>
                  {subCounts[s]} {s}
                </span>
              ))}
            </span>
          </div>
        )}

        {!current ? (
          <div className="review-empty">
            {bucket ? `No groups in ${STATUS_LABELS[bucket]}.` : "No groups to review."}
          </div>
        ) : (
          <div className={cn("review-single", singleClass)}>
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
                  {bucket && currentStatus && (
                    <>
                      {" · "}
                      <span className={`review-single-tag review-single-tag-${currentStatus}`}>
                        {STATUS_LABELS[currentStatus]}
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
              {actionItems.map(({ s, label, activeClass }, idx) => (
                <button
                  type="button"
                  key={s}
                  className={cn(
                    "btn review-single-status-btn",
                    activeSlot === s ? activeClass : "btn-secondary",
                  )}
                  onClick={() => chooseAndAdvance(idx)}
                >
                  <span className="review-single-status-key">{idx + 1}</span>
                  {label}
                </button>
              ))}
            </div>
          </div>
        )}
      </Modal>
      {lightbox && (
        <Lightbox
          images={lightbox.images}
          initialIndex={lightbox.index}
          onClose={() => setLightbox(null)}
        />
      )}
    </>
  );
}
