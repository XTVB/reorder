import { useEffect, useMemo, useRef, useState } from "react";
import { useGroupStore } from "../stores/groupStore.ts";
import { useImageStore } from "../stores/imageStore.ts";
import type { ImageGroup, ImageInfo } from "../types.ts";
import { cn, imageUrl, reorderImagesByGroups } from "../utils/helpers.ts";
import { Lightbox } from "./Lightbox.tsx";
import { Modal } from "./Modal.tsx";

type ReviewStatus = "keep" | "maybe" | "delete";

const STATUS_ORDER: ReviewStatus[] = ["keep", "maybe", "delete"];

const STATUS_LABELS: Record<ReviewStatus, string> = {
  keep: "Keep",
  maybe: "Maybe",
  delete: "Delete",
};

const STATUS_ACTIVE_CLASS: Record<ReviewStatus, string> = {
  keep: "review-status-keep-active",
  maybe: "review-status-maybe-active",
  delete: "review-status-delete-active",
};

interface ReviewModalProps {
  onClose: () => void;
}

export function ReviewModal({ onClose }: ReviewModalProps) {
  const [snapshot] = useState<ImageGroup[]>(() =>
    useGroupStore.getState().groups.map((g) => ({ ...g, images: g.images.slice() })),
  );
  const [statuses, setStatuses] = useState<Map<string, ReviewStatus>>(() => new Map());
  const [currentIndex, setCurrentIndex] = useState(0);
  const [lightbox, setLightbox] = useState<{ images: ImageInfo[]; index: number } | null>(null);

  const current = snapshot[currentIndex];
  const total = snapshot.length;

  function setStatusFor(id: string, status: ReviewStatus) {
    setStatuses((prev) => {
      const next = new Map(prev);
      if (prev.get(id) === status) next.delete(id);
      else next.set(id, status);
      return next;
    });
  }

  function advance(delta: number) {
    setCurrentIndex((i) => Math.min(total - 1, Math.max(0, i + delta)));
  }

  function chooseAndAdvance(status: ReviewStatus) {
    if (!current) return;
    const wasSame = statuses.get(current.id) === status;
    setStatusFor(current.id, status);
    if (!wasSame) setCurrentIndex((i) => Math.min(total - 1, i + 1));
  }

  // Stash latest handlers so the keyboard effect can stay subscribed across renders
  // without re-adding the listener every time statuses change.
  const handlersRef = useRef({ chooseAndAdvance, advance, onClose });
  handlersRef.current = { chooseAndAdvance, advance, onClose };

  // Keyboard shortcuts — disabled while lightbox is open (it owns its own keys).
  useEffect(() => {
    if (lightbox) return;
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLElement) {
        const tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") return;
      }
      const h = handlersRef.current;
      if (e.key === "Escape") {
        h.onClose();
        return;
      }
      if (e.key === "1") h.chooseAndAdvance("keep");
      else if (e.key === "2") h.chooseAndAdvance("maybe");
      else if (e.key === "3") h.chooseAndAdvance("delete");
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

  const counts = useMemo(() => {
    let keep = 0;
    let maybe = 0;
    let del = 0;
    for (const s of statuses.values()) {
      if (s === "keep") keep++;
      else if (s === "maybe") maybe++;
      else if (s === "delete") del++;
    }
    return { keep, maybe, delete: del };
  }, [statuses]);

  function handleApply() {
    if (snapshot.length === 0) {
      onClose();
      return;
    }
    // Unreviewed groups default to "maybe" — keeps are explicit opt-ins.
    const keeps: ImageGroup[] = [];
    const maybes: ImageGroup[] = [];
    const deletes: ImageGroup[] = [];
    for (const g of snapshot) {
      const s = statuses.get(g.id);
      if (s === "keep") keeps.push(g);
      else if (s === "delete") deletes.push(g);
      else maybes.push(g);
    }
    const newOrder = [...keeps, ...maybes, ...deletes];

    const { images, imageMap, setImages } = useImageStore.getState();
    setImages(reorderImagesByGroups(images, imageMap, newOrder));
    useGroupStore.getState().updateGroups(() => newOrder);

    onClose();
  }

  const currentStatus = current ? statuses.get(current.id) : undefined;

  const title = (
    <>
      <span>Review Groups</span>
      <span className="review-progress">
        <span className="review-progress-chip review-progress-keep">{counts.keep} keep</span>
        <span className="review-progress-chip review-progress-maybe">{counts.maybe} maybe</span>
        <span className="review-progress-chip review-progress-delete">{counts.delete} delete</span>
        <span className="review-progress-total">
          {currentIndex + 1} / {total}
        </span>
      </span>
    </>
  );

  const footer = (
    <>
      <span className="review-footer-hint modal-footer-spacer">
        1 keep · 2 maybe · 3 delete · ← → navigate
      </span>
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
        {!current ? (
          <div className="review-empty">No groups to review.</div>
        ) : (
          <div className={cn("review-single", currentStatus && `review-single-${currentStatus}`)}>
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
              {STATUS_ORDER.map((s) => {
                const active = currentStatus === s;
                return (
                  <button
                    type="button"
                    key={s}
                    className={cn(
                      "btn review-single-status-btn",
                      active ? STATUS_ACTIVE_CLASS[s] : "btn-secondary",
                    )}
                    onClick={() => chooseAndAdvance(s)}
                  >
                    <span className="review-single-status-key">{STATUS_ORDER.indexOf(s) + 1}</span>
                    {STATUS_LABELS[s]}
                  </button>
                );
              })}
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
