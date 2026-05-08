import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { FloatingPopoverContent } from "../shared/GroupPopover.tsx";
import { ImageThumb } from "../shared/ImageThumb.tsx";

const MARGIN = 16;
const GAP_BELOW_CARD = 8;

interface MergePopoverProps {
  anchorRect: DOMRect;
  displayName: string;
  images: string[];
  onClose: () => void;
}

/**
 * Portal-rendered floating popover for the merge-suggestions page. Escapes
 * the row's overflow-x clipping, anchors visually below the clicked card's
 * rect, clamps to viewport, closes on escape / click-outside / scroll.
 */
export function MergePopover({ anchorRect, displayName, images, onClose }: MergePopoverProps) {
  const popRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const lightboxOpen = useLightboxStore((s) => s.open);

  // Position below the anchor card, clamped to viewport. Use layout effect so
  // the popover doesn't flicker at (0,0) on its first paint.
  useLayoutEffect(() => {
    const pop = popRef.current;
    if (!pop) return;
    const popWidth = pop.offsetWidth;
    const popHeight = pop.offsetHeight;
    const cardCenterX = anchorRect.left + anchorRect.width / 2;
    let left = cardCenterX - popWidth / 2;
    left = Math.max(MARGIN, Math.min(left, window.innerWidth - popWidth - MARGIN));

    // Prefer below, fall back to above if it would overflow
    let top = anchorRect.bottom + GAP_BELOW_CARD;
    if (top + popHeight > window.innerHeight - MARGIN) {
      const above = anchorRect.top - GAP_BELOW_CARD - popHeight;
      if (above >= MARGIN) top = above;
      else top = Math.max(MARGIN, window.innerHeight - popHeight - MARGIN);
    }
    setPos({ top, left });
  }, [anchorRect]);

  // Close on escape, outside click, or ancestor scroll. Defer to the lightbox
  // when it's open so its escape/click consumes the event first.
  useEffect(() => {
    if (lightboxOpen) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    function handleScroll(e: Event) {
      // Ignore scrolls inside the popover's own grid; only close when an
      // ancestor scrolls, which moves the anchor out from under the popover.
      if (popRef.current?.contains(e.target as Node)) return;
      onClose();
    }
    function handlePointerDown(e: PointerEvent) {
      if (!popRef.current?.contains(e.target as Node)) onClose();
    }
    window.addEventListener("keydown", handleKey);
    // Capture phase catches scrolling of nested overflow containers
    window.addEventListener("scroll", handleScroll, true);
    window.addEventListener("pointerdown", handlePointerDown, true);
    return () => {
      window.removeEventListener("keydown", handleKey);
      window.removeEventListener("scroll", handleScroll, true);
      window.removeEventListener("pointerdown", handlePointerDown, true);
    };
  }, [onClose, lightboxOpen]);

  return createPortal(
    <div
      ref={popRef}
      className="merge-popover-portal"
      style={{
        position: "fixed",
        top: pos?.top ?? 0,
        left: pos?.left ?? 0,
        visibility: pos ? "visible" : "hidden",
        zIndex: 200,
        maxHeight: `calc(100vh - ${MARGIN * 2}px)`,
        overflowY: "auto",
      }}
    >
      <FloatingPopoverContent
        displayName={displayName}
        imageCount={images.length}
        actions={
          <button className="btn btn-small btn-secondary" onClick={onClose}>
            Close
          </button>
        }
      >
        {images.map((fn, i) => (
          <ImageThumb
            key={fn}
            filename={fn}
            lightboxImages={images}
            lightboxIndex={i}
            footer={
              <span className="image-thumb-name" title={fn}>
                {fn}
              </span>
            }
          />
        ))}
      </FloatingPopoverContent>
    </div>,
    document.body,
  );
}
