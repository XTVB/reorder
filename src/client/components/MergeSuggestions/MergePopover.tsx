import type React from "react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { imageUrl } from "../../utils/helpers.ts";
import { FloatingPopoverContent } from "../GroupPopover.tsx";

const MARGIN = 16;
const GAP_BELOW_CARD = 8;

interface MergePopoverProps {
  anchorRect: DOMRect;
  displayName: string;
  images: string[];
  onOpenLightbox: (images: string[], index: number) => void;
  onClose: () => void;
}

/**
 * Portal-rendered floating popover for the merge-suggestions page. Escapes
 * the row's overflow-x clipping, anchors visually below the clicked card's
 * rect, clamps to viewport, closes on escape / click-outside / scroll.
 */
export function MergePopover({
  anchorRect,
  displayName,
  images,
  onOpenLightbox,
  onClose,
}: MergePopoverProps) {
  const popRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);

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

  // Close on escape, outside click, or ancestor scroll
  useEffect(() => {
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
  }, [onClose]);

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
          <MergeExpandedThumb key={fn} filename={fn} onClick={() => onOpenLightbox(images, i)} />
        ))}
      </FloatingPopoverContent>
    </div>,
    document.body,
  );
}

function MergeExpandedThumb({
  filename,
  onClick,
}: {
  filename: string;
  onClick: (e: React.MouseEvent) => void;
}) {
  return (
    <div className="card group-image-card merge-expanded-thumb" onClick={onClick}>
      <img
        className="card-thumb"
        src={imageUrl(filename)}
        alt={filename}
        loading="lazy"
        draggable={false}
      />
      <div className="card-info">
        <span className="card-name" title={filename}>
          {filename}
        </span>
      </div>
    </div>
  );
}
