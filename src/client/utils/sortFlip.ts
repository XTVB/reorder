// FLIP transition for bulk reorders (sort apply, ⌥-peek toggle).
//
// Callers snapshot the on-screen card positions with captureFlipRects() just
// before mutating the order; ReorderView calls playPendingFlip() in a layout
// effect once the reordered grid is in the DOM, animating every card from its
// old position to its new one. Animations run through WAAPI (el.animate) so
// they never touch inline styles and can't fight dnd-kit's transform handling.
//
// Cards are tagged via the data-flip-id attribute (group/folder sort id, or
// the image filename). The grid is virtualized, so only cards visible in both
// states slide; cards that newly enter the viewport fade in instead.

import { useGroupStore } from "../stores/groupStore.ts";
import { useImageStore } from "../stores/imageStore.ts";
import { openSortWriteWindow, useSortHistoryStore } from "../stores/sortHistoryStore.ts";

export const FLIP_ATTR = "data-flip-id";

// A capture is only valid for the render it was taken for; if that render
// never happens (no-op sort), drop the snapshot rather than replaying it
// against some unrelated later update.
const CAPTURE_MAX_AGE_MS = 1000;

// The sort-apply flip is a "settle into place" gesture, so it decelerates hard.
// The peek is a *comparison*: it wants near-constant speed through the middle
// so a card can be followed with the eye rather than snapping at one end.
const GESTURES = {
  sort: { duration: 500, easing: "cubic-bezier(0.22, 1, 0.36, 1)" },
  peek: { duration: 700, easing: "cubic-bezier(0.45, 0.05, 0.55, 0.95)" },
} as const;

export type FlipGesture = keyof typeof GESTURES;

interface PendingFlip {
  rects: Map<string, DOMRect>;
  duration: number;
  easing: string;
  capturedAt: number;
}

let pending: PendingFlip | null = null;

/** Snapshot the current position of every flip-tagged card. */
export function captureFlipRects(gesture: FlipGesture = "sort"): void {
  const rects = new Map<string, DOMRect>();
  for (const el of document.querySelectorAll<HTMLElement>(`[${FLIP_ATTR}]`)) {
    const id = el.getAttribute(FLIP_ATTR);
    if (id) rects.set(id, el.getBoundingClientRect());
  }
  pending = { rects, ...GESTURES[gesture], capturedAt: performance.now() };
}

/** Animate flip-tagged cards from their captured positions to where they are now. */
export function playPendingFlip(): void {
  const captured = pending;
  pending = null;
  if (!captured || performance.now() - captured.capturedAt > CAPTURE_MAX_AGE_MS) return;
  for (const el of document.querySelectorAll<HTMLElement>(`[${FLIP_ATTR}]`)) {
    const id = el.getAttribute(FLIP_ATTR);
    if (!id) continue;
    const prev = captured.rects.get(id);
    if (!prev) {
      // Was outside the virtualized viewport before the reorder.
      el.animate([{ opacity: 0 }, { opacity: 1 }], {
        duration: captured.duration,
        easing: "ease-out",
      });
      continue;
    }
    const now = el.getBoundingClientRect();
    const dx = prev.left - now.left;
    const dy = prev.top - now.top;
    if (Math.abs(dx) < 1 && Math.abs(dy) < 1) continue;
    el.animate(
      [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "translate(0px, 0px)" }],
      { duration: captured.duration, easing: captured.easing },
    );
  }
}

/**
 * Call right before applying a sort: captures the FLIP snapshot for the
 * slide-to-new-position animation and stashes the current order so holding
 * ⌥ afterwards can show the pre-sort state for comparison. Must be called in
 * the same synchronous block as the order write(s) it describes, which it
 * marks as sort-owned so they don't invalidate the snapshot.
 */
export function beginSortTransition(): void {
  captureFlipRects("sort");
  const { images } = useImageStore.getState();
  const { groups } = useGroupStore.getState();
  openSortWriteWindow();
  useSortHistoryStore.getState().setPreviousOrder({ images, groups });
}
