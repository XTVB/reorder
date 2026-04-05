import type { Virtualizer } from "@tanstack/react-virtual";
import { type RefObject, useLayoutEffect } from "react";

/**
 * Synchronously re-measure all mounted virtualized rows whenever `deps` change.
 *
 * React doesn't re-fire ref callbacks on update, and the virtualizer's own
 * ResizeObserver fires asynchronously (after paint), which causes a visible
 * flash of stale layout after toggle/rename/split/merge. We force a synchronous
 * re-measure in useLayoutEffect so corrected positions land before paint.
 *
 * IMPORTANT: do NOT call virtualizer.measure() here. It clears the entire
 * itemSizeCache, and resizeItem() then only writes back entries whose size
 * *changed* (delta !== 0). Unchanged rows lose their cached size and fall back
 * to estimateSize() — which produces wrong positions for rows whose actual
 * height differs from the estimate.
 */
export function useRemeasureVirtualRows(
  // biome-ignore lint/suspicious/noExplicitAny: virtualizer generics vary per call site
  virtualizer: Virtualizer<any, any>,
  containerRef: RefObject<HTMLElement | null>,
  deps: unknown[],
) {
  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    for (const el of container.querySelectorAll<HTMLElement>("[data-index]")) {
      virtualizer.measureElement(el);
    }
    // biome-ignore lint/correctness/useExhaustiveDependencies: deps is the intentional external trigger, virtualizer/containerRef are stable
  }, deps);
}
