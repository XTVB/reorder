import React, { useCallback, useRef } from "react";

const INFO_HEIGHT = 40; // card-info bar: 8px padding + 24px badge + 8px padding

/**
 * Reads column count and row height from actual CSS grid layout.
 * An empty hidden div with the same auto-fill grid rule is rendered in the
 * flow; CSS decides columns and we read them via getComputedStyle.
 * Row height is derived from the resolved track width (thumbnail is
 * aspect-ratio:1 so height = width) + the fixed info bar + gap.
 */
export function useGridLayout() {
  const [layout, setLayout] = React.useState({ columnCount: 6, rowHeight: 220 });
  const observerRef = useRef<ResizeObserver | null>(null);

  const measureRowRef = useCallback((node: HTMLDivElement | null) => {
    if (observerRef.current) {
      observerRef.current.disconnect();
      observerRef.current = null;
    }

    if (!node) return;

    function measure() {
      if (!node) return;
      const tracks = getComputedStyle(node).gridTemplateColumns;
      const trackValues = tracks.split(" ");
      const cols = Math.max(1, trackValues.length);
      const trackWidth = parseFloat(trackValues[0]!) || 160;
      const gap = parseFloat(getComputedStyle(node).gap) || 16;
      const rowHeight = Math.ceil(trackWidth + INFO_HEIGHT + gap);

      setLayout((prev) =>
        prev.columnCount === cols && prev.rowHeight === rowHeight
          ? prev
          : { columnCount: cols, rowHeight }
      );
    }

    const observer = new ResizeObserver(() => measure());
    observer.observe(node);
    observerRef.current = observer;
    requestAnimationFrame(measure);
  }, []);

  return { ...layout, measureRowRef };
}
