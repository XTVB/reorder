// Points the hovered card at its pre-sort slot or viewport edge if out of view

import { type RefObject, useLayoutEffect, useState } from "react";
import type { SortOrigin } from "../../hooks/useSortOrigin.ts";

interface Props {
  origin: SortOrigin;
  cardEl: HTMLElement;
  containerRef: RefObject<HTMLElement | null>;
  rowHeight: number;
  gap: number;
}

interface Geometry {
  from: { x: number; y: number };
  to: { x: number; y: number };
  ghost: { left: number; top: number; width: number; height: number } | null;
  offscreen: "above" | "below" | null;
  label: string;
}

const EDGE_INSET = 44;

function rowLabel(rows: number, direction: "above" | "below"): string {
  const n = Math.abs(rows);
  return `${direction === "above" ? "↑" : "↓"} ${n} ${n === 1 ? "row" : "rows"} ${direction}`;
}

function sameGeometry(a: Geometry, b: Geometry): boolean {
  return (
    a.from.x === b.from.x &&
    a.from.y === b.from.y &&
    a.to.x === b.to.x &&
    a.to.y === b.to.y &&
    a.offscreen === b.offscreen &&
    a.label === b.label &&
    a.ghost?.left === b.ghost?.left &&
    a.ghost?.top === b.ghost?.top &&
    a.ghost?.width === b.ghost?.width &&
    a.ghost?.height === b.ghost?.height
  );
}

export function SortOriginArrow({ origin, cardEl, containerRef, rowHeight, gap }: Props) {
  const [geo, setGeo] = useState<Geometry | null>(null);

  useLayoutEffect(() => {
    if (!containerRef.current) return;
    const containerEl: HTMLElement = containerRef.current;
    const rowEl = cardEl.closest<HTMLElement>(".grid-row");
    // Coordinates are relative to the virtualizer's sized div (the overlay's
    // offset parent), not the scroll container, which adds its own padding.
    const originEl = rowEl?.parentElement ?? containerEl;

    function compute(container: HTMLElement): Geometry {
      const cardRect = cardEl.getBoundingClientRect();
      const originRect = originEl.getBoundingClientRect();
      const rowRect = (rowEl ?? originEl).getBoundingClientRect();

      const from = {
        x: cardRect.left - originRect.left + cardRect.width / 2,
        y: cardRect.top - originRect.top + cardRect.height / 2,
      };

      // Track width comes from the card's own rect plus the gap; dividing the
      // container width instead folds the gaps in and drifts across columns.
      const ghost = {
        left: rowRect.left - originRect.left + origin.column * (cardRect.width + gap),
        top: origin.row * rowHeight,
        width: cardRect.width,
        height: Math.max(24, rowHeight - gap),
      };
      const to = { x: ghost.left + ghost.width / 2, y: ghost.top + ghost.height / 2 };

      const viewTop = container.getBoundingClientRect().top - originRect.top;
      const viewBottom = viewTop + container.clientHeight;
      const rowsAway = Math.round((to.y - from.y) / rowHeight);

      if (to.y < viewTop + 8) {
        return {
          from,
          to: { x: from.x, y: viewTop + EDGE_INSET },
          ghost: null,
          offscreen: "above",
          label: rowLabel(rowsAway, "above"),
        };
      }
      if (to.y > viewBottom - 8) {
        return {
          from,
          to: { x: from.x, y: viewBottom - EDGE_INSET },
          ghost: null,
          offscreen: "below",
          label: rowLabel(rowsAway, "below"),
        };
      }
      return { from, to, ghost, offscreen: null, label: "" };
    }

    function update() {
      const next = compute(containerEl);
      setGeo((prev) => (prev && sameGeometry(prev, next) ? prev : next));
    }

    update();

    let frame = 0;
    function schedule() {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        update();
      });
    }
    containerEl.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      if (frame) cancelAnimationFrame(frame);
      containerEl.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, [origin, cardEl, containerRef, rowHeight, gap]);

  if (!geo) return null;

  const { from, to, ghost, offscreen, label } = geo;
  // Bow perpendicular to travel so the path reads as an arc rather than
  // disappearing under the cards it crosses.
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy) || 1;
  const bow = Math.min(70, len * 0.22);
  const ctrlX = (from.x + to.x) / 2 - (dy / len) * bow;
  const ctrlY = (from.y + to.y) / 2 + (dx / len) * bow;

  const kind = origin.displaced ? "displaced" : "moved";

  return (
    <svg className={`sort-origin-overlay sort-origin-${kind}`} aria-hidden>
      <title>{origin.displaced ? "Pushed along by other moves" : "Moved by the sort"}</title>
      <defs>
        <marker
          id={`sort-origin-head-${kind}`}
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" className="sort-origin-arrowhead" />
        </marker>
      </defs>

      {ghost && (
        <rect
          className="sort-origin-ghost"
          x={ghost.left}
          y={ghost.top}
          width={ghost.width}
          height={ghost.height}
          rx="8"
        />
      )}

      <path
        className="sort-origin-path"
        d={`M ${from.x} ${from.y} Q ${ctrlX} ${ctrlY} ${to.x} ${to.y}`}
        markerEnd={`url(#sort-origin-head-${kind})`}
      />

      {offscreen && (
        <g transform={`translate(${to.x}, ${to.y + (offscreen === "above" ? -22 : 22)})`}>
          <rect className="sort-origin-label-bg" x={-86} y={-13} width={172} height={26} rx="13" />
          <text className="sort-origin-label" x={0} y={5} textAnchor="middle">
            {origin.displaced ? `${label} (pushed)` : label}
          </text>
        </g>
      )}
    </svg>
  );
}
