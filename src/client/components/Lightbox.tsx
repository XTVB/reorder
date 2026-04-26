import type React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useTrashStore } from "../stores/trashStore.ts";
import type { ImageInfo } from "../types.ts";
import { cn, fullImageUrl } from "../utils/helpers.ts";
import { TrashIcon } from "./TrashIcon.tsx";

export function Lightbox({
  images,
  initialIndex,
  onClose,
  enableTrashMark = false,
}: {
  images: ImageInfo[];
  initialIndex: number;
  onClose: () => void;
  enableTrashMark?: boolean;
}) {
  const [index, setIndex] = useState(initialIndex);
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [dimensions, setDimensions] = useState<{ w: number; h: number } | null>(null);
  const panStart = useRef({ x: 0, y: 0 });
  const translateStart = useRef({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const image = images[index]!;
  const isZoomed = scale > 1;

  const markedIds = useTrashStore((s) => s.markedIds);
  const toggleTrashMark = useTrashStore((s) => s.toggle);
  const isMarked = enableTrashMark && markedIds.has(image.filename);
  const trashButtonLabel = isMarked ? "Unmark for deletion" : "Mark for deletion";

  const handleImageLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setDimensions({ w: img.naturalWidth, h: img.naturalHeight });
  }, []);

  function resetView() {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  }

  const indexRef = useRef(index);
  indexRef.current = index;
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  const trashEnabledRef = useRef(enableTrashMark);
  trashEnabledRef.current = enableTrashMark;

  // biome-ignore lint/correctness/useExhaustiveDependencies: mount-only — uses refs for current index/onClose/trash to avoid re-registering on every navigation
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      switch (e.key) {
        case "Escape":
          onCloseRef.current();
          break;
        case "ArrowLeft": {
          const prev = indexRef.current - 1;
          if (prev >= 0) {
            setIndex(prev);
            resetView();
          }
          break;
        }
        case "ArrowRight": {
          const next = indexRef.current + 1;
          if (next < images.length) {
            setIndex(next);
            resetView();
          }
          break;
        }
        case "+":
        case "=":
          setScale((s) => Math.min(8, s + 0.5));
          break;
        case "-":
          setScale((s) => {
            const next = Math.max(1, s - 0.5);
            if (next === 1) setTranslate({ x: 0, y: 0 });
            return next;
          });
          break;
        case "0":
          resetView();
          break;
        case "d":
        case "D":
          if (trashEnabledRef.current) {
            const fn = images[indexRef.current]?.filename;
            if (fn) useTrashStore.getState().toggle(fn);
          }
          break;
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [images.length]);

  function go(dir: -1 | 1) {
    const next = index + dir;
    if (next >= 0 && next < images.length) {
      setIndex(next);
      resetView();
    }
  }

  function handleWheel(e: React.WheelEvent) {
    e.stopPropagation();
    const delta = e.deltaY > 0 ? -0.2 : 0.2;
    setScale((s) => {
      const next = Math.min(8, Math.max(1, s + delta));
      if (next === 1) setTranslate({ x: 0, y: 0 });
      return next;
    });
  }

  function handleDoubleClick() {
    if (isZoomed) resetView();
    else setScale(3);
  }

  function handlePointerDown(e: React.PointerEvent) {
    if (!isZoomed) return;
    e.preventDefault();
    setIsPanning(true);
    panStart.current = { x: e.clientX, y: e.clientY };
    translateStart.current = { ...translate };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }

  function handlePointerMove(e: React.PointerEvent) {
    if (!isPanning) return;
    setTranslate({
      x: translateStart.current.x + (e.clientX - panStart.current.x),
      y: translateStart.current.y + (e.clientY - panStart.current.y),
    });
  }

  return (
    <div
      className="lightbox-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <button className="lightbox-close" onClick={onClose} aria-label="Close">
        &times;
      </button>

      {index > 0 && (
        <button className="lightbox-nav lightbox-prev" onClick={() => go(-1)} aria-label="Previous">
          &#8249;
        </button>
      )}
      {index < images.length - 1 && (
        <button className="lightbox-nav lightbox-next" onClick={() => go(1)} aria-label="Next">
          &#8250;
        </button>
      )}

      <div
        ref={containerRef}
        className="lightbox-image-container"
        onWheel={handleWheel}
        onDoubleClick={handleDoubleClick}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={() => setIsPanning(false)}
        style={{ cursor: isZoomed ? (isPanning ? "grabbing" : "grab") : "zoom-in" }}
      >
        <img
          className="lightbox-image"
          src={fullImageUrl(image.filename)}
          alt={image.filename}
          draggable={false}
          onLoad={handleImageLoad}
          style={{ transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})` }}
        />
      </div>

      {enableTrashMark && (
        <button
          type="button"
          className={cn("lightbox-trash", isMarked && "lightbox-trash-active")}
          onClick={() => toggleTrashMark(image.filename)}
          aria-label={trashButtonLabel}
          title={`${trashButtonLabel} (D)`}
        >
          <TrashIcon size={22} />
        </button>
      )}

      <div className="lightbox-bar">
        <span className="lightbox-filename">
          {image.filename}
          {dimensions && (
            <span className="lightbox-dimensions">
              {dimensions.w} &times; {dimensions.h}
            </span>
          )}
          {isMarked && <span className="lightbox-marked">marked for deletion</span>}
        </span>
        <span className="lightbox-counter">
          {index + 1} / {images.length}
        </span>
        {isZoomed && <span className="lightbox-zoom">{Math.round(scale * 100)}%</span>}
      </div>
    </div>
  );
}
