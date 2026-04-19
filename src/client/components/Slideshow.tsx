import type React from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type SlideshowTransition, useUIStore } from "../stores/uiStore.ts";
import type { ImageInfo } from "../types.ts";
import { fullImageUrl } from "../utils/helpers.ts";

const IDLE_MS = 1000;
const INTERVAL_MIN = 0.25;
const INTERVAL_MAX = 60;

function shuffledIndices(n: number, firstIndex: number): number[] {
  const arr = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i]!, arr[j]!] = [arr[j]!, arr[i]!];
  }
  if (n > 0) {
    const pos = arr.indexOf(firstIndex);
    if (pos > 0) {
      [arr[0]!, arr[pos]!] = [arr[pos]!, arr[0]!];
    }
  }
  return arr;
}

export function Slideshow({
  images,
  initialIndex,
  onClose,
}: {
  images: ImageInfo[];
  initialIndex: number;
  onClose: () => void;
}) {
  const playing = useUIStore((s) => s.slideshow.playing);
  const intervalMs = useUIStore((s) => s.slideshow.intervalMs);
  const shuffle = useUIStore((s) => s.slideshow.shuffle);
  const transition = useUIStore((s) => s.slideshow.transition);
  const setPlaying = useUIStore((s) => s.setSlideshowPlaying);
  const setInterval_ = useUIStore((s) => s.setSlideshowInterval);
  const setShuffle = useUIStore((s) => s.setSlideshowShuffle);
  const setTransition = useUIStore((s) => s.setSlideshowTransition);

  const safeInitial = Math.max(0, Math.min(initialIndex, images.length - 1));

  const [playOrder, setPlayOrder] = useState<number[]>(() =>
    shuffle
      ? shuffledIndices(images.length, safeInitial)
      : Array.from({ length: images.length }, (_, i) => i),
  );
  const [position, setPosition] = useState<number>(() => (shuffle ? 0 : safeInitial));
  const [dimensions, setDimensions] = useState<{ w: number; h: number } | null>(null);
  const [uiVisible, setUiVisible] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(() => document.fullscreenElement != null);
  const [intervalInput, setIntervalInput] = useState(() => (intervalMs / 1000).toFixed(2));
  const backdropRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setIntervalInput((intervalMs / 1000).toFixed(2).replace(/\.?0+$/, "") || "0");
  }, [intervalMs]);

  const imageCount = images.length;
  const currentImageIndex = playOrder[position] ?? 0;
  const currentImage = images[currentImageIndex];
  const currentUrl = currentImage ? fullImageUrl(currentImage.filename) : "";

  const [layers, setLayers] = useState<[string, string]>(() => [currentUrl, ""]);
  const [topLayer, setTopLayer] = useState<0 | 1>(0);

  useEffect(() => {
    if (!currentUrl) return;
    setLayers((prev) => {
      if (prev[topLayer] === currentUrl) return prev;
      const nextTop: 0 | 1 = topLayer === 0 ? 1 : 0;
      const copy: [string, string] = [...prev];
      copy[nextTop] = currentUrl;
      setTopLayer(nextTop);
      return copy;
    });
  }, [currentUrl, topLayer]);

  const go = useCallback(
    (dir: 1 | -1) => {
      if (imageCount === 0) return;
      setPosition((p) => (p + dir + imageCount) % imageCount);
    },
    [imageCount],
  );

  const goRef = useRef(go);
  goRef.current = go;
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  const handleToggleShuffle = useCallback(() => {
    const nextShuffle = !shuffle;
    setShuffle(nextShuffle);
    const currentIdx = playOrder[position] ?? 0;
    if (nextShuffle) {
      setPlayOrder(shuffledIndices(imageCount, currentIdx));
      setPosition(0);
    } else {
      setPlayOrder(Array.from({ length: imageCount }, (_, i) => i));
      setPosition(currentIdx);
    }
  }, [shuffle, setShuffle, playOrder, position, imageCount]);

  // `position` is intentionally in deps: manual nav restarts the timer so the
  // next auto-advance gets the full interval from the click, not a remainder.
  // biome-ignore lint/correctness/useExhaustiveDependencies: see above
  useEffect(() => {
    if (!playing || imageCount < 2) return;
    const id = window.setInterval(() => goRef.current(1), intervalMs);
    return () => window.clearInterval(id);
  }, [playing, intervalMs, imageCount, position]);

  const idleTimerRef = useRef<number | null>(null);
  const resetIdleTimer = useCallback(() => {
    setUiVisible(true);
    if (idleTimerRef.current != null) window.clearTimeout(idleTimerRef.current);
    idleTimerRef.current = window.setTimeout(() => setUiVisible(false), IDLE_MS);
  }, []);

  useEffect(() => {
    resetIdleTimer();
    return () => {
      if (idleTimerRef.current != null) window.clearTimeout(idleTimerRef.current);
    };
  }, [resetIdleTimer]);

  const toggleFullscreen = useCallback(async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else if (backdropRef.current) {
        await backdropRef.current.requestFullscreen();
      }
    } catch {
      // User denied or browser doesn't support it
    }
  }, []);
  const toggleFullscreenRef = useRef(toggleFullscreen);
  toggleFullscreenRef.current = toggleFullscreen;

  useEffect(() => {
    function onChange() {
      setIsFullscreen(document.fullscreenElement != null);
    }
    document.addEventListener("fullscreenchange", onChange);
    return () => {
      document.removeEventListener("fullscreenchange", onChange);
      if (document.fullscreenElement) {
        document.exitFullscreen().catch(() => {});
      }
    };
  }, []);

  // Deliberately skips resetIdleTimer so keyboard-driven nav stays immersive —
  // only mouse movement brings chrome back.
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement) return;
      switch (e.key) {
        case "Escape":
          onCloseRef.current();
          break;
        case "ArrowLeft":
          e.preventDefault();
          goRef.current(-1);
          break;
        case "ArrowRight":
          e.preventDefault();
          goRef.current(1);
          break;
        case " ":
          e.preventDefault();
          setPlaying(!playing);
          break;
        case "s":
        case "S":
          if (e.metaKey || e.ctrlKey) return;
          handleToggleShuffle();
          break;
        case "f":
        case "F":
          if (e.metaKey || e.ctrlKey) return;
          toggleFullscreenRef.current();
          break;
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleToggleShuffle, setPlaying, playing]);

  const preloadUrls = useMemo(() => {
    if (imageCount < 2) return [];
    const urls: string[] = [];
    for (const offset of [1, -1, 2]) {
      const idx = playOrder[(position + offset + imageCount) % imageCount];
      if (idx != null && images[idx]) urls.push(fullImageUrl(images[idx].filename));
    }
    return urls;
  }, [images, playOrder, position, imageCount]);

  function handleImageLoad(e: React.SyntheticEvent<HTMLImageElement>) {
    const img = e.currentTarget;
    // Only record dims for the layer currently in front (compare by attribute, not resolved URL)
    if (img.getAttribute("src") !== currentUrl) return;
    setDimensions({ w: img.naturalWidth, h: img.naturalHeight });
  }

  function commitIntervalInput() {
    const seconds = parseFloat(intervalInput);
    if (!Number.isFinite(seconds)) {
      setIntervalInput((intervalMs / 1000).toString());
      return;
    }
    const clamped = Math.min(INTERVAL_MAX, Math.max(INTERVAL_MIN, seconds));
    setInterval_(Math.round(clamped * 1000));
  }

  if (!currentImage) {
    return (
      <div className="slideshow-backdrop" onClick={onClose}>
        <div className="slideshow-empty">No images to display</div>
      </div>
    );
  }

  const backdropClass =
    `slideshow-backdrop slideshow-transition-${transition}` +
    (uiVisible ? " slideshow-ui-visible" : " slideshow-ui-hidden");

  return (
    <div
      ref={backdropRef}
      className={backdropClass}
      onMouseMove={resetIdleTimer}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      {(["a", "b"] as const).map((slot, idx) => {
        const url = layers[idx as 0 | 1];
        if (!url) return null;
        const isTop = topLayer === idx;
        return (
          <img
            key={slot}
            className={`slideshow-image${isTop ? " slideshow-image-top" : ""}`}
            src={url}
            alt={isTop && currentImage ? currentImage.filename : ""}
            draggable={false}
            onLoad={handleImageLoad}
          />
        );
      })}

      <div className="slideshow-preload" aria-hidden>
        {preloadUrls.map((url) => (
          <img key={url} src={url} alt="" />
        ))}
      </div>

      <div className="slideshow-chrome slideshow-top-right">
        <button
          type="button"
          className="slideshow-icon-btn"
          onClick={toggleFullscreen}
          aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          aria-pressed={isFullscreen}
          title={isFullscreen ? "Exit fullscreen (F)" : "Fullscreen (F)"}
        >
          {isFullscreen ? (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" role="presentation">
              <path
                d="M8 3v5H3M16 3v5h5M8 21v-5H3M16 21v-5h5"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" role="presentation">
              <path
                d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </button>
        <button
          type="button"
          className="slideshow-icon-btn slideshow-close"
          onClick={onClose}
          aria-label="Close slideshow"
          title="Close (Esc)"
        >
          &times;
        </button>
      </div>

      {imageCount > 1 && (
        <>
          <button
            type="button"
            className="slideshow-chrome slideshow-nav slideshow-prev"
            onClick={() => {
              go(-1);
              resetIdleTimer();
            }}
            aria-label="Previous"
            title="Previous (←)"
          >
            &#8249;
          </button>
          <button
            type="button"
            className="slideshow-chrome slideshow-nav slideshow-next"
            onClick={() => {
              go(1);
              resetIdleTimer();
            }}
            aria-label="Next"
            title="Next (→)"
          >
            &#8250;
          </button>
        </>
      )}

      <div
        className="slideshow-chrome slideshow-controls"
        onClick={(e) => e.stopPropagation()}
        onMouseMove={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          className="slideshow-play-btn"
          onClick={() => setPlaying(!playing)}
          aria-label={playing ? "Pause" : "Play"}
          title={playing ? "Pause (Space)" : "Play (Space)"}
          disabled={imageCount < 2}
        >
          {playing ? "❚❚" : "▶"}
        </button>

        <label className="slideshow-interval">
          <span>Every</span>
          <input
            type="number"
            min={INTERVAL_MIN}
            max={INTERVAL_MAX}
            step={0.25}
            value={intervalInput}
            onChange={(e) => setIntervalInput(e.target.value)}
            onBlur={commitIntervalInput}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                commitIntervalInput();
                (e.currentTarget as HTMLInputElement).blur();
              }
            }}
          />
          <span>s</span>
        </label>

        <button
          type="button"
          className={`slideshow-toggle${shuffle ? " slideshow-toggle-on" : ""}`}
          onClick={handleToggleShuffle}
          aria-pressed={shuffle}
          title="Shuffle (S)"
        >
          ⤨ Shuffle
        </button>

        <label className="slideshow-select-field">
          <span>Transition</span>
          <select
            value={transition}
            onChange={(e) => setTransition(e.target.value as SlideshowTransition)}
          >
            <option value="fade">Fade</option>
            <option value="none">None</option>
          </select>
        </label>

        <span className="slideshow-counter">
          {position + 1} / {imageCount}
        </span>
      </div>

      <div className="slideshow-chrome slideshow-caption">
        <span className="slideshow-filename">{currentImage.filename}</span>
        {dimensions && (
          <span className="slideshow-dimensions">
            {dimensions.w} &times; {dimensions.h}
          </span>
        )}
      </div>
    </div>
  );
}
