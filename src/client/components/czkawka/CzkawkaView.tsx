import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { postJson } from "../../api/client.ts";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { useSessionStore } from "../../stores/core/sessionStore.ts";
import { useToastStore } from "../../stores/core/toastStore.ts";
import { computeChain, useCzkawkaRankingStore } from "../../stores/czkawkaRanking.ts";
import { imgPath, useCzkawkaStore } from "../../stores/czkawkaStore.ts";
import type { CzkawkaImage, CzkawkaOperation } from "../../types.ts";
import { cn, fullImageUrl, imageUrl } from "../../utils/helpers.ts";
import { RankingSidebar } from "./RankingSidebar.tsx";

// ── Small helpers ───────────────────────────────────────────────────────

function formatSize(bytes: number): string {
  const units = ["B", "KB", "MB", "GB"];
  let s = bytes;
  for (const u of units) {
    if (s < 1024) return `${s.toFixed(1)} ${u}`;
    s /= 1024;
  }
  return `${s.toFixed(1)} TB`;
}

const dirBase = (dir: string) => dir.slice(dir.lastIndexOf("/") + 1) || dir;

const czkawkaFileUrl = (path: string) => `/api/czkawka/file?path=${encodeURIComponent(path)}`;

interface DiffParts {
  prefix: string;
  diff: string;
  suffix: string;
}

/** Char-level common prefix/suffix split across a set of names, so the part
 * that actually differs (a digit, a "_1" suffix, the directory) highlights. */
function diffParts(names: string[]): DiffParts[] {
  if (names.length < 2) return names.map((n) => ({ prefix: n, diff: "", suffix: "" }));
  const minLen = Math.min(...names.map((n) => n.length));
  let p = 0;
  while (p < minLen && names.every((n) => n[p] === names[0]![p])) p++;
  let sfx = 0;
  while (
    sfx < minLen - p &&
    names.every((n) => n[n.length - 1 - sfx] === names[0]![names[0]!.length - 1 - sfx])
  ) {
    sfx++;
  }
  return names.map((n) => ({
    prefix: n.slice(0, p),
    diff: n.slice(p, n.length - sfx),
    suffix: n.slice(n.length - sfx),
  }));
}

function FilenameDiff({ parts }: { parts: DiffParts }) {
  if (!parts.diff) return <>{parts.prefix + parts.suffix}</>;
  return (
    <>
      {parts.prefix}
      <span className="czkawka-name-diff">{parts.diff}</span>
      {parts.suffix}
    </>
  );
}

// ── Size/resolution diff banner (exactly 2 compared images) ─────────────

function SizeDiffBanner({ left, right }: { left: CzkawkaImage; right: CzkawkaImage }) {
  const sizeDiff = Math.abs(left.size - right.size);
  const sameSize = sizeDiff === 0;
  const sizeArrow = left.size > right.size ? "←" : "→";
  const sizePct = sameSize
    ? ""
    : `${((sizeDiff / Math.min(left.size, right.size)) * 100).toFixed(0)}%`;

  const leftPx = left.width * left.height;
  const rightPx = right.width * right.height;
  const sameRes = left.width === right.width && left.height === right.height;
  const resArrow = leftPx > rightPx ? "←" : "→";
  const resPct = sameRes
    ? ""
    : `${Math.round((Math.max(leftPx, rightPx) / Math.min(leftPx, rightPx) - 1) * 100)}%`;

  return (
    <div className="czkawka-size-banner">
      {sameSize ? (
        <span className="diff-text">Same file size</span>
      ) : (
        <span>
          <span className="diff-arrow">{sizeArrow}</span>{" "}
          <span className="diff-bigger">{formatSize(sizeDiff)} larger</span>{" "}
          <span className="diff-text">({sizePct})</span>
        </span>
      )}
      <span className="diff-sep">·</span>
      {sameRes ? (
        <span className="diff-text">Same resolution</span>
      ) : (
        <span>
          <span className="diff-arrow">{resArrow}</span>{" "}
          <span className="diff-bigger">{resPct} more pixels</span>{" "}
          <span className="diff-text">
            {left.width}×{left.height} vs {right.width}×{right.height}
          </span>
        </span>
      )}
    </div>
  );
}

// ── Slider overlay: two images stacked with a draggable divider ─────────

function SliderComparison({
  leftUrl,
  rightUrl,
  leftLabel,
  rightLabel,
}: {
  leftUrl: string;
  rightUrl: string;
  leftLabel: string;
  rightLabel: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(50);
  const dragging = useRef(false);

  const onMove = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setPos(Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100)));
  }, []);

  return (
    <div
      ref={containerRef}
      className="czkawka-slider"
      onPointerDown={(e) => {
        dragging.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
        onMove(e.clientX);
      }}
      onPointerMove={(e) => {
        if (dragging.current) onMove(e.clientX);
      }}
      onPointerUp={() => {
        dragging.current = false;
      }}
      onPointerLeave={() => {
        dragging.current = false;
      }}
    >
      <div className="czkawka-slider-base">
        <img src={rightUrl} alt={rightLabel} draggable={false} />
      </div>
      <div className="czkawka-slider-overlay" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img src={leftUrl} alt={leftLabel} draggable={false} />
      </div>
      <div className="czkawka-slider-handle" style={{ left: `${pos}%` }}>
        <div className="czkawka-slider-grip" />
      </div>
      <span className="czkawka-slider-label czkawka-slider-label-left">{leftLabel}</span>
      <span className="czkawka-slider-label czkawka-slider-label-right">{rightLabel}</span>
    </div>
  );
}

// ── Thumbnail strip (pick the compare subset, exclude from Y/W) ─────────

interface DisplayImage {
  img: CzkawkaImage;
  path: string;
  displayName: string;
  thumbUrl: string;
  fullUrl: string;
  isRef: boolean;
}

function ThumbnailBar({
  items,
  compared,
  excluded,
  winnerPath,
  onToggleCompare,
  onToggleExclude,
}: {
  items: DisplayImage[];
  compared: string[];
  excluded: Set<string>;
  winnerPath: string | null;
  onToggleCompare: (path: string) => void;
  onToggleExclude: (path: string) => void;
}) {
  return (
    <div className="czkawka-thumb-bar">
      {items.map(({ img, path, displayName, thumbUrl, isRef }) => {
        const pos = compared.indexOf(path);
        const isExcluded = excluded.has(path);
        const posLabel =
          compared.length === 2 && pos !== -1 ? (pos === 0 ? "Left" : "Right") : null;
        return (
          <button
            key={path}
            type="button"
            className={cn(
              "czkawka-thumb",
              pos !== -1 && "is-compared",
              isExcluded && "is-excluded",
              winnerPath === path && "is-winner",
            )}
            title="Click to compare · Right-click to exclude from Y/W"
            onClick={(e) => {
              if (e.altKey) onToggleExclude(path);
              else onToggleCompare(path);
            }}
            onContextMenu={(e) => {
              e.preventDefault();
              onToggleExclude(path);
            }}
          >
            <span className="czkawka-thumb-imgwrap">
              <img src={thumbUrl} alt={displayName} loading="lazy" />
              {winnerPath === path && (
                <span className="czkawka-thumb-star" title="Computed keeper (Y)">
                  ★
                </span>
              )}
            </span>
            <span className="czkawka-thumb-info">
              <span className="czkawka-thumb-toprow">
                {isRef && <span className="czkawka-ref-badge">REF</span>}
                {posLabel && <span className="czkawka-thumb-pos">{posLabel}</span>}
                {isExcluded && <span className="czkawka-thumb-skip">skip Y/W</span>}
              </span>
              <span className="czkawka-thumb-name" title={path}>
                {displayName}
              </span>
              <span className="czkawka-thumb-meta">
                {formatSize(img.size)} · {img.width}×{img.height}
                {img.difference > 0 && ` · Δ${img.difference}`}
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

// ── Step-through view ───────────────────────────────────────────────────

function StepThroughView({ onRankingToggle }: { onRankingToggle: () => void }) {
  const groups = useCzkawkaStore((s) => s.groups);
  const currentIndex = useCzkawkaStore((s) => s.currentIndex);
  const loading = useCzkawkaStore((s) => s.loading);
  const progress = useCzkawkaStore((s) => s.progress);
  const error = useCzkawkaStore((s) => s.error);
  const computeTimeMs = useCzkawkaStore((s) => s.computeTimeMs);
  const runStats = useCzkawkaStore((s) => s.runStats);
  const sliderMode = useCzkawkaStore((s) => s.sliderMode);
  const trashedCount = useCzkawkaStore((s) => s.trashedCount);
  const historyLen = useCzkawkaStore((s) => s.history.length);
  const undoDepth = useCzkawkaStore((s) => s.undoDepth);
  const excluded = useCzkawkaStore((s) => s.excludedPaths);
  const targetDir = useCzkawkaStore((s) => s.targetDir);
  const dirs = useCzkawkaStore((s) => s.dirs);

  const runComparison = useCzkawkaStore((s) => s.runComparison);
  const applyOperations = useCzkawkaStore((s) => s.applyOperations);
  const advance = useCzkawkaStore((s) => s.advance);
  const goBack = useCzkawkaStore((s) => s.goBack);
  const goToGroup = useCzkawkaStore((s) => s.goToGroup);
  const undo = useCzkawkaStore((s) => s.undo);
  const setSliderMode = useCzkawkaStore((s) => s.setSliderMode);
  const toggleExclude = useCzkawkaStore((s) => s.toggleExclude);
  const clearExclusions = useCzkawkaStore((s) => s.clearExclusions);

  const contentConditions = useCzkawkaRankingStore((s) => s.contentConditions);
  const targetConditions = useCzkawkaRankingStore((s) => s.targetConditions);

  const openLightbox = useLightboxStore((s) => s.openLightbox);
  const updateLightboxFilenames = useLightboxStore((s) => s.updateFilenames);
  const setHeaderSubtitle = useSessionStore((s) => s.setHeaderSubtitle);
  const showToast = useToastStore((s) => s.showToast);

  const group = groups[currentIndex];
  const groupKey = group?.map(imgPath).join(" ") ?? "";
  // Prefix names with their folder when comparing multiple dirs, or when a
  // recursive scan means same-named files can live in different sub-folders.
  const multiDir = dirs.length > 1 || dirs.some((d) => d.recursive);
  const refDir = dirs.find((d) => d.reference)?.path ?? null;

  // Total matched files — the number that tracks the similarity threshold
  // intuitively (group count alone is misleading: a looser threshold can
  // merge groups, so fewer groups can mean more matched files).
  const totalFiles = useMemo(() => groups.reduce((n, g) => n + g.length, 0), [groups]);

  // Per-image display data (name, urls, ref flag) for the current group.
  const items: DisplayImage[] = useMemo(() => {
    if (!group) return [];
    return group.map((img) => {
      const path = imgPath(img);
      const local = img.dir === targetDir;
      return {
        img,
        path,
        displayName: multiDir ? `${dirBase(img.dir)}/${img.filename}` : img.filename,
        thumbUrl: local ? imageUrl(img.filename) : czkawkaFileUrl(path),
        fullUrl: local ? fullImageUrl(img.filename) : czkawkaFileUrl(path),
        isRef: refDir !== null && img.dir === refDir,
      };
    });
  }, [group, targetDir, multiDir, refDir]);

  // Compare subset — which images render side by side. Defaults to the whole
  // group, capped at 3 when the group is mostly landscape (wide panels).
  const [compared, setCompared] = useState<string[]>([]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: reset only when the group's identity changes
  useEffect(() => {
    const s = useCzkawkaStore.getState();
    const g = s.groups[s.currentIndex];
    if (!g) {
      setCompared([]);
      return;
    }
    const landscape = g.filter((i) => i.width > i.height).length > g.length / 2;
    const subset = landscape && g.length > 3 ? g.slice(0, 3) : g;
    setCompared(subset.map(imgPath));
    clearExclusions();
  }, [groupKey, clearExclusions]);

  const toggleCompare = useCallback(
    (path: string) => {
      if (!group) return;
      setCompared((prev) => {
        if (prev.includes(path)) {
          return prev.length > 1 ? prev.filter((p) => p !== path) : prev;
        }
        const next = [...prev, path];
        const order = new Map(group.map((img, i) => [imgPath(img), i]));
        next.sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0));
        return next;
      });
    },
    [group],
  );

  const comparedItems = useMemo(
    () => items.filter((it) => compared.includes(it.path)),
    [items, compared],
  );

  // Ranking results react to condition edits, exclusions, and group changes.
  const considered = useMemo(
    () => (group ? group.filter((i) => !excluded.has(imgPath(i))) : []),
    [group, excluded],
  );
  const contentResult = useMemo(
    () => computeChain(contentConditions, considered),
    [contentConditions, considered],
  );
  const targetResult = useMemo(
    () => computeChain(targetConditions, considered),
    [targetConditions, considered],
  );

  const computedWinner =
    contentResult?.winners.length === 1 ? considered[contentResult.winners[0]!]! : null;
  const winnerPath = computedWinner ? imgPath(computedWinner) : null;
  const computedTied = contentResult
    ? contentResult.winners.length > 1 && contentResult.winners.length < contentResult.total
    : false;
  const computedLabel = computedWinner
    ? `Keep ${computedWinner.filename}`
    : computedTied
      ? `Trash ${contentResult!.total - contentResult!.winners.length} losers, ${contentResult!.winners.length} tied`
      : "All tied";
  const computedDisabled = !contentResult || contentResult.winners.length === contentResult.total;

  const copyTarget =
    targetResult?.winners.length === 1 ? considered[targetResult.winners[0]!]! : null;
  const copyReplaceReady = computedWinner !== null && copyTarget !== null;
  const copyReplaceLabel = copyReplaceReady
    ? winnerPath === imgPath(copyTarget!)
      ? `Keep ${copyTarget!.filename} in place`
      : `${computedWinner!.filename} → ${copyTarget!.filename}`
    : "Tied — resolve manually";

  // ── Action handlers ──────────────────────────────────────────────────

  const trashPaths = useCallback(
    (paths: string[], toastMsg: string, keep?: string) => {
      if (paths.length === 0) return;
      void applyOperations([{ type: "trash", paths, ...(keep ? { keep } : {}) }], toastMsg);
    },
    [applyOperations],
  );

  const trashComparedAt = useCallback(
    (pos: number) => {
      const it = comparedItems[pos];
      if (it) trashPaths([it.path], `Trashed ${it.displayName}`);
    },
    [comparedItems, trashPaths],
  );

  const keepFirstCompared = useCallback(() => {
    const keep = comparedItems[0];
    if (!group || !keep) return;
    const toDelete = group.map(imgPath).filter((p) => p !== keep.path);
    trashPaths(toDelete, `Kept ${keep.displayName}, trashed ${toDelete.length}`, keep.path);
  }, [group, comparedItems, trashPaths]);

  const trashAll = useCallback(() => {
    if (!group) return;
    trashPaths(group.map(imgPath), `Trashed all ${group.length} files`);
  }, [group, trashPaths]);

  const keepComputed = useCallback(() => {
    if (!group || !contentResult || computedDisabled) return;
    if (computedWinner && winnerPath) {
      const toDelete = group.map(imgPath).filter((p) => p !== winnerPath && !excluded.has(p));
      trashPaths(
        toDelete,
        `Kept ${computedWinner.filename}, trashed ${toDelete.length}`,
        winnerPath,
      );
    } else {
      // Tie: trash the losers, stay on the group to choose between winners.
      const winnerSet = new Set(contentResult.winners);
      const losers = considered.filter((_, idx) => !winnerSet.has(idx)).map(imgPath);
      trashPaths(losers, `Trashed ${losers.length} losers — choose between the tied`);
    }
  }, [
    group,
    contentResult,
    computedDisabled,
    computedWinner,
    winnerPath,
    considered,
    excluded,
    trashPaths,
  ]);

  const copyReplace = useCallback(() => {
    if (!group || !computedWinner || !copyTarget || !winnerPath) return;
    const source = winnerPath;
    const target = imgPath(copyTarget);
    const others = group
      .map(imgPath)
      .filter((p) => p !== source && p !== target && !excluded.has(p));
    void applyOperations(
      [{ type: "copy_replace", source, target, others }],
      source === target
        ? `Kept ${copyTarget.filename} in place, trashed ${others.length}`
        : `Copied ${computedWinner.filename} → ${copyTarget.filename}`,
    );
  }, [group, computedWinner, copyTarget, winnerPath, excluded, applyOperations]);

  // Auto-resolve consecutive groups with a clear winner; stop at the first
  // tie so it lands there for manual review. One batched request → one undo.
  const applyAllComputed = useCallback(() => {
    const s = useCzkawkaStore.getState();
    const conditions = useCzkawkaRankingStore.getState().contentConditions;
    const ops: CzkawkaOperation[] = [];
    for (let gi = s.currentIndex; gi < s.groups.length; gi++) {
      const g = gi === s.currentIndex ? considered : s.groups[gi]!;
      const res = computeChain(conditions, g);
      if (!res || res.winners.length !== 1) break;
      const keep = imgPath(g[res.winners[0]!]!);
      const toDelete = g.map(imgPath).filter((p) => p !== keep);
      if (toDelete.length === 0) break;
      ops.push({ type: "trash", paths: toDelete, keep });
    }
    if (ops.length === 0) {
      showToast("No clear winner here — resolve manually", "warning");
      return;
    }
    void applyOperations(ops, `Auto-resolved ${ops.length} group(s)`);
  }, [considered, applyOperations, showToast]);

  const applyAllCopyReplace = useCallback(() => {
    const s = useCzkawkaStore.getState();
    const ranking = useCzkawkaRankingStore.getState();
    const ops: CzkawkaOperation[] = [];
    for (let gi = s.currentIndex; gi < s.groups.length; gi++) {
      const g = gi === s.currentIndex ? considered : s.groups[gi]!;
      const content = computeChain(ranking.contentConditions, g);
      const target = computeChain(ranking.targetConditions, g);
      if (!content || content.winners.length !== 1) break;
      if (!target || target.winners.length !== 1) break;
      const source = imgPath(g[content.winners[0]!]!);
      const tgt = imgPath(g[target.winners[0]!]!);
      const others = g.map(imgPath).filter((p) => p !== source && p !== tgt);
      ops.push({ type: "copy_replace", source, target: tgt, others });
    }
    if (ops.length === 0) {
      showToast("No clear source+target here — resolve manually", "warning");
      return;
    }
    void applyOperations(ops, `Auto copy-replaced ${ops.length} group(s)`);
  }, [considered, applyOperations, showToast]);

  const revealInFinder = useCallback(
    (path: string) => {
      postJson("/api/czkawka/reveal", { path }).catch(() =>
        showToast("Could not reveal in Finder", "error"),
      );
    },
    [showToast],
  );

  // The lightbox navigates only the images under comparison, minus any
  // excluded from consideration — arrow keys never land on something the
  // user has dismissed from the compare strip or the Y/W ranking. A directly
  // clicked image is always included even when excluded.
  const lightboxItems = useMemo(
    () => comparedItems.filter((it) => !excluded.has(it.path)),
    [comparedItems, excluded],
  );

  const pinnedPathRef = useRef<string | null>(null);
  const openLightboxAt = useCallback(
    (path: string) => {
      const pinned = !lightboxItems.some((it) => it.path === path);
      pinnedPathRef.current = pinned ? path : null;
      const candidates = pinned
        ? comparedItems.filter((it) => !excluded.has(it.path) || it.path === path)
        : lightboxItems;
      const idx = candidates.findIndex((it) => it.path === path);
      if (idx === -1) return;
      openLightbox(
        candidates.map((it) => it.displayName),
        idx,
        { trashMark: false, urls: candidates.map((it) => it.fullUrl) },
      );
    },
    [lightboxItems, comparedItems, excluded, openLightbox],
  );

  // ── Effects: header subtitle, lightbox sync, preloading ─────────────

  useEffect(() => {
    if (groups.length > 0 && currentIndex < groups.length) {
      setHeaderSubtitle(`Group ${currentIndex + 1} of ${groups.length}`);
    } else {
      setHeaderSubtitle("");
    }
    return () => setHeaderSubtitle("");
  }, [groups.length, currentIndex, setHeaderSubtitle]);

  // Keep an open lightbox in sync when actions mutate the current group or
  // the compare/exclusion sets change — updateFilenames preserves zoom/pan
  // (no remount) and closes on empty. A directly-clicked excluded image
  // (pinned at open) stays in the set rather than vanishing mid-view.
  useEffect(() => {
    if (!useLightboxStore.getState().open) return;
    const pinned = pinnedPathRef.current;
    const items = pinned
      ? comparedItems.filter((it) => !excluded.has(it.path) || it.path === pinned)
      : lightboxItems;
    updateLightboxFilenames(
      items.map((it) => it.displayName),
      items.map((it) => it.fullUrl),
    );
  }, [lightboxItems, comparedItems, excluded, updateLightboxFilenames]);

  // Preload the next group's full-size images so stepping feels instant.
  useEffect(() => {
    const next = groups[currentIndex + 1];
    if (!next) return;
    for (const img of next) {
      const local = img.dir === targetDir;
      new Image().src = local ? fullImageUrl(img.filename) : czkawkaFileUrl(imgPath(img));
    }
  }, [groups, currentIndex, targetDir]);

  // ── Keyboard (registered once; handler reads current closures via ref) ─

  const keyHandlerRef = useRef<(e: KeyboardEvent) => void>(() => {});
  keyHandlerRef.current = (e: KeyboardEvent) => {
    if (
      e.target instanceof HTMLInputElement ||
      e.target instanceof HTMLTextAreaElement ||
      e.target instanceof HTMLSelectElement
    ) {
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    // The lightbox owns arrows (image nav, zoom preserved), 0 (reset view),
    // +/- (zoom) and Escape. Everything else falls through so actions work
    // while zoomed in — the group mutates underneath and the lightbox syncs.
    const lightboxOpen = useLightboxStore.getState().open;
    if (
      lightboxOpen &&
      ["ArrowLeft", "ArrowRight", "0", "+", "-", "=", "Escape", "d", "D"].includes(e.key)
    ) {
      return;
    }

    // No current group (done screen, or everything resolved): only
    // navigation/undo apply.
    if (!group) {
      if (e.key === "b" || e.key === "B") goBack();
      else if (e.key === "u" || e.key === "U") void undo();
      return;
    }

    switch (e.key) {
      case "ArrowLeft":
        goToGroup(currentIndex - 1);
        break;
      case "ArrowRight":
        goToGroup(currentIndex + 1);
        break;
      case "k":
      case "K":
        advance();
        break;
      case "s":
      case "S":
        advance();
        break;
      case "b":
      case "B":
        goBack();
        break;
      case "u":
      case "U":
        void undo();
        break;
      case "v":
      case "V":
        setSliderMode(!sliderMode);
        break;
      case "h":
      case "H":
        keepFirstCompared();
        break;
      case "d":
      case "D":
        trashAll();
        break;
      case "y":
      case "Y":
        keepComputed();
        break;
      case "w":
      case "W":
        if (copyReplaceReady) copyReplace();
        break;
      case "r":
      case "R":
        onRankingToggle();
        break;
      default: {
        const num = Number(e.key);
        if (num >= 1 && num <= 9) trashComparedAt(num - 1);
        break;
      }
    }
  };

  useEffect(() => {
    const fn = (e: KeyboardEvent) => keyHandlerRef.current(e);
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, []);

  // ── Screens ──────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="czkawka-view">
        <div className="czkawka-center">
          <div className="czkawka-spinner" />
          <div className="czkawka-center-title">Finding duplicates</div>
          <div className="czkawka-center-desc">{progress ?? "Working..."}</div>
        </div>
      </div>
    );
  }

  if (error && groups.length === 0) {
    return (
      <div className="czkawka-view">
        <div className="czkawka-center">
          <div className="czkawka-center-title">Comparison failed</div>
          <div className="czkawka-center-desc czkawka-error-text">{error}</div>
          <button className="btn btn-primary" onClick={runComparison}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Before the first run this is the start screen (nothing runs until the
  // button is clicked); after a run that found nothing (and where nothing was
  // resolved this session) it reads as "no duplicates". An emptied-out group
  // list after actions is the done screen below.
  if (groups.length === 0 && historyLen === 0 && trashedCount === 0 && undoDepth === 0) {
    return (
      <div className="czkawka-view">
        <div className="czkawka-center">
          <div className="czkawka-center-title">
            {runStats ? "No similar images found" : "Find duplicate images"}
          </div>
          <div className="czkawka-center-desc">
            {runStats
              ? `Compared ${runStats.cached + runStats.computed} images — all distinct at this threshold.`
              : "Run a comparison to hash the configured folders and find duplicate and near-duplicate images."}
          </div>
          <button className="btn btn-primary" onClick={runComparison}>
            Run Comparison
          </button>
        </div>
      </div>
    );
  }

  if (!group) {
    return (
      <div className="czkawka-view">
        <div className="czkawka-center">
          <div className="czkawka-done-icon">✓</div>
          <div className="czkawka-center-title">All groups reviewed</div>
          <div className="czkawka-center-desc">
            {trashedCount > 0
              ? `${trashedCount} file(s) moved to Trash this session.`
              : "No files were deleted this session."}
          </div>
          <div className="czkawka-center-actions">
            <button className="btn" onClick={goBack}>
              Back <kbd>B</kbd>
            </button>
            <button
              className="btn"
              onClick={() => void undo()}
              disabled={historyLen === 0 && undoDepth === 0}
            >
              Undo <kbd>U</kbd>
            </button>
            <button className="btn btn-primary" onClick={runComparison}>
              Re-run Comparison
            </button>
          </div>
        </div>
      </div>
    );
  }

  const nameParts = diffParts(comparedItems.map((it) => it.displayName));
  const showSlider = sliderMode && comparedItems.length >= 2;

  return (
    <div className="czkawka-view">
      {/* Progress bar */}
      <div className="czkawka-progress">
        <div
          className="czkawka-progress-fill"
          style={{ width: `${((currentIndex + 1) / groups.length) * 100}%` }}
        />
        <span className="czkawka-progress-text">
          {currentIndex + 1} / {groups.length}
          <span className="czkawka-progress-stats">
            {" "}
            · {totalFiles} file{totalFiles === 1 ? "" : "s"} in {groups.length} group
            {groups.length === 1 ? "" : "s"}
          </span>
          {refDir !== null && (
            <span className="czkawka-progress-stats"> · ref: {dirBase(refDir)}</span>
          )}
          {computeTimeMs != null && (
            <span className="czkawka-progress-stats">
              {" "}
              ·{" "}
              {computeTimeMs < 1000
                ? `${computeTimeMs}ms`
                : `${(computeTimeMs / 1000).toFixed(1)}s`}
              {runStats && ` · ${runStats.cached} cached, ${runStats.computed} hashed`}
            </span>
          )}
        </span>
      </div>

      {/* Thumbnail strip */}
      <ThumbnailBar
        items={items}
        compared={compared}
        excluded={excluded}
        winnerPath={winnerPath}
        onToggleCompare={toggleCompare}
        onToggleExclude={toggleExclude}
      />

      {/* Size diff banner (pair comparisons only) */}
      {comparedItems.length === 2 && (
        <SizeDiffBanner left={comparedItems[0]!.img} right={comparedItems[1]!.img} />
      )}

      {/* Comparison area */}
      <div className="czkawka-compare-area">
        {showSlider ? (
          <SliderComparison
            leftUrl={comparedItems[0]!.fullUrl}
            rightUrl={comparedItems[1]!.fullUrl}
            leftLabel={comparedItems[0]!.displayName}
            rightLabel={comparedItems[1]!.displayName}
          />
        ) : (
          <div
            className="czkawka-panels"
            style={{ gridTemplateColumns: `repeat(${comparedItems.length}, 1fr)` }}
          >
            {comparedItems.map((it, pos) => (
              <div
                key={it.path}
                className={cn("czkawka-panel", winnerPath === it.path && "is-winner")}
              >
                <div className="czkawka-panel-imgwrap">
                  <img
                    src={it.fullUrl}
                    alt={it.displayName}
                    onClick={() => openLightboxAt(it.path)}
                  />
                </div>
                <div className="czkawka-panel-info">
                  {it.isRef && <span className="czkawka-ref-badge">REF</span>}
                  <button
                    type="button"
                    className="czkawka-panel-name"
                    onClick={() => revealInFinder(it.path)}
                    title={`${it.path} — Reveal in Finder`}
                  >
                    <FilenameDiff parts={nameParts[pos]!} />
                  </button>
                  <span className="czkawka-panel-meta">
                    {it.img.width}×{it.img.height} · {formatSize(it.img.size)}
                    {it.img.difference > 0 && ` · Δ${it.img.difference}`}
                  </span>
                  <button
                    type="button"
                    className="czkawka-panel-trash"
                    onClick={() => trashPaths([it.path], `Trashed ${it.displayName}`)}
                    title={`Move ${it.displayName} to Trash (${pos + 1})`}
                  >
                    Trash <kbd>{pos + 1}</kbd>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Action bar */}
      <div className="czkawka-actions">
        <div className="czkawka-action-group">
          <button className="btn" onClick={goBack} disabled={currentIndex === 0}>
            Back <kbd>B</kbd>
          </button>
          <button
            className="btn"
            onClick={() => void undo()}
            disabled={historyLen === 0 && undoDepth === 0}
          >
            Undo <kbd>U</kbd>
          </button>
        </div>

        <div className="czkawka-action-group">
          <button
            className="btn btn-primary"
            onClick={keepComputed}
            disabled={computedDisabled}
            title="Keep the ranking winner, trash the rest (configure via R)"
          >
            {computedLabel} <kbd>Y</kbd>
          </button>
          <button
            className="btn"
            onClick={applyAllComputed}
            disabled={computedDisabled}
            title="Keep the ranking winner in every consecutive group with a clear winner; stops at the first tie"
          >
            Auto-keep All
          </button>
        </div>

        <div className="czkawka-action-group">
          <button
            className="btn"
            onClick={copyReplace}
            disabled={!copyReplaceReady}
            title="Copy the best content over the target filename, trash the rest"
          >
            {copyReplaceLabel} <kbd>W</kbd>
          </button>
          <button
            className="btn"
            onClick={applyAllCopyReplace}
            disabled={!copyReplaceReady}
            title="Copy-replace every consecutive group with a clear source and target; stops at the first tie"
          >
            Auto-replace All
          </button>
        </div>

        <div className="czkawka-action-group">
          <button
            className="btn"
            onClick={keepFirstCompared}
            disabled={comparedItems.length === 0}
            title={
              comparedItems[0]
                ? `Keep ${comparedItems[0].displayName}, trash the rest of the group`
                : ""
            }
          >
            Keep left, trash rest <kbd>H</kbd>
          </button>
          <button
            className="btn btn-danger"
            onClick={trashAll}
            title="Trash every file in this group"
          >
            Trash all <kbd>D</kbd>
          </button>
        </div>

        <div className="czkawka-action-group">
          <button className="btn" onClick={advance} title="Keep every file, next group">
            Keep all <kbd>K</kbd>
          </button>
          <button className="btn" onClick={advance} title="Decide later, next group">
            Skip <kbd>S</kbd>
          </button>
          <button
            className="btn"
            onClick={() => setSliderMode(!sliderMode)}
            disabled={comparedItems.length < 2}
            title="Toggle overlay slider comparison"
          >
            {showSlider ? "Side-by-side" : "Slider"} <kbd>V</kbd>
          </button>
          <button className="btn" onClick={onRankingToggle} title="Configure Y/W ranking rules">
            Ranking <kbd>R</kbd>
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Mount wrapper: restore the persisted session (no auto-run — hashing
// only starts from the Run Comparison button) ───────────────────────────

export function CzkawkaView() {
  const [rankingOpen, setRankingOpen] = useState(false);

  useEffect(() => {
    void useCzkawkaStore.getState().loadExisting();
  }, []);

  return (
    <>
      <StepThroughView onRankingToggle={() => setRankingOpen((v) => !v)} />
      <RankingSidebar open={rankingOpen} onClose={() => setRankingOpen(false)} />
    </>
  );
}
