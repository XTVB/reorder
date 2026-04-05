import { useVirtualizer } from "@tanstack/react-virtual";
import { useCallback, useEffect, useRef, useState } from "react";
import { useRemeasureVirtualRows } from "../../hooks/useRemeasureVirtualRows.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useMergeSuggestionsStore } from "../../stores/mergeSuggestionsStore.ts";
import { useUIStore } from "../../stores/uiStore.ts";
import type { MergeSuggestionRow as MergeSuggestionRowType } from "../../types.ts";
import { Lightbox } from "../Lightbox.tsx";
import { MergePopover } from "./MergePopover.tsx";
import type { OpenCardHandler } from "./MergeSuggestionCard.tsx";
import { MergeSuggestionRow } from "./MergeSuggestionRow.tsx";
import { MergeSuggestionsToolbar } from "./MergeSuggestionsToolbar.tsx";

interface ExpandedCard {
  refGroupId: string;
  /** null means the ref card itself is expanded */
  candidateId: string | null;
  anchorRect: DOMRect;
  displayName: string;
  images: string[];
}

interface LightboxState {
  images: string[];
  index: number;
}

const EMPTY_ROWS: MergeSuggestionRowType[] = [];

export function MergeSuggestions() {
  const suggestions = useMergeSuggestionsStore((s) => s.suggestions);
  const loading = useMergeSuggestionsStore((s) => s.loading);
  const error = useMergeSuggestionsStore((s) => s.error);
  const computeTimeMs = useMergeSuggestionsStore((s) => s.computeTimeMs);
  const threshold = useMergeSuggestionsStore((s) => s.threshold);
  const collapsedRows = useMergeSuggestionsStore((s) => s.collapsedRows);
  const pendingMerges = useMergeSuggestionsStore((s) => s.pendingMerges);
  const undoStack = useMergeSuggestionsStore((s) => s.undoStack);

  const setThreshold = useMergeSuggestionsStore((s) => s.setThreshold);
  const fetchSuggestions = useMergeSuggestionsStore((s) => s.fetchSuggestions);
  const clearPendingMerges = useMergeSuggestionsStore((s) => s.clearPendingMerges);
  const applyMerges = useMergeSuggestionsStore((s) => s.applyMerges);
  const undo = useMergeSuggestionsStore((s) => s.undo);
  const pendingMergeCount = useMergeSuggestionsStore((s) => s.pendingMergeCount);
  const collapseAllRows = useMergeSuggestionsStore((s) => s.collapseAllRows);
  const expandAllRows = useMergeSuggestionsStore((s) => s.expandAllRows);

  const fetchGroups = useGroupStore((s) => s.fetchGroups);
  const setHeaderSubtitle = useUIStore((s) => s.setHeaderSubtitle);

  const [expandedCard, setExpandedCard] = useState<ExpandedCard | null>(null);
  const [lightbox, setLightbox] = useState<LightboxState | null>(null);

  const handleOpenCard = useCallback<OpenCardHandler>(
    ({ anchorEl, displayName, images, refGroupId, candidateId }) => {
      setExpandedCard((prev) => {
        if (prev && prev.refGroupId === refGroupId && prev.candidateId === candidateId) {
          return null;
        }
        return {
          refGroupId,
          candidateId,
          anchorRect: anchorEl.getBoundingClientRect(),
          displayName,
          images,
        };
      });
    },
    [],
  );

  const handleClosePopover = useCallback(() => setExpandedCard(null), []);
  const handleCloseLightbox = useCallback(() => setLightbox(null), []);
  const handleOpenLightbox = useCallback((images: string[], index: number) => {
    setLightbox({ images, index });
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: mount-only — fetchGroups is a stable Zustand action
  useEffect(() => {
    fetchGroups();
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: setHeaderSubtitle is a stable Zustand action
  useEffect(() => {
    if (suggestions) {
      setHeaderSubtitle(`${suggestions.length} groups with merge candidates`);
    } else {
      setHeaderSubtitle("");
    }
    return () => setHeaderSubtitle("");
  }, [suggestions]);

  // Stable reference for empty state
  const rows = suggestions ?? EMPTY_ROWS;

  // Virtualization
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollContainerRef.current,
    getItemKey: (index) => rows[index]?.refGroupId ?? index,
    estimateSize: (index) => {
      const row = rows[index];
      if (!row || collapsedRows.has(row.refGroupId)) return 48;
      return 220;
    },
    measureElement: (el) => el.getBoundingClientRect().height,
    overscan: 3,
  });

  useRemeasureVirtualRows(virtualizer, scrollContainerRef, [rows, collapsedRows]);

  const count = pendingMergeCount();

  return (
    <div className="merge-suggestions-page">
      <MergeSuggestionsToolbar
        threshold={threshold}
        loading={loading}
        computeTimeMs={computeTimeMs}
        suggestionCount={rows.length}
        pendingCount={count}
        canUndo={undoStack.length > 0}
        onThresholdChange={setThreshold}
        onCompute={fetchSuggestions}
        onApply={applyMerges}
        onUndo={undo}
        onClear={clearPendingMerges}
        onExpandAll={expandAllRows}
        onCollapseAll={collapseAllRows}
      />

      {error && <div className="merge-error">{error}</div>}

      {!suggestions && !loading && !error && (
        <div className="merge-empty">
          Click "Compute" to find groups that may be similar enough to merge.
        </div>
      )}

      <div className="merge-scroll-container" ref={scrollContainerRef}>
        <div
          style={{
            height: `${virtualizer.getTotalSize()}px`,
            width: "100%",
            position: "relative",
          }}
        >
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const row = rows[virtualRow.index]!;
            const pending = pendingMerges.get(row.refGroupId) ?? new Set();
            const rowHasExpansion = expandedCard?.refGroupId === row.refGroupId;
            return (
              <div
                key={virtualRow.key}
                data-index={virtualRow.index}
                ref={virtualizer.measureElement}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              >
                <MergeSuggestionRow
                  refGroupId={row.refGroupId}
                  refGroupName={row.refGroupName}
                  refGroupImages={row.refGroupImages}
                  similar={row.similar}
                  collapsed={collapsedRows.has(row.refGroupId)}
                  pendingCandidates={pending}
                  refCardExpanded={rowHasExpansion && expandedCard!.candidateId === null}
                  expandedCandidateId={rowHasExpansion ? expandedCard!.candidateId : null}
                  onOpenCard={handleOpenCard}
                />
              </div>
            );
          })}
        </div>
      </div>

      {expandedCard && (
        <MergePopover
          anchorRect={expandedCard.anchorRect}
          displayName={expandedCard.displayName}
          images={expandedCard.images}
          onOpenLightbox={handleOpenLightbox}
          onClose={handleClosePopover}
        />
      )}

      {lightbox && (
        <Lightbox
          images={lightbox.images.map((f) => ({ filename: f }))}
          initialIndex={lightbox.index}
          onClose={handleCloseLightbox}
        />
      )}
    </div>
  );
}
