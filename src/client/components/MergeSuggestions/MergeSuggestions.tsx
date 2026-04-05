import { useVirtualizer } from "@tanstack/react-virtual";
import React, { useEffect, useLayoutEffect, useRef } from "react";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useMergeSuggestionsStore } from "../../stores/mergeSuggestionsStore.ts";
import { useUIStore } from "../../stores/uiStore.ts";
import type { MergeSuggestionRow as MergeSuggestionRowType } from "../../types.ts";
import { MergeSuggestionRow } from "./MergeSuggestionRow.tsx";
import { MergeSuggestionsToolbar } from "./MergeSuggestionsToolbar.tsx";

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

  // Fetch groups on mount
  useEffect(() => {
    fetchGroups();
  }, []);

  // Update header subtitle
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

  // Re-measure row heights when suggestions or collapse state changes.
  // measure() clears the virtualizer's size cache, but existing visible DOM elements
  // won't re-fire their ref callbacks (React only fires refs on mount, not update),
  // so we must manually re-measure all visible elements from the DOM.
  // biome-ignore lint/correctness/useExhaustiveDependencies: rows and collapsedRows are intentional triggers
  useLayoutEffect(() => {
    virtualizer.measure();
    const container = scrollContainerRef.current;
    if (container) {
      for (const el of container.querySelectorAll<HTMLElement>("[data-index]")) {
        virtualizer.measureElement(el);
      }
    }
  }, [rows, collapsedRows]);

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
                />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
