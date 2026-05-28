import { useVirtualizer } from "@tanstack/react-virtual";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRemeasureVirtualRows } from "../../hooks/useRemeasureVirtualRows.ts";
import { useConstraintsStore } from "../../stores/constraintsStore.ts";
import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useSessionStore } from "../../stores/core/sessionStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useMergeSuggestionsStore } from "../../stores/mergeSuggestionsStore.ts";
import type { MergeSuggestionRow as MergeSuggestionRowType } from "../../types.ts";
import { SearchOverlay, useSearchOverlayState } from "../shared/SearchBar.tsx";
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

const EMPTY_ROWS: MergeSuggestionRowType[] = [];

export function MergeSuggestions() {
  const suggestions = useMergeSuggestionsStore((s) => s.suggestions);
  const loading = useMergeSuggestionsStore((s) => s.loading);
  const error = useMergeSuggestionsStore((s) => s.error);
  const computeTimeMs = useMergeSuggestionsStore((s) => s.computeTimeMs);
  const progress = useMergeSuggestionsStore((s) => s.progress);
  const threshold = useMergeSuggestionsStore((s) => s.threshold);
  const fullResolution = useMergeSuggestionsStore((s) => s.fullResolution);
  const maxCombinedSize = useMergeSuggestionsStore((s) => s.maxCombinedSize);
  const sortMode = useMergeSuggestionsStore((s) => s.sortMode);
  const collapsedRows = useMergeSuggestionsStore((s) => s.collapsedRows);
  const pendingMerges = useSelectionStore((s) => s.rowSelections["merge-suggestions"]);
  const undoStack = useMergeSuggestionsStore((s) => s.undoStack);
  const rejectedMerges = useConstraintsStore((s) => s.rejectedMerges);
  const addRejectedMerges = useConstraintsStore((s) => s.addRejectedMerges);

  const setThreshold = useMergeSuggestionsStore((s) => s.setThreshold);
  const setFullResolution = useMergeSuggestionsStore((s) => s.setFullResolution);
  const setMaxCombinedSize = useMergeSuggestionsStore((s) => s.setMaxCombinedSize);
  const setSortMode = useMergeSuggestionsStore((s) => s.setSortMode);
  const fetchSuggestions = useMergeSuggestionsStore((s) => s.fetchSuggestions);
  const clearPendingMerges = useMergeSuggestionsStore((s) => s.clearPendingMerges);
  const applyMerges = useMergeSuggestionsStore((s) => s.applyMerges);
  const undo = useMergeSuggestionsStore((s) => s.undo);
  const pendingMergeCount = useMergeSuggestionsStore((s) => s.pendingMergeCount);
  const collapseAllRows = useMergeSuggestionsStore((s) => s.collapseAllRows);
  const expandAllRows = useMergeSuggestionsStore((s) => s.expandAllRows);

  const fetchGroups = useGroupStore((s) => s.fetchGroups);
  // Subscribe to a string key of the group id ordering so renames / image
  // mutations don't re-render this component. Only add / remove / reorder
  // changes the key.
  const groupOrderKey = useGroupStore((s) =>
    sortMode === "groupOrder" ? s.groups.map((g) => g.id).join("|") : "",
  );
  const setHeaderSubtitle = useSessionStore((s) => s.setHeaderSubtitle);

  const [expandedCard, setExpandedCard] = useState<ExpandedCard | null>(null);
  const search = useSearchOverlayState();
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);

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

  // No recompute — the client-side filter below hides rejected pairs until the next Compute.
  const handleRejectSelected = useCallback(() => {
    const sel = useSelectionStore.getState().rowSelections["merge-suggestions"];
    const pairs: { groupA: string; groupB: string }[] = [];
    for (const [refId, candidateIds] of sel) {
      for (const candId of candidateIds) {
        pairs.push({ groupA: refId, groupB: candId });
      }
    }
    if (pairs.length === 0) return;
    void addRejectedMerges(pairs);
    useSelectionStore.getState().clearRowContext("merge-suggestions");
  }, [addRejectedMerges]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: mount-only — fetchGroups is a stable Zustand action
  useEffect(() => {
    fetchGroups();
  }, []);

  // Filter client-side so a freshly-rejected pair disappears without recomputing.
  const filteredSuggestions = useMemo(() => {
    if (!suggestions || rejectedMerges.size === 0) return suggestions;
    const isRejected = useConstraintsStore.getState().isMergeRejected;
    const out: MergeSuggestionRowType[] = [];
    for (const row of suggestions) {
      const kept = row.similar.filter((c) => !isRejected(row.refGroupId, c.groupId));
      if (kept.length === 0) continue;
      out.push(kept.length === row.similar.length ? row : { ...row, similar: kept });
    }
    return out;
  }, [suggestions, rejectedMerges]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: setHeaderSubtitle is a stable Zustand action
  useEffect(() => {
    if (filteredSuggestions) {
      setHeaderSubtitle(`${filteredSuggestions.length} groups with merge candidates`);
    } else {
      setHeaderSubtitle("");
    }
    return () => setHeaderSubtitle("");
  }, [filteredSuggestions]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: groupOrderKey already encodes the group ordering we care about
  const sortedSuggestions = useMemo(() => {
    if (!filteredSuggestions || sortMode !== "groupOrder") return filteredSuggestions;
    const groups = useGroupStore.getState().groups;
    const orderById = new Map(groups.map((g, i) => [g.id, i]));
    return [...filteredSuggestions].sort((a, b) => {
      const ai = orderById.get(a.refGroupId) ?? Number.MAX_SAFE_INTEGER;
      const bi = orderById.get(b.refGroupId) ?? Number.MAX_SAFE_INTEGER;
      return ai - bi;
    });
  }, [filteredSuggestions, sortMode, groupOrderKey]);

  // Stable reference for empty state
  const rows = sortedSuggestions ?? EMPTY_ROWS;

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

  const matchRowIndices = useMemo(() => {
    const q = search.query.trim().toLowerCase();
    if (!q) return [] as number[];
    const indices: number[] = [];
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i]!;
      if (
        row.refGroupName.toLowerCase().includes(q) ||
        row.similar.some((c) => c.groupName.toLowerCase().includes(q))
      ) {
        indices.push(i);
      }
    }
    return indices;
  }, [rows, search.query]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: query is the intentional trigger; body only resets the cursor
  useEffect(() => {
    setCurrentMatchIndex(0);
  }, [search.query]);

  const clampedMatchIndex =
    matchRowIndices.length === 0 ? 0 : Math.min(currentMatchIndex, matchRowIndices.length - 1);
  const currentMatchRowIndex = matchRowIndices[clampedMatchIndex];
  const currentMatchRefId =
    currentMatchRowIndex !== undefined ? (rows[currentMatchRowIndex]?.refGroupId ?? null) : null;

  useEffect(() => {
    if (currentMatchRowIndex !== undefined) {
      virtualizer.scrollToIndex(currentMatchRowIndex, { align: "center" });
    }
  }, [currentMatchRowIndex, virtualizer]);

  const goNextMatch = useCallback(() => {
    if (matchRowIndices.length === 0) return;
    setCurrentMatchIndex((i) => (i + 1) % matchRowIndices.length);
  }, [matchRowIndices.length]);

  const goPrevMatch = useCallback(() => {
    if (matchRowIndices.length === 0) return;
    setCurrentMatchIndex((i) => (i - 1 + matchRowIndices.length) % matchRowIndices.length);
  }, [matchRowIndices.length]);

  const count = pendingMergeCount();

  return (
    <div className="merge-suggestions-page">
      <MergeSuggestionsToolbar
        threshold={threshold}
        loading={loading}
        computeTimeMs={computeTimeMs}
        progress={progress}
        suggestionCount={rows.length}
        pendingCount={count}
        canUndo={undoStack.length > 0}
        fullResolution={fullResolution}
        maxCombinedSize={maxCombinedSize}
        sortMode={sortMode}
        onThresholdChange={setThreshold}
        onFullResolutionChange={setFullResolution}
        onMaxCombinedSizeChange={setMaxCombinedSize}
        onSortModeChange={setSortMode}
        onCompute={fetchSuggestions}
        onApply={applyMerges}
        onRejectSelected={handleRejectSelected}
        onUndo={undo}
        onClear={clearPendingMerges}
        onExpandAll={expandAllRows}
        onCollapseAll={collapseAllRows}
      />

      <SearchOverlay
        isOpen={search.isOpen}
        query={search.query}
        setQuery={search.setQuery}
        open={search.open}
        close={search.close}
        matchCount={matchRowIndices.length}
        currentMatchIndex={clampedMatchIndex}
        onNext={goNextMatch}
        onPrev={goPrevMatch}
        placeholder="Search group names..."
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
                  isCurrentSearchMatch={row.refGroupId === currentMatchRefId}
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
          onClose={handleClosePopover}
        />
      )}
    </div>
  );
}
