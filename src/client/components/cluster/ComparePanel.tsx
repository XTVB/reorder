import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDismissOnOutside } from "../../hooks/useDismissOnOutside.ts";
import { useSelectionStore } from "../../stores/core/selectionStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import {
  type CompareState,
  findClusterEverywhere,
  useCompareStore,
  useListStore,
} from "../../stores/modes/cluster/index.ts";
import type { ClusterResultData, SplitChildren } from "../../types.ts";
import { cn, imageUrl } from "../../utils/helpers.ts";
import { Modal } from "../shared/Modal.tsx";

const MAX_INITIAL_CANDIDATES = 15;

export function ComparePanel() {
  const compare = useCompareStore((s) => s.compare);
  const clusterData = useListStore((s) => s.clusterData);
  const splitChildren = useListStore((s) => s.splitChildren);
  const checked = useSelectionStore((s) => s.contexts.compare);
  const closeCompare = useCompareStore((s) => s.closeCompare);
  const toggleCompareCandidate = useCompareStore((s) => s.toggleCompareCandidate);
  const setIncludeConfirmedGroups = useCompareStore((s) => s.setCompareIncludeConfirmedGroups);
  const addClusterToCompare = useCompareStore((s) => s.addClusterToCompare);
  const confirmMerge = useCompareStore((s) => s.confirmMerge);

  // Allow Escape to dismiss, Cmd+Enter to confirm.
  useEffect(() => {
    if (!compare) return;
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        if (e.key === "Escape") closeCompare();
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        closeCompare();
      } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        confirmMerge();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [compare, closeCompare, confirmMerge]);

  if (!compare || !clusterData) return null;

  const source = findClusterEverywhere(
    clusterData.clusters,
    splitChildren,
    compare.sourceClusterId,
  );
  if (!source) return null;

  const visibleCandidates = filterCandidates(compare, checked, clusterData.clusters, splitChildren);

  return (
    <div className="compare-panel">
      <div className="compare-header">
        <div className="compare-header-info">
          <button
            className="compare-back-btn"
            onClick={closeCompare}
            title="Cancel and return to cluster list"
            aria-label="Close compare"
          >
            ←
          </button>
          <div>
            <div className="compare-title">
              Merge candidates for{" "}
              <span className="compare-source-name">
                {source.confirmedGroup?.name ?? source.autoName}
              </span>
            </div>
            <div className="compare-subtitle">
              {source.images.length} images · ranked by similarity
            </div>
          </div>
        </div>
        <div className="compare-header-actions">
          <button className="btn btn-primary" onClick={confirmMerge} disabled={checked.size === 0}>
            merge selected ({checked.size})
          </button>
          <button className="btn" onClick={closeCompare}>
            cancel
          </button>
        </div>
      </div>

      <div className="compare-source-pinned">
        <div className="compare-section-label">Source</div>
        <ContactSheet images={source.images} />
      </div>

      <div className="compare-controls">
        <label className="compare-toggle">
          <input
            type="checkbox"
            checked={compare.includeConfirmedGroups}
            onChange={(e) => setIncludeConfirmedGroups(e.target.checked)}
          />
          <span>Include confirmed groups</span>
        </label>
        <CompareSearch
          existingIds={new Set(compare.candidateOrder)}
          sourceId={compare.sourceClusterId}
          onPick={addClusterToCompare}
        />
      </div>

      <div className="compare-candidates">
        {compare.loading ? (
          <div className="compare-empty">Computing similarity…</div>
        ) : visibleCandidates.length === 0 ? (
          <div className="compare-empty">
            No similar clusters found.
            {!compare.includeConfirmedGroups ? " Try enabling Include confirmed groups." : ""}
          </div>
        ) : (
          <CandidateList
            visibleCandidates={visibleCandidates}
            checked={checked}
            distances={compare.candidateDistance}
            onToggle={toggleCompareCandidate}
          />
        )}
      </div>

      {compare.pendingWinnerChoice && <WinnerChoiceModal />}
    </div>
  );
}

function WinnerChoiceModal() {
  const pending = useCompareStore((s) => s.compare?.pendingWinnerChoice);
  const groups = useGroupStore((s) => s.groups);
  const cancelWinnerChoice = useCompareStore((s) => s.cancelWinnerChoice);
  const commitMergeWithWinner = useCompareStore((s) => s.commitMergeWithWinner);
  const groupChoices = useMemo(() => {
    if (!pending) return [];
    const ids = new Set(pending.confirmedGroupIds);
    return groups.filter((g) => ids.has(g.id));
  }, [pending, groups]);
  const [picked, setPicked] = useState<string | null>(null);
  useEffect(() => {
    if (picked || groupChoices.length === 0) return;
    const largest = [...groupChoices].sort((a, b) => b.images.length - a.images.length)[0]!;
    setPicked(largest.id);
  }, [groupChoices, picked]);
  if (!pending) return null;

  const title = "Multiple confirmed groups";
  const footer = (
    <>
      <button className="btn" onClick={cancelWinnerChoice}>
        Back
      </button>
      <button
        className="btn btn-primary"
        onClick={() => picked && commitMergeWithWinner(picked)}
        disabled={!picked}
      >
        Merge into selected
      </button>
    </>
  );
  return (
    <Modal
      title={title}
      onClose={cancelWinnerChoice}
      footer={footer}
      className="winner-choice-modal"
    >
      <div className="winner-choice-intro">
        Pick the group that should keep the merged images. The other group(s) will be deleted and
        their images moved into the chosen group.
      </div>
      <div className="winner-choice-list">
        {groupChoices.map((g) => (
          <label
            key={g.id}
            className={cn("winner-choice-row", picked === g.id && "winner-choice-row-picked")}
          >
            <input
              type="radio"
              name="winner-choice"
              value={g.id}
              checked={picked === g.id}
              onChange={() => setPicked(g.id)}
            />
            <div className="winner-choice-name">{g.name}</div>
            <div className="winner-choice-count">{g.images.length} images</div>
          </label>
        ))}
      </div>
    </Modal>
  );
}

function filterCandidates(
  compare: CompareState,
  _checked: Set<string>,
  clusters: ClusterResultData[],
  splitChildren: Record<string, SplitChildren>,
): ClusterResultData[] {
  const { candidateOrder, includeConfirmedGroups, manuallyAdded } = compare;
  const out: ClusterResultData[] = [];
  for (const id of candidateOrder) {
    const c = findClusterEverywhere(clusters, splitChildren, id);
    if (!c) continue;
    if (!includeConfirmedGroups && c.confirmedGroup && !manuallyAdded.has(id)) continue;
    out.push(c);
  }
  return out;
}

function CandidateList({
  visibleCandidates,
  checked,
  distances,
  onToggle,
}: {
  visibleCandidates: ClusterResultData[];
  checked: Set<string>;
  distances: Record<string, number>;
  onToggle: (id: string) => void;
}) {
  const [shown, setShown] = useState(MAX_INITIAL_CANDIDATES);
  const sentinelRef = useRef<HTMLDivElement>(null);

  // Reset when the list source changes.
  const candidateIdsKey = useMemo(
    () => visibleCandidates.map((c) => c.id).join(","),
    [visibleCandidates],
  );
  // biome-ignore lint/correctness/useExhaustiveDependencies: candidateIdsKey is the only intentional trigger
  useEffect(() => {
    setShown(MAX_INITIAL_CANDIDATES);
  }, [candidateIdsKey]);

  const hasMore = visibleCandidates.length > shown;
  const loadMore = useCallback(() => {
    setShown((n) => n + MAX_INITIAL_CANDIDATES);
  }, []);

  // Auto-load more when the sentinel scrolls into view.
  useEffect(() => {
    if (!hasMore) return;
    const el = sentinelRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) loadMore();
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasMore, loadMore]);

  const slice = visibleCandidates.slice(0, shown);

  return (
    <>
      {slice.map((c) => (
        <CandidateRow
          key={c.id}
          candidate={c}
          checked={checked.has(c.id)}
          distance={distances[c.id]}
          onToggle={() => onToggle(c.id)}
        />
      ))}
      {hasMore && <div ref={sentinelRef} className="compare-load-sentinel" aria-hidden />}
    </>
  );
}

function CandidateRow({
  candidate,
  checked,
  distance,
  onToggle,
}: {
  candidate: ClusterResultData;
  checked: boolean;
  distance: number | undefined;
  onToggle: () => void;
}) {
  const isConfirmed = !!candidate.confirmedGroup;
  return (
    <div
      className={cn(
        "compare-candidate",
        checked && "compare-candidate-checked",
        isConfirmed && "compare-candidate-confirmed",
      )}
      onClick={onToggle}
    >
      <input
        type="checkbox"
        className="compare-candidate-check"
        checked={checked}
        onChange={onToggle}
        onClick={(e) => e.stopPropagation()}
      />
      <div className="compare-candidate-meta">
        <div className="compare-candidate-name">
          {isConfirmed && (
            <span className="compare-lock" title="Confirmed group">
              🔒
            </span>
          )}
          {candidate.confirmedGroup?.name ?? candidate.autoName}
        </div>
        <div className="compare-candidate-stats">
          {candidate.images.length} images
          {distance !== undefined && ` · d=${distance.toFixed(3)}`}
        </div>
      </div>
      <ContactSheet images={candidate.images} />
    </div>
  );
}

function ContactSheet({ images }: { images: string[] }) {
  // Justified row layout — let CSS flex handle the row wrapping; we cap how
  // many we render to keep the DOM small for huge clusters.
  const MAX_THUMBS = 60;
  const trimmed = images.length > MAX_THUMBS ? images.slice(0, MAX_THUMBS) : images;
  return (
    <div className="compare-contact-sheet">
      {trimmed.map((f) => (
        <div key={f} className="compare-thumb">
          <img src={imageUrl(f)} loading="lazy" decoding="async" alt={f} />
        </div>
      ))}
      {images.length > MAX_THUMBS && (
        <div className="compare-thumb-more">+{images.length - MAX_THUMBS}</div>
      )}
    </div>
  );
}

function CompareSearch({
  existingIds,
  sourceId,
  onPick,
}: {
  existingIds: Set<string>;
  sourceId: string;
  onPick: (id: string) => void;
}) {
  const clusterData = useListStore((s) => s.clusterData);
  const splitChildren = useListStore((s) => s.splitChildren);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  useDismissOnOutside(containerRef, open, () => setOpen(false));

  const matches = useMemo(() => {
    if (!query.trim() || !clusterData) return [] as ClusterResultData[];
    const q = query.trim().toLowerCase();
    const all: ClusterResultData[] = [...clusterData.clusters];
    for (const k of Object.values(splitChildren)) {
      all.push(k.childA, k.childB);
    }
    return all
      .filter((c) => c.id !== sourceId)
      .filter((c) => {
        const name = c.confirmedGroup?.name ?? c.autoName;
        return name.toLowerCase().includes(q);
      })
      .slice(0, 8);
  }, [query, clusterData, splitChildren, sourceId]);

  return (
    <div className="compare-search" ref={containerRef}>
      <input
        type="search"
        placeholder="Compare with another cluster…"
        className="compare-search-input"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
      />
      {open && matches.length > 0 && (
        <div className="compare-search-results">
          {matches.map((c) => {
            const inList = existingIds.has(c.id);
            const name = c.confirmedGroup?.name ?? c.autoName;
            return (
              <div
                key={c.id}
                className={cn("compare-search-item", inList && "is-already-in-list")}
                onMouseDown={(e) => {
                  e.preventDefault();
                  onPick(c.id);
                  setQuery("");
                  setOpen(false);
                }}
              >
                <span className="compare-search-name">{name}</span>
                <span className="compare-search-count">{c.images.length}</span>
                {inList && <span className="compare-search-flag">already shown</span>}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
