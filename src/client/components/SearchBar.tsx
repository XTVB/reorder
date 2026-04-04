import type React from "react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useGroupStore } from "../stores/groupStore.ts";
import type { GridItem } from "../types.ts";
import { gridItemId } from "../utils/gridItems.ts";

// Context to share search state between SearchBar and App
type SearchContextType = ReturnType<typeof useSearchState>;
export const SearchContext = createContext<SearchContextType | null>(null);
export const useSearchContext = () => useContext(SearchContext);

interface SearchBarProps {
  gridItems: GridItem[];
  onScrollToRow: (rowIndex: number) => void;
  columnCount: number;
}

export function useSearchState() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [matchIds, setMatchIds] = useState<Set<string>>(new Set());
  const [currentMatchId, setCurrentMatchId] = useState<string | null>(null);

  const close = useCallback(() => {
    setIsOpen(false);
    setQuery("");
    setMatchIds(new Set());
    setCurrentMatchId(null);
  }, []);

  const open = useCallback(() => setIsOpen(true), []);

  return {
    isOpen,
    query,
    setQuery,
    matchIds,
    setMatchIds,
    currentMatchId,
    setCurrentMatchId,
    open,
    close,
  };
}

export function SearchBar({ gridItems, onScrollToRow, columnCount }: SearchBarProps) {
  const { isOpen, query, setQuery, matchIds, setMatchIds, setCurrentMatchId, open, close } =
    useSearchContext()!;
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const groupMap = useGroupStore((s) => s.groupMap);

  // Compute matches (pure — no state updates)
  const {
    gridIndices: matches,
    orderedIds,
    ids,
  } = useMemo(() => {
    if (!query.trim()) {
      return { gridIndices: [] as number[], orderedIds: [] as string[], ids: new Set<string>() };
    }
    const q = query.toLowerCase();
    const gridIndices: number[] = [];
    const orderedIds: string[] = [];
    const ids = new Set<string>();
    for (let i = 0; i < gridItems.length; i++) {
      const item = gridItems[i]!;
      if (item.type === "group-image") continue;
      if (item.type === "image") {
        if (item.filename.toLowerCase().includes(q)) {
          gridIndices.push(i);
          orderedIds.push(item.filename);
          ids.add(item.filename);
        }
      } else if (item.type === "group") {
        const group = groupMap.get(item.groupId);
        if (group && group.name.toLowerCase().includes(q)) {
          const id = gridItemId(item);
          gridIndices.push(i);
          orderedIds.push(id);
          ids.add(id);
        }
      }
    }
    return { gridIndices, orderedIds, ids };
  }, [query, gridItems, groupMap]);

  // Sync shared state after matches change
  useEffect(() => {
    setMatchIds(ids);
    setCurrentMatchIndex(0);
  }, [ids, setMatchIds]);

  // Update currentMatchId when index or matches change
  useEffect(() => {
    setCurrentMatchId(orderedIds[currentMatchIndex] ?? null);
  }, [currentMatchIndex, orderedIds, setCurrentMatchId]);

  // Scroll to current match
  useEffect(() => {
    if (matches.length === 0) return;
    const gridIndex = matches[currentMatchIndex];
    if (gridIndex === undefined) return;
    // Compute which visible row this is (excluding group-image items)
    let visibleIdx = 0;
    for (let i = 0; i < gridIndex; i++) {
      if (gridItems[i]!.type !== "group-image") visibleIdx++;
    }
    const rowIndex = Math.floor(visibleIdx / columnCount);
    onScrollToRow(rowIndex);
  }, [currentMatchIndex, matches, columnCount, gridItems, onScrollToRow]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) inputRef.current?.focus();
  }, [isOpen]);

  // Keyboard shortcut
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "f") {
        e.preventDefault();
        open();
      }
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [open]);

  if (!isOpen) return null;

  function goNext() {
    if (matches.length === 0) return;
    setCurrentMatchIndex((i) => (i + 1) % matches.length);
  }

  function goPrev() {
    if (matches.length === 0) return;
    setCurrentMatchIndex((i) => (i - 1 + matches.length) % matches.length);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      close();
    } else if (e.key === "Enter") {
      if (e.shiftKey) goPrev();
      else goNext();
    }
  }

  return (
    <div className="search-bar">
      <input
        ref={inputRef}
        type="text"
        className="search-input"
        placeholder="Search images..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <span className="search-count">
        {query.trim()
          ? matches.length > 0
            ? `${currentMatchIndex + 1} of ${matches.length}`
            : "No matches"
          : ""}
      </span>
      <button
        className="search-nav-btn"
        onClick={goPrev}
        disabled={matches.length === 0}
        title="Previous (Shift+Enter)"
      >
        &#8593;
      </button>
      <button
        className="search-nav-btn"
        onClick={goNext}
        disabled={matches.length === 0}
        title="Next (Enter)"
      >
        &#8595;
      </button>
      <button className="search-close-btn" onClick={close} title="Close (Esc)">
        &times;
      </button>
    </div>
  );
}
