import type React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useDismissOnOutside } from "../hooks/useDismissOnOutside.ts";
import { useUIStore } from "../stores/uiStore.ts";
import type { ImageGroup } from "../types.ts";

interface GroupPickerProps {
  groups: ImageGroup[];
  onSelect: (groupId: string) => void;
}

function fuzzyMatch(query: string, text: string): boolean {
  const lq = query.toLowerCase();
  const lt = text.toLowerCase();
  let qi = 0;
  for (let ti = 0; ti < lt.length && qi < lq.length; ti++) {
    if (lt[ti] === lq[qi]) qi++;
  }
  return qi === lq.length;
}

export function GroupPicker({ groups, onSelect }: GroupPickerProps) {
  const open = useUIStore((s) => s.showGroupPicker);
  const setOpen = useUIStore((s) => s.setShowGroupPicker);
  const [query, setQuery] = useState("");
  const [highlightIdx, setHighlightIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const filtered = query ? groups.filter((g) => fuzzyMatch(query, g.name)) : groups;

  // Reset highlight when filter changes
  // biome-ignore lint/correctness/useExhaustiveDependencies: query is the intentional trigger to reset highlight position
  useEffect(() => {
    setHighlightIdx(0);
  }, [query]);

  // Focus input when opened
  useEffect(() => {
    if (open) {
      setQuery("");
      setHighlightIdx(0);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  useDismissOnOutside(containerRef, open, () => setOpen(false));

  useEffect(() => () => setOpen(false), [setOpen]);

  const handleSelect = useCallback(
    (groupId: string) => {
      onSelect(groupId);
      setOpen(false);
    },
    [onSelect, setOpen],
  );

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightIdx((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (filtered[highlightIdx]) handleSelect(filtered[highlightIdx].id);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div className="group-picker" ref={containerRef}>
      <button
        className="btn btn-secondary"
        onClick={() => setOpen(!open)}
        title="Add selection to an existing group (H)"
      >
        Add to Group
      </button>
      {open && (
        <div className="group-picker-dropdown">
          <input
            ref={inputRef}
            className="group-picker-search"
            type="text"
            placeholder="Search groups..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <div className="group-picker-list">
            {filtered.length === 0 ? (
              <div className="group-picker-empty">No matching groups</div>
            ) : (
              filtered.map((g, i) => (
                <button
                  key={g.id}
                  className={`group-picker-item${i === highlightIdx ? " group-picker-item-active" : ""}`}
                  onClick={() => handleSelect(g.id)}
                  onMouseEnter={() => setHighlightIdx(i)}
                >
                  <span className="group-picker-item-name">{g.name}</span>
                  <span className="group-picker-item-count">{g.images.length}</span>
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
