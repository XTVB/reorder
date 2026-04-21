import { useEffect, useMemo, useState } from "react";
import { useClusterStore } from "../../stores/clusterStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { cn } from "../../utils/helpers.ts";
import { Modal } from "../Modal.tsx";

interface Props {
  onClose: () => void;
}

export function ScopePickerModal({ onClose }: Props) {
  const groups = useGroupStore((s) => s.groups);
  const runScopedCluster = useClusterStore((s) => s.runScopedCluster);

  const [query, setQuery] = useState("");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        if (!submitting) onClose();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose, submitting]);

  const filtered = useMemo(() => {
    if (!query.trim()) return groups;
    const lq = query.toLowerCase();
    return groups.filter((g) => g.name.toLowerCase().includes(lq));
  }, [groups, query]);

  const totalImages = useMemo(() => {
    let n = 0;
    for (const g of groups) {
      if (selectedIds.has(g.id)) n += g.images.length;
    }
    return n;
  }, [groups, selectedIds]);

  function toggle(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleRun() {
    if (selectedIds.size < 1 || submitting) return;
    setSubmitting(true);
    try {
      await runScopedCluster([...selectedIds]);
      onClose();
    } finally {
      setSubmitting(false);
    }
  }

  const title = (
    <>
      <span className="modal-title-main">Scoped clustering</span>
      <span className="scope-picker-subtitle">
        {selectedIds.size} selected · {totalImages} images
      </span>
      <button
        type="button"
        className="btn btn-icon modal-close-btn"
        onClick={() => !submitting && onClose()}
        aria-label="Close"
      >
        ×
      </button>
    </>
  );

  return (
    <Modal
      title={title}
      className="scope-picker-modal"
      onClose={() => !submitting && onClose()}
      footer={
        <>
          <span className="nn-footer-status">
            Re-cluster over the images in the chosen groups only
          </span>
          <button type="button" className="btn" onClick={() => !submitting && onClose()}>
            Cancel
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleRun}
            disabled={selectedIds.size < 1 || submitting}
          >
            {submitting
              ? "Running…"
              : `Re-cluster ${selectedIds.size} group${selectedIds.size === 1 ? "" : "s"}`}
          </button>
        </>
      }
    >
      <input
        type="text"
        autoFocus
        placeholder="Search groups…"
        className="scope-picker-search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {groups.length === 0 ? (
        <div className="scope-picker-empty">
          No groups yet. Create some groups first, then come back to cluster within them.
        </div>
      ) : filtered.length === 0 ? (
        <div className="scope-picker-empty">No matching groups</div>
      ) : (
        <div className="scope-picker-list">
          {filtered.map((g) => {
            const selected = selectedIds.has(g.id);
            return (
              <button
                type="button"
                key={g.id}
                className={cn("scope-picker-item", selected && "scope-picker-item-selected")}
                onClick={() => toggle(g.id)}
              >
                <span className="scope-picker-check">{selected ? "✓" : ""}</span>
                <span className="scope-picker-name">{g.name}</span>
                <span className="scope-picker-count">{g.images.length}</span>
              </button>
            );
          })}
        </div>
      )}
    </Modal>
  );
}
