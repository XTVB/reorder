import { arrayMove } from "@dnd-kit/sortable";
import { useMemo, useState } from "react";
import { cn } from "../../utils/helpers.ts";
import {
  colorForId,
  makeId,
  type ReviewCategory,
  type ReviewConfig,
  type ReviewSubcategory,
  reviewColorVar,
  shortcutForSlot,
} from "../../utils/reviewConfigs.ts";
import { Modal } from "./Modal.tsx";

interface ReviewConfigEditorProps {
  initial: ReviewConfig;
  isNew: boolean;
  onSave: (config: ReviewConfig) => void;
  onDelete?: () => void;
  onClose: () => void;
}

export function ReviewConfigEditor({
  initial,
  isNew,
  onSave,
  onDelete,
  onClose,
}: ReviewConfigEditorProps) {
  const [name, setName] = useState(initial.name);
  const [categories, setCategories] = useState<ReviewCategory[]>(() =>
    initial.categories.map((c) => ({
      ...c,
      subcategories: c.subcategories.map((s) => ({ ...s })),
    })),
  );
  const [defaultCategoryId, setDefaultCategoryId] = useState(initial.defaultCategoryId);
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());

  function toggleExpanded(id: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function addCategory() {
    const id = makeId("cat");
    setCategories((prev) => [
      ...prev,
      { id, label: `Category ${prev.length + 1}`, subcategories: [] },
    ]);
    setExpanded((prev) => new Set(prev).add(id));
  }

  function updateCategory(id: string, patch: Partial<ReviewCategory>) {
    setCategories((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  }

  function removeCategory(id: string) {
    setCategories((prev) => {
      const next = prev.filter((c) => c.id !== id);
      // Re-point default if it pointed at the removed category.
      if (id === defaultCategoryId && next[0]) setDefaultCategoryId(next[0].id);
      return next;
    });
  }

  function reorderCategory(id: string, delta: number) {
    setCategories((prev) => {
      const idx = prev.findIndex((c) => c.id === id);
      const to = idx + delta;
      if (idx === -1 || to < 0 || to >= prev.length) return prev;
      return arrayMove(prev, idx, to);
    });
  }

  function addSubcategory(catId: string) {
    setCategories((prev) =>
      prev.map((c) =>
        c.id === catId
          ? {
              ...c,
              subcategories: [
                ...c.subcategories,
                { id: makeId("sub"), label: `Subgroup ${c.subcategories.length + 1}` },
              ],
            }
          : c,
      ),
    );
  }

  function updateSubcategory(catId: string, subId: string, patch: Partial<ReviewSubcategory>) {
    setCategories((prev) =>
      prev.map((c) =>
        c.id === catId
          ? {
              ...c,
              subcategories: c.subcategories.map((s) => (s.id === subId ? { ...s, ...patch } : s)),
            }
          : c,
      ),
    );
  }

  function removeSubcategory(catId: string, subId: string) {
    setCategories((prev) =>
      prev.map((c) => {
        if (c.id !== catId) return c;
        const next = c.subcategories.filter((s) => s.id !== subId);
        const nextDefault = c.defaultSubcategoryId === subId ? undefined : c.defaultSubcategoryId;
        return { ...c, subcategories: next, defaultSubcategoryId: nextDefault };
      }),
    );
  }

  function reorderSubcategory(catId: string, subId: string, delta: number) {
    setCategories((prev) =>
      prev.map((c) => {
        if (c.id !== catId) return c;
        const idx = c.subcategories.findIndex((s) => s.id === subId);
        const to = idx + delta;
        if (idx === -1 || to < 0 || to >= c.subcategories.length) return c;
        return { ...c, subcategories: arrayMove(c.subcategories, idx, to) };
      }),
    );
  }

  const validation = useMemo<string | null>(() => {
    if (!name.trim()) return "Name is required.";
    if (categories.length === 0) return "Add at least one category.";
    const seen = new Set<string>();
    for (const c of categories) {
      if (!c.label.trim()) return "Every category needs a label.";
      const key = c.label.trim().toLowerCase();
      if (seen.has(key)) return `Duplicate category label: "${c.label}".`;
      seen.add(key);
      const subSeen = new Set<string>();
      for (const s of c.subcategories) {
        if (!s.label.trim()) return `Every subcategory in "${c.label}" needs a label.`;
        const sk = s.label.trim().toLowerCase();
        if (subSeen.has(sk)) return `Duplicate subcategory in "${c.label}": "${s.label}".`;
        subSeen.add(sk);
      }
    }
    if (!categories.some((c) => c.id === defaultCategoryId)) return "Pick a default category.";
    return null;
  }, [name, categories, defaultCategoryId]);

  function handleSave() {
    if (validation) return;
    onSave({
      id: initial.id,
      name: name.trim(),
      defaultCategoryId,
      categories: categories.map((c) => ({
        id: c.id,
        label: c.label.trim(),
        subcategories: c.subcategories.map((s) => ({ id: s.id, label: s.label.trim() })),
        defaultSubcategoryId: c.defaultSubcategoryId,
      })),
    });
  }

  const title = isNew ? "New grouping" : "Edit grouping";

  const footer = (
    <>
      <span className="review-config-validation modal-footer-spacer">{validation ?? ""}</span>
      {onDelete && !initial.builtIn && (
        <button
          type="button"
          className="btn btn-danger"
          onClick={() => {
            if (confirm(`Delete grouping "${initial.name}"?`)) onDelete();
          }}
        >
          Delete
        </button>
      )}
      <button type="button" className="btn btn-secondary" onClick={onClose}>
        Cancel
      </button>
      <button
        type="button"
        className="btn btn-primary"
        onClick={handleSave}
        disabled={validation !== null}
      >
        Save
      </button>
    </>
  );

  return (
    <Modal
      title={title}
      onClose={onClose}
      footer={footer}
      className="review-config-modal"
      bodyClassName="review-config-body"
    >
      <label className="review-config-name-row">
        <span className="review-config-label">Name</span>
        <input
          type="text"
          className="review-config-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. By era, By status…"
        />
      </label>

      <div className="review-config-section-header">
        <span className="review-config-label">Categories</span>
        <button type="button" className="btn btn-secondary btn-small" onClick={addCategory}>
          + Add category
        </button>
      </div>

      <ol className="review-config-cat-list">
        {categories.map((cat, ci) => {
          const shortcut = shortcutForSlot(ci);
          const color = colorForId(cat.id, ci);
          const isExpanded = expanded.has(cat.id);
          return (
            <li key={cat.id} className="review-config-cat-row" style={reviewColorVar(color)}>
              <div className="review-config-cat-head">
                <button
                  type="button"
                  className="review-config-expand-btn"
                  onClick={() => toggleExpanded(cat.id)}
                  aria-label={isExpanded ? "Collapse" : "Expand"}
                  title={isExpanded ? "Collapse subcategories" : "Expand subcategories"}
                >
                  {isExpanded ? "▾" : "▸"}
                </button>
                <span className="review-config-shortcut" title="Number-key shortcut">
                  {shortcut ?? "—"}
                </span>
                <span className="review-config-swatch" aria-hidden style={{ background: color }} />
                <input
                  type="text"
                  className="review-config-input"
                  value={cat.label}
                  onChange={(e) => updateCategory(cat.id, { label: e.target.value })}
                  placeholder="Category label"
                />
                <label
                  className={cn(
                    "review-config-default-pill",
                    defaultCategoryId === cat.id && "review-config-default-pill-active",
                  )}
                  title="Unassigned groups fall into the default category on Apply"
                >
                  <input
                    type="radio"
                    name="default-category"
                    checked={defaultCategoryId === cat.id}
                    onChange={() => setDefaultCategoryId(cat.id)}
                  />
                  default
                </label>
                <div className="review-config-row-actions">
                  <button
                    type="button"
                    className="btn-icon"
                    onClick={() => reorderCategory(cat.id, -1)}
                    disabled={ci === 0}
                    aria-label="Move up"
                    title="Move up"
                  >
                    ↑
                  </button>
                  <button
                    type="button"
                    className="btn-icon"
                    onClick={() => reorderCategory(cat.id, 1)}
                    disabled={ci === categories.length - 1}
                    aria-label="Move down"
                    title="Move down"
                  >
                    ↓
                  </button>
                  <button
                    type="button"
                    className="btn-icon btn-icon-danger"
                    onClick={() => removeCategory(cat.id)}
                    disabled={categories.length === 1}
                    aria-label="Remove category"
                    title="Remove category"
                  >
                    ×
                  </button>
                </div>
              </div>

              {isExpanded && (
                <div className="review-config-sub-block">
                  {cat.subcategories.length === 0 ? (
                    <p className="review-config-sub-empty">
                      No subcategories. Groups in this category won't be refinable.
                    </p>
                  ) : (
                    <ol className="review-config-sub-list">
                      {cat.subcategories.map((sub, si) => {
                        const subShortcut = shortcutForSlot(si);
                        const subColor = colorForId(sub.id, si);
                        const isDefaultSub = cat.defaultSubcategoryId === sub.id;
                        return (
                          <li
                            key={sub.id}
                            className="review-config-sub-row"
                            style={reviewColorVar(subColor)}
                          >
                            <span className="review-config-shortcut">{subShortcut ?? "—"}</span>
                            <span
                              className="review-config-swatch"
                              aria-hidden
                              style={{ background: subColor }}
                            />
                            <input
                              type="text"
                              className="review-config-input"
                              value={sub.label}
                              onChange={(e) =>
                                updateSubcategory(cat.id, sub.id, { label: e.target.value })
                              }
                              placeholder="Subcategory label"
                            />
                            <label
                              className={cn(
                                "review-config-default-pill",
                                isDefaultSub && "review-config-default-pill-active",
                              )}
                              title="Unassigned groups inside this category fall into this subcategory on Apply"
                            >
                              <input
                                type="radio"
                                name={`default-sub-${cat.id}`}
                                checked={isDefaultSub}
                                onChange={() =>
                                  updateCategory(cat.id, { defaultSubcategoryId: sub.id })
                                }
                              />
                              default
                            </label>
                            <div className="review-config-row-actions">
                              <button
                                type="button"
                                className="btn-icon"
                                onClick={() => reorderSubcategory(cat.id, sub.id, -1)}
                                disabled={si === 0}
                                aria-label="Move up"
                                title="Move up"
                              >
                                ↑
                              </button>
                              <button
                                type="button"
                                className="btn-icon"
                                onClick={() => reorderSubcategory(cat.id, sub.id, 1)}
                                disabled={si === cat.subcategories.length - 1}
                                aria-label="Move down"
                                title="Move down"
                              >
                                ↓
                              </button>
                              <button
                                type="button"
                                className="btn-icon btn-icon-danger"
                                onClick={() => removeSubcategory(cat.id, sub.id)}
                                aria-label="Remove subcategory"
                                title="Remove subcategory"
                              >
                                ×
                              </button>
                            </div>
                          </li>
                        );
                      })}
                    </ol>
                  )}
                  <button
                    type="button"
                    className="btn btn-secondary btn-small review-config-add-sub"
                    onClick={() => addSubcategory(cat.id)}
                  >
                    + Add subcategory
                  </button>
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </Modal>
  );
}
