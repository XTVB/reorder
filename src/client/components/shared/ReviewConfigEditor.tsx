import {
  closestCenter,
  DndContext,
  type DragEndEvent,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
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

interface SortableSubRowProps {
  sub: ReviewSubcategory;
  index: number;
  catId: string;
  isDefault: boolean;
  onUpdate: (catId: string, subId: string, patch: Partial<ReviewSubcategory>) => void;
  onRemove: (catId: string, subId: string) => void;
  onSetDefault: (catId: string, subId: string) => void;
}

function SortableSubRow({
  sub,
  index,
  catId,
  isDefault,
  onUpdate,
  onRemove,
  onSetDefault,
}: SortableSubRowProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: sub.id,
  });
  const subShortcut = shortcutForSlot(index);
  const subColor = colorForId(sub.id, index);
  const style = {
    ...reviewColorVar(subColor),
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : undefined,
  };
  return (
    <li ref={setNodeRef} className="review-config-sub-row" style={style}>
      <button
        type="button"
        className="review-config-drag-handle"
        aria-label="Drag to reorder subcategory"
        title="Drag to reorder"
        {...attributes}
        {...listeners}
      >
        ⠿
      </button>
      <span className="review-config-shortcut">{subShortcut ?? "—"}</span>
      <span className="review-config-swatch" aria-hidden style={{ background: subColor }} />
      <input
        type="text"
        className="review-config-input"
        value={sub.label}
        onChange={(e) => onUpdate(catId, sub.id, { label: e.target.value })}
        placeholder="Subcategory label"
      />
      <label
        className={cn(
          "review-config-default-pill",
          isDefault && "review-config-default-pill-active",
        )}
        title="Unassigned groups inside this category fall into this subcategory on Apply"
      >
        <input
          type="radio"
          name={`default-sub-${catId}`}
          checked={isDefault}
          onChange={() => onSetDefault(catId, sub.id)}
        />
        default
      </label>
      <div className="review-config-row-actions">
        <button
          type="button"
          className="btn-icon btn-icon-danger"
          onClick={() => onRemove(catId, sub.id)}
          aria-label="Remove subcategory"
          title="Remove subcategory"
        >
          ×
        </button>
      </div>
    </li>
  );
}

interface SortableCategoryRowProps {
  cat: ReviewCategory;
  index: number;
  total: number;
  defaultCategoryId: string;
  isExpanded: boolean;
  onToggleExpand: (id: string) => void;
  onUpdate: (id: string, patch: Partial<ReviewCategory>) => void;
  onRemove: (id: string) => void;
  onSetDefault: (id: string) => void;
  onAddSub: (catId: string) => void;
  onUpdateSub: (catId: string, subId: string, patch: Partial<ReviewSubcategory>) => void;
  onRemoveSub: (catId: string, subId: string) => void;
  onSetDefaultSub: (catId: string, subId: string) => void;
  onSubDragEnd: (catId: string, e: DragEndEvent) => void;
  subSensors: ReturnType<typeof useSensors>;
}

function SortableCategoryRow({
  cat,
  index,
  total,
  defaultCategoryId,
  isExpanded,
  onToggleExpand,
  onUpdate,
  onRemove,
  onSetDefault,
  onAddSub,
  onUpdateSub,
  onRemoveSub,
  onSetDefaultSub,
  onSubDragEnd,
  subSensors,
}: SortableCategoryRowProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: cat.id,
  });
  const shortcut = shortcutForSlot(index);
  const color = colorForId(cat.id, index);
  const style = {
    ...reviewColorVar(color),
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : undefined,
  };
  return (
    <li ref={setNodeRef} className="review-config-cat-row" style={style}>
      <div className="review-config-cat-head">
        <button
          type="button"
          className="review-config-drag-handle"
          aria-label="Drag to reorder category"
          title="Drag to reorder"
          {...attributes}
          {...listeners}
        >
          ⠿
        </button>
        <button
          type="button"
          className="review-config-expand-btn"
          onClick={() => onToggleExpand(cat.id)}
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
          onChange={(e) => onUpdate(cat.id, { label: e.target.value })}
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
            onChange={() => onSetDefault(cat.id)}
          />
          default
        </label>
        <div className="review-config-row-actions">
          <button
            type="button"
            className="btn-icon btn-icon-danger"
            onClick={() => onRemove(cat.id)}
            disabled={total === 1}
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
            <DndContext
              sensors={subSensors}
              collisionDetection={closestCenter}
              onDragEnd={(e) => onSubDragEnd(cat.id, e)}
            >
              <SortableContext
                items={cat.subcategories.map((s) => s.id)}
                strategy={verticalListSortingStrategy}
              >
                <ol className="review-config-sub-list">
                  {cat.subcategories.map((sub, si) => (
                    <SortableSubRow
                      key={sub.id}
                      sub={sub}
                      index={si}
                      catId={cat.id}
                      isDefault={cat.defaultSubcategoryId === sub.id}
                      onUpdate={onUpdateSub}
                      onRemove={onRemoveSub}
                      onSetDefault={onSetDefaultSub}
                    />
                  ))}
                </ol>
              </SortableContext>
            </DndContext>
          )}
          <button
            type="button"
            className="btn btn-secondary btn-small review-config-add-sub"
            onClick={() => onAddSub(cat.id)}
          >
            + Add subcategory
          </button>
        </div>
      )}
    </li>
  );
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

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

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

  function handleCategoryDragEnd(e: DragEndEvent) {
    const { active, over } = e;
    if (!over || active.id === over.id) return;
    setCategories((prev) => {
      const from = prev.findIndex((c) => c.id === active.id);
      const to = prev.findIndex((c) => c.id === over.id);
      if (from === -1 || to === -1) return prev;
      return arrayMove(prev, from, to);
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

  function setDefaultSubcategory(catId: string, subId: string) {
    updateCategory(catId, { defaultSubcategoryId: subId });
  }

  function handleSubDragEnd(catId: string, e: DragEndEvent) {
    const { active, over } = e;
    if (!over || active.id === over.id) return;
    setCategories((prev) =>
      prev.map((c) => {
        if (c.id !== catId) return c;
        const from = c.subcategories.findIndex((s) => s.id === active.id);
        const to = c.subcategories.findIndex((s) => s.id === over.id);
        if (from === -1 || to === -1) return c;
        return { ...c, subcategories: arrayMove(c.subcategories, from, to) };
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

      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragEnd={handleCategoryDragEnd}
      >
        <SortableContext items={categories.map((c) => c.id)} strategy={verticalListSortingStrategy}>
          <ol className="review-config-cat-list">
            {categories.map((cat, ci) => (
              <SortableCategoryRow
                key={cat.id}
                cat={cat}
                index={ci}
                total={categories.length}
                defaultCategoryId={defaultCategoryId}
                isExpanded={expanded.has(cat.id)}
                onToggleExpand={toggleExpanded}
                onUpdate={updateCategory}
                onRemove={removeCategory}
                onSetDefault={setDefaultCategoryId}
                onAddSub={addSubcategory}
                onUpdateSub={updateSubcategory}
                onRemoveSub={removeSubcategory}
                onSetDefaultSub={setDefaultSubcategory}
                onSubDragEnd={handleSubDragEnd}
                subSensors={sensors}
              />
            ))}
          </ol>
        </SortableContext>
      </DndContext>
    </Modal>
  );
}
