import { useEffect, useRef, useState } from "react";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { useToastStore } from "../../stores/core/toastStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import type { ImageGroup } from "../../types.ts";
import { cn, imageUrl } from "../../utils/helpers.ts";
import {
  deleteRuleSetById,
  duplicateRuleSet,
  emptyRuleSetDraft,
  loadAllRuleSets,
  loadLastRuleSetId,
  type NamingRuleSet,
  renderTemplate,
  saveLastRuleSetId,
  upsertRuleSet,
} from "../../utils/namingRules.ts";
import { shortcutForSlot, slotForKeyEvent } from "../../utils/reviewConfigs.ts";
import { Modal } from "./Modal.tsx";
import { NamingRuleEditor } from "./NamingRuleEditor.tsx";

const NEW_RULESET_OPTION = "__new_naming__";
// Sentinel selection value meaning "use this group's free-text name". Stored in
// the same `chosen` map as rule ids, so a group has exactly one active choice.
const CUSTOM_CHOICE = "__custom__";

interface NamingRulesModalProps {
  onClose: () => void;
}

export function NamingRulesModal({ onClose }: NamingRulesModalProps) {
  // Snapshot groups once so stepping/preview stays stable while we work; Apply
  // writes back into the live store keyed by id.
  const [snapshot] = useState<ImageGroup[]>(() =>
    useGroupStore.getState().groups.map((g) => ({ ...g, images: g.images.slice() })),
  );

  const [boot] = useState(() => {
    const sets = loadAllRuleSets();
    return { sets, lastId: loadLastRuleSetId(sets) };
  });
  const [ruleSets, setRuleSets] = useState<NamingRuleSet[]>(boot.sets);
  const [activeId, setActiveId] = useState(boot.lastId);
  const activeSet = ruleSets.find((s) => s.id === activeId) ?? ruleSets[0]!;

  const [editor, setEditor] = useState<{ initial: NamingRuleSet; isNew: boolean } | null>(null);

  // Per group: the active choice — a rule id or CUSTOM_CHOICE. Map<groupId, choice>.
  const [chosen, setChosen] = useState<Map<string, string>>(new Map());
  // Per group: the free-text name, kept independently of selection so switching
  // to a rule and back preserves whatever was typed. Map<groupId, text>.
  const [customNames, setCustomNames] = useState<Map<string, string>>(new Map());
  const [index, setIndex] = useState(0);
  const customInputRef = useRef<HTMLInputElement | null>(null);

  const showToast = useToastStore((s) => s.showToast);
  const lightboxOpen = useLightboxStore((s) => s.open);

  const total = snapshot.length;
  const current = snapshot[index];
  const rules = activeSet.rules;

  function selectRuleSet(id: string) {
    setActiveId(id);
    saveLastRuleSetId(id);
  }

  function setChoice(id: string, choice: string) {
    setChosen((prev) => {
      const next = new Map(prev);
      if (next.get(id) === choice) next.delete(id);
      else next.set(id, choice);
      return next;
    });
  }

  // Slot keys 0..rules.length-1 pick a rule (and advance); the trailing slot
  // selects the free-text choice (and focuses the field instead of advancing).
  function chooseAndAdvance(slot: number) {
    if (!current) return;
    if (slot === rules.length) {
      selectCustom();
      return;
    }
    const rule = rules[slot];
    if (!rule) return;
    const wasSame = chosen.get(current.id) === rule.id;
    setChoice(current.id, rule.id);
    if (!wasSame) setIndex((i) => Math.min(total - 1, i + 1));
  }

  // Toggle the free-text choice for the current group (mirrors rule toggling);
  // focus the field when turning it on. The text itself is left untouched.
  function selectCustom() {
    if (!current) return;
    const willSelect = chosen.get(current.id) !== CUSTOM_CHOICE;
    setChoice(current.id, CUSTOM_CHOICE);
    if (willSelect) requestAnimationFrame(() => customInputRef.current?.focus());
  }

  // Make the free-text choice active without toggling (e.g. on field focus).
  function ensureCustomSelected(id: string) {
    setChosen((prev) => {
      if (prev.get(id) === CUSTOM_CHOICE) return prev;
      const next = new Map(prev);
      next.set(id, CUSTOM_CHOICE);
      return next;
    });
  }

  function setCustomName(id: string, text: string) {
    setCustomNames((prev) => {
      const next = new Map(prev);
      if (text === "") next.delete(id);
      else next.set(id, text);
      return next;
    });
    // Editing the field implies you want it — make it the active choice.
    if (text !== "") {
      setChosen((prev) => {
        if (prev.get(id) === CUSTOM_CHOICE) return prev;
        const next = new Map(prev);
        next.set(id, CUSTOM_CHOICE);
        return next;
      });
    }
  }

  // Insert a metadata field's *value* (not the token) at the custom input's
  // caret, so the user can build a name from real text and edit it.
  function insertFieldValue(field: "name" | "title" | "subtitle" | "short_sub") {
    if (!current) return;
    const value = current[field];
    if (!value) return;
    const el = customInputRef.current;
    const existing = customNames.get(current.id) ?? "";
    const start = el?.selectionStart ?? existing.length;
    const end = el?.selectionEnd ?? existing.length;
    const next = existing.slice(0, start) + value + existing.slice(end);
    setCustomName(current.id, next);
    if (el) {
      const caret = start + value.length;
      requestAnimationFrame(() => {
        el.focus();
        el.setSelectionRange(caret, caret);
      });
    }
  }

  function advance(delta: number) {
    setIndex((i) => Math.min(total - 1, Math.max(0, i + delta)));
  }

  function openGroupLightbox(groupImages: string[], i: number) {
    const imageMap = useImageStore.getState().imageMap;
    const items = groupImages.filter((fn) => imageMap.has(fn));
    if (items.length === 0) return;
    useLightboxStore.getState().openLightbox(items, i);
  }

  // Keyboard: digits assign a rule + advance (kept live under the lightbox too);
  // arrows navigate; Esc closes. Mirrors GroupingSortModal's handling.
  const handlersRef = useRef<{
    chooseAndAdvance: (slot: number) => void;
    advance: (delta: number) => void;
    onClose: () => void;
    slotCount: number;
  }>(null!);
  // +1 trailing slot for the free-text choice.
  handlersRef.current = { chooseAndAdvance, advance, onClose, slotCount: rules.length + 1 };

  useEffect(() => {
    if (editor) return;
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLElement) {
        const tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      }
      const h = handlersRef.current;
      const slot = slotForKeyEvent(e);
      if (slot !== null && slot < h.slotCount) {
        e.preventDefault();
        h.chooseAndAdvance(slot);
        return;
      }
      if (lightboxOpen) return;
      if (e.key === "Escape") {
        h.onClose();
      } else if (e.key === "ArrowLeft" || e.key === "h") {
        e.preventDefault();
        h.advance(-1);
      } else if (e.key === "ArrowRight" || e.key === "l") {
        e.preventDefault();
        h.advance(1);
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [lightboxOpen, editor]);

  // The name a group would be renamed to given its current choice, or "" if it
  // has no choice (or the choice resolves to empty text).
  function effectiveName(g: ImageGroup): string {
    const choice = chosen.get(g.id);
    if (!choice) return "";
    if (choice === CUSTOM_CHOICE) return (customNames.get(g.id) ?? "").trim();
    const rule = activeSet.rules.find((r) => r.id === choice);
    return rule ? renderTemplate(rule.template, g) : "";
  }

  // Cheap (few groups) — recompute each render rather than memoise an unstable
  // closure over chosen/customNames/activeSet.
  const assignedCount = snapshot.reduce((n, g) => n + (effectiveName(g) === "" ? 0 : 1), 0);

  function commitSet(next: NamingRuleSet) {
    setRuleSets((prev) => upsertRuleSet(prev, next));
  }

  function handleEditorSave(next: NamingRuleSet) {
    commitSet(next);
    if (next.id !== activeId) selectRuleSet(next.id);
    setEditor(null);
  }

  function handleEditorDelete() {
    if (!editor) return;
    setRuleSets((prev) => deleteRuleSetById(prev, editor.initial.id));
    selectRuleSet(loadAllRuleSets()[0]!.id);
    setEditor(null);
  }

  function openEditEditor() {
    if (activeSet.builtIn) {
      setEditor({ initial: duplicateRuleSet(activeSet), isNew: true });
    } else {
      setEditor({ initial: activeSet, isNew: false });
    }
  }

  function handleApply() {
    const renamed = new Map<string, string>();
    for (const g of snapshot) {
      const name = effectiveName(g);
      if (name) renamed.set(g.id, name);
    }
    if (renamed.size === 0) {
      onClose();
      return;
    }
    useGroupStore
      .getState()
      .updateGroups((prev) =>
        prev.map((g) => (renamed.has(g.id) ? { ...g, name: renamed.get(g.id)! } : g)),
      );
    showToast(
      `Renamed ${renamed.size} group${renamed.size === 1 ? "" : "s"} from naming rules`,
      "success",
    );
    onClose();
  }

  const currentChoice = current ? chosen.get(current.id) : undefined;
  const currentCustom = current ? (customNames.get(current.id) ?? "") : "";
  const customSelected = currentChoice === CUSTOM_CHOICE;

  const headerControls = (
    <div className="review-config-picker">
      <select
        className="review-config-select"
        value={activeId}
        onChange={(e) => {
          const v = e.target.value;
          if (v === NEW_RULESET_OPTION) setEditor({ initial: emptyRuleSetDraft(), isNew: true });
          else selectRuleSet(v);
        }}
        title="Select naming rule set"
      >
        {ruleSets.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name}
            {s.builtIn ? " (built-in)" : ""}
          </option>
        ))}
        <option disabled>──────</option>
        <option value={NEW_RULESET_OPTION}>+ New rule set…</option>
      </select>
      <button
        type="button"
        className="btn-icon"
        onClick={openEditEditor}
        title={activeSet.builtIn ? "Duplicate this rule set to edit" : "Edit this rule set"}
        aria-label="Edit rule set"
      >
        ⚙
      </button>
    </div>
  );

  const title = (
    <>
      <span className="review-modal-title-text">Apply Naming Rules</span>
      {headerControls}
      <span className="review-progress">
        <span className="review-progress-total">{assignedCount} chosen</span>
        <span className="review-progress-total">
          {total === 0 ? "0 / 0" : `${index + 1} / ${total}`}
        </span>
      </span>
    </>
  );

  const customShortcut = shortcutForSlot(rules.length);
  const hintText = [
    ...rules
      .map((_, i) => {
        const k = shortcutForSlot(i);
        return k ? `${k} rule ${i + 1}` : null;
      })
      .filter(Boolean),
    customShortcut ? `${customShortcut} custom` : null,
    "← → navigate",
    "Esc cancel",
  ]
    .filter(Boolean)
    .join(" · ");

  const footer = (
    <>
      <span className="review-footer-hint modal-footer-spacer">{hintText}</span>
      <button type="button" className="btn btn-secondary" onClick={onClose}>
        Cancel
      </button>
      <button
        type="button"
        className="btn btn-primary"
        onClick={handleApply}
        disabled={assignedCount === 0}
        title="Set each chosen group's name to its selected rule's rendered text"
      >
        {assignedCount === 0 ? "Apply" : `Apply (${assignedCount})`}
      </button>
    </>
  );

  return (
    <>
      <Modal
        title={title}
        onClose={onClose}
        footer={footer}
        className="review-modal"
        headerClassName="review-modal-header"
        bodyClassName="review-modal-body"
        closeOnEscape={false}
      >
        {!current ? (
          <div className="review-empty">No groups to rename.</div>
        ) : (
          <div className="review-single">
            <div className="review-single-header">
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={() => advance(-1)}
                disabled={index === 0}
                aria-label="Previous"
              >
                ← Prev
              </button>
              <div className="review-single-title">
                <span className="review-single-name">{current.name}</span>
                <span className="review-single-count">
                  {current.images.length} image{current.images.length === 1 ? "" : "s"}
                </span>
              </div>
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={() => advance(1)}
                disabled={index === total - 1}
                aria-label="Next"
              >
                Next →
              </button>
            </div>

            <dl className="naming-meta">
              {(["title", "subtitle", "short_sub"] as const).map((field) => (
                <div key={field} className="naming-meta-row">
                  <dt className="naming-meta-key">{field}</dt>
                  <dd className={cn("naming-meta-val", !current[field] && "naming-meta-val-empty")}>
                    {current[field] || "—"}
                  </dd>
                </div>
              ))}
            </dl>

            <div className="review-single-thumbs">
              {current.images.map((fn, i) => (
                <button
                  type="button"
                  key={fn}
                  className="review-single-thumb"
                  onClick={() => openGroupLightbox(current.images, i)}
                  aria-label={`Open ${fn}`}
                >
                  <img src={imageUrl(fn)} alt="" loading="lazy" draggable={false} />
                </button>
              ))}
            </div>

            <div className="review-single-actions naming-rule-actions">
              {rules.length === 0 && (
                <span className="review-single-no-subs">
                  No rules in this set — open ⚙ to add one.
                </span>
              )}
              {rules.map((rule, idx) => {
                const shortcut = shortcutForSlot(idx);
                const preview = renderTemplate(rule.template, current);
                const active = currentChoice === rule.id;
                return (
                  <button
                    type="button"
                    key={rule.id}
                    className={cn(
                      "btn naming-rule-btn",
                      active ? "review-status-active" : "btn-secondary",
                    )}
                    onClick={() => chooseAndAdvance(idx)}
                    title={rule.template}
                  >
                    <span className="naming-rule-btn-head">
                      {shortcut && <span className="review-single-status-key">{shortcut}</span>}
                      <span className="naming-rule-btn-template">{rule.template}</span>
                    </span>
                    <span
                      className={cn(
                        "naming-rule-btn-preview",
                        !preview && "naming-rule-btn-preview-empty",
                      )}
                    >
                      {preview || "(empty)"}
                    </span>
                  </button>
                );
              })}
            </div>

            <div className={cn("naming-custom", customSelected && "naming-custom-selected")}>
              <div className="naming-custom-head">
                <button
                  type="button"
                  className="naming-custom-select"
                  onClick={selectCustom}
                  title={
                    customShortcut
                      ? `Use this free-text name (press ${customShortcut})`
                      : "Use this free-text name"
                  }
                >
                  {customShortcut && (
                    <span className="review-single-status-key">{customShortcut}</span>
                  )}
                  <span className="naming-custom-label">Custom name</span>
                  <span className="naming-custom-hint">
                    {customSelected ? "selected" : "click or press the key to use"}
                  </span>
                </button>
                <div className="naming-custom-fill">
                  {(["title", "subtitle", "short_sub", "name"] as const).map((field) => (
                    <button
                      key={field}
                      type="button"
                      className="btn btn-secondary btn-small naming-token-btn"
                      onClick={() => insertFieldValue(field)}
                      disabled={!current[field]}
                      title={
                        current[field]
                          ? `Insert ${field}: ${current[field]}`
                          : `No ${field} on this group`
                      }
                    >
                      + {field}
                    </button>
                  ))}
                </div>
              </div>
              <div className="naming-custom-input-row">
                <input
                  type="text"
                  ref={customInputRef}
                  className="review-config-input naming-custom-input"
                  value={currentCustom}
                  onChange={(e) => current && setCustomName(current.id, e.target.value)}
                  onFocus={() => current && ensureCustomSelected(current.id)}
                  placeholder="Type a name, or use the buttons to insert field values…"
                />
                {currentCustom && (
                  <button
                    type="button"
                    className="btn-icon btn-icon-danger"
                    onClick={() => current && setCustomName(current.id, "")}
                    aria-label="Clear custom name"
                    title="Clear custom name"
                  >
                    ×
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </Modal>

      {editor && (
        <NamingRuleEditor
          initial={editor.initial}
          isNew={editor.isNew}
          sample={current}
          onSave={handleEditorSave}
          onClose={() => setEditor(null)}
          onDelete={editor.isNew ? undefined : handleEditorDelete}
        />
      )}
    </>
  );
}
