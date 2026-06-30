import { useMemo, useRef, useState } from "react";
import type { ImageGroup } from "../../types.ts";
import { cn } from "../../utils/helpers.ts";
import {
  makeRuleId,
  NAMING_TOKENS,
  type NamingRule,
  type NamingRuleSet,
  renderTemplate,
} from "../../utils/namingRules.ts";
import { shortcutForSlot } from "../../utils/reviewConfigs.ts";
import { Modal } from "./Modal.tsx";

interface NamingRuleEditorProps {
  initial: NamingRuleSet;
  isNew: boolean;
  /** A representative group used to show a live preview of each template. */
  sample?: ImageGroup;
  onSave: (set: NamingRuleSet) => void;
  onDelete?: () => void;
  onClose: () => void;
}

export function NamingRuleEditor({
  initial,
  isNew,
  sample,
  onSave,
  onDelete,
  onClose,
}: NamingRuleEditorProps) {
  const [name, setName] = useState(initial.name);
  const [rules, setRules] = useState<NamingRule[]>(() => initial.rules.map((r) => ({ ...r })));
  const inputRefs = useRef<Map<string, HTMLInputElement>>(new Map());

  function updateRule(id: string, template: string) {
    setRules((prev) => prev.map((r) => (r.id === id ? { ...r, template } : r)));
  }

  function addRule() {
    setRules((prev) => [...prev, { id: makeRuleId(), template: "" }]);
  }

  function removeRule(id: string) {
    setRules((prev) => prev.filter((r) => r.id !== id));
  }

  // Insert a `<token>` at the input's caret (or append if it isn't focused),
  // then restore the caret just after the inserted text.
  function insertToken(rule: NamingRule, token: string) {
    const snippet = `<${token}>`;
    const el = inputRefs.current.get(rule.id);
    const start = el?.selectionStart ?? rule.template.length;
    const end = el?.selectionEnd ?? rule.template.length;
    const next = rule.template.slice(0, start) + snippet + rule.template.slice(end);
    updateRule(rule.id, next);
    if (el) {
      const caret = start + snippet.length;
      requestAnimationFrame(() => {
        el.focus();
        el.setSelectionRange(caret, caret);
      });
    }
  }

  const validation = useMemo<string | null>(() => {
    if (!name.trim()) return "Name is required.";
    if (rules.length === 0) return "Add at least one rule.";
    if (rules.every((r) => !r.template.trim())) return "At least one rule needs a template.";
    return null;
  }, [name, rules]);

  function handleSave() {
    if (validation) return;
    const cleaned = rules
      .map((r) => ({ id: r.id, template: r.template.trim() }))
      .filter((r) => r.template !== "");
    onSave({ id: initial.id, name: name.trim(), rules: cleaned });
  }

  const footer = (
    <>
      <span className="review-config-validation modal-footer-spacer">{validation ?? ""}</span>
      {onDelete && !initial.builtIn && (
        <button
          type="button"
          className="btn btn-danger"
          onClick={() => {
            if (confirm(`Delete rule set "${initial.name}"?`)) onDelete();
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
      title={isNew ? "New rule set" : "Edit rule set"}
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
          placeholder="e.g. Subtitle-first, Short + name…"
        />
      </label>

      <div className="review-config-section-header">
        <span className="review-config-label">Rules</span>
        <button type="button" className="btn btn-secondary btn-small" onClick={addRule}>
          + Add rule
        </button>
      </div>

      <p className="naming-rule-tokens-hint">
        Insert a token to pull from the group's metadata:
        {NAMING_TOKENS.map((t) => (
          <code key={t} className="naming-token">{`<${t}>`}</code>
        ))}
      </p>

      <ol className="naming-rule-list">
        {rules.map((rule, i) => {
          const shortcut = shortcutForSlot(i);
          const preview = sample ? renderTemplate(rule.template, sample) : "";
          return (
            <li key={rule.id} className="naming-rule-row">
              <div className="naming-rule-head">
                <span className="review-config-shortcut" title="Number-key shortcut">
                  {shortcut ?? "—"}
                </span>
                <input
                  type="text"
                  className="review-config-input naming-rule-input"
                  value={rule.template}
                  ref={(el) => {
                    if (el) inputRefs.current.set(rule.id, el);
                    else inputRefs.current.delete(rule.id);
                  }}
                  onChange={(e) => updateRule(rule.id, e.target.value)}
                  placeholder="e.g. <subtitle> : <title>"
                />
                <div className="review-config-row-actions">
                  <button
                    type="button"
                    className="btn-icon btn-icon-danger"
                    onClick={() => removeRule(rule.id)}
                    disabled={rules.length === 1}
                    aria-label="Remove rule"
                    title="Remove rule"
                  >
                    ×
                  </button>
                </div>
              </div>
              <div className="naming-rule-token-buttons">
                {NAMING_TOKENS.map((t) => (
                  <button
                    key={t}
                    type="button"
                    className="btn btn-secondary btn-small naming-token-btn"
                    onClick={() => insertToken(rule, t)}
                    title={`Insert <${t}>`}
                  >
                    {`<${t}>`}
                  </button>
                ))}
              </div>
              {sample && (
                <div className={cn("naming-rule-preview", !preview && "naming-rule-preview-empty")}>
                  {preview || "(empty)"}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </Modal>
  );
}
