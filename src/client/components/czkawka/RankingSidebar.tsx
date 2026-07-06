import { useCallback, useRef, useState } from "react";
import type {
  ConditionType,
  GuardAttribute,
  GuardConfig,
  GuardMode,
  GuardOp,
  RankingCondition,
} from "../../stores/czkawkaRanking.ts";
import {
  CONDITION_TYPES,
  GUARD_ATTRS,
  GUARD_MODES,
  GUARD_OPS,
  makeCondition,
  useCzkawkaRankingStore,
} from "../../stores/czkawkaRanking.ts";

function GuardRow({ guard, onChange }: { guard: GuardConfig; onChange: (g: GuardConfig) => void }) {
  return (
    <div className="ranking-guard-row">
      <label className="ranking-guard-toggle">
        <input
          type="checkbox"
          checked={guard.enabled}
          onChange={(e) => onChange({ ...guard, enabled: e.target.checked })}
        />
      </label>
      <select
        className="ranking-guard-mode"
        value={guard.mode}
        onChange={(e) => onChange({ ...guard, mode: e.target.value as GuardMode })}
        disabled={!guard.enabled}
      >
        {GUARD_MODES.map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>
      <select
        className="ranking-guard-attr"
        value={guard.attribute}
        onChange={(e) => onChange({ ...guard, attribute: e.target.value as GuardAttribute })}
        disabled={!guard.enabled}
      >
        {GUARD_ATTRS.map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>
      <select
        className="ranking-guard-op"
        value={guard.op}
        onChange={(e) => onChange({ ...guard, op: e.target.value as GuardOp })}
        disabled={!guard.enabled}
      >
        {GUARD_OPS.map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>
      <input
        type="text"
        className="ranking-guard-value"
        value={guard.value}
        onChange={(e) => onChange({ ...guard, value: e.target.value })}
        disabled={!guard.enabled}
        placeholder="val"
      />
      <span className="ranking-guard-fallback">else skip</span>
    </div>
  );
}

function ConditionRow({
  condition,
  index,
  onUpdate,
  onRemove,
  onDragStart,
  onDragOver,
  onDragEnd,
  onDrop,
}: {
  condition: RankingCondition;
  index: number;
  onUpdate: (c: RankingCondition) => void;
  onRemove: () => void;
  onDragStart: (e: React.DragEvent, idx: number) => void;
  onDragOver: (e: React.DragEvent, idx: number) => void;
  onDragEnd: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent, idx: number) => void;
}) {
  const typeNeedsPattern = condition.type === "path_prefix" || condition.type === "path_regex";

  const configSelectOptions = (() => {
    switch (condition.type) {
      case "resolution":
        return [
          ["higher", "Higher"],
          ["lower", "Lower"],
        ];
      case "file_size":
        return [
          ["bigger", "Bigger"],
          ["smaller", "Smaller"],
        ];
      case "filename_order":
        return [
          ["lower", "Lower"],
          ["higher", "Higher"],
        ];
      default:
        return [];
    }
  })();

  return (
    <div
      className="ranking-condition-row"
      draggable
      onDragStart={(e) => onDragStart(e, index)}
      onDragOver={(e) => onDragOver(e, index)}
      onDragEnd={onDragEnd}
      onDrop={(e) => onDrop(e, index)}
    >
      <div className="ranking-condition-main">
        <span className="ranking-drag-handle">☰</span>
        <label className="ranking-checkbox">
          <input
            type="checkbox"
            checked={condition.enabled}
            onChange={(e) => onUpdate({ ...condition, enabled: e.target.checked })}
          />
        </label>
        <span className="ranking-label">{condition.label}</span>
        {typeNeedsPattern ? (
          <input
            type="text"
            className="ranking-config-input"
            value={condition.config.pattern ?? ""}
            onChange={(e) =>
              onUpdate({ ...condition, config: { ...condition.config, pattern: e.target.value } })
            }
            placeholder="pattern"
          />
        ) : (
          <select
            className="ranking-config-select"
            value={condition.config.prefer ?? "higher"}
            onChange={(e) =>
              onUpdate({
                ...condition,
                config: {
                  ...condition.config,
                  prefer: e.target.value as "higher" | "lower" | "bigger" | "smaller",
                },
              })
            }
          >
            {configSelectOptions.map(([v, l]) => (
              <option key={v} value={v}>
                {l}
              </option>
            ))}
          </select>
        )}
        <span className="ranking-priority">{index + 1}</span>
        <button className="ranking-remove-btn" onClick={onRemove} title="Remove condition">
          ✕
        </button>
      </div>
      <GuardRow guard={condition.guard} onChange={(g) => onUpdate({ ...condition, guard: g })} />
    </div>
  );
}

function ConditionList({
  conditions,
  onChange,
  title,
}: {
  conditions: RankingCondition[];
  onChange: (c: RankingCondition[]) => void;
  title: string;
}) {
  const [newType, setNewType] = useState<ConditionType>("path_prefix");
  const dragIndex = useRef<number | null>(null);

  const handleDragStart = useCallback((e: React.DragEvent, idx: number) => {
    dragIndex.current = idx;
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", String(idx));
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent, _idx: number) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent, idx: number) => {
      e.preventDefault();
      const fromIdx = dragIndex.current;
      if (fromIdx === null || fromIdx === idx) return;
      const updated = [...conditions];
      const [moved] = updated.splice(fromIdx, 1);
      updated.splice(idx, 0, moved!);
      onChange(updated);
      dragIndex.current = null;
    },
    [conditions, onChange],
  );

  const handleDragEnd = useCallback(() => {
    dragIndex.current = null;
  }, []);

  const handleUpdate = useCallback(
    (updated: RankingCondition) => {
      const next = conditions.map((c) => (c.id === updated.id ? updated : c));
      onChange(next);
    },
    [conditions, onChange],
  );

  const handleRemove = useCallback(
    (id: string) => {
      const next = conditions.filter((c) => c.id !== id);
      onChange(next);
    },
    [conditions, onChange],
  );

  const handleAdd = useCallback(() => {
    const next = [...conditions, makeCondition(newType)];
    onChange(next);
  }, [conditions, onChange, newType]);

  return (
    <div className="ranking-section">
      <div className="ranking-section-title">{title}</div>
      <div className="ranking-condition-list">
        {conditions.map((c, i) => (
          <ConditionRow
            key={c.id}
            condition={c}
            index={i}
            onUpdate={handleUpdate}
            onRemove={() => handleRemove(c.id)}
            onDragStart={handleDragStart}
            onDragOver={handleDragOver}
            onDragEnd={handleDragEnd}
            onDrop={handleDrop}
          />
        ))}
      </div>
      <div className="ranking-add-row">
        <select value={newType} onChange={(e) => setNewType(e.target.value as ConditionType)}>
          {CONDITION_TYPES.map(([v, l]) => (
            <option key={v} value={v}>
              {l}
            </option>
          ))}
        </select>
        <button className="btn btn-sm" onClick={handleAdd}>
          Add condition
        </button>
      </div>
    </div>
  );
}

export function RankingSidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const contentConditions = useCzkawkaRankingStore((s) => s.contentConditions);
  const targetConditions = useCzkawkaRankingStore((s) => s.targetConditions);
  const setContentConditions = useCzkawkaRankingStore((s) => s.setContentConditions);
  const setTargetConditions = useCzkawkaRankingStore((s) => s.setTargetConditions);

  if (!open) return null;

  return (
    <>
      <div className="ranking-overlay" onClick={onClose} />
      <div className="ranking-sidebar">
        <div className="ranking-sidebar-header">
          <span className="ranking-sidebar-title">Ranking Rules</span>
          <button className="ranking-close-btn" onClick={onClose} title="Close">
            ✕
          </button>
        </div>
        <div className="ranking-sidebar-body">
          <p className="ranking-help">
            Top condition wins; ties fall through to the next. Content picks which image's
            <em> bytes</em> survive (Y/W); target picks which <em>filename</em> survives (W).
          </p>
          <ConditionList
            conditions={contentConditions}
            onChange={setContentConditions}
            title="Content Source (Y / W)"
          />
          <ConditionList
            conditions={targetConditions}
            onChange={setTargetConditions}
            title="Target Location (W)"
          />
        </div>
      </div>
    </>
  );
}
