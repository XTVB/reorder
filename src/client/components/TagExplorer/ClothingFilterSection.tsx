import React, { useState, useMemo } from "react";
import type { ActiveFilter, ClothingOption } from "../../types.ts";
import { formatClothingValue, cn } from "../../utils/helpers.ts";

interface ClothingFilterSectionProps {
  clothingOptions: ClothingOption[];
  filters: ActiveFilter[];
  onAdd: (piece: string, color: string) => void;
  onRemove: (value: string) => void;
}

export function ClothingFilterSection({
  clothingOptions,
  filters,
  onAdd,
  onRemove,
}: ClothingFilterSectionProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [selectedPiece, setSelectedPiece] = useState("");
  const [selectedColor, setSelectedColor] = useState("");

  const activeStructured = filters.filter((f) => f.category === "__clothing_structured");

  const pieceOptions = useMemo(
    () => clothingOptions.map((o) => o.piece).sort(),
    [clothingOptions],
  );

  const colorsForPiece = useMemo(() => {
    if (!selectedPiece) return [];
    const opt = clothingOptions.find((o) => o.piece === selectedPiece);
    return opt?.colors ?? [];
  }, [selectedPiece, clothingOptions]);

  function handleAdd() {
    if (!selectedPiece) return;
    onAdd(selectedPiece, selectedColor);
    setSelectedColor("");
  }

  if (clothingOptions.length === 0) return null;

  return (
    <div className="tag-category">
      <div className="tag-category-header" onClick={() => setCollapsed(!collapsed)}>
        <span>Clothing Filter</span>
        <span className={`tag-category-arrow ${collapsed ? "collapsed" : ""}`}>▼</span>
      </div>
      {!collapsed && (
        <div className="clothing-filter">
          {activeStructured.length > 0 && (
            <div className="clothing-filter-active">
              {activeStructured.map((f) => (
                <span key={f.value} className="tag-chip tag-chip-active-and">
                  {formatClothingValue(f.value)}
                  <button
                    className="tag-chip-remove"
                    onClick={() => onRemove(f.value)}
                  >
                    x
                  </button>
                </span>
              ))}
            </div>
          )}
          <div className="clothing-filter-controls">
            <select
              className="clothing-filter-select"
              value={selectedPiece}
              onChange={(e) => { setSelectedPiece(e.target.value); setSelectedColor(""); }}
            >
              <option value="">Piece...</option>
              {pieceOptions.map((p) => (
                <option key={p} value={p}>{p.replace(/_/g, " ")}</option>
              ))}
            </select>
            {selectedPiece && colorsForPiece.length > 0 && (
              <select
                className="clothing-filter-select"
                value={selectedColor}
                onChange={(e) => setSelectedColor(e.target.value)}
              >
                <option value="">Any color</option>
                {colorsForPiece.map((c) => (
                  <option key={c} value={c}>{c.replace(/_/g, " ")}</option>
                ))}
              </select>
            )}
            <button
              className="btn btn-primary clothing-filter-add"
              onClick={handleAdd}
              disabled={!selectedPiece}
            >
              +
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
