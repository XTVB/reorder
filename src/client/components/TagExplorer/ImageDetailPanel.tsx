import React from "react";
import type { ImageTagData } from "../../types.ts";
import { imageUrl, formatCategoryName } from "../../utils/helpers.ts";
import { categoryOrderIndex } from "../../utils/tagIndex.ts";
import { TagChip } from "./TagChip.tsx";

interface ImageDetailPanelProps {
  tagData: ImageTagData;
  onTagClick: (category: string, value: string, e: React.MouseEvent) => void;
  onClose: () => void;
}

export function ImageDetailPanel({ tagData, onTagClick, onClose }: ImageDetailPanelProps) {
  const categories = Object.entries(tagData.tags);
  // Add denormalized clothing fields
  if (tagData.clothing.length > 0) {
    const pieces = [...new Set(tagData.clothing.map((c) => c.piece))];
    const colors = [...new Set(tagData.clothing.flatMap((c) => c.colors))];
    const styles = [...new Set(tagData.clothing.flatMap((c) => c.styles))];
    if (pieces.length) categories.push(["clothing_piece", pieces]);
    if (colors.length) categories.push(["clothing_color", colors]);
    if (styles.length) categories.push(["clothing_style (items)", styles]);
  }

  categories.sort((a, b) => categoryOrderIndex(a[0]) - categoryOrderIndex(b[0]));

  return (
    <div className="image-detail-panel">
      <div className="image-detail-header">
        <img className="image-detail-thumb" src={imageUrl(tagData.filename)} alt="" />
        <div>
          <div className="image-detail-filename">{tagData.filename}</div>
          <button
            className="btn btn-secondary"
            style={{ padding: "2px 8px", fontSize: "11px", marginTop: 4 }}
            onClick={onClose}
          >
            Close
          </button>
        </div>
      </div>
      <div className="image-detail-tags">
        {categories.length === 0 && (
          <div style={{ color: "var(--text-muted)", fontSize: 13 }}>No tags for this image</div>
        )}
        {categories.map(([cat, values]) => {
          if (!Array.isArray(values) || values.length === 0) return null;
          return (
            <div key={cat}>
              <div className="image-detail-category">{formatCategoryName(cat)}</div>
              <div className="image-detail-values">
                {values.map((v: string) => (
                  <TagChip
                    key={v}
                    category={cat}
                    value={v}
                    onClick={(e) => onTagClick(cat, v, e)}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
