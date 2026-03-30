import React from "react";
import type { ImageTagData } from "../../types.ts";
import { imageUrl, formatCategoryName, formatClothingValue } from "../../utils/helpers.ts";
import { categoryOrderIndex } from "../../utils/tagIndex.ts";
import { TagChip } from "./TagChip.tsx";

interface ImageDetailPanelProps {
  tagData: ImageTagData;
  onTagClick: (category: string, value: string, e: React.MouseEvent) => void;
  onClothingClick: (piece: string, color: string) => void;
  onClose: () => void;
  onOpenLightbox: (filename: string) => void;
}

export function ImageDetailPanel({ tagData, onTagClick, onClothingClick, onClose, onOpenLightbox }: ImageDetailPanelProps) {
  const categories = Object.entries(tagData.tags);
  categories.sort((a, b) => categoryOrderIndex(a[0]) - categoryOrderIndex(b[0]));

  return (
    <div className="image-detail-panel">
      <div className="image-detail-header">
        <img
          className="image-detail-thumb"
          src={imageUrl(tagData.filename)}
          alt=""
          onClick={() => onOpenLightbox(tagData.filename)}
          style={{ cursor: "zoom-in" }}
        />
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
        {categories.length === 0 && tagData.clothing.length === 0 && (
          <div style={{ color: "var(--text-muted)", fontSize: 13 }}>No tags for this image</div>
        )}

        {tagData.clothing.length > 0 && (
          <div>
            <div className="image-detail-category">Clothing Items</div>
            <div className="clothing-items-list">
              {tagData.clothing.map((item, i) => (
                <div key={i} className="clothing-item-row">
                  <span className="clothing-item-piece">{formatClothingValue(item.piece)}</span>
                  <div className="clothing-item-chips">
                    {item.colors.map((color) => (
                      <span
                        key={`${item.piece}|${color}`}
                        className="tag-chip clothing-item-chip"
                        onClick={() => onClothingClick(item.piece, color)}
                        title={`Filter: ${formatClothingValue(`${item.piece}|${color}`)}`}
                      >
                        {formatClothingValue(`${item.piece}|${color}`)}
                      </span>
                    ))}
                    {item.colors.length === 0 && (
                      <span
                        className="tag-chip clothing-item-chip"
                        onClick={() => onClothingClick(item.piece, "")}
                        title={`Filter: ${formatClothingValue(item.piece)}`}
                      >
                        {formatClothingValue(item.piece)}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
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
