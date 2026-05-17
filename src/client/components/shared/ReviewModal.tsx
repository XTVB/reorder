import { useState } from "react";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import type { ImageGroup } from "../../types.ts";
import { imageUrl, reorderImagesByGroups } from "../../utils/helpers.ts";
import {
  colorForId,
  type ReviewCategory,
  reviewColorVar,
  reviewConfigStore,
} from "../../utils/reviewConfigs.ts";
import { GroupingSortModal, type SortContext } from "./GroupingSortModal.tsx";

interface ReviewModalProps {
  onClose: () => void;
}

export function ReviewModal({ onClose }: ReviewModalProps) {
  const [snapshot] = useState<ImageGroup[]>(() =>
    useGroupStore.getState().groups.map((g) => ({ ...g, images: g.images.slice() })),
  );

  function openGroupLightbox(groupImages: string[], index: number) {
    const imageMap = useImageStore.getState().imageMap;
    const items = groupImages.filter((fn) => imageMap.has(fn));
    if (items.length === 0) return;
    useLightboxStore.getState().openLightbox(items, index);
  }

  function applyOrder({ items, config, statuses, subs }: SortContext<ImageGroup>) {
    if (items.length === 0) {
      onClose();
      return;
    }

    const effectiveCategoryId = (id: string): string =>
      statuses.get(id) ?? config.defaultCategoryId;

    const buckets = new Map<string, ImageGroup[]>();
    for (const c of config.categories) buckets.set(c.id, []);
    for (const g of items) buckets.get(effectiveCategoryId(g.id))?.push(g);

    const sortBucket = (cat: ReviewCategory, gs: ImageGroup[]): ImageGroup[] => {
      if (cat.subcategories.length === 0) return gs;
      const subRank = new Map(cat.subcategories.map((s, i) => [s.id, i] as const));
      const fallbackId = cat.defaultSubcategoryId ?? null;
      // Unassigned groups sort to the default sub's rank, else after all subs.
      const fallbackRank =
        fallbackId !== null && subRank.has(fallbackId)
          ? subRank.get(fallbackId)!
          : cat.subcategories.length;
      return gs
        .map((g, originalIdx) => {
          const sa = subs.get(g.id);
          const rank =
            sa && sa.categoryId === cat.id && subRank.has(sa.subId)
              ? subRank.get(sa.subId)!
              : fallbackRank;
          return { g, rank, originalIdx };
        })
        .sort((a, b) => a.rank - b.rank || a.originalIdx - b.originalIdx)
        .map(({ g }) => g);
    };

    const newOrder: ImageGroup[] = [];
    for (const cat of config.categories) {
      newOrder.push(...sortBucket(cat, buckets.get(cat.id) ?? []));
    }

    const { images, imageMap, setImages } = useImageStore.getState();
    setImages(reorderImagesByGroups(images, imageMap, newOrder));
    useGroupStore.getState().updateGroups(() => newOrder);

    onClose();
  }

  return (
    <GroupingSortModal<ImageGroup>
      store={reviewConfigStore}
      items={snapshot}
      getId={(g) => g.id}
      getName={(g) => g.name}
      defaultCategoryId={(cfg) => cfg.defaultCategoryId}
      terms={{ group: "category", sub: "subcategory" }}
      titleText={(b) => (b ? `Refine ${b}` : "Review Groups")}
      emptyText={(b) => (b ? `No groups in ${b}.` : "No groups to review.")}
      subBannerLabel={(label) => (
        <>
          Refining <strong>{label}</strong>
        </>
      )}
      chipTitle={(label, isActive, hasSubs) =>
        isActive
          ? `Exit ${label}`
          : hasSubs
            ? `Refine ${label} ordering`
            : `Open ${label} (press n to add a subcategory)`
      }
      tailHint="click chip to refine"
      renderSubtitle={(group, info) => (
        <>
          {group.images.length} image{group.images.length === 1 ? "" : "s"}
          {info.inBucket && info.category && (
            <>
              {" · "}
              <span
                className="review-single-tag"
                style={reviewColorVar(
                  colorForId(info.category.id, info.config.categories.indexOf(info.category)),
                )}
              >
                {info.category.label}
              </span>
            </>
          )}
        </>
      )}
      renderMedia={(group) => (
        <div className="review-single-thumbs">
          {group.images.map((fn, i) => (
            <button
              type="button"
              key={fn}
              className="review-single-thumb"
              onClick={() => openGroupLightbox(group.images, i)}
              aria-label={`Open ${fn}`}
            >
              <img src={imageUrl(fn)} alt="" loading="lazy" draggable={false} />
            </button>
          ))}
        </div>
      )}
      apply={{
        describe: ({ items }) => ({ label: "Apply Order", disabled: items.length === 0 }),
        run: applyOrder,
      }}
      onClose={onClose}
    />
  );
}
