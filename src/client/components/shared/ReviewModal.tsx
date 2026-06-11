import { useState } from "react";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import type { ImageGroup } from "../../types.ts";
import { groupsInGalleryOrder, withLockedGroupsInPlace } from "../../utils/groups.ts";
import { imageUrl, reorderImagesByGroups } from "../../utils/helpers.ts";
import {
  assignmentFromTags,
  colorForId,
  configOwnedTags,
  explicitGroupTags,
  type ReviewCategory,
  type ReviewConfig,
  reviewColorVar,
  reviewConfigStore,
} from "../../utils/reviewConfigs.ts";
import { beginSortTransition } from "../../utils/sortFlip.ts";
import {
  GroupingSortModal,
  type SortContext,
  type SortState,
  type SubAssignment,
} from "./GroupingSortModal.tsx";

interface ReviewModalProps {
  onClose: () => void;
}

export function ReviewModal({ onClose }: ReviewModalProps) {
  const [snapshot] = useState<ImageGroup[]>(() =>
    useGroupStore.getState().groups.map((g) => ({ ...g, images: g.images.slice() })),
  );

  // Pre-seed a config's assignments from each group's persisted tags so groups
  // already bucketed by a previous Apply open on their category/subcategory.
  function initialStateFor(config: ReviewConfig): SortState {
    const statuses = new Map<string, string>();
    const subs = new Map<string, SubAssignment>();
    for (const g of snapshot) {
      const a = assignmentFromTags(config, g.tags);
      if (!a) continue;
      statuses.set(g.id, a.categoryId);
      if (a.sub) subs.set(g.id, a.sub);
    }
    return { statuses, subs };
  }

  function groupLightboxItems(groupImages: string[]): string[] {
    const imageMap = useImageStore.getState().imageMap;
    return groupImages.filter((fn) => imageMap.has(fn));
  }

  function openGroupLightbox(groupImages: string[], index: number) {
    const items = groupLightboxItems(groupImages);
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

    // Replace this config's slice of each group's tags while keeping tags owned
    // by other configs (see configOwnedTags / explicitGroupTags).
    const owned = configOwnedTags(config);
    const tagged = newOrder.map((g) => {
      const newTags = explicitGroupTags(config, statuses, subs, g.id);
      const kept = (g.tags ?? []).filter((t) => !owned.has(t));
      const finalTags = Array.from(new Set([...kept, ...newTags]));
      if (finalTags.length === 0) {
        return g.tags && g.tags.length > 0 ? { ...g, tags: undefined } : g;
      }
      return { ...g, tags: finalTags };
    });

    beginSortTransition();
    const { images, imageMap, setImages } = useImageStore.getState();
    // Locked groups keep their current gallery slot; the categorised order
    // fills in around them.
    const finalOrder = withLockedGroupsInPlace(
      groupsInGalleryOrder(useGroupStore.getState().groups, images),
      tagged,
    );
    setImages(reorderImagesByGroups(images, imageMap, finalOrder));
    useGroupStore.getState().updateGroups(() => finalOrder);

    onClose();
  }

  return (
    <GroupingSortModal<ImageGroup>
      store={reviewConfigStore}
      items={snapshot}
      getId={(g) => g.id}
      getName={(g) => g.name}
      initialStateFor={initialStateFor}
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
      getLightboxTarget={(group) => {
        const items = groupLightboxItems(group.images);
        return items.length > 0 ? { filenames: items, index: 0 } : null;
      }}
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
