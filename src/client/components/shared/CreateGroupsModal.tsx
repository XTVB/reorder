import { useState } from "react";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { useToastStore } from "../../stores/core/toastStore.ts";
import { useGroupStore } from "../../stores/groupStore.ts";
import { useImageStore } from "../../stores/imageStore.ts";
import type { ImageGroup } from "../../types.ts";
import { appendNewGroups, groupedFilenameSet } from "../../utils/groups.ts";
import { fullImageUrl, reorderImagesByGroups } from "../../utils/helpers.ts";
import {
  colorForId,
  createGroupsConfigStore,
  type ReviewConfig,
  reviewColorVar,
} from "../../utils/reviewConfigs.ts";
import { GroupingSortModal, type SortContext } from "./GroupingSortModal.tsx";

interface CreateGroupsModalProps {
  onClose: () => void;
}

// One created ImageGroup per non-empty leaf bucket. A category with no
// subgroups is itself a leaf; otherwise each subgroup is a leaf, plus one more
// leaf for photos assigned to the category but no subgroup (grouped under the
// category's own label).
function buildNewGroups({ items, config, statuses, subs }: SortContext<string>): ImageGroup[] {
  const out: ImageGroup[] = [];
  for (const cat of config.categories) {
    if (cat.subcategories.length === 0) {
      const images = items.filter((fn) => statuses.get(fn) === cat.id);
      if (images.length > 0) out.push({ id: crypto.randomUUID(), name: cat.label, images });
      continue;
    }
    const unsubbed: string[] = [];
    for (const sub of cat.subcategories) {
      const images = items.filter((fn) => {
        const sa = subs.get(fn);
        return sa?.categoryId === cat.id && sa.subId === sub.id;
      });
      if (images.length > 0) {
        out.push({ id: crypto.randomUUID(), name: `${cat.label} - ${sub.label}`, images });
      }
    }
    for (const fn of items) {
      const sa = subs.get(fn);
      if (statuses.get(fn) === cat.id && sa?.categoryId !== cat.id) unsubbed.push(fn);
    }
    if (unsubbed.length > 0) {
      out.push({ id: crypto.randomUUID(), name: cat.label, images: unsubbed });
    }
  }
  return out;
}

export function CreateGroupsModal({ onClose }: CreateGroupsModalProps) {
  // Ungrouped photos at open time, in current image order.
  const [snapshot] = useState<string[]>(() => {
    const grouped = groupedFilenameSet(useGroupStore.getState().groups);
    return useImageStore
      .getState()
      .images.map((i) => i.filename)
      .filter((fn) => !grouped.has(fn));
  });

  function imageLightboxTarget(current: string): { filenames: string[]; index: number } | null {
    const imageMap = useImageStore.getState().imageMap;
    const items = snapshot.filter((fn) => imageMap.has(fn));
    const index = items.indexOf(current);
    return index >= 0 ? { filenames: items, index } : null;
  }

  function openImageLightbox(current: string) {
    const target = imageLightboxTarget(current);
    if (target) useLightboxStore.getState().openLightbox(target.filenames, target.index);
  }

  function applyCreate(ctx: SortContext<string>) {
    const newGroups = buildNewGroups(ctx);
    if (newGroups.length === 0) {
      onClose();
      return;
    }
    const existing = useGroupStore.getState().groups;
    const merged = appendNewGroups(existing, newGroups);
    const { images, imageMap, setImages } = useImageStore.getState();
    setImages(reorderImagesByGroups(images, imageMap, merged));
    useGroupStore.getState().updateGroups(() => merged);
    useToastStore
      .getState()
      .showToast(
        `Created ${newGroups.length} group${newGroups.length === 1 ? "" : "s"}`,
        "success",
      );
    onClose();
  }

  const subgroupLabel = (config: ReviewConfig, sub: { categoryId: string; subId: string }) =>
    config.categories
      .find((c) => c.id === sub.categoryId)
      ?.subcategories.find((s) => s.id === sub.subId)?.label;

  return (
    <GroupingSortModal<string>
      store={createGroupsConfigStore}
      items={snapshot}
      modalClassName="cg-modal"
      getId={(fn) => fn}
      getName={(fn) => fn}
      defaultCategoryId={() => null}
      terms={{ group: "group", sub: "subgroup" }}
      titleText={(b) => (b ? `Assign subgroups · ${b}` : "Create Groups")}
      emptyText={(b) => (b ? `No photos assigned to ${b}.` : "No ungrouped photos to sort.")}
      subBannerLabel={(label) => (
        <>
          Subgroups of <strong>{label}</strong>
        </>
      )}
      chipTitle={(label, isActive, hasSubs) =>
        isActive
          ? `Exit ${label}`
          : hasSubs
            ? `Assign subgroups for ${label}`
            : `Open ${label} (press n to add a subgroup)`
      }
      tailHint="click chip for subgroups"
      renderProgressExtra={({ items, statuses }) => {
        const unassigned = items.reduce((n, fn) => (statuses.has(fn) ? n : n + 1), 0);
        return (
          <span className="review-progress-total" title="Photos left unassigned (stay ungrouped)">
            {unassigned} unassigned
          </span>
        );
      }}
      renderSubtitle={(_fn, info) =>
        info.category ? (
          <span
            className="review-single-tag"
            style={reviewColorVar(
              colorForId(info.category.id, info.config.categories.indexOf(info.category)),
            )}
          >
            {info.category.label}
            {info.sub && info.category.subcategories.length > 0
              ? ` - ${subgroupLabel(info.config, info.sub) ?? "?"}`
              : ""}
          </span>
        ) : (
          "unassigned"
        )
      }
      getLightboxTarget={(fn) => imageLightboxTarget(fn)}
      renderMedia={(fn) => (
        <div className="cg-single-image">
          <button type="button" onClick={() => openImageLightbox(fn)} aria-label={`Open ${fn}`}>
            <img src={fullImageUrl(fn)} alt="" loading="lazy" draggable={false} />
          </button>
        </div>
      )}
      apply={{
        describe: (ctx) => {
          const n = buildNewGroups(ctx).length;
          return {
            label: n === 0 ? "Create Groups" : `Create ${n} Group${n === 1 ? "" : "s"}`,
            disabled: n === 0,
            title:
              n === 0
                ? "Assign at least one photo to a group"
                : `Create ${n} group${n === 1 ? "" : "s"}`,
          };
        },
        run: applyCreate,
      }}
      onClose={onClose}
    />
  );
}
