import { useMemo, useState } from "react";
import { useImageStore } from "../stores/imageStore.ts";
import { useTrashStore } from "../stores/trashStore.ts";
import { useUIStore } from "../stores/uiStore.ts";
import type { ImageInfo } from "../types.ts";
import { getErrorMessage, imageUrl } from "../utils/helpers.ts";
import { Lightbox } from "./Lightbox.tsx";
import { Modal } from "./Modal.tsx";

interface TrashModalProps {
  onClose: () => void;
}

export function TrashModal({ onClose }: TrashModalProps) {
  const markedIds = useTrashStore((s) => s.markedIds);
  const unmark = useTrashStore((s) => s.unmark);
  const clear = useTrashStore((s) => s.clear);
  const confirmDelete = useTrashStore((s) => s.confirmDelete);

  const imageMap = useImageStore((s) => s.imageMap);
  const fetchImages = useImageStore((s) => s.fetchImages);

  const showToast = useUIStore((s) => s.showToast);

  const [deleting, setDeleting] = useState(false);
  const [lightbox, setLightbox] = useState<{ images: ImageInfo[]; index: number } | null>(null);

  const items = useMemo<ImageInfo[]>(() => {
    const result: ImageInfo[] = [];
    for (const fn of markedIds) {
      const img = imageMap.get(fn);
      if (img) result.push(img);
    }
    return result;
  }, [markedIds, imageMap]);

  const count = items.length;

  async function handleConfirm() {
    if (count === 0 || deleting) return;
    setDeleting(true);
    try {
      const res = await confirmDelete();
      const warnings = res.warnings ?? [];
      const hasWarnings = warnings.length > 0;
      const baseMsg = `Moved ${res.deleted.length} file${res.deleted.length === 1 ? "" : "s"} to Trash`;
      const text = hasWarnings
        ? `${baseMsg} (${warnings.length} warning${warnings.length === 1 ? "" : "s"})`
        : baseMsg;
      showToast(text, hasWarnings ? "warning" : "success");
      await fetchImages();
      onClose();
    } catch (err) {
      showToast(getErrorMessage(err, "Delete failed"), "error");
    } finally {
      setDeleting(false);
    }
  }

  const title = (
    <>
      <span>Delete to Trash</span>
      <span className="trash-modal-count">
        {count} file{count === 1 ? "" : "s"}
      </span>
    </>
  );

  const footer = (
    <>
      <span className="modal-footer-spacer trash-footer-hint">
        Files are moved to macOS Trash — you can restore from Finder
      </span>
      <button
        type="button"
        className="btn btn-secondary"
        onClick={() => clear()}
        disabled={count === 0 || deleting}
        title="Remove all marks"
      >
        Clear marks
      </button>
      <button type="button" className="btn btn-secondary" onClick={onClose} disabled={deleting}>
        Cancel
      </button>
      <button
        type="button"
        className="btn btn-danger"
        onClick={handleConfirm}
        disabled={count === 0 || deleting}
      >
        {deleting ? "Deleting..." : `Delete ${count} file${count === 1 ? "" : "s"}`}
      </button>
    </>
  );

  return (
    <>
      <Modal
        title={title}
        onClose={onClose}
        footer={footer}
        className="trash-modal"
        headerClassName="trash-modal-header"
        bodyClassName="trash-modal-body"
      >
        {count === 0 ? (
          <div className="trash-empty">No files marked for deletion.</div>
        ) : (
          <div className="trash-grid">
            {items.map((img, i) => (
              <div className="trash-item" key={img.filename}>
                <button
                  type="button"
                  className="trash-thumb"
                  onClick={() => setLightbox({ images: items, index: i })}
                  aria-label={`Open ${img.filename}`}
                >
                  <img src={imageUrl(img.filename)} alt="" loading="lazy" draggable={false} />
                </button>
                <div className="trash-item-row">
                  <span className="trash-item-name" title={img.filename}>
                    {img.filename}
                  </span>
                  <button
                    type="button"
                    className="trash-item-unmark"
                    onClick={() => unmark([img.filename])}
                    title="Unmark"
                    aria-label={`Unmark ${img.filename}`}
                  >
                    ×
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Modal>
      {lightbox && (
        <Lightbox
          images={lightbox.images}
          initialIndex={lightbox.index}
          onClose={() => setLightbox(null)}
        />
      )}
    </>
  );
}
