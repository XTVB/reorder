import type React from "react";
import { useEffect } from "react";
import { useLightboxStore } from "../../stores/core/lightboxStore.ts";
import { cn } from "../../utils/helpers.ts";

export function Modal({
  title,
  onClose,
  children,
  footer,
  className,
  headerClassName,
  bodyClassName,
  closeOnEscape = true,
}: {
  title: React.ReactNode;
  onClose: () => void;
  children: React.ReactNode;
  footer: React.ReactNode;
  className?: string;
  headerClassName?: string;
  bodyClassName?: string;
  /** Opt out when the caller runs its own Escape handling (e.g. backing out
   * of a sub-view first) — otherwise both listeners fire on one keypress. */
  closeOnEscape?: boolean;
}) {
  // Escape closes the modal — unless the lightbox is layered on top of it.
  useEffect(() => {
    if (!closeOnEscape) return;
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      if (useLightboxStore.getState().open) return;
      onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, closeOnEscape]);

  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className={cn("modal", className)}>
        <div className={cn("modal-header", headerClassName)}>{title}</div>
        <div className={cn("modal-body", bodyClassName)}>{children}</div>
        <div className="modal-footer">{footer}</div>
      </div>
    </div>
  );
}
