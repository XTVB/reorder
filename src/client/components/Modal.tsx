import type React from "react";
import { cn } from "../utils/helpers.ts";

export function Modal({
  title,
  onClose,
  children,
  footer,
  className,
  headerClassName,
  bodyClassName,
}: {
  title: React.ReactNode;
  onClose: () => void;
  children: React.ReactNode;
  footer: React.ReactNode;
  className?: string;
  headerClassName?: string;
  bodyClassName?: string;
}) {
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
