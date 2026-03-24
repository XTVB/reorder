import React from "react";

export function Modal({
  title,
  onClose,
  children,
  footer,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  footer: React.ReactNode;
}) {
  return (
    <div
      className="modal-backdrop"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="modal">
        <div className="modal-header">{title}</div>
        <div className="modal-body">{children}</div>
        <div className="modal-footer">{footer}</div>
      </div>
    </div>
  );
}
