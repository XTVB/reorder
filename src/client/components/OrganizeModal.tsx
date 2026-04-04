import React from "react";
import type { OrganizeMapping } from "../types.ts";
import { Modal } from "./Modal.tsx";

interface OrganizeModalProps {
  mappings: OrganizeMapping[];
  onClose: () => void;
  onConfirm: () => void;
}

export function OrganizeModal({ mappings, onClose, onConfirm }: OrganizeModalProps) {
  return (
    <Modal
      title="Organize into Folders"
      onClose={onClose}
      footer={
        <>
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={onConfirm}>
            Confirm
          </button>
        </>
      }
    >
      <p className="organize-description">The following groups will be moved into subfolders:</p>
      {mappings.map((m) => (
        <div key={m.folder} className="organize-group">
          <div className="organize-folder">{m.folder}/</div>
          <div className="organize-files">
            {m.files.map((f) => (
              <div key={f.from} className="organize-file">
                {f.from === f.to ? (
                  f.to
                ) : (
                  <>
                    {f.from} <span className="rename-arrow">→</span> {f.to}
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
    </Modal>
  );
}
