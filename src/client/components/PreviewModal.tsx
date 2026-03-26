import React from "react";
import type { RenameMapping } from "../types.ts";
import { Modal } from "./Modal.tsx";

interface PreviewModalProps {
  renames: RenameMapping[];
  onClose: () => void;
  onConfirm: () => void;
}

export function PreviewModal({ renames, onClose, onConfirm }: PreviewModalProps) {
  return (
    <Modal title="Preview Renames" onClose={onClose} footer={
      <>
        <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
        <button className="btn btn-primary" onClick={onConfirm}>Confirm Rename</button>
      </>
    }>
      <table className="rename-table">
        <thead>
          <tr>
            <th>Current Name</th>
            <th className="rename-arrow"></th>
            <th>New Name</th>
          </tr>
        </thead>
        <tbody>
          {renames.map((r) => (
            <tr key={r.from}>
              <td>{r.from}</td>
              <td className="rename-arrow">→</td>
              <td>{r.to}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Modal>
  );
}
