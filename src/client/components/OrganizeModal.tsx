import { useUIStore } from "../stores/uiStore.ts";
import type { OrganizeMapping } from "../types.ts";
import { stripFolderNumber } from "../utils/helpers.ts";
import { Modal } from "./Modal.tsx";

interface OrganizeModalProps {
  mappings: OrganizeMapping[];
  onClose: () => void;
  onConfirm: () => void;
}

export function OrganizeModal({ mappings, onClose, onConfirm }: OrganizeModalProps) {
  const numbered = useUIStore((s) => s.numberedFolderPrefix);
  const setNumbered = useUIStore((s) => s.setNumberedFolderPrefix);

  const displayFolder = (folder: string) =>
    numbered ? folder : stripFolderNumber(folder) || folder;

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
      <div className="organize-options">
        <label className="organize-toggle">
          <input
            type="checkbox"
            checked={numbered}
            onChange={(e) => setNumbered(e.target.checked)}
          />
          <span>Number folder names (e.g. "001 - {"{name}"}")</span>
        </label>
      </div>
      <p className="organize-description">The following groups will be moved into subfolders:</p>
      {mappings.map((m) => (
        <div key={m.folder} className="organize-group">
          <div className="organize-folder">{displayFolder(m.folder)}/</div>
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
