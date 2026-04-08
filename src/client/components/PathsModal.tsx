import { Modal } from "./Modal.tsx";

interface PathsModalProps {
  pathsText: string;
  onClose: () => void;
  onCopy: () => void;
}

export function PathsModal({ pathsText, onClose, onCopy }: PathsModalProps) {
  return (
    <Modal
      title="Copy Paths"
      onClose={onClose}
      footer={
        <>
          <button className="btn btn-secondary" onClick={onClose}>
            Close
          </button>
          <button className="btn btn-primary" onClick={onCopy}>
            Copy
          </button>
        </>
      }
    >
      <pre className="paths-list">{pathsText}</pre>
    </Modal>
  );
}
