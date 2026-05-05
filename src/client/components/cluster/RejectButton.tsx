export function RejectButton({
  filename,
  onReject,
}: {
  filename: string;
  onReject: (filename: string) => void;
}) {
  return (
    <button
      type="button"
      className="thumb-reject-btn"
      onClick={(e) => {
        e.stopPropagation();
        onReject(filename);
      }}
      title="Never suggest this image for this group"
      aria-label={`Reject ${filename} from this group`}
    >
      ×
    </button>
  );
}
