import { TrashIcon } from "./TrashIcon.tsx";

export function TrashBadge({ size = 14 }: { size?: number }) {
  return (
    <span className="trash-badge" aria-label="Marked for deletion" title="Marked for deletion">
      <TrashIcon size={size} />
    </span>
  );
}
