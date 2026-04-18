import { type RefObject, useEffect, useRef } from "react";

/**
 * Close a popover/menu when the user clicks outside its container or presses Escape.
 * Listeners are only installed while `open` is true. `onDismiss` is captured via
 * ref so callers can pass an inline arrow without triggering listener churn.
 */
export function useDismissOnOutside(
  ref: RefObject<HTMLElement | null>,
  open: boolean,
  onDismiss: () => void,
) {
  const onDismissRef = useRef(onDismiss);
  onDismissRef.current = onDismiss;

  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onDismissRef.current();
    }
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onDismissRef.current();
    }
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [ref, open]);
}
