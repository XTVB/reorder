import { useToastStore } from "../../stores/core/toastStore.ts";

export function Toast() {
  const toast = useToastStore((s) => s.toast);
  if (!toast) return null;
  return <div className={`toast toast-${toast.type}`}>{toast.message}</div>;
}
