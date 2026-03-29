import React from "react";
import { useUIStore } from "../stores/uiStore.ts";

export function Toast() {
  const toast = useUIStore((s) => s.toast);
  if (!toast) return null;
  return <div className={`toast toast-${toast.type}`}>{toast.message}</div>;
}
