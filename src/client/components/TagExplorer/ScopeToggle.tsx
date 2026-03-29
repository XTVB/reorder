import React from "react";
import { cn } from "../../utils/helpers.ts";

interface ScopeToggleProps {
  scope: "all" | "ungrouped";
  onChange: (scope: "all" | "ungrouped") => void;
  resultCount: number;
}

export function ScopeToggle({ scope, onChange, resultCount }: ScopeToggleProps) {
  return (
    <div className="scope-toggle">
      <button
        className={cn("scope-toggle-btn", scope === "ungrouped" && "scope-active")}
        onClick={() => onChange("ungrouped")}
      >
        Ungrouped{scope === "ungrouped" ? ` (${resultCount})` : ""}
      </button>
      <button
        className={cn("scope-toggle-btn", scope === "all" && "scope-active")}
        onClick={() => onChange("all")}
      >
        All{scope === "all" ? ` (${resultCount})` : ""}
      </button>
    </div>
  );
}
