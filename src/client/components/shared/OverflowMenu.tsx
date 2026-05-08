import { createContext, type ReactNode, useContext, useMemo, useRef, useState } from "react";
import { useDismissOnOutside } from "../../hooks/useDismissOnOutside.ts";

interface MenuContextValue {
  close: () => void;
  checkable: boolean;
}
const MenuContext = createContext<MenuContextValue | null>(null);

interface MenuProps {
  label: string;
  align?: "left" | "right";
  /** When true, every item reserves a leading column for a checkmark. */
  checkable?: boolean;
  children: ReactNode;
}

export function OverflowMenu({ label, align = "left", checkable = false, children }: MenuProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  useDismissOnOutside(containerRef, open, () => setOpen(false));

  const ctx = useMemo<MenuContextValue>(
    () => ({ close: () => setOpen(false), checkable }),
    [checkable],
  );

  return (
    <div className="overflow-menu" ref={containerRef}>
      <button
        className="overflow-menu-trigger"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label={label}
        title={label}
      >
        <span aria-hidden="true">⋮</span>
      </button>
      {open && (
        <div
          className={`overflow-menu-panel${align === "right" ? " overflow-menu-panel--right" : ""}`}
          role="menu"
        >
          <MenuContext.Provider value={ctx}>{children}</MenuContext.Provider>
        </div>
      )}
    </div>
  );
}

interface ItemProps {
  onClick: () => void;
  disabled?: boolean;
  /** Only meaningful when the parent OverflowMenu has checkable=true. */
  checked?: boolean;
  danger?: boolean;
  title?: string;
  /** Close the menu before invoking onClick (e.g. handler opens a confirm dialog). */
  closeBeforeClick?: boolean;
  /** Don't auto-close the menu after onClick. */
  keepOpen?: boolean;
  children: ReactNode;
}

export function OverflowMenuItem({
  onClick,
  disabled,
  checked,
  danger,
  title,
  closeBeforeClick,
  keepOpen,
  children,
}: ItemProps) {
  const ctx = useContext(MenuContext);
  function handleClick() {
    if (closeBeforeClick) {
      ctx?.close();
      onClick();
      return;
    }
    onClick();
    if (!keepOpen) ctx?.close();
  }
  return (
    <button
      className={`overflow-menu-item${danger ? " overflow-menu-danger" : ""}`}
      onClick={handleClick}
      disabled={disabled}
      title={title}
    >
      {ctx?.checkable && <span className="overflow-menu-check">{checked ? "✓" : ""}</span>}
      {children}
    </button>
  );
}

export function OverflowMenuDivider() {
  return <div className="overflow-menu-divider" />;
}
