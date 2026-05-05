interface Props {
  size?: number;
  variant?: "plain" | "plus" | "minus";
}

export function TrashIcon({ size = 16, variant = "plain" }: Props) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" role="presentation">
      <path
        d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 11a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-11"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {variant === "plus" && (
        <path d="M12 11v6M9 14h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      )}
      {variant === "minus" && (
        <path d="M9.5 14h5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      )}
    </svg>
  );
}
