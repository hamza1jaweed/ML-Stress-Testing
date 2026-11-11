import type { LucideIcon } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  icon: LucideIcon;
  /** Accent for the icon only — text always stays in ink tokens. */
  accent?: string;
}

export default function MetricCard({
  label,
  value,
  sub,
  icon: Icon,
  accent = "#8b949e",
}: MetricCardProps): JSX.Element {
  return (
    <div className="panel flex items-start gap-3 p-3.5">
      <Icon size={16} style={{ color: accent }} className="mt-0.5 shrink-0" aria-hidden />
      <div className="min-w-0">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-terminal-muted">
          {label}
        </p>
        <p className="mt-0.5 truncate font-mono text-lg tabular-nums text-terminal-ink">
          {value}
        </p>
        {sub && <p className="text-[11px] text-terminal-faint">{sub}</p>}
      </div>
    </div>
  );
}
