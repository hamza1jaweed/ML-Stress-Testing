/**
 * Chart palette — validated dark-mode steps from the reference dataviz
 * palette. The first three categorical slots are all-pairs CVD-validated on
 * dark surfaces; slots 1–5 are adjacent-pair validated for line charts.
 * Regime identity is never carried by color alone: every chart ships a
 * legend with text labels.
 */

/** Categorical series slots (dark mode), fixed order — never cycled. */
export const SERIES: readonly string[] = [
  "#3987e5", // blue
  "#d95926", // orange
  "#199e70", // aqua
  "#c98500", // yellow
  "#d55181", // magenta
] as const;

/** Regime id -> color. Uses only the all-pairs-validated first three slots. */
export const REGIME_COLORS: Record<number, string> = {
  0: "#199e70", // Bull        (aqua, slot 3)
  1: "#d95926", // High Vol Bear (orange, slot 2)
  2: "#3987e5", // Consolidation (blue, slot 1)
};

/** Sequential blue ramp (magnitude) — heatmap cells, fan-chart bands. */
export const SEQ_BLUE: readonly string[] = [
  "#cde2fb",
  "#9ec5f4",
  "#6da7ec",
  "#3987e5",
  "#256abf",
  "#184f95",
  "#0d366b",
] as const;

/** Chrome / ink tokens (dark terminal theme). */
export const INK = {
  primary: "#f0efec",
  secondary: "#c3c2b7",
  muted: "#8b949e",
  grid: "#262c36",
  axis: "#3a4250",
  surface: "#14181f",
  page: "#0d0f13",
} as const;

export function formatCurrency(value: number): string {
  if (Math.abs(value) >= 1_000_000)
    return `$${(value / 1_000_000).toFixed(2)}M`;
  if (Math.abs(value) >= 1_000) return `$${(value / 1_000).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
}

export function formatPct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}
