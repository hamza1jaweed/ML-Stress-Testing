import { useMemo } from "react";
import type { CSSProperties } from "react";
import {
  Area,
  Bar,
  BarChart,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertTriangle,
  ArrowDownRight,
  Crosshair,
  Download,
  Flame,
  Gauge,
  GitBranch,
  TrendingDown,
} from "lucide-react";
import type { ScenarioInfo, StressTestResponse } from "../types";
import { INK, SEQ_BLUE, formatCurrency, formatPct } from "../palette";

export interface ShockState {
  equity: number; // simple return, e.g. -0.20
  bond: number;
  commodity: number;
  crypto: number;
}

export const ZERO_SHOCKS: ShockState = { equity: 0, bond: 0, commodity: 0, crypto: 0 };

interface StressCockpitProps {
  result: StressTestResponse | null;
  running: boolean;
  method: "parametric" | "bootstrap";
  onMethodChange: (m: "parametric" | "bootstrap") => void;
  scenarios: ScenarioInfo[];
  activeScenario: string | null;
  onScenarioChange: (key: string | null) => void;
  shocks: ShockState;
  onShocksChange: (s: ShockState) => void;
  horizonDays: number;
  onHorizonChange: (d: number) => void;
  regimeAware: boolean;
  onRegimeAwareChange: (enabled: boolean) => void;
  regimeTrained: boolean;
  regimeNames: Record<string, string> | null;
  correlationStress: number;
  onCorrelationStressChange: (v: number) => void;
}

/** Trigger a client-side JSON download of the current result. */
function downloadResult(result: StressTestResponse): void {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `stress-test-${result.method}-${Date.now()}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

const SHOCK_CONTROLS: {
  key: keyof ShockState;
  label: string;
  min: number;
  max: number;
  format: (v: number) => string;
}[] = [
  { key: "equity", label: "Equity shock", min: -50, max: 20, format: (v) => `${v}%` },
  {
    key: "bond",
    label: "Rate shock (duration-adjusted)",
    min: -20,
    max: 20,
    format: (v) => `${(-v * 10).toFixed(0)} bps ≈ ${v}% px`,
  },
  { key: "commodity", label: "Commodity shock", min: -40, max: 40, format: (v) => `${v}%` },
  { key: "crypto", label: "Crypto shock", min: -70, max: 50, format: (v) => `${v}%` },
];

/**
 * Stress-testing cockpit: Monte Carlo fan chart (5/25/50/75/95 percentile
 * bands on a single sequential hue), VaR/CVaR metric cards, and interactive
 * day-0 shock sliders. Results re-run automatically upstream (debounced).
 */
export default function StressCockpit(props: StressCockpitProps): JSX.Element {
  const {
    result,
    running,
    method,
    onMethodChange,
    scenarios,
    activeScenario,
    onScenarioChange,
    shocks,
    onShocksChange,
    horizonDays,
    onHorizonChange,
    regimeAware,
    onRegimeAwareChange,
    regimeTrained,
    regimeNames,
    correlationStress,
    onCorrelationStressChange,
  } = props;

  const corrStressAvailable = activeScenario === null && (regimeAware || method === "parametric");

  const fanData = useMemo(() => {
    if (!result) return [];
    const fan = result.fan_chart;
    const days = fan.days ?? [];
    return days.map((day, i) => ({
      day,
      p5: fan.p5[i],
      p25: fan.p25[i],
      p50: fan.p50[i],
      p75: fan.p75[i],
      p95: fan.p95[i],
      // Stacked band geometry: base is invisible, bands render as deltas.
      base: fan.p5[i],
      band5_25: fan.p25[i] - fan.p5[i],
      band25_75: fan.p75[i] - fan.p25[i],
      band75_95: fan.p95[i] - fan.p75[i],
    }));
  }, [result]);

  const metrics = result?.metrics ?? null;

  return (
    <section className="panel">
      <header className="panel-title justify-between">
        <span className="flex items-center gap-2">
          <Gauge size={14} aria-hidden />
          Stress-Testing Cockpit
        </span>
        <span className="flex items-center gap-2">
          {running && (
            <span className="animate-pulse text-[10px] normal-case tracking-normal text-series-1">
              simulating…
            </span>
          )}
          {result && (
            <button
              type="button"
              onClick={() => downloadResult(result)}
              title="Download full result as JSON"
              className="flex items-center gap-1 rounded border border-terminal-border px-2 py-0.5 text-[10px] normal-case tracking-normal text-terminal-muted hover:text-terminal-ink"
            >
              <Download size={11} aria-hidden />
              Export
            </button>
          )}
        </span>
      </header>

      <div className="grid gap-4 p-4 lg:grid-cols-[260px,1fr]">
        {/* ----- Controls column ----- */}
        <div className="space-y-5">
          <div>
            <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-terminal-muted">
              Mode
            </p>
            <div className="flex flex-wrap gap-1.5">
              {(["parametric", "bootstrap"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => {
                    onScenarioChange(null);
                    onRegimeAwareChange(false);
                    onMethodChange(m);
                  }}
                  className={`rounded border px-2 py-1 text-xs ${
                    activeScenario === null && !regimeAware && method === m
                      ? "border-series-1 text-terminal-ink"
                      : "border-terminal-border text-terminal-muted hover:text-terminal-ink"
                  }`}
                >
                  MC {m}
                </button>
              ))}
              <button
                type="button"
                disabled={!regimeTrained}
                onClick={() => {
                  onScenarioChange(null);
                  onRegimeAwareChange(!regimeAware);
                }}
                title={
                  regimeTrained
                    ? "Paths switch between fitted regime distributions via the Markov chain"
                    : "Train the regime model first"
                }
                className={`flex items-center gap-1 rounded border px-2 py-1 text-xs disabled:opacity-40 ${
                  regimeAware && activeScenario === null
                    ? "border-series-3 text-terminal-ink"
                    : "border-terminal-border text-terminal-muted hover:text-terminal-ink"
                }`}
              >
                <GitBranch size={11} aria-hidden />
                MC regime-aware
              </button>
            </div>
          </div>

          <div>
            <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-terminal-muted">
              Historical scenario replay
            </p>
            <div className="flex flex-col gap-1.5">
              {scenarios.map((s) => (
                <button
                  key={s.key}
                  type="button"
                  onClick={() => onScenarioChange(activeScenario === s.key ? null : s.key)}
                  title={s.description}
                  className={`rounded border px-2 py-1 text-left text-xs ${
                    activeScenario === s.key
                      ? "border-series-2 text-terminal-ink"
                      : "border-terminal-border text-terminal-muted hover:text-terminal-ink"
                  }`}
                >
                  <Flame size={11} className="mr-1 inline" aria-hidden />
                  {s.name}
                </button>
              ))}
            </div>
          </div>

          <div className={activeScenario ? "pointer-events-none opacity-40" : ""}>
            <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-terminal-muted">
              Day-0 shocks {activeScenario && "(disabled in replay)"}
            </p>
            <div className="space-y-3">
              {SHOCK_CONTROLS.map((ctl) => (
                <div key={ctl.key}>
                  <div className="mb-0.5 flex justify-between text-[11px]">
                    <span className="text-terminal-muted">{ctl.label}</span>
                    <span className="font-mono tabular-nums">
                      {ctl.format(Math.round(shocks[ctl.key] * 100))}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={ctl.min}
                    max={ctl.max}
                    step={1}
                    value={Math.round(shocks[ctl.key] * 100)}
                    onChange={(e) =>
                      onShocksChange({ ...shocks, [ctl.key]: Number(e.target.value) / 100 })
                    }
                    aria-label={ctl.label}
                  />
                </div>
              ))}
              <div>
                <div className="mb-0.5 flex justify-between text-[11px]">
                  <span className="text-terminal-muted">Horizon</span>
                  <span className="font-mono tabular-nums">{horizonDays}d</span>
                </div>
                <input
                  type="range"
                  min={21}
                  max={756}
                  step={21}
                  value={horizonDays}
                  onChange={(e) => onHorizonChange(Number(e.target.value))}
                  aria-label="Simulation horizon in trading days"
                />
              </div>
            </div>
          </div>

          <div className={corrStressAvailable ? "" : "pointer-events-none opacity-40"}>
            <div className="mb-0.5 flex justify-between text-[11px]">
              <span
                className="text-terminal-muted"
                title="Blends the correlation matrix toward 1 — in a crisis, diversification disappears. Parametric & regime-aware modes only."
              >
                Correlation breakdown
              </span>
              <span className="font-mono tabular-nums">
                {correlationStress === 0
                  ? "historical"
                  : correlationStress === 1
                    ? "full panic (ρ→1)"
                    : `${Math.round(correlationStress * 100)}%`}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={Math.round(correlationStress * 100)}
              onChange={(e) => onCorrelationStressChange(Number(e.target.value) / 100)}
              style={{ "--thumb-color": "#d55181" } as CSSProperties}
              aria-label="Correlation stress toward crisis matrix"
              disabled={!corrStressAvailable}
            />
          </div>
        </div>

        {/* ----- Chart + metrics column ----- */}
        <div className="min-w-0">
          {metrics && (
            <div className="mb-4 grid grid-cols-2 gap-2 xl:grid-cols-4">
              <MetricInline
                icon={TrendingDown}
                accent="#d95926"
                label="VaR 95%"
                value={formatCurrency(metrics.var_95)}
              />
              <MetricInline
                icon={AlertTriangle}
                accent="#d95926"
                label="VaR 99%"
                value={formatCurrency(metrics.var_99)}
              />
              <MetricInline
                icon={Crosshair}
                accent="#c98500"
                label="CVaR 95% (ES)"
                value={formatCurrency(metrics.cvar_95)}
              />
              <MetricInline
                icon={ArrowDownRight}
                accent="#c98500"
                label="Median Max DD"
                value={formatPct(metrics.max_drawdown_median)}
              />
            </div>
          )}

          <div className="h-72">
            {fanData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={fanData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
                  <XAxis
                    dataKey="day"
                    tick={{ fill: INK.muted, fontSize: 10 }}
                    tickLine={false}
                    axisLine={{ stroke: INK.axis }}
                    label={{
                      value: "trading days",
                      position: "insideBottomRight",
                      fill: INK.muted,
                      fontSize: 10,
                      dy: -2,
                    }}
                  />
                  <YAxis
                    tick={{ fill: INK.muted, fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                    width={56}
                    domain={["auto", "auto"]}
                    tickFormatter={(v: number) => formatCurrency(v)}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: INK.surface,
                      border: `1px solid ${INK.grid}`,
                      borderRadius: 6,
                      fontSize: 12,
                    }}
                    labelFormatter={(d) => `Day ${d}`}
                    formatter={(value: number | string) => formatCurrency(Number(value))}
                  />
                  {/* Invisible base + translucent single-hue bands (5–25, 25–75, 75–95) */}
                  <Area dataKey="base" stackId="fan" fill="transparent" stroke="none" isAnimationActive={false} tooltipType="none" />
                  <Area dataKey="band5_25" stackId="fan" fill={SEQ_BLUE[5]} fillOpacity={0.35} stroke="none" isAnimationActive={false} tooltipType="none" />
                  <Area dataKey="band25_75" stackId="fan" fill={SEQ_BLUE[3]} fillOpacity={0.4} stroke="none" isAnimationActive={false} tooltipType="none" />
                  <Area dataKey="band75_95" stackId="fan" fill={SEQ_BLUE[5]} fillOpacity={0.35} stroke="none" isAnimationActive={false} tooltipType="none" />
                  {/* Invisible lines expose exact percentile values to the tooltip */}
                  <Line dataKey="p95" name="95th pct" stroke="none" dot={false} activeDot={false} isAnimationActive={false} />
                  <Line dataKey="p75" name="75th pct" stroke="none" dot={false} activeDot={false} isAnimationActive={false} />
                  <Line
                    dataKey="p50"
                    name="Median"
                    stroke={INK.primary}
                    strokeWidth={1.8}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line dataKey="p25" name="25th pct" stroke="none" dot={false} activeDot={false} isAnimationActive={false} />
                  <Line dataKey="p5" name="5th pct" stroke="none" dot={false} activeDot={false} isAnimationActive={false} />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-terminal-faint">
                Generate data to run simulations
              </div>
            )}
          </div>

          {/* Band legend */}
          <div className="mt-2 flex flex-wrap items-center gap-4 text-[11px] text-terminal-muted">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-4 rounded-sm" style={{ backgroundColor: SEQ_BLUE[3], opacity: 0.6 }} aria-hidden />
              25th–75th percentile
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-4 rounded-sm" style={{ backgroundColor: SEQ_BLUE[5], opacity: 0.6 }} aria-hidden />
              5th–95th percentile
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4" style={{ backgroundColor: INK.primary }} aria-hidden />
              Median path
            </span>
            {metrics && (
              <span className="ml-auto font-mono tabular-nums">
                P(loss) {formatPct(metrics.prob_loss)} · worst{" "}
                {formatCurrency(metrics.worst_terminal)} · {result?.n_paths.toLocaleString()}{" "}
                paths
              </span>
            )}
          </div>

          {result && <TerminalHistogram result={result} />}

          {result?.scenario && (
            <p className="mt-3 rounded border border-terminal-border bg-terminal-bg p-2.5 text-[11px] leading-relaxed text-terminal-muted">
              <span className="font-semibold text-terminal-ink">{result.scenario.name}:</span>{" "}
              {result.scenario.description}
              {result.scenario.regime_occupancy && regimeNames && (
                <span className="mt-1 block font-mono tabular-nums">
                  time in regime:{" "}
                  {Object.entries(result.scenario.regime_occupancy)
                    .map(
                      ([id, share]) =>
                        `${regimeNames[id] ?? `R${id}`} ${formatPct(share, 0)}`,
                    )
                    .join(" · ")}
                </span>
              )}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

/**
 * Distribution of terminal portfolio values: losses (below initial value)
 * in orange, gains in blue — a polarity (diverging) encoding with a
 * reference line at the starting value.
 */
function TerminalHistogram({ result }: { result: StressTestResponse }): JSX.Element {
  const { bin_edges: edges, counts } = result.terminal_histogram;
  const data = counts.map((count, i) => ({
    center: (edges[i] + edges[i + 1]) / 2,
    count,
    isLoss: (edges[i] + edges[i + 1]) / 2 < result.initial_value,
  }));

  return (
    <div className="mt-4">
      <h3 className="mb-1.5 text-[11px] font-semibold uppercase tracking-widest text-terminal-muted">
        Terminal value distribution ({result.n_paths.toLocaleString()} paths)
      </h3>
      <div className="h-36">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 8 }} barCategoryGap={1}>
            <XAxis
              dataKey="center"
              tick={{ fill: INK.muted, fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: INK.axis }}
              tickFormatter={(v: number) => formatCurrency(v)}
              minTickGap={60}
            />
            <YAxis
              tick={{ fill: INK.muted, fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              width={40}
            />
            <Tooltip
              cursor={{ fill: INK.grid, fillOpacity: 0.4 }}
              contentStyle={{
                backgroundColor: INK.surface,
                border: `1px solid ${INK.grid}`,
                borderRadius: 6,
                fontSize: 12,
              }}
              labelFormatter={(v) => `≈ ${formatCurrency(Number(v))}`}
              formatter={(value: number | string) => [`${value} paths`, "Count"]}
            />
            <ReferenceLine
              x={result.initial_value}
              stroke={INK.primary}
              strokeDasharray="4 3"
            />
            <Bar dataKey="count" isAnimationActive={false} radius={[2, 2, 0, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.isLoss ? "#d95926" : "#3987e5"} fillOpacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1 flex gap-4 text-[11px] text-terminal-muted">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "#d95926" }} aria-hidden />
          Loss vs. initial
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "#3987e5" }} aria-hidden />
          Gain vs. initial
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4 border-t border-dashed" style={{ borderColor: INK.primary }} aria-hidden />
          Initial value
        </span>
      </div>
    </div>
  );
}

function MetricInline({
  icon: Icon,
  accent,
  label,
  value,
}: {
  icon: typeof TrendingDown;
  accent: string;
  label: string;
  value: string;
}): JSX.Element {
  return (
    <div className="rounded border border-terminal-border bg-terminal-bg px-3 py-2">
      <p className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest text-terminal-muted">
        <Icon size={11} style={{ color: accent }} aria-hidden />
        {label}
      </p>
      <p className="mt-0.5 font-mono text-base tabular-nums text-terminal-ink">{value}</p>
    </div>
  );
}
