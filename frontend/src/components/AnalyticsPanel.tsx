import { useMemo } from "react";
import {
  Area,
  AreaChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { LineChart as LineChartIcon } from "lucide-react";
import type { PortfolioAnalyticsResponse } from "../types";
import { INK, REGIME_COLORS, SERIES, formatPct } from "../palette";

interface AnalyticsPanelProps {
  analytics: PortfolioAnalyticsResponse;
}

/**
 * Historical portfolio analytics: equity curve, underwater drawdown,
 * performance metric strip, per-asset risk contributions, and the asset
 * correlation matrix (diverging blue/red heatmap).
 */
export default function AnalyticsPanel({ analytics }: AnalyticsPanelProps): JSX.Element {
  const series = useMemo(
    () =>
      analytics.dates.map((date, i) => ({
        date,
        equity: analytics.equity_curve[i],
        drawdown: analytics.drawdown[i],
      })),
    [analytics],
  );
  const m = analytics.metrics;

  return (
    <section className="panel">
      <header className="panel-title">
        <LineChartIcon size={14} aria-hidden />
        Portfolio Analytics (historical, daily-rebalanced)
      </header>

      <div className="p-4">
        {/* Metric strip */}
        <div className="mb-4 grid grid-cols-3 gap-2 text-center sm:grid-cols-5 xl:grid-cols-9">
          <Stat label="CAGR" value={formatPct(m.cagr)} />
          <Stat label="Vol (ann.)" value={formatPct(m.annual_vol)} />
          <Stat label="Sharpe" value={m.sharpe.toFixed(2)} />
          <Stat label="Sortino" value={m.sortino.toFixed(2)} />
          <Stat label="Calmar" value={m.calmar.toFixed(2)} />
          <Stat label="Max DD" value={formatPct(m.max_drawdown)} />
          <Stat label="VaR95 1d" value={formatPct(m.hist_var_95_daily)} />
          <Stat label="Skew" value={m.skew.toFixed(2)} />
          <Stat label="Ex. kurt" value={m.excess_kurtosis.toFixed(1)} />
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          {/* Equity curve */}
          <div>
            <ChartLabel text="Equity curve (start = 100)" />
            <div className="h-44">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                  <XAxis
                    dataKey="date"
                    tick={{ fill: INK.muted, fontSize: 10 }}
                    tickLine={false}
                    axisLine={{ stroke: INK.axis }}
                    minTickGap={70}
                  />
                  <YAxis
                    tick={{ fill: INK.muted, fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                    width={44}
                    domain={["auto", "auto"]}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelStyle={{ color: INK.secondary }}
                    formatter={(v: number | string) => [Number(v).toFixed(1), "Index"]}
                  />
                  <Line
                    dataKey="equity"
                    stroke={SERIES[0]}
                    strokeWidth={1.6}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Underwater drawdown */}
          <div>
            <ChartLabel text="Drawdown (underwater)" />
            <div className="h-44">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={series} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                  <XAxis
                    dataKey="date"
                    tick={{ fill: INK.muted, fontSize: 10 }}
                    tickLine={false}
                    axisLine={{ stroke: INK.axis }}
                    minTickGap={70}
                  />
                  <YAxis
                    tick={{ fill: INK.muted, fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                    width={44}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelStyle={{ color: INK.secondary }}
                    formatter={(v: number | string) => [formatPct(Number(v)), "Drawdown"]}
                  />
                  <Area
                    dataKey="drawdown"
                    stroke={SERIES[1]}
                    strokeWidth={1.4}
                    fill={SERIES[1]}
                    fillOpacity={0.22}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="mt-5 grid gap-6 lg:grid-cols-2">
          <RiskContributions analytics={analytics} />
          <CorrelationMatrix analytics={analytics} />
        </div>

        <div className="mt-5 grid gap-6 lg:grid-cols-2">
          {analytics.regime_performance && analytics.regime_performance.length > 0 && (
            <RegimePerformance rows={analytics.regime_performance} />
          )}
          {analytics.var_backtest && <VarBacktestCard backtest={analytics.var_backtest} />}
        </div>
      </div>
    </section>
  );
}

const tooltipStyle = {
  backgroundColor: INK.surface,
  border: `1px solid ${INK.grid}`,
  borderRadius: 6,
  fontSize: 12,
} as const;

function ChartLabel({ text }: { text: string }): JSX.Element {
  return (
    <h3 className="mb-1.5 text-[11px] font-semibold uppercase tracking-widest text-terminal-muted">
      {text}
    </h3>
  );
}

function Stat({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div className="rounded border border-terminal-border bg-terminal-bg px-2 py-1.5">
      <p className="text-[9px] font-semibold uppercase tracking-wider text-terminal-faint">
        {label}
      </p>
      <p className="font-mono text-sm tabular-nums">{value}</p>
    </div>
  );
}

function RiskContributions({
  analytics,
}: {
  analytics: PortfolioAnalyticsResponse;
}): JSX.Element {
  return (
    <div>
      <ChartLabel text="Risk contribution (share of portfolio variance)" />
      <div className="space-y-2">
        {analytics.tickers.map((ticker, i) => {
          const rc = analytics.risk_contributions[ticker] ?? 0;
          return (
            <div key={ticker} className="flex items-center gap-2 text-xs">
              <span className="w-24 shrink-0 text-terminal-muted">{ticker}</span>
              <div className="h-3 flex-1 overflow-hidden rounded-sm bg-terminal-bg">
                <div
                  className="h-full rounded-sm"
                  style={{
                    width: `${Math.max(0, Math.min(100, rc * 100))}%`,
                    backgroundColor: SERIES[i % SERIES.length],
                  }}
                  aria-hidden
                />
              </div>
              <span className="w-14 shrink-0 text-right font-mono tabular-nums">
                {formatPct(rc)}
              </span>
            </div>
          );
        })}
      </div>
      <p className="mt-2 text-[11px] leading-relaxed text-terminal-faint">
        RC<sub>i</sub> = w<sub>i</sub>(Σw)<sub>i</sub> / w′Σw — a high-vol sleeve can
        dominate risk even at a small weight.
      </p>
    </div>
  );
}

function RegimePerformance({
  rows,
}: {
  rows: NonNullable<PortfolioAnalyticsResponse["regime_performance"]>;
}): JSX.Element {
  return (
    <div>
      <ChartLabel text="Performance by detected regime" />
      <table className="w-full text-xs">
        <thead>
          <tr className="text-terminal-faint">
            <th className="p-1 text-left font-normal">regime</th>
            <th className="p-1 text-right font-normal">days</th>
            <th className="p-1 text-right font-normal">ann. ret</th>
            <th className="p-1 text-right font-normal">ann. vol</th>
            <th className="p-1 text-right font-normal">Sharpe</th>
            <th className="p-1 text-right font-normal">worst day</th>
            <th className="p-1 text-right font-normal">hit rate</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.regime} className="border-t border-terminal-border">
              <td className="flex items-center gap-1.5 p-1.5">
                <span
                  className="inline-block h-2 w-2 rounded-sm"
                  style={{ backgroundColor: REGIME_COLORS[row.regime] ?? INK.muted }}
                  aria-hidden
                />
                {row.name}
              </td>
              <td className="p-1.5 text-right font-mono tabular-nums">{row.days}</td>
              <td className="p-1.5 text-right font-mono tabular-nums">
                {formatPct(row.ann_return)}
              </td>
              <td className="p-1.5 text-right font-mono tabular-nums">
                {formatPct(row.ann_vol)}
              </td>
              <td className="p-1.5 text-right font-mono tabular-nums">
                {row.sharpe.toFixed(2)}
              </td>
              <td className="p-1.5 text-right font-mono tabular-nums">
                {formatPct(row.worst_day)}
              </td>
              <td className="p-1.5 text-right font-mono tabular-nums">
                {formatPct(row.hit_rate, 0)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] leading-relaxed text-terminal-faint">
        How this exact allocation behaved inside each regime the model detected —
        the case for (or against) regime-conditional allocation.
      </p>
    </div>
  );
}

function VarBacktestCard({
  backtest,
}: {
  backtest: NonNullable<PortfolioAnalyticsResponse["var_backtest"]>;
}): JSX.Element {
  return (
    <div>
      <ChartLabel text={`VaR ${Math.round(backtest.confidence * 100)}% backtest (Kupiec POF)`} />
      <div className="rounded border border-terminal-border bg-terminal-bg p-3 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-terminal-muted">
            Rolling {backtest.window}-day historical VaR, {backtest.observations} test days
          </span>
          <span
            className={`rounded px-2 py-0.5 font-mono text-[10px] font-semibold ${
              backtest.model_ok ? "text-series-3" : "text-series-2"
            }`}
          >
            {backtest.model_ok ? "MODEL ACCEPTED" : "MODEL REJECTED"}
          </span>
        </div>
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1.5">
          <dt className="text-terminal-muted">Breaches observed</dt>
          <dd className="text-right font-mono tabular-nums">{backtest.breaches}</dd>
          <dt className="text-terminal-muted">Breaches expected</dt>
          <dd className="text-right font-mono tabular-nums">{backtest.expected_breaches}</dd>
          <dt className="text-terminal-muted">Breach rate</dt>
          <dd className="text-right font-mono tabular-nums">{formatPct(backtest.breach_rate)}</dd>
          <dt className="text-terminal-muted">Kupiec LR / p-value</dt>
          <dd className="text-right font-mono tabular-nums">
            {backtest.kupiec_lr.toFixed(2)} / {backtest.p_value.toFixed(3)}
          </dd>
        </dl>
        <p className="mt-2 text-[11px] leading-relaxed text-terminal-faint">
          Tests whether realized losses breach the daily VaR estimate about as often as
          the confidence level promises (p &gt; 0.05 ⇒ the VaR model is statistically
          consistent with the data).
        </p>
      </div>
    </div>
  );
}

/** Diverging blue (positive) / red (negative) correlation heatmap. */
function corrColor(rho: number): { bg: string; ink: string } {
  const NEUTRAL = "#383835";
  const BLUES = ["#383835", "#1c5cab", "#256abf", "#3987e5", "#6da7ec"]; // 0 -> +1
  const REDS = ["#383835", "#8c2f2e", "#b23c3b", "#e34948", "#e66767"]; // 0 -> -1
  const magnitude = Math.min(Math.abs(rho), 1);
  const step = Math.min(4, Math.floor(magnitude * 4 + 0.5));
  if (step === 0) return { bg: NEUTRAL, ink: "#f0efec" };
  const bg = rho > 0 ? BLUES[step] : REDS[step];
  const ink = (rho > 0 && step >= 4) ? "#0b0b0b" : "#f0efec";
  return { bg, ink };
}

function CorrelationMatrix({
  analytics,
}: {
  analytics: PortfolioAnalyticsResponse;
}): JSX.Element {
  return (
    <div>
      <ChartLabel text="Return correlation matrix" />
      <table className="w-full border-separate border-spacing-0.5 text-[10px]">
        <thead>
          <tr>
            <th aria-hidden />
            {analytics.tickers.map((t) => (
              <th key={t} className="p-0.5 font-normal text-terminal-muted">
                {t.replace("_", " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {analytics.correlation.map((row, i) => (
            <tr key={analytics.tickers[i]}>
              <td className="pr-1 text-terminal-muted">{analytics.tickers[i]}</td>
              {row.map((rho, j) => {
                const { bg, ink } = corrColor(rho);
                return (
                  <td
                    key={j}
                    className="rounded p-1 text-center font-mono tabular-nums"
                    style={{ backgroundColor: bg, color: ink }}
                    title={`ρ(${analytics.tickers[i]}, ${analytics.tickers[j]}) = ${rho.toFixed(2)}`}
                  >
                    {rho.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 flex items-center gap-3 text-[11px] text-terminal-faint">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "#e34948" }} aria-hidden />
          −1
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "#383835" }} aria-hidden />
          0
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "#3987e5" }} aria-hidden />
          +1
        </span>
      </p>
    </div>
  );
}
