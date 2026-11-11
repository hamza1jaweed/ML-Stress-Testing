import { useMemo } from "react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { GitBranch, Siren } from "lucide-react";
import type { EarlyWarningResponse, RegimeForecastResponse } from "../types";
import { INK, REGIME_COLORS, formatPct } from "../palette";

interface RegimeForecastProps {
  forecast: RegimeForecastResponse;
  earlyWarning: EarlyWarningResponse | null;
}

/**
 * Markov-chain regime nowcast: stacked probability bands for each regime
 * over the forecast horizon, decaying from the current decoded state toward
 * the stationary distribution.
 */
export default function RegimeForecast({
  forecast,
  earlyWarning,
}: RegimeForecastProps): JSX.Element {
  const regimeIds = Object.keys(forecast.state_names)
    .map(Number)
    .sort((a, b) => a - b);

  const data = useMemo(
    () =>
      forecast.probabilities.map((row, t) => {
        const point: Record<string, number> = { day: t + 1 };
        regimeIds.forEach((id, k) => {
          point[`r${id}`] = row[k];
        });
        return point;
      }),
    [forecast, regimeIds],
  );

  return (
    <section className="panel">
      <header className="panel-title">
        <GitBranch size={14} aria-hidden />
        Regime Forecast
        <span className="ml-auto font-mono text-[10px] normal-case tracking-normal">
          nowcast: {forecast.current_state_name}
        </span>
      </header>

      <div className="p-4">
        {earlyWarning && (
          <div className="mb-4">
            <h3 className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-widest text-terminal-muted">
              <Siren size={12} aria-hidden />
              Bear-regime early warning
            </h3>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {Object.entries(earlyWarning.prob_within).map(([days, prob]) => (
                <div
                  key={days}
                  className="rounded border border-terminal-border bg-terminal-bg px-2 py-1.5"
                >
                  <p className="text-[9px] font-semibold uppercase tracking-wider text-terminal-faint">
                    P(bear ≤ {days}d)
                  </p>
                  <p
                    className="font-mono text-sm tabular-nums"
                    style={{ color: prob > 0.5 ? "#d95926" : INK.primary }}
                  >
                    {formatPct(prob)}
                  </p>
                </div>
              ))}
              <div className="rounded border border-terminal-border bg-terminal-bg px-2 py-1.5">
                <p className="text-[9px] font-semibold uppercase tracking-wider text-terminal-faint">
                  E[days to bear]
                </p>
                <p className="font-mono text-sm tabular-nums">
                  {earlyWarning.already_in_bear
                    ? "in bear now"
                    : earlyWarning.expected_days_to_bear.toFixed(0)}
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="mb-3 flex flex-wrap gap-4 text-xs">
          {regimeIds.map((id) => (
            <span key={id} className="flex items-center gap-1.5 text-terminal-muted">
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ backgroundColor: REGIME_COLORS[id] ?? INK.muted }}
                aria-hidden
              />
              {forecast.state_names[String(id)]}
            </span>
          ))}
        </div>

        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <XAxis
                dataKey="day"
                tick={{ fill: INK.muted, fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: INK.axis }}
                label={{
                  value: "days ahead",
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
                width={40}
                domain={[0, 1]}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: INK.surface,
                  border: `1px solid ${INK.grid}`,
                  borderRadius: 6,
                  fontSize: 12,
                }}
                labelFormatter={(d) => `Day +${d}`}
                formatter={(value: number | string, name: string) => {
                  const id = Number(String(name).replace("r", ""));
                  return [formatPct(Number(value)), forecast.state_names[String(id)] ?? name];
                }}
              />
              {regimeIds.map((id) => (
                <Area
                  key={id}
                  dataKey={`r${id}`}
                  stackId="prob"
                  stroke={REGIME_COLORS[id] ?? INK.muted}
                  strokeWidth={1}
                  fill={REGIME_COLORS[id] ?? INK.muted}
                  fillOpacity={0.45}
                  isAnimationActive={false}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <p className="mt-2 text-[11px] leading-relaxed text-terminal-faint">
          P(regime at day t) = e<sub>now</sub>·T<sup>t</sup>; converges to the stationary
          distribution π = (
          {forecast.stationary_distribution.map((p) => formatPct(p, 0)).join(", ")}
          ).
        </p>
      </div>
    </section>
  );
}
