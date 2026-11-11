import { useMemo } from "react";
import {
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity } from "lucide-react";
import type { TrainRegimesResponse } from "../types";
import { INK, REGIME_COLORS } from "../palette";

interface RegimeVisualizerProps {
  regimes: TrainRegimesResponse;
}

interface RegimeBand {
  startDate: string;
  endDate: string;
  state: number;
}

/** Collapse the per-day state sequence into contiguous colored bands. */
function toBands(dates: string[], states: number[]): RegimeBand[] {
  const bands: RegimeBand[] = [];
  let start = 0;
  for (let i = 1; i <= states.length; i++) {
    if (i === states.length || states[i] !== states[start]) {
      bands.push({ startDate: dates[start], endDate: dates[i - 1], state: states[start] });
      start = i;
    }
  }
  return bands;
}

/**
 * Market index line over regime-colored background bands, plus the Markov
 * transition probability heatmap (sequential blue ramp).
 */
export default function RegimeVisualizer({ regimes }: RegimeVisualizerProps): JSX.Element {
  const chartData = useMemo(
    () =>
      regimes.dates.map((date, i) => ({
        date,
        index: regimes.market_index[i],
        state: regimes.states[i],
      })),
    [regimes],
  );
  const bands = useMemo(() => toBands(regimes.dates, regimes.states), [regimes]);
  const regimeIds = Object.keys(regimes.state_names).map(Number).sort((a, b) => a - b);

  return (
    <section className="panel">
      <header className="panel-title">
        <Activity size={14} aria-hidden />
        Regime Classification
        {regimes.accuracy_vs_true !== null && (
          <span className="ml-auto font-mono text-[10px] normal-case tracking-normal">
            decode accuracy vs. generator truth:{" "}
            {(regimes.accuracy_vs_true * 100).toFixed(1)}%
          </span>
        )}
      </header>

      <div className="p-4">
        {/* Legend — identity never carried by color alone */}
        <div className="mb-3 flex flex-wrap gap-4 text-xs">
          {regimeIds.map((id) => (
            <span key={id} className="flex items-center gap-1.5 text-terminal-muted">
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ backgroundColor: REGIME_COLORS[id] ?? INK.muted }}
                aria-hidden
              />
              {regimes.state_names[String(id)]}
            </span>
          ))}
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              {bands.map((band, i) => (
                <ReferenceArea
                  key={i}
                  x1={band.startDate}
                  x2={band.endDate}
                  fill={REGIME_COLORS[band.state] ?? INK.muted}
                  fillOpacity={0.16}
                  strokeOpacity={0}
                />
              ))}
              <XAxis
                dataKey="date"
                tick={{ fill: INK.muted, fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: INK.axis }}
                minTickGap={60}
              />
              <YAxis
                tick={{ fill: INK.muted, fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                width={44}
                domain={["auto", "auto"]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: INK.surface,
                  border: `1px solid ${INK.grid}`,
                  borderRadius: 6,
                  fontSize: 12,
                }}
                labelStyle={{ color: INK.secondary }}
                formatter={(value: number | string, _name, item) => {
                  const state = (item.payload as { state: number }).state;
                  return [
                    `${Number(value).toFixed(1)} — ${regimes.state_names[String(state)]}`,
                    "Market index",
                  ];
                }}
              />
              <Line
                type="monotone"
                dataKey="index"
                stroke={INK.primary}
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <TransitionHeatmap regimes={regimes} regimeIds={regimeIds} />
      </div>
    </section>
  );
}

const SEQ_STOPS: readonly [number, string][] = [
  [0.0, "#14181f"],
  [0.15, "#0d366b"],
  [0.35, "#184f95"],
  [0.55, "#256abf"],
  [0.75, "#3987e5"],
  [0.9, "#6da7ec"],
  [1.0, "#9ec5f4"],
];

/** Map p in [0,1] onto the sequential blue ramp (dark surface -> light). */
function heatColor(p: number): string {
  for (let i = SEQ_STOPS.length - 1; i >= 0; i--) {
    if (p >= SEQ_STOPS[i][0]) return SEQ_STOPS[i][1];
  }
  return SEQ_STOPS[0][1];
}

function TransitionHeatmap({
  regimes,
  regimeIds,
}: {
  regimes: TrainRegimesResponse;
  regimeIds: number[];
}): JSX.Element {
  const shortName = (id: number): string => {
    const name = regimes.state_names[String(id)] ?? `R${id}`;
    return name === "High Volatility Bear" ? "HV Bear" : name;
  };

  return (
    <div className="mt-5 grid gap-6 md:grid-cols-2">
      <div>
        <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-widest text-terminal-muted">
          Transition probabilities (daily)
        </h3>
        <table className="w-full border-separate border-spacing-0.5 text-xs">
          <thead>
            <tr>
              <th className="p-1 text-left font-normal text-terminal-faint">from \ to</th>
              {regimeIds.map((id) => (
                <th key={id} className="p-1 font-normal text-terminal-muted">
                  {shortName(id)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {regimes.transition_matrix.map((row, i) => (
              <tr key={i}>
                <td className="p-1 text-terminal-muted">{shortName(regimeIds[i])}</td>
                {row.map((p, j) => {
                  // Off-diagonal probabilities are tiny; sqrt-stretch so they register.
                  const intensity = Math.sqrt(p);
                  const bg = heatColor(intensity);
                  const lightCell = intensity > 0.82;
                  return (
                    <td
                      key={j}
                      className="rounded p-1.5 text-center font-mono tabular-nums"
                      style={{
                        backgroundColor: bg,
                        color: lightCell ? "#0b0b0b" : "#f0efec",
                      }}
                      title={`P(${shortName(regimeIds[i])} → ${shortName(regimeIds[j])}) = ${(p * 100).toFixed(2)}%`}
                    >
                      {(p * 100).toFixed(1)}%
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div>
        <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-widest text-terminal-muted">
          Regime signatures
        </h3>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-terminal-faint">
              <th className="p-1 text-left font-normal">regime</th>
              <th className="p-1 text-right font-normal">rel. vol</th>
              <th className="p-1 text-right font-normal">daily ret</th>
              <th className="p-1 text-right font-normal">stationary π</th>
            </tr>
          </thead>
          <tbody>
            {regimeIds.map((id, i) => {
              const means = regimes.state_means[String(id)];
              return (
                <tr key={id} className="border-t border-terminal-border">
                  <td className="flex items-center gap-1.5 p-1.5">
                    <span
                      className="inline-block h-2 w-2 rounded-sm"
                      style={{ backgroundColor: REGIME_COLORS[id] ?? INK.muted }}
                      aria-hidden
                    />
                    {regimes.state_names[String(id)]}
                  </td>
                  <td className="p-1.5 text-right font-mono tabular-nums">
                    {means ? `${means.rolling_vol.toFixed(2)}×` : "—"}
                  </td>
                  <td className="p-1.5 text-right font-mono tabular-nums">
                    {means ? `${(means.mean_log_return * 10000).toFixed(1)} bps` : "—"}
                  </td>
                  <td className="p-1.5 text-right font-mono tabular-nums">
                    {(regimes.stationary_distribution[i] * 100).toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-2 text-[11px] leading-relaxed text-terminal-faint">
          Fit quality: BIC {regimes.bic.toFixed(0)}, log-likelihood{" "}
          {regimes.log_likelihood.toFixed(0)}
        </p>
      </div>
    </div>
  );
}
