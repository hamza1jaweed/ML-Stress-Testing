import { useMemo } from "react";
import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { Target } from "lucide-react";
import type { EfficientFrontierResponse } from "../types";
import { INK, SEQ_BLUE, formatPct } from "../palette";

interface EfficientFrontierProps {
  frontier: EfficientFrontierResponse;
  /** Current portfolio marker (CAGR, annualized vol) from the analytics panel. */
  currentPoint: { ret: number; vol: number } | null;
  onApplyWeights: (weights: Record<string, number>) => void;
}

/** Bucket Sharpe into the sequential blue ramp (magnitude encoding). */
function sharpeColor(sharpe: number, lo: number, hi: number): string {
  if (hi <= lo) return SEQ_BLUE[3];
  const t = Math.max(0, Math.min(1, (sharpe - lo) / (hi - lo)));
  const steps = [SEQ_BLUE[0], SEQ_BLUE[1], SEQ_BLUE[2], SEQ_BLUE[3], SEQ_BLUE[4]];
  return steps[Math.min(steps.length - 1, Math.floor(t * steps.length))];
}

/**
 * Mean-variance landscape: random long-only portfolio cloud colored by
 * Sharpe ratio, the SLSQP minimum-variance frontier, and markers for the
 * max-Sharpe, min-vol, and current portfolios (distinct shapes + legend).
 */
export default function EfficientFrontier({
  frontier,
  currentPoint,
  onApplyWeights,
}: EfficientFrontierProps): JSX.Element {
  const [sharpeLo, sharpeHi] = useMemo(() => {
    const values = frontier.cloud.map((p) => p.sharpe ?? 0);
    return [Math.min(...values), Math.max(...values)];
  }, [frontier]);

  const maxSharpePoint = [{ vol: frontier.max_sharpe.vol, ret: frontier.max_sharpe.ret }];
  const minVolPoint = [{ vol: frontier.min_vol.vol, ret: frontier.min_vol.ret }];

  return (
    <section className="panel">
      <header className="panel-title">
        <Target size={14} aria-hidden />
        Efficient Frontier
      </header>

      <div className="p-4">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
              <CartesianGrid stroke={INK.grid} strokeDasharray="2 4" />
              <XAxis
                type="number"
                dataKey="vol"
                name="Volatility"
                tick={{ fill: INK.muted, fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: INK.axis }}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                domain={["auto", "auto"]}
                label={{
                  value: "annualized volatility",
                  position: "insideBottomRight",
                  fill: INK.muted,
                  fontSize: 10,
                  dy: 8,
                }}
              />
              <YAxis
                type="number"
                dataKey="ret"
                name="Return"
                tick={{ fill: INK.muted, fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                width={48}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                domain={["auto", "auto"]}
              />
              <ZAxis range={[14, 14]} />
              <Tooltip
                cursor={{ stroke: INK.axis, strokeDasharray: "3 3" }}
                contentStyle={{
                  backgroundColor: INK.surface,
                  border: `1px solid ${INK.grid}`,
                  borderRadius: 6,
                  fontSize: 12,
                }}
                formatter={(value: number | string, name: string) => [
                  formatPct(Number(value)),
                  name,
                ]}
              />
              <Scatter name="Random portfolios" data={frontier.cloud} isAnimationActive={false}>
                {frontier.cloud.map((p, i) => (
                  <Cell key={i} fill={sharpeColor(p.sharpe ?? 0, sharpeLo, sharpeHi)} fillOpacity={0.55} />
                ))}
              </Scatter>
              <Scatter
                name="Frontier"
                data={frontier.frontier}
                fill={INK.primary}
                line={{ stroke: INK.primary, strokeWidth: 1.6 }}
                shape={() => <g />}
                isAnimationActive={false}
              />
              <Scatter
                name="Max Sharpe"
                data={maxSharpePoint}
                fill="#199e70"
                shape="diamond"
                isAnimationActive={false}
              />
              <Scatter
                name="Min vol"
                data={minVolPoint}
                fill="#d55181"
                shape="square"
                isAnimationActive={false}
              />
              {currentPoint && (
                <Scatter
                  name="Current"
                  data={[currentPoint]}
                  fill={INK.primary}
                  shape="triangle"
                  isAnimationActive={false}
                />
              )}
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Legend + actions */}
        <div className="mt-2 flex flex-wrap items-center gap-4 text-[11px] text-terminal-muted">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-6 rounded-sm" style={{ background: `linear-gradient(90deg, ${SEQ_BLUE[0]}, ${SEQ_BLUE[4]})` }} aria-hidden />
            Sharpe {sharpeLo.toFixed(1)} → {sharpeHi.toFixed(1)}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-5" style={{ backgroundColor: INK.primary }} aria-hidden />
            Frontier
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rotate-45" style={{ backgroundColor: "#199e70" }} aria-hidden />
            Max Sharpe
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5" style={{ backgroundColor: "#d55181" }} aria-hidden />
            Min vol
          </span>
          {currentPoint && (
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-0 w-0 border-x-[5px] border-b-[9px] border-x-transparent"
                style={{ borderBottomColor: INK.primary }}
                aria-hidden
              />
              Current
            </span>
          )}
        </div>

        <div className="mt-3 grid gap-2 sm:grid-cols-3">
          <OptimalCard
            title="Max Sharpe"
            portfolio={frontier.max_sharpe}
            onApply={() => onApplyWeights(frontier.max_sharpe.weights)}
          />
          <OptimalCard
            title="Min volatility"
            portfolio={frontier.min_vol}
            onApply={() => onApplyWeights(frontier.min_vol.weights)}
          />
          <OptimalCard
            title="Risk parity"
            portfolio={frontier.risk_parity}
            onApply={() => onApplyWeights(frontier.risk_parity.weights)}
          />
        </div>
      </div>
    </section>
  );
}

function OptimalCard({
  title,
  portfolio,
  onApply,
}: {
  title: string;
  portfolio: EfficientFrontierResponse["max_sharpe"];
  onApply: () => void;
}): JSX.Element {
  return (
    <div className="rounded border border-terminal-border bg-terminal-bg p-2.5 text-xs">
      <div className="flex items-center justify-between">
        <span className="font-semibold">{title}</span>
        <button
          type="button"
          onClick={onApply}
          className="rounded border border-terminal-border px-2 py-0.5 text-[10px] text-terminal-muted hover:border-series-1 hover:text-terminal-ink"
        >
          Apply weights
        </button>
      </div>
      <p className="mt-1 font-mono tabular-nums text-terminal-muted">
        ret {formatPct(portfolio.ret)} · vol {formatPct(portfolio.vol)} · SR{" "}
        {portfolio.sharpe.toFixed(2)}
      </p>
      <p className="mt-1 text-[10px] leading-relaxed text-terminal-faint">
        {Object.entries(portfolio.weights)
          .filter(([, w]) => w > 0.001)
          .map(([t, w]) => `${t} ${(w * 100).toFixed(0)}%`)
          .join(" · ")}
      </p>
    </div>
  );
}
