import type { CSSProperties } from "react";
import { SlidersHorizontal } from "lucide-react";
import type { AssetInfo } from "../types";
import { SERIES, formatPct } from "../palette";

interface AllocationMatrixProps {
  assets: AssetInfo[];
  weights: Record<string, number>;
  onChange: (weights: Record<string, number>) => void;
}

/**
 * Interactive multi-asset weight sliders. Moving one slider rebalances the
 * others proportionally so the allocation always sums to exactly 100%.
 */
export default function AllocationMatrix({
  assets,
  weights,
  onChange,
}: AllocationMatrixProps): JSX.Element {
  const handleSlider = (ticker: string, rawValue: number): void => {
    const target = Math.min(Math.max(rawValue / 100, 0), 1);
    const others = assets.filter((a) => a.ticker !== ticker);
    const othersTotal = others.reduce((s, a) => s + (weights[a.ticker] ?? 0), 0);
    const remaining = 1 - target;

    const next: Record<string, number> = { [ticker]: target };
    if (othersTotal <= 1e-9) {
      // Everything was on the moved asset: spread the remainder equally.
      for (const a of others) next[a.ticker] = remaining / others.length;
    } else {
      for (const a of others)
        next[a.ticker] = ((weights[a.ticker] ?? 0) / othersTotal) * remaining;
    }
    onChange(next);
  };

  const setEqualWeight = (): void => {
    const w = 1 / assets.length;
    onChange(Object.fromEntries(assets.map((a) => [a.ticker, w])));
  };

  const total = assets.reduce((s, a) => s + (weights[a.ticker] ?? 0), 0);

  return (
    <section className="panel">
      <header className="panel-title justify-between">
        <span className="flex items-center gap-2">
          <SlidersHorizontal size={14} aria-hidden />
          Asset Allocation
        </span>
        <button
          type="button"
          onClick={setEqualWeight}
          className="rounded border border-terminal-border px-2 py-0.5 text-[10px] normal-case tracking-normal text-terminal-muted hover:text-terminal-ink"
        >
          Equal weight
        </button>
      </header>

      <div className="space-y-4 p-4">
        {assets.map((asset, i) => {
          const weight = weights[asset.ticker] ?? 0;
          const color = SERIES[i % SERIES.length];
          return (
            <div key={asset.ticker}>
              <div className="mb-1 flex items-baseline justify-between text-sm">
                <span className="flex items-center gap-2">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-sm"
                    style={{ backgroundColor: color }}
                    aria-hidden
                  />
                  <span className="font-medium">{asset.ticker}</span>
                  <span className="text-xs text-terminal-muted">{asset.name}</span>
                </span>
                <span className="font-mono tabular-nums text-terminal-ink">
                  {formatPct(weight)}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={Math.round(weight * 100)}
                onChange={(e) => handleSlider(asset.ticker, Number(e.target.value))}
                style={{ "--thumb-color": color } as CSSProperties}
                aria-label={`${asset.name} allocation percent`}
              />
            </div>
          );
        })}

        <div className="flex justify-between border-t border-terminal-border pt-3 text-sm">
          <span className="text-terminal-muted">Total</span>
          <span
            className={`font-mono tabular-nums ${
              Math.abs(total - 1) < 0.005 ? "text-terminal-ink" : "text-series-2"
            }`}
          >
            {formatPct(total)}
          </span>
        </div>
      </div>
    </section>
  );
}
