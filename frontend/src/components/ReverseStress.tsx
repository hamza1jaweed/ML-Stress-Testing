import { useEffect, useState } from "react";
import { Rewind, ShieldAlert } from "lucide-react";
import { api } from "../api";
import type { ReverseStressResponse } from "../types";
import { formatPct } from "../palette";

interface ReverseStressProps {
  weights: Record<string, number>;
}

const HORIZON_OPTIONS = [5, 21, 63] as const;

/**
 * Reverse stress test: instead of asking what a shock does to the
 * portfolio, asks which joint market move is the *most likely* one to
 * produce a chosen loss (minimum Mahalanobis distance under the historical
 * covariance). Diversifiers can move up in the answer — that is the point.
 */
export default function ReverseStress({ weights }: ReverseStressProps): JSX.Element {
  const [targetLoss, setTargetLoss] = useState<number>(0.25);
  const [horizon, setHorizon] = useState<number>(21);
  const [result, setResult] = useState<ReverseStressResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (Object.keys(weights).length === 0) return;
    const id = window.setTimeout(() => {
      api
        .reverseStress(weights, targetLoss, horizon)
        .then((r) => {
          setResult(r);
          setError(null);
        })
        .catch((err: unknown) =>
          setError(err instanceof Error ? err.message : "request failed"),
        );
    }, 350);
    return () => window.clearTimeout(id);
  }, [weights, targetLoss, horizon]);

  const maxAbsShock = result
    ? Math.max(...Object.values(result.shocks).map((s) => Math.abs(s)), 1e-9)
    : 1;

  return (
    <section className="panel">
      <header className="panel-title justify-between">
        <span className="flex items-center gap-2">
          <Rewind size={14} aria-hidden />
          Reverse Stress Test
        </span>
        {result && (
          <span
            className={`flex items-center gap-1 font-mono text-[10px] normal-case tracking-normal ${
              result.plausible ? "text-series-3" : "text-series-2"
            }`}
          >
            <ShieldAlert size={11} aria-hidden />
            {result.sigma_distance.toFixed(1)}σ move —{" "}
            {result.plausible ? "plausible" : "extreme tail"}
          </span>
        )}
      </header>

      <div className="p-4">
        <p className="mb-3 text-[11px] leading-relaxed text-terminal-faint">
          The most likely joint market move that loses exactly the target —
          min s′Σ⁻¹s subject to w′s = −L. Assets that hedge the book can
          legitimately move <em>up</em> in the answer.
        </p>

        <div className="mb-4 flex flex-wrap items-end gap-4">
          <div className="min-w-44 flex-1">
            <div className="mb-0.5 flex justify-between text-[11px]">
              <span className="text-terminal-muted">Target loss</span>
              <span className="font-mono tabular-nums">{formatPct(targetLoss, 0)}</span>
            </div>
            <input
              type="range"
              min={5}
              max={60}
              step={1}
              value={Math.round(targetLoss * 100)}
              onChange={(e) => setTargetLoss(Number(e.target.value) / 100)}
              aria-label="Target portfolio loss percent"
            />
          </div>
          <div>
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-terminal-muted">
              Over
            </p>
            <div className="flex gap-1.5">
              {HORIZON_OPTIONS.map((h) => (
                <button
                  key={h}
                  type="button"
                  onClick={() => setHorizon(h)}
                  className={`rounded border px-2 py-1 text-xs ${
                    horizon === h
                      ? "border-series-1 text-terminal-ink"
                      : "border-terminal-border text-terminal-muted hover:text-terminal-ink"
                  }`}
                >
                  {h}d
                </button>
              ))}
            </div>
          </div>
        </div>

        {error && <p className="text-xs text-series-2">{error}</p>}

        {result && (
          <div className="space-y-2">
            {Object.entries(result.shocks).map(([ticker, shock]) => {
              const widthPct = (Math.abs(shock) / maxAbsShock) * 50;
              const negative = shock < 0;
              return (
                <div key={ticker} className="flex items-center gap-2 text-xs">
                  <span className="w-24 shrink-0 text-terminal-muted">{ticker}</span>
                  {/* Mirrored bar around a center axis: losses left/orange, gains right/blue */}
                  <div className="relative h-3.5 flex-1 rounded-sm bg-terminal-bg">
                    <div
                      className="absolute bottom-0 top-0 w-px bg-terminal-border"
                      style={{ left: "50%" }}
                      aria-hidden
                    />
                    <div
                      className="absolute bottom-0 top-0 rounded-sm"
                      style={{
                        left: negative ? `${50 - widthPct}%` : "50%",
                        width: `${widthPct}%`,
                        backgroundColor: negative ? "#d95926" : "#3987e5",
                      }}
                      aria-hidden
                    />
                  </div>
                  <span
                    className="w-16 shrink-0 text-right font-mono tabular-nums"
                  >
                    {shock >= 0 ? "+" : ""}
                    {formatPct(shock)}
                  </span>
                </div>
              );
            })}
            <p className="pt-1 text-[11px] text-terminal-faint">
              A {formatPct(result.target_loss, 0)} portfolio loss over{" "}
              {result.horizon_days} trading days is a{" "}
              {result.sigma_distance.toFixed(1)}-sigma event for this allocation
              {result.plausible
                ? " — within normal market experience."
                : " — far outside Gaussian experience (tail / crisis territory)."}
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
