import { useCallback, useEffect, useRef, useState } from "react";
import { BrainCircuit, Database, HeartPulse, TerminalSquare } from "lucide-react";
import AllocationMatrix from "./components/AllocationMatrix";
import AnalyticsPanel from "./components/AnalyticsPanel";
import EfficientFrontier from "./components/EfficientFrontier";
import RegimeForecast from "./components/RegimeForecast";
import RegimeVisualizer from "./components/RegimeVisualizer";
import ReverseStress from "./components/ReverseStress";
import StressCockpit, { ShockState, ZERO_SHOCKS } from "./components/StressCockpit";
import { api, ApiError } from "./api";
import type {
  EarlyWarningResponse,
  EfficientFrontierResponse,
  GenerateDataResponse,
  HealthResponse,
  PortfolioAnalyticsResponse,
  RegimeForecastResponse,
  ScenarioInfo,
  StressTestResponse,
  TrainRegimesResponse,
} from "./types";

const INITIAL_VALUE = 1_000_000;
const N_PATHS = 10_000;

export default function App(): JSX.Element {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [market, setMarket] = useState<GenerateDataResponse | null>(null);
  const [regimes, setRegimes] = useState<TrainRegimesResponse | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [stress, setStress] = useState<StressTestResponse | null>(null);
  const [analytics, setAnalytics] = useState<PortfolioAnalyticsResponse | null>(null);
  const [frontier, setFrontier] = useState<EfficientFrontierResponse | null>(null);
  const [forecast, setForecast] = useState<RegimeForecastResponse | null>(null);
  const [earlyWarning, setEarlyWarning] = useState<EarlyWarningResponse | null>(null);

  const [weights, setWeights] = useState<Record<string, number>>({});
  const [shocks, setShocks] = useState<ShockState>(ZERO_SHOCKS);
  const [method, setMethod] = useState<"parametric" | "bootstrap">("parametric");
  const [activeScenario, setActiveScenario] = useState<string | null>(null);
  const [horizonDays, setHorizonDays] = useState<number>(252);
  const [regimeAware, setRegimeAware] = useState<boolean>(false);
  const [correlationStress, setCorrelationStress] = useState<number>(0);

  const [busy, setBusy] = useState<"generate" | "train" | null>(null);
  const [simRunning, setSimRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stressRunId = useRef(0);

  // ---- health poll -------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    const poll = async (): Promise<void> => {
      try {
        const h = await api.health();
        if (!cancelled) setHealth(h);
      } catch {
        if (!cancelled) setHealth(null);
      }
    };
    void poll();
    const id = window.setInterval(poll, 15_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  useEffect(() => {
    api
      .scenarios()
      .then((r) => setScenarios(r.scenarios))
      .catch(() => setScenarios([]));
  }, []);

  // ---- actions -----------------------------------------------------------
  const handleError = (err: unknown): void => {
    if (err instanceof ApiError) setError(`API ${err.status}: ${err.message}`);
    else if (err instanceof Error) setError(err.message);
    else setError("Unknown error");
  };

  const generateData = async (): Promise<void> => {
    setBusy("generate");
    setError(null);
    try {
      const data = await api.generateData(1500, 42);
      setMarket(data);
      setRegimes(null);
      setStress(null);
      setAnalytics(null);
      setForecast(null);
      setFrontier(null);
      setEarlyWarning(null);
      setRegimeAware(false);
      setCorrelationStress(0);
      const equal = 1 / data.tickers.length;
      setWeights(Object.fromEntries(data.tickers.map((t) => [t, equal])));
      // The frontier depends only on the dataset — compute it once per seed.
      api.efficientFrontier().then(setFrontier).catch(handleError);
    } catch (err) {
      handleError(err);
    } finally {
      setBusy(null);
    }
  };

  const trainRegimes = async (): Promise<void> => {
    setBusy("train");
    setError(null);
    try {
      setRegimes(await api.trainRegimes(3, 7));
      api.regimeForecast(126).then(setForecast).catch(handleError);
      api.earlyWarning().then(setEarlyWarning).catch(handleError);
      // Analytics gains the regime-performance table once trained.
      if (Object.keys(weights).length > 0) {
        api.portfolioAnalytics(weights).then(setAnalytics).catch(handleError);
      }
    } catch (err) {
      handleError(err);
    } finally {
      setBusy(null);
    }
  };

  const runStress = useCallback(async (): Promise<void> => {
    if (!market || Object.keys(weights).length === 0) return;
    const runId = ++stressRunId.current;
    setSimRunning(true);
    try {
      const nonZeroShocks = Object.fromEntries(
        Object.entries(shocks).filter(([, v]) => Math.abs(v) > 1e-9),
      );
      const result = await api.stressTest({
        weights,
        n_paths: N_PATHS,
        horizon_days: horizonDays,
        method,
        initial_value: INITIAL_VALUE,
        seed: 123,
        scenario: activeScenario,
        regime_aware: activeScenario ? false : regimeAware,
        correlation_stress:
          activeScenario === null && (regimeAware || method === "parametric")
            ? correlationStress
            : 0,
        shocks: activeScenario
          ? null
          : Object.keys(nonZeroShocks).length > 0
            ? nonZeroShocks
            : null,
      });
      if (runId === stressRunId.current) {
        setStress(result);
        setError(null);
      }
    } catch (err) {
      if (runId === stressRunId.current) handleError(err);
    } finally {
      if (runId === stressRunId.current) setSimRunning(false);
    }
  }, [market, weights, shocks, method, activeScenario, horizonDays, regimeAware, correlationStress]);

  // Debounced auto-rerun whenever simulation inputs change.
  useEffect(() => {
    if (!market) return;
    const id = window.setTimeout(() => void runStress(), 350);
    return () => window.clearTimeout(id);
  }, [market, runStress]);

  // Debounced historical analytics on weight changes.
  useEffect(() => {
    if (!market || Object.keys(weights).length === 0) return;
    const id = window.setTimeout(() => {
      api.portfolioAnalytics(weights).then(setAnalytics).catch(handleError);
    }, 400);
    return () => window.clearTimeout(id);
  }, [market, weights]);

  // ---- layout ------------------------------------------------------------
  return (
    <div className="mx-auto min-h-screen max-w-7xl px-4 py-5">
      <header className="mb-5 flex flex-wrap items-center gap-3">
        <TerminalSquare size={22} className="text-series-1" aria-hidden />
        <div>
          <h1 className="text-lg font-semibold tracking-tight">
            Regime Terminal
          </h1>
          <p className="text-xs text-terminal-muted">
            Market regime detection · Monte Carlo portfolio stress testing
          </p>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <HealthBadge health={health} />
          <button
            type="button"
            onClick={() => void generateData()}
            disabled={busy !== null}
            className="flex items-center gap-1.5 rounded border border-terminal-border bg-terminal-panel px-3 py-1.5 text-xs font-medium hover:border-series-1 disabled:opacity-50"
          >
            <Database size={13} aria-hidden />
            {busy === "generate" ? "Generating…" : "Generate data"}
          </button>
          <button
            type="button"
            onClick={() => void trainRegimes()}
            disabled={busy !== null || !market}
            className="flex items-center gap-1.5 rounded border border-terminal-border bg-terminal-panel px-3 py-1.5 text-xs font-medium hover:border-series-1 disabled:opacity-50"
          >
            <BrainCircuit size={13} aria-hidden />
            {busy === "train" ? "Training…" : "Train regimes"}
          </button>
        </div>
      </header>

      {error && (
        <div
          role="alert"
          className="mb-4 rounded border border-series-2 bg-terminal-panel px-3 py-2 text-xs text-terminal-ink"
        >
          {error}
        </div>
      )}

      {!market && (
        <div className="panel flex h-48 flex-col items-center justify-center gap-2 text-sm text-terminal-muted">
          <p>No market data loaded.</p>
          <p className="text-xs text-terminal-faint">
            Click <span className="font-semibold text-terminal-ink">Generate data</span> to
            seed a synthetic 1,500-day multi-asset history, then{" "}
            <span className="font-semibold text-terminal-ink">Train regimes</span>.
          </p>
        </div>
      )}

      {market && (
        <div className="space-y-4">
          <div className="grid gap-4 xl:grid-cols-[320px,1fr]">
            <div className="space-y-4">
              <AllocationMatrix
                assets={market.assets}
                weights={weights}
                onChange={setWeights}
              />
              <DataSummary market={market} />
            </div>

            <div className="space-y-4">
              {regimes && <RegimeVisualizer regimes={regimes} />}
              <StressCockpit
                result={stress}
                running={simRunning}
                method={method}
                onMethodChange={setMethod}
                scenarios={scenarios}
                activeScenario={activeScenario}
                onScenarioChange={setActiveScenario}
                shocks={shocks}
                onShocksChange={setShocks}
                horizonDays={horizonDays}
                onHorizonChange={setHorizonDays}
                regimeAware={regimeAware}
                onRegimeAwareChange={setRegimeAware}
                regimeTrained={regimes !== null}
                regimeNames={regimes?.state_names ?? null}
                correlationStress={correlationStress}
                onCorrelationStressChange={setCorrelationStress}
              />
            </div>
          </div>

          {analytics && <AnalyticsPanel analytics={analytics} />}

          <div className="grid gap-4 xl:grid-cols-2">
            {frontier && (
              <EfficientFrontier
                frontier={frontier}
                currentPoint={
                  analytics
                    ? { ret: analytics.metrics.cagr, vol: analytics.metrics.annual_vol }
                    : null
                }
                onApplyWeights={setWeights}
              />
            )}
            {forecast && <RegimeForecast forecast={forecast} earlyWarning={earlyWarning} />}
          </div>

          <ReverseStress weights={weights} />
        </div>
      )}
    </div>
  );
}

function HealthBadge({ health }: { health: HealthResponse | null }): JSX.Element {
  const online = health?.status === "ok";
  return (
    <span
      className="flex items-center gap-1.5 rounded border border-terminal-border px-2.5 py-1.5 text-[11px] text-terminal-muted"
      title={
        online
          ? `Python ${health.python_version} · NumPy ${health.numpy_version} · ${health.cpu_count} CPUs` +
            (health.memory_used_mb !== null ? ` · ${health.memory_used_mb} MB RSS` : "")
          : "Backend unreachable"
      }
    >
      <HeartPulse
        size={12}
        className={online ? "text-series-3" : "text-series-2"}
        aria-hidden
      />
      {online ? "API online" : "API offline"}
    </span>
  );
}

function DataSummary({ market }: { market: GenerateDataResponse }): JSX.Element {
  return (
    <section className="panel">
      <header className="panel-title">
        <Database size={14} aria-hidden />
        Dataset
      </header>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-2 p-4 text-xs">
        <dt className="text-terminal-muted">Span</dt>
        <dd className="text-right font-mono tabular-nums">
          {market.start_date} → {market.end_date}
        </dd>
        <dt className="text-terminal-muted">Trading days</dt>
        <dd className="text-right font-mono tabular-nums">{market.n_days.toLocaleString()}</dd>
        <dt className="text-terminal-muted">Assets</dt>
        <dd className="text-right font-mono tabular-nums">{market.tickers.length}</dd>
        <dt className="text-terminal-muted">Seed</dt>
        <dd className="text-right font-mono tabular-nums">{market.seed}</dd>
        {Object.entries(market.true_regime_counts).map(([name, count]) => (
          <SummaryRow key={name} label={`True ${name} days`} value={count.toLocaleString()} />
        ))}
      </dl>
    </section>
  );
}

function SummaryRow({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <>
      <dt className="text-terminal-muted">{label}</dt>
      <dd className="text-right font-mono tabular-nums">{value}</dd>
    </>
  );
}
