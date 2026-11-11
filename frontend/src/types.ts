/** API contract types — mirror backend/schemas.py exactly. */

export interface AssetInfo {
  ticker: string;
  name: string;
  asset_class: string;
}

export interface PricePoint {
  date: string;
  values: Record<string, number>;
}

export interface GenerateDataResponse {
  n_days: number;
  tickers: string[];
  assets: AssetInfo[];
  start_date: string;
  end_date: string;
  seed: number;
  true_regime_counts: Record<string, number>;
  prices: PricePoint[];
}

export interface RegimeStateParams {
  mean_log_return: number;
  rolling_vol: number;
  norm_atr: number;
}

export interface TrainRegimesResponse {
  dates: string[];
  states: number[];
  state_names: Record<string, string>;
  transition_matrix: number[][];
  state_means: Record<string, RegimeStateParams>;
  stationary_distribution: number[];
  log_likelihood: number;
  bic: number;
  accuracy_vs_true: number | null;
  feature_names: string[];
  market_index: number[];
}

export interface RiskMetrics {
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  expected_terminal: number;
  worst_terminal: number;
  best_terminal: number;
  max_drawdown_median: number;
  max_drawdown_p95: number;
  prob_loss: number;
}

export interface ScenarioInfo {
  key: string;
  name: string;
  description: string;
  horizon_days: number;
  shocks: Record<string, number>;
  start_state?: number | null;
  regime_occupancy?: Record<string, number> | null;
}

export interface StressTestResponse {
  method: string;
  n_paths: number;
  horizon_days: number;
  initial_value: number;
  metrics: RiskMetrics;
  fan_chart: Record<string, number[]>;
  terminal_histogram: { bin_edges: number[]; counts: number[] };
  scenario: ScenarioInfo | null;
}

export interface HealthResponse {
  status: "ok";
  python_version: string;
  numpy_version: string;
  cpu_count: number;
  memory_used_mb: number | null;
  data_loaded: boolean;
  model_trained: boolean;
  uptime_seconds: number;
}

export interface StressTestRequest {
  weights: Record<string, number>;
  n_paths: number;
  horizon_days: number;
  method: "parametric" | "bootstrap";
  initial_value: number;
  seed: number;
  scenario?: string | null;
  shocks?: Record<string, number> | null;
  regime_aware?: boolean;
  correlation_stress?: number;
}

export interface PerformanceMetrics {
  cagr: number;
  annual_vol: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  max_drawdown: number;
  hist_var_95_daily: number;
  skew: number;
  excess_kurtosis: number;
}

export interface VarBacktest {
  confidence: number;
  window: number;
  observations: number;
  breaches: number;
  expected_breaches: number;
  breach_rate: number;
  kupiec_lr: number;
  p_value: number;
  model_ok: boolean;
}

export interface RegimePerformanceRow {
  regime: number;
  name: string;
  days: number;
  ann_return: number;
  ann_vol: number;
  sharpe: number;
  worst_day: number;
  hit_rate: number;
}

export interface PortfolioAnalyticsResponse {
  dates: string[];
  equity_curve: number[];
  drawdown: number[];
  metrics: PerformanceMetrics;
  risk_contributions: Record<string, number>;
  correlation: number[][];
  tickers: string[];
  var_backtest: VarBacktest | null;
  regime_performance: RegimePerformanceRow[] | null;
}

export interface FrontierPoint {
  ret: number;
  vol: number;
  sharpe?: number | null;
}

export interface OptimalPortfolio {
  weights: Record<string, number>;
  ret: number;
  vol: number;
  sharpe: number;
}

export interface EfficientFrontierResponse {
  cloud: FrontierPoint[];
  frontier: FrontierPoint[];
  max_sharpe: OptimalPortfolio;
  min_vol: OptimalPortfolio;
  risk_parity: OptimalPortfolio;
  tickers: string[];
}

export interface ReverseStressResponse {
  target_loss: number;
  horizon_days: number;
  shocks: Record<string, number>;
  achieved_loss: number;
  sigma_distance: number;
  plausible: boolean;
}

export interface EarlyWarningResponse {
  current_state: number;
  current_state_name: string;
  bear_state: number;
  already_in_bear: boolean;
  prob_within: Record<string, number>;
  expected_days_to_bear: number;
}

export interface RegimeForecastResponse {
  current_state: number;
  current_state_name: string;
  horizon_days: number;
  probabilities: number[][];
  state_names: Record<string, string>;
  stationary_distribution: number[];
}
