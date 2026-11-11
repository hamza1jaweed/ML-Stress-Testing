"""Pydantic v2 request/response schemas for the API layer."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class GenerateDataRequest(BaseModel):
    """Parameters for seeding a synthetic market history."""

    model_config = ConfigDict(extra="forbid")

    n_days: Annotated[int, Field(ge=60, le=10_000)] = 1500
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 42


class TrainRegimesRequest(BaseModel):
    """Parameters for fitting the regime detection model."""

    model_config = ConfigDict(extra="forbid")

    n_regimes: Annotated[int, Field(ge=2, le=6)] = 3
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 7


class StressTestRequest(BaseModel):
    """Parameters for one stress-test run.

    ``weights`` maps ticker -> weight; values are renormalized to sum to 1.
    ``scenario`` switches to historical replay mode and ignores ``method``.
    ``shocks`` applies instantaneous day-0 asset-class shocks (simple
    returns, e.g. ``{"equity": -0.20}``) in Monte Carlo mode.
    """

    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float]
    n_paths: Annotated[int, Field(ge=100, le=100_000)] = 10_000
    horizon_days: Annotated[int, Field(ge=1, le=2520)] = 252
    method: Literal["parametric", "bootstrap"] = "parametric"
    initial_value: Annotated[float, Field(gt=0, le=1e12)] = 1_000_000.0
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 123
    scenario: str | None = None
    shocks: dict[str, float] | None = None
    # Simulate with the fitted Markov regime chain (requires a trained model).
    regime_aware: bool = False
    # Blend correlations toward the all-ones crisis matrix (parametric /
    # regime-aware methods only).
    correlation_stress: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    @field_validator("weights")
    @classmethod
    def _weights_valid(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("weights must not be empty")
        if any(w < 0 for w in v.values()):
            raise ValueError("weights must be non-negative")
        if sum(v.values()) <= 0:
            raise ValueError("weights must sum to a positive value")
        return v

    @field_validator("shocks")
    @classmethod
    def _shocks_valid(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is None:
            return v
        for cls_name, shock in v.items():
            if not -0.99 <= shock <= 2.0:
                raise ValueError(
                    f"shock for {cls_name!r} must be within [-0.99, 2.0], got {shock}"
                )
        return v


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


class AssetInfo(BaseModel):
    ticker: str
    name: str
    asset_class: str


class PricePoint(BaseModel):
    date: str
    values: dict[str, float]


class GenerateDataResponse(BaseModel):
    n_days: int
    tickers: list[str]
    assets: list[AssetInfo]
    start_date: str
    end_date: str
    seed: int
    true_regime_counts: dict[str, int]
    prices: list[PricePoint]


class RegimeStateParams(BaseModel):
    mean_log_return: float
    rolling_vol: float
    norm_atr: float


class TrainRegimesResponse(BaseModel):
    dates: list[str]
    states: list[int]
    state_names: dict[str, str]
    transition_matrix: list[list[float]]
    state_means: dict[str, RegimeStateParams]
    stationary_distribution: list[float]
    log_likelihood: float
    bic: float
    accuracy_vs_true: float | None
    feature_names: list[str]
    # Equal-weight market index aligned to `dates`, for the colored chart.
    market_index: list[float]


class RiskMetrics(BaseModel):
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_terminal: float
    worst_terminal: float
    best_terminal: float
    max_drawdown_median: float
    max_drawdown_p95: float
    prob_loss: float


class ScenarioInfo(BaseModel):
    key: str
    name: str
    description: str
    horizon_days: int
    shocks: dict[str, float]
    # Present only for regime-aware runs:
    start_state: int | None = None
    regime_occupancy: dict[str, float] | None = None


class StressTestResponse(BaseModel):
    method: str
    n_paths: int
    horizon_days: int
    initial_value: float
    metrics: RiskMetrics
    fan_chart: dict[str, list[float]]
    terminal_histogram: dict[str, list[float]]
    scenario: ScenarioInfo | None = None


class ScenarioListResponse(BaseModel):
    scenarios: list[ScenarioInfo]


class PortfolioAnalyticsRequest(BaseModel):
    """Weights for historical performance and risk decomposition."""

    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float]

    @field_validator("weights")
    @classmethod
    def _weights_valid(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("weights must not be empty")
        if any(w < 0 for w in v.values()):
            raise ValueError("weights must be non-negative")
        if sum(v.values()) <= 0:
            raise ValueError("weights must sum to a positive value")
        return v


class PerformanceMetrics(BaseModel):
    cagr: float
    annual_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    hist_var_95_daily: float
    skew: float
    excess_kurtosis: float


class VarBacktest(BaseModel):
    confidence: float
    window: int
    observations: int
    breaches: int
    expected_breaches: float
    breach_rate: float
    kupiec_lr: float
    p_value: float
    model_ok: bool


class RegimePerformanceRow(BaseModel):
    regime: int
    name: str
    days: int
    ann_return: float
    ann_vol: float
    sharpe: float
    worst_day: float
    hit_rate: float


class PortfolioAnalyticsResponse(BaseModel):
    dates: list[str]
    equity_curve: list[float]
    drawdown: list[float]
    metrics: PerformanceMetrics
    risk_contributions: dict[str, float]
    correlation: list[list[float]]
    tickers: list[str]
    var_backtest: VarBacktest | None = None
    # Present only when a regime model has been trained.
    regime_performance: list[RegimePerformanceRow] | None = None


class EfficientFrontierRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_samples: Annotated[int, Field(ge=100, le=20_000)] = 3000
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 11


class FrontierPoint(BaseModel):
    ret: float
    vol: float
    sharpe: float | None = None


class OptimalPortfolio(BaseModel):
    weights: dict[str, float]
    ret: float
    vol: float
    sharpe: float


class EfficientFrontierResponse(BaseModel):
    cloud: list[FrontierPoint]
    frontier: list[FrontierPoint]
    max_sharpe: OptimalPortfolio
    min_vol: OptimalPortfolio
    risk_parity: OptimalPortfolio
    tickers: list[str]


class ReverseStressRequest(BaseModel):
    """Solve for the most likely market move producing a target loss."""

    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float]
    target_loss: Annotated[float, Field(gt=0.0, le=0.9)] = 0.25
    horizon_days: Annotated[int, Field(ge=1, le=504)] = 21

    @field_validator("weights")
    @classmethod
    def _weights_valid(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("weights must not be empty")
        if any(w < 0 for w in v.values()):
            raise ValueError("weights must be non-negative")
        if sum(v.values()) <= 0:
            raise ValueError("weights must sum to a positive value")
        return v


class ReverseStressResponse(BaseModel):
    target_loss: float
    horizon_days: int
    shocks: dict[str, float]
    achieved_loss: float
    sigma_distance: float
    plausible: bool


class EarlyWarningResponse(BaseModel):
    current_state: int
    current_state_name: str
    bear_state: int
    already_in_bear: bool
    prob_within: dict[str, float]   # horizon (days) -> P(first bear visit)
    expected_days_to_bear: float


class RegimeForecastResponse(BaseModel):
    current_state: int
    current_state_name: str
    horizon_days: int
    # probabilities[t][k] = P(regime k at day t+1)
    probabilities: list[list[float]]
    state_names: dict[str, str]
    stationary_distribution: list[float]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    python_version: str
    numpy_version: str
    cpu_count: int
    memory_used_mb: float | None
    data_loaded: bool
    model_trained: bool
    uptime_seconds: float


class ErrorResponse(BaseModel):
    detail: str
