"""API routes. State is held in a process-local store guarded by a lock.

The heavy NumPy work runs in a thread pool (``run_in_threadpool``) so the
event loop stays responsive while 10k+ paths are simulated.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from backend.schemas import (
    AssetInfo,
    EarlyWarningResponse,
    EfficientFrontierRequest,
    EfficientFrontierResponse,
    GenerateDataRequest,
    GenerateDataResponse,
    HealthResponse,
    PortfolioAnalyticsRequest,
    PortfolioAnalyticsResponse,
    PricePoint,
    RegimeForecastResponse,
    RegimeStateParams,
    ReverseStressRequest,
    ReverseStressResponse,
    ScenarioInfo,
    ScenarioListResponse,
    StressTestRequest,
    StressTestResponse,
    TrainRegimesRequest,
    TrainRegimesResponse,
)
from engine.data_generator import (
    DEFAULT_UNIVERSE,
    REGIME_BEAR_HIGH_VOL,
    GeneratedMarket,
    SyntheticMarketGenerator,
)
from engine.portfolio_analytics import (
    bear_hitting_stats,
    compute_efficient_frontier,
    compute_portfolio_analytics,
    compute_regime_performance,
    compute_risk_parity,
    compute_var_backtest,
    forecast_regime_probabilities,
)
from engine.regime_model import RegimeDetector, RegimeFitResult, build_features
from engine.stress_engine import HISTORICAL_SCENARIOS, StressTester

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

_START_TIME = time.monotonic()


@dataclass(slots=True)
class AppState:
    """Process-local mutable state shared across requests."""

    market: GeneratedMarket | None = None
    regime_result: RegimeFitResult | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


STATE = AppState()


def _require_market() -> GeneratedMarket:
    if STATE.market is None:
        raise HTTPException(
            status_code=409,
            detail="No market data loaded. Call POST /api/generate-data first.",
        )
    return STATE.market


def _require_regimes() -> RegimeFitResult:
    if STATE.regime_result is None:
        raise HTTPException(
            status_code=409,
            detail="No regime model trained. Call POST /api/train-regimes first.",
        )
    return STATE.regime_result


def _weight_vector(weights: dict[str, float], market: GeneratedMarket) -> np.ndarray:
    unknown = set(weights) - set(market.tickers)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown tickers in weights: {sorted(unknown)}. Valid: {market.tickers}",
        )
    return np.array([weights.get(t, 0.0) for t in market.tickers])


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post("/generate-data", response_model=GenerateDataResponse)
async def generate_data(req: GenerateDataRequest) -> GenerateDataResponse:
    """Seed a synthetic multi-asset market history."""

    def _work() -> GeneratedMarket:
        generator = SyntheticMarketGenerator()
        return generator.generate(n_days=req.n_days, seed=req.seed)

    market = await run_in_threadpool(_work)
    with STATE.lock:
        STATE.market = market
        STATE.regime_result = None  # stale model invalidated

    # Downsample prices for the wire (~400 points keeps charts snappy).
    stride = max(1, len(market.dates) // 400)
    idx = list(range(0, len(market.dates), stride))
    if idx[-1] != len(market.dates) - 1:
        idx.append(len(market.dates) - 1)
    prices = [
        PricePoint(
            date=market.dates[i],
            values={t: round(float(market.close.iloc[i][t]), 4) for t in market.tickers},
        )
        for i in idx
    ]
    summary = market.to_summary()
    return GenerateDataResponse(
        n_days=int(summary["n_days"]),  # type: ignore[arg-type]
        tickers=market.tickers,
        assets=[
            AssetInfo(ticker=a.ticker, name=a.name, asset_class=a.asset_class)
            for a in DEFAULT_UNIVERSE
        ],
        start_date=str(summary["start_date"]),
        end_date=str(summary["end_date"]),
        seed=market.seed,
        true_regime_counts=summary["true_regime_counts"],  # type: ignore[arg-type]
        prices=prices,
    )


@router.post("/train-regimes", response_model=TrainRegimesResponse)
async def train_regimes(req: TrainRegimesRequest) -> TrainRegimesResponse:
    """Fit the GMM regime model on the loaded market history."""
    market = _require_market()

    def _work() -> RegimeFitResult:
        features = build_features(market.close, market.high, market.low)
        detector = RegimeDetector(n_regimes=req.n_regimes, seed=req.seed)
        return detector.fit(features, true_regimes=market.true_regimes)

    try:
        result = await run_in_threadpool(_work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with STATE.lock:
        STATE.regime_result = result

    payload = result.to_payload()
    # Equal-weight normalized market index aligned to feature dates.
    aligned_close = market.close.loc[result.dates]
    index_values = (aligned_close / aligned_close.iloc[0]).mean(axis=1) * 100.0

    return TrainRegimesResponse(
        dates=payload["dates"],  # type: ignore[arg-type]
        states=payload["states"],  # type: ignore[arg-type]
        state_names=payload["state_names"],  # type: ignore[arg-type]
        transition_matrix=payload["transition_matrix"],  # type: ignore[arg-type]
        state_means={
            k: RegimeStateParams(**v) for k, v in payload["state_means"].items()  # type: ignore[union-attr]
        },
        stationary_distribution=payload["stationary_distribution"],  # type: ignore[arg-type]
        log_likelihood=float(payload["log_likelihood"]),  # type: ignore[arg-type]
        bic=float(payload["bic"]),  # type: ignore[arg-type]
        accuracy_vs_true=payload["accuracy_vs_true"],  # type: ignore[arg-type]
        feature_names=payload["feature_names"],  # type: ignore[arg-type]
        market_index=[round(float(v), 3) for v in index_values],
    )


@router.post("/stress-test", response_model=StressTestResponse)
async def stress_test(req: StressTestRequest) -> StressTestResponse:
    """Run Monte Carlo or historical-replay stress tests."""
    market = _require_market()
    weight_vector = _weight_vector(req.weights, market)
    regimes = _require_regimes() if req.regime_aware else None

    def _work() -> dict[str, object]:
        tester = StressTester(market.log_returns, market.asset_classes)
        if regimes is not None:
            result = tester.monte_carlo_regime_aware(
                weights=weight_vector,
                regime_states=regimes.states,
                transition=regimes.transition_matrix,
                start_state=int(regimes.states[-1]),
                n_paths=req.n_paths,
                horizon_days=req.horizon_days,
                initial_value=req.initial_value,
                seed=req.seed,
                shock_overrides=req.shocks,
                correlation_stress=req.correlation_stress,
            )
        elif req.scenario is not None:
            result = tester.historical_scenario(
                weights=weight_vector,
                scenario_key=req.scenario,
                n_paths=min(req.n_paths, 20_000),
                initial_value=req.initial_value,
                seed=req.seed,
            )
        else:
            result = tester.monte_carlo(
                weights=weight_vector,
                n_paths=req.n_paths,
                horizon_days=req.horizon_days,
                method=req.method,
                initial_value=req.initial_value,
                seed=req.seed,
                shock_overrides=req.shocks,
                correlation_stress=req.correlation_stress,
            )
        return result.to_payload()

    try:
        payload = await run_in_threadpool(_work)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc.args[0])) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return StressTestResponse.model_validate(payload)


@router.post("/portfolio-analytics", response_model=PortfolioAnalyticsResponse)
async def portfolio_analytics(req: PortfolioAnalyticsRequest) -> PortfolioAnalyticsResponse:
    """Historical performance, risk decomposition, and correlation matrix."""
    market = _require_market()
    weight_vector = _weight_vector(req.weights, market)
    regimes = STATE.regime_result  # optional: enriches the payload if trained

    def _work() -> dict[str, object]:
        payload = compute_portfolio_analytics(market.log_returns, weight_vector).to_payload()
        try:
            payload["var_backtest"] = compute_var_backtest(market.log_returns, weight_vector)
        except ValueError:
            payload["var_backtest"] = None  # dataset too short for the window
        if regimes is not None:
            payload["regime_performance"] = compute_regime_performance(
                market.log_returns, weight_vector, regimes.states, regimes.state_names
            )
        return payload

    try:
        payload = await run_in_threadpool(_work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PortfolioAnalyticsResponse.model_validate(payload)


@router.post("/efficient-frontier", response_model=EfficientFrontierResponse)
async def efficient_frontier(req: EfficientFrontierRequest) -> EfficientFrontierResponse:
    """Random-portfolio cloud plus SLSQP max-Sharpe / min-vol portfolios."""
    market = _require_market()

    def _work() -> dict[str, object]:
        payload = compute_efficient_frontier(
            market.log_returns, n_samples=req.n_samples, seed=req.seed
        ).to_payload()
        payload["risk_parity"] = compute_risk_parity(market.log_returns)
        return payload

    try:
        payload = await run_in_threadpool(_work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return EfficientFrontierResponse.model_validate(payload)


@router.post("/reverse-stress", response_model=ReverseStressResponse)
async def reverse_stress(req: ReverseStressRequest) -> ReverseStressResponse:
    """Most likely joint market move that produces the target loss."""
    market = _require_market()
    weight_vector = _weight_vector(req.weights, market)

    def _work() -> dict[str, object]:
        tester = StressTester(market.log_returns, market.asset_classes)
        return tester.reverse_stress(
            weights=weight_vector,
            target_loss=req.target_loss,
            horizon_days=req.horizon_days,
        )

    try:
        payload = await run_in_threadpool(_work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ReverseStressResponse.model_validate(payload)


@router.get("/early-warning", response_model=EarlyWarningResponse)
async def early_warning() -> EarlyWarningResponse:
    """Probability of entering the bear regime soon, from the nowcast."""
    regimes = _require_regimes()
    current = int(regimes.states[-1])

    def _work() -> dict[str, object]:
        return bear_hitting_stats(
            regimes.transition_matrix, current, REGIME_BEAR_HIGH_VOL
        )

    stats = await run_in_threadpool(_work)
    return EarlyWarningResponse(
        current_state=current,
        current_state_name=regimes.state_names.get(current, f"Regime {current}"),
        bear_state=REGIME_BEAR_HIGH_VOL,
        already_in_bear=bool(stats["already_in_bear"]),
        prob_within=stats["prob_within"],  # type: ignore[arg-type]
        expected_days_to_bear=float(stats["expected_days_to_bear"]),  # type: ignore[arg-type]
    )


@router.get("/regime-forecast", response_model=RegimeForecastResponse)
async def regime_forecast(horizon_days: int = 126) -> RegimeForecastResponse:
    """Regime probability evolution from the latest decoded state."""
    if not 1 <= horizon_days <= 2520:
        raise HTTPException(status_code=422, detail="horizon_days must be within [1, 2520]")
    regimes = _require_regimes()
    current = int(regimes.states[-1])

    def _work() -> np.ndarray:
        return forecast_regime_probabilities(
            regimes.transition_matrix, current, horizon_days
        )

    probs = await run_in_threadpool(_work)
    return RegimeForecastResponse(
        current_state=current,
        current_state_name=regimes.state_names.get(current, f"Regime {current}"),
        horizon_days=horizon_days,
        probabilities=np.round(probs, 6).tolist(),
        state_names={str(k): v for k, v in regimes.state_names.items()},
        stationary_distribution=np.round(regimes.stationary_distribution, 6).tolist(),
    )


@router.get("/scenarios", response_model=ScenarioListResponse)
async def list_scenarios() -> ScenarioListResponse:
    """Enumerate the built-in historical crisis scenarios."""
    return ScenarioListResponse(
        scenarios=[
            ScenarioInfo(
                key=s.key,
                name=s.name,
                description=s.description,
                horizon_days=s.horizon_days,
                shocks=s.shocks,
            )
            for s in HISTORICAL_SCENARIOS.values()
        ]
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Healthcheck with lightweight hardware metrics."""
    import os

    memory_mb: float | None = None
    try:
        import psutil

        memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    return HealthResponse(
        status="ok",
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        cpu_count=os.cpu_count() or 1,
        memory_used_mb=round(memory_mb, 1) if memory_mb is not None else None,
        data_loaded=STATE.market is not None,
        model_trained=STATE.regime_result is not None,
        uptime_seconds=round(time.monotonic() - _START_TIME, 1),
    )
