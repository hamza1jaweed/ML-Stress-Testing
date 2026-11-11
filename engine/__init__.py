"""Market Regime Detection & Portfolio Stress-Testing engine."""

from engine.data_generator import SyntheticMarketGenerator, AssetSpec, GeneratedMarket
from engine.portfolio_analytics import (
    FrontierResult,
    PortfolioAnalytics,
    compute_efficient_frontier,
    compute_portfolio_analytics,
    forecast_regime_probabilities,
)
from engine.regime_model import RegimeDetector, RegimeFitResult
from engine.stress_engine import StressTester, StressTestResult, HISTORICAL_SCENARIOS

__all__ = [
    "SyntheticMarketGenerator",
    "AssetSpec",
    "GeneratedMarket",
    "RegimeDetector",
    "RegimeFitResult",
    "StressTester",
    "StressTestResult",
    "HISTORICAL_SCENARIOS",
    "PortfolioAnalytics",
    "FrontierResult",
    "compute_portfolio_analytics",
    "compute_efficient_frontier",
    "forecast_regime_probabilities",
]
