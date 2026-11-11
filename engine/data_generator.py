"""Synthetic multi-asset market data generator.

Produces correlated geometric Brownian motion with regime-switching drift,
volatility clustering (GARCH(1,1)-style variance recursion), and synthetic
OHLC bars so that ATR-style features can be computed downstream.

The generator embeds a *true* hidden Markov regime sequence (bull /
high-volatility bear / consolidation) so the unsupervised regime model has
genuine structure to recover. The whole pipeline therefore runs fully
offline with zero external API dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR: Final[int] = 252

# Semantic regime ids used throughout the engine.
REGIME_BULL: Final[int] = 0
REGIME_BEAR_HIGH_VOL: Final[int] = 1
REGIME_CONSOLIDATION: Final[int] = 2
REGIME_NAMES: Final[dict[int, str]] = {
    REGIME_BULL: "Bull",
    REGIME_BEAR_HIGH_VOL: "High Volatility Bear",
    REGIME_CONSOLIDATION: "Consolidation",
}


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """Static description of one synthetic asset.

    Attributes:
        ticker: Display ticker, e.g. ``"EQ_US"``.
        name: Human-readable name.
        annual_drift: Baseline annualized drift in the bull regime.
        annual_vol: Baseline annualized volatility in the bull regime.
        start_price: Initial price level.
        asset_class: One of ``equity | bond | commodity | crypto``; used by
            historical scenario shock mapping.
    """

    ticker: str
    name: str
    annual_drift: float
    annual_vol: float
    start_price: float
    asset_class: str


DEFAULT_UNIVERSE: Final[tuple[AssetSpec, ...]] = (
    AssetSpec("EQ_US", "US Large-Cap Equity", 0.09, 0.16, 450.0, "equity"),
    AssetSpec("EQ_EM", "Emerging Markets Equity", 0.07, 0.22, 42.0, "equity"),
    AssetSpec("BOND_10Y", "10Y Treasury Bond Fund", 0.03, 0.07, 98.0, "bond"),
    AssetSpec("GOLD", "Gold", 0.04, 0.14, 1900.0, "commodity"),
    AssetSpec("CRYPTO_BTC", "Bitcoin", 0.25, 0.65, 45_000.0, "crypto"),
)

# Target cross-asset return correlation (bull/normal conditions).
_DEFAULT_CORRELATION: Final[np.ndarray] = np.array(
    [
        [1.00, 0.75, -0.25, 0.10, 0.35],
        [0.75, 1.00, -0.15, 0.15, 0.30],
        [-0.25, -0.15, 1.00, 0.20, -0.05],
        [0.10, 0.15, 0.20, 1.00, 0.10],
        [0.35, 0.30, -0.05, 0.10, 1.00],
    ]
)

# Per-regime multipliers applied to (drift, vol) of every asset class.
# Bear regime flips equity/crypto drift sign and inflates volatility.
_REGIME_DRIFT_MULT: Final[dict[int, dict[str, float]]] = {
    REGIME_BULL: {"equity": 1.0, "bond": 1.0, "commodity": 1.0, "crypto": 1.0},
    REGIME_BEAR_HIGH_VOL: {"equity": -3.5, "bond": 1.6, "commodity": 1.8, "crypto": -2.5},
    REGIME_CONSOLIDATION: {"equity": 0.15, "bond": 0.8, "commodity": 0.4, "crypto": 0.1},
}
_REGIME_VOL_MULT: Final[dict[int, dict[str, float]]] = {
    REGIME_BULL: {"equity": 1.0, "bond": 1.0, "commodity": 1.0, "crypto": 1.0},
    REGIME_BEAR_HIGH_VOL: {"equity": 2.6, "bond": 1.4, "commodity": 1.7, "crypto": 2.2},
    REGIME_CONSOLIDATION: {"equity": 0.45, "bond": 0.7, "commodity": 0.55, "crypto": 0.5},
}

# Row-stochastic transition matrix of the *true* generating regime chain.
_TRUE_TRANSITION: Final[np.ndarray] = np.array(
    [
        [0.990, 0.004, 0.006],
        [0.015, 0.975, 0.010],
        [0.012, 0.006, 0.982],
    ]
)


@dataclass(slots=True)
class GeneratedMarket:
    """Container for one generated market history."""

    dates: list[str]
    tickers: list[str]
    asset_classes: dict[str, str]
    close: pd.DataFrame          # (T, N) close prices, columns = tickers
    high: pd.DataFrame           # (T, N) intraday highs
    low: pd.DataFrame            # (T, N) intraday lows
    log_returns: pd.DataFrame    # (T-1, N) daily log returns
    true_regimes: np.ndarray     # (T,) int regime labels used by the generator
    seed: int
    correlation: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    def to_summary(self) -> dict[str, object]:
        """Compact JSON-safe summary for API responses."""
        return {
            "n_days": len(self.dates),
            "tickers": self.tickers,
            "start_date": self.dates[0],
            "end_date": self.dates[-1],
            "seed": self.seed,
            "true_regime_counts": {
                REGIME_NAMES[r]: int((self.true_regimes == r).sum())
                for r in sorted(REGIME_NAMES)
            },
        }


def _nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the positive-definite cone.

    Uses eigenvalue clipping; adequate for correlation matrices that are
    nearly PD after regime perturbations.
    """
    sym = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    clipped = np.clip(eigenvalues, 1e-8, None)
    rebuilt = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    # Re-normalize diagonal to exactly 1.
    d = np.sqrt(np.diag(rebuilt))
    rebuilt = rebuilt / np.outer(d, d)
    np.fill_diagonal(rebuilt, 1.0)
    return rebuilt


class SyntheticMarketGenerator:
    """Correlated regime-switching GBM generator with volatility clustering.

    The daily log-return of asset ``i`` on day ``t`` is

    ``r_it = (mu_i(s_t) - 0.5 * sigma_it^2) * dt + sigma_it * sqrt(dt) * z_it``

    where ``s_t`` follows a Markov chain, ``z_t`` is multivariate normal
    with the configured cross-asset correlation, and ``sigma_it`` follows a
    GARCH(1,1)-style recursion around the regime's target volatility so
    large shocks cluster.
    """

    def __init__(
        self,
        universe: tuple[AssetSpec, ...] = DEFAULT_UNIVERSE,
        correlation: np.ndarray | None = None,
        garch_alpha: float = 0.08,
        garch_beta: float = 0.88,
    ) -> None:
        if not universe:
            raise ValueError("universe must contain at least one asset")
        if not 0.0 < garch_alpha + garch_beta < 1.0:
            raise ValueError("garch_alpha + garch_beta must lie in (0, 1)")
        self.universe = universe
        n = len(universe)
        corr = _DEFAULT_CORRELATION[:n, :n] if correlation is None else np.asarray(correlation, dtype=float)
        if corr.shape != (n, n):
            raise ValueError(f"correlation must be {(n, n)}, got {corr.shape}")
        self.correlation = _nearest_positive_definite(corr)
        self._chol = np.linalg.cholesky(self.correlation)
        self.garch_alpha = garch_alpha
        self.garch_beta = garch_beta

    def generate(
        self,
        n_days: int = 1500,
        seed: int = 42,
        start: date | None = None,
    ) -> GeneratedMarket:
        """Generate ``n_days`` of correlated OHLC market history.

        Args:
            n_days: Number of trading days (minimum 60 so rolling features
                are computable downstream).
            seed: RNG seed; identical inputs yield identical output.
            start: First calendar date; defaults to today minus the span.

        Raises:
            ValueError: If ``n_days`` is below the minimum.
        """
        if n_days < 60:
            raise ValueError("n_days must be >= 60 for meaningful features")

        rng = np.random.default_rng(seed)
        n_assets = len(self.universe)
        dt = 1.0 / TRADING_DAYS_PER_YEAR

        regimes = self._simulate_regime_chain(n_days, rng)

        base_vol = np.array([a.annual_vol for a in self.universe])
        base_drift = np.array([a.annual_drift for a in self.universe])
        classes = [a.asset_class for a in self.universe]

        # Per-day regime-adjusted targets.
        drift_mult = np.array(
            [[_REGIME_DRIFT_MULT[int(s)][c] for c in classes] for s in regimes]
        )
        vol_mult = np.array(
            [[_REGIME_VOL_MULT[int(s)][c] for c in classes] for s in regimes]
        )
        target_var = (base_vol[None, :] * vol_mult) ** 2  # annualized variance targets

        log_returns = np.zeros((n_days, n_assets))
        alpha, beta = self.garch_alpha, self.garch_beta
        omega_weight = 1.0 - alpha - beta

        shocks = rng.standard_normal((n_days, n_assets)) @ self._chol.T

        # Volatility clustering as a mean-one GARCH(1,1) multiplier on top of
        # the regime's variance target: the regime switch moves volatility
        # immediately, while the multiplier makes large shocks cluster.
        cluster = np.ones(n_assets)
        for t in range(n_days):
            if t > 0:
                cluster = omega_weight + alpha * shocks[t - 1] ** 2 + beta * cluster
            variance = target_var[t] * cluster
            sigma = np.sqrt(variance)
            mu = base_drift * drift_mult[t]
            log_returns[t] = (mu - 0.5 * variance) * dt + sigma * np.sqrt(dt) * shocks[t]

        start_prices = np.array([a.start_price for a in self.universe])
        log_price = np.cumsum(log_returns, axis=0) + np.log(start_prices)[None, :]
        close = np.exp(log_price)

        # Synthetic intraday range: scaled by instantaneous volatility so ATR
        # expands in stressed regimes.
        daily_sigma = np.sqrt(target_var) * np.sqrt(dt)
        range_frac = np.abs(rng.standard_normal((n_days, n_assets))) * daily_sigma * 1.6
        high = close * (1.0 + range_frac / 2.0)
        low = close * (1.0 - range_frac / 2.0)
        low = np.minimum(low, close)  # guard against inverted bars
        high = np.maximum(high, close)

        if start is None:
            start = date.today() - timedelta(days=int(n_days * 1.45))
        dates = pd.bdate_range(start=start, periods=n_days)
        date_strings = [d.strftime("%Y-%m-%d") for d in dates]
        tickers = [a.ticker for a in self.universe]

        close_df = pd.DataFrame(close, index=date_strings, columns=tickers)
        returns_df = pd.DataFrame(
            log_returns[1:], index=date_strings[1:], columns=tickers
        )

        logger.info(
            "Generated %d days x %d assets (seed=%d)", n_days, n_assets, seed
        )
        return GeneratedMarket(
            dates=date_strings,
            tickers=tickers,
            asset_classes={a.ticker: a.asset_class for a in self.universe},
            close=close_df,
            high=pd.DataFrame(high, index=date_strings, columns=tickers),
            low=pd.DataFrame(low, index=date_strings, columns=tickers),
            log_returns=returns_df,
            true_regimes=regimes,
            seed=seed,
            correlation=self.correlation,
        )

    def _simulate_regime_chain(self, n_days: int, rng: np.random.Generator) -> np.ndarray:
        """Simulate the hidden Markov regime path."""
        regimes = np.zeros(n_days, dtype=np.int64)
        regimes[0] = REGIME_BULL
        cumulative = np.cumsum(_TRUE_TRANSITION, axis=1)
        draws = rng.random(n_days)
        for t in range(1, n_days):
            regimes[t] = int(np.searchsorted(cumulative[regimes[t - 1]], draws[t]))
        return regimes


class LiveDataFetcher:
    """Thin cached wrapper around ``yfinance`` for pulling real tickers.

    The import is deferred so the rest of the platform has zero hard
    dependency on network access or the ``yfinance`` package.
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        from pathlib import Path
        import tempfile

        self._cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "regime_platform_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        tickers: list[str],
        period: str = "5y",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices, caching to parquet on disk.

        Args:
            tickers: Yahoo Finance ticker symbols.
            period: yfinance period string, e.g. ``"5y"``.
            force_refresh: Skip the cache and re-download.

        Returns:
            DataFrame of close prices indexed by date, one column per ticker.

        Raises:
            RuntimeError: If ``yfinance`` is unavailable or the download
                returns no data.
        """
        if not tickers:
            raise ValueError("tickers must be non-empty")
        key = "_".join(sorted(t.upper() for t in tickers)) + f"_{period}"
        cache_file = self._cache_dir / f"{key}.parquet"

        if cache_file.exists() and not force_refresh:
            logger.info("Loading cached prices from %s", cache_file)
            return pd.read_parquet(cache_file)

        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "yfinance is not installed; install it or use the synthetic generator"
            ) from exc

        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no data for {tickers}")
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        close = close.dropna(how="all").ffill().dropna()
        close.to_parquet(cache_file)
        logger.info("Cached %d rows to %s", len(close), cache_file)
        return close
