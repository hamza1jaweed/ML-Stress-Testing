"""Portfolio stress-testing engine.

Capabilities:
  * Parametric Monte Carlo — multivariate normal daily log returns drawn
    from the historical mean vector and covariance matrix (Cholesky).
  * Non-parametric Monte Carlo — stationary block bootstrap of historical
    daily return rows, preserving cross-asset correlation and short-range
    autocorrelation / volatility clustering.
  * Historical scenario replay — canonical crisis shock templates (2008
    GFC, 2020 liquidity crunch, 2013 taper tantrum, stagflation) mapped
    onto the portfolio via asset-class betas, plus user-defined custom
    shocks (equity %, rate bps, etc.).
  * Risk metrics — VaR (95/99), CVaR / Expected Shortfall, maximum
    drawdown, and percentile fan paths for charting.

All simulation is fully vectorized NumPy with explicit seed locking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MCMethod = Literal["parametric", "bootstrap"]

FAN_PERCENTILES: Final[tuple[float, ...]] = (5.0, 25.0, 50.0, 75.0, 95.0)


@dataclass(frozen=True, slots=True)
class HistoricalScenario:
    """A crisis template expressed as total shock per asset class.

    ``shocks`` are cumulative simple returns applied over ``horizon_days``
    (e.g. -0.55 means the asset class loses 55% across the scenario), and
    ``vol_multiplier`` scales the path noise around the shock trajectory.
    """

    key: str
    name: str
    description: str
    horizon_days: int
    shocks: dict[str, float]
    vol_multiplier: float


HISTORICAL_SCENARIOS: Final[dict[str, HistoricalScenario]] = {
    "gfc_2008": HistoricalScenario(
        key="gfc_2008",
        name="2008 Global Financial Crisis",
        description="Sep 2008 – Mar 2009: equity collapse, flight to treasuries, broad deleveraging.",
        horizon_days=126,
        shocks={"equity": -0.45, "bond": 0.08, "commodity": -0.25, "crypto": -0.60},
        vol_multiplier=2.8,
    ),
    "covid_2020": HistoricalScenario(
        key="covid_2020",
        name="2020 COVID Liquidity Crunch",
        description="Feb–Mar 2020: fastest 30% equity drawdown on record, everything-sells-off liquidity spiral.",
        horizon_days=23,
        shocks={"equity": -0.34, "bond": -0.02, "commodity": -0.12, "crypto": -0.50},
        vol_multiplier=3.5,
    ),
    "taper_2013": HistoricalScenario(
        key="taper_2013",
        name="2013 Taper Tantrum",
        description="May–Sep 2013: rapid rate rise, duration and EM pain, equities range-bound.",
        horizon_days=85,
        shocks={"equity": -0.06, "bond": -0.10, "commodity": -0.08, "crypto": -0.15},
        vol_multiplier=1.6,
    ),
    "stagflation": HistoricalScenario(
        key="stagflation",
        name="Stagflation Shock (1970s-style)",
        description="Hypothetical replay: persistent inflation, rising yields, real-asset bid.",
        horizon_days=252,
        shocks={"equity": -0.20, "bond": -0.18, "commodity": 0.35, "crypto": -0.10},
        vol_multiplier=1.8,
    ),
}


@dataclass(slots=True)
class StressTestResult:
    """Full output of one stress-test run (JSON-safe via ``to_payload``)."""

    method: str
    n_paths: int
    horizon_days: int
    initial_value: float
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
    fan_chart: dict[str, list[float]] = field(default_factory=dict)
    terminal_histogram: dict[str, list[float]] = field(default_factory=dict)
    scenario: dict[str, object] | None = None

    def to_payload(self) -> dict[str, object]:
        payload = {
            "method": self.method,
            "n_paths": self.n_paths,
            "horizon_days": self.horizon_days,
            "initial_value": round(self.initial_value, 2),
            "metrics": {
                "var_95": round(self.var_95, 2),
                "var_99": round(self.var_99, 2),
                "cvar_95": round(self.cvar_95, 2),
                "cvar_99": round(self.cvar_99, 2),
                "expected_terminal": round(self.expected_terminal, 2),
                "worst_terminal": round(self.worst_terminal, 2),
                "best_terminal": round(self.best_terminal, 2),
                "max_drawdown_median": round(self.max_drawdown_median, 4),
                "max_drawdown_p95": round(self.max_drawdown_p95, 4),
                "prob_loss": round(self.prob_loss, 4),
            },
            "fan_chart": self.fan_chart,
            "terminal_histogram": self.terminal_histogram,
        }
        if self.scenario is not None:
            payload["scenario"] = self.scenario
        return payload


def _validate_weights(weights: np.ndarray, n_assets: int) -> np.ndarray:
    """Validate and renormalize a weight vector."""
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != n_assets:
        raise ValueError(f"expected {n_assets} weights, got {w.shape[0]}")
    if np.any(w < -1e-9):
        raise ValueError("short positions are not supported; weights must be >= 0")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    if not np.isclose(total, 1.0, atol=1e-6):
        w = w / total
    return w


def max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Max drawdown per path for a (P, T) matrix of portfolio values."""
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = paths / running_max - 1.0
    return drawdowns.min(axis=1)


def blend_correlation(cov: np.ndarray, stress: float) -> np.ndarray:
    """Blend a covariance matrix's correlation structure toward 1.

    ``stress`` in [0, 1] linearly interpolates the correlation matrix
    between its historical value (0) and the all-ones "crisis" matrix (1),
    keeping marginal variances fixed. Both endpoints are PSD, so every
    convex combination is too.
    """
    if not 0.0 <= stress <= 1.0:
        raise ValueError("correlation stress must be within [0, 1]")
    if stress == 0.0:
        return cov
    std = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    blended = (1.0 - stress) * corr + stress * np.ones_like(corr)
    return blended * np.outer(std, std)


class StressTester:
    """Monte Carlo and scenario stress-testing against custom weights."""

    def __init__(self, log_returns: pd.DataFrame, asset_classes: dict[str, str]) -> None:
        if log_returns.empty:
            raise ValueError("log_returns must be non-empty")
        self.log_returns = log_returns
        self.tickers: list[str] = list(log_returns.columns)
        self.asset_classes = asset_classes
        self._returns = log_returns.to_numpy(dtype=float)
        self._mu = self._returns.mean(axis=0)
        self._cov = np.cov(self._returns, rowvar=False)
        # Jitter guards Cholesky against near-singular covariance.
        self._chol = np.linalg.cholesky(
            self._cov + np.eye(len(self.tickers)) * 1e-12
        )
        std = np.sqrt(np.maximum(np.diag(self._cov), 1e-18))
        corr = self._cov / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        self._chol_corr = np.linalg.cholesky(corr + np.eye(len(self.tickers)) * 1e-10)

    # ------------------------------------------------------------------ #
    # Monte Carlo
    # ------------------------------------------------------------------ #

    def monte_carlo(
        self,
        weights: np.ndarray,
        n_paths: int = 10_000,
        horizon_days: int = 252,
        method: MCMethod = "parametric",
        initial_value: float = 1_000_000.0,
        seed: int = 123,
        shock_overrides: dict[str, float] | None = None,
        correlation_stress: float = 0.0,
    ) -> StressTestResult:
        """Simulate portfolio paths and compute the risk metric suite.

        Args:
            weights: Portfolio weights aligned to ``self.tickers``.
            n_paths: Number of simulated paths (>= 1000 recommended).
            horizon_days: Simulation horizon in trading days.
            method: ``"parametric"`` (Gaussian) or ``"bootstrap"``
                (stationary block bootstrap of historical rows).
            initial_value: Starting portfolio value in currency units.
            seed: RNG seed for exact reproducibility.
            shock_overrides: Optional per-asset-class instantaneous shocks
                (simple returns) applied on day 0, e.g.
                ``{"equity": -0.20, "bond": -0.05}``.
            correlation_stress: Blend of the correlation matrix toward the
                all-ones "crisis" matrix (parametric method only; the
                bootstrap resamples history and cannot be re-correlated).
        """
        if n_paths < 100:
            raise ValueError("n_paths must be >= 100")
        if not 1 <= horizon_days <= 2520:
            raise ValueError("horizon_days must be within [1, 2520]")
        if initial_value <= 0:
            raise ValueError("initial_value must be positive")

        w = _validate_weights(weights, len(self.tickers))
        rng = np.random.default_rng(seed)

        if method == "parametric":
            sim = self._simulate_parametric(rng, n_paths, horizon_days, correlation_stress)
        elif method == "bootstrap":
            if correlation_stress > 0:
                raise ValueError(
                    "correlation_stress requires the parametric method; the "
                    "bootstrap resamples history and cannot be re-correlated"
                )
            sim = self._simulate_bootstrap(rng, n_paths, horizon_days)
        else:
            raise ValueError(f"unknown method: {method!r}")

        # Day-0 shock: deform starting asset levels by class shocks.
        shock_factor = 1.0
        if shock_overrides:
            per_asset = np.array(
                [1.0 + shock_overrides.get(self.asset_classes.get(t, ""), 0.0) for t in self.tickers]
            )
            if np.any(per_asset <= 0):
                raise ValueError("a shock of -100% or worse is not supported")
            shock_factor = float(np.dot(w, per_asset))

        # Portfolio simple-return paths from per-asset log returns.
        asset_growth = np.exp(np.cumsum(sim, axis=1))          # (P, T, N)
        port_values = initial_value * shock_factor * np.einsum(
            "ptn,n->pt", asset_growth, w
        )
        start_col = np.full((n_paths, 1), initial_value * shock_factor)
        port_values = np.concatenate([start_col, port_values], axis=1)  # (P, T+1)

        return self._build_result(
            port_values,
            method=method,
            n_paths=n_paths,
            horizon_days=horizon_days,
            initial_value=initial_value,
        )

    def _simulate_parametric(
        self,
        rng: np.random.Generator,
        n_paths: int,
        horizon: int,
        correlation_stress: float = 0.0,
    ) -> np.ndarray:
        """(P, T, N) correlated Gaussian daily log returns."""
        if correlation_stress > 0:
            cov = blend_correlation(self._cov, correlation_stress)
            chol = np.linalg.cholesky(cov + np.eye(len(self.tickers)) * 1e-12)
        else:
            chol = self._chol
        z = rng.standard_normal((n_paths, horizon, len(self.tickers)))
        return self._mu[None, None, :] + z @ chol.T

    def _simulate_bootstrap(
        self, rng: np.random.Generator, n_paths: int, horizon: int, block: int = 10
    ) -> np.ndarray:
        """(P, T, N) block-bootstrapped historical return rows.

        Sampling whole rows preserves the empirical cross-asset copula
        (fat tails included); contiguous blocks preserve volatility
        clustering.
        """
        t_hist = self._returns.shape[0]
        n_blocks = int(np.ceil(horizon / block))
        starts = rng.integers(0, t_hist, size=(n_paths, n_blocks))
        offsets = np.arange(block)
        idx = (starts[:, :, None] + offsets[None, None, :]) % t_hist  # (P, B, block)
        idx = idx.reshape(n_paths, -1)[:, :horizon]
        return self._returns[idx]  # fancy-index -> (P, T, N)

    # ------------------------------------------------------------------ #
    # Regime-aware Monte Carlo
    # ------------------------------------------------------------------ #

    def monte_carlo_regime_aware(
        self,
        weights: np.ndarray,
        regime_states: np.ndarray,
        transition: np.ndarray,
        start_state: int,
        n_paths: int = 10_000,
        horizon_days: int = 252,
        initial_value: float = 1_000_000.0,
        seed: int = 123,
        shock_overrides: dict[str, float] | None = None,
        correlation_stress: float = 0.0,
    ) -> StressTestResult:
        """Monte Carlo that switches return distributions with a Markov chain.

        Each path simulates its own regime sequence from the fitted
        transition matrix (starting at ``start_state``, the latest decoded
        regime), and draws daily log returns from that regime's historical
        Gaussian (per-regime mean vector + covariance). Fat regime tails are
        therefore endogenous: paths that wander into the bear state inherit
        its vol and drift.

        Args:
            regime_states: Decoded regime label per historical return row,
                aligned to the *tail* of ``self.log_returns`` (feature
                warm-up rows at the start may be unlabeled).
            transition: (K, K) row-stochastic matrix from the regime fit.
            start_state: Regime to start every path in (the nowcast).
        """
        if n_paths < 100:
            raise ValueError("n_paths must be >= 100")
        if not 1 <= horizon_days <= 2520:
            raise ValueError("horizon_days must be within [1, 2520]")
        if initial_value <= 0:
            raise ValueError("initial_value must be positive")

        w = _validate_weights(weights, len(self.tickers))
        t_matrix = np.asarray(transition, dtype=float)
        k = t_matrix.shape[0]
        if t_matrix.shape != (k, k):
            raise ValueError("transition matrix must be square")
        if not 0 <= start_state < k:
            raise ValueError(f"start_state must be in [0, {k})")

        states = np.asarray(regime_states, dtype=np.int64).ravel()
        if states.shape[0] > self._returns.shape[0]:
            raise ValueError("more regime labels than return rows")
        aligned = self._returns[-states.shape[0]:]

        # Per-regime moments; fall back to unconditional for sparse regimes.
        n_assets = len(self.tickers)
        mus = np.empty((k, n_assets))
        chols = np.empty((k, n_assets, n_assets))
        jitter = np.eye(n_assets) * 1e-12
        for regime in range(k):
            sub = aligned[states == regime]
            if sub.shape[0] < n_assets + 2:
                logger.warning(
                    "regime %d has only %d observations; using unconditional moments",
                    regime,
                    sub.shape[0],
                )
                mus[regime] = self._mu
                chols[regime] = np.linalg.cholesky(
                    blend_correlation(self._cov, correlation_stress) + jitter
                )
                continue
            mus[regime] = sub.mean(axis=0)
            regime_cov = blend_correlation(np.cov(sub, rowvar=False), correlation_stress)
            chols[regime] = np.linalg.cholesky(regime_cov + jitter)

        rng = np.random.default_rng(seed)

        # Simulate every path's regime chain (vectorized across paths).
        cumulative = np.cumsum(t_matrix, axis=1)
        path_states = np.empty((n_paths, horizon_days), dtype=np.int64)
        current = np.full(n_paths, start_state, dtype=np.int64)
        draws = rng.random((horizon_days, n_paths))
        for t in range(horizon_days):
            current = (draws[t][:, None] > cumulative[current]).sum(axis=1)
            path_states[:, t] = current

        # Draw returns regime-by-regime through boolean masks.
        z = rng.standard_normal((n_paths, horizon_days, n_assets))
        sim = np.empty_like(z)
        for regime in range(k):
            mask = path_states == regime
            if mask.any():
                sim[mask] = mus[regime] + z[mask] @ chols[regime].T

        shock_factor = 1.0
        if shock_overrides:
            per_asset = np.array(
                [1.0 + shock_overrides.get(self.asset_classes.get(t, ""), 0.0) for t in self.tickers]
            )
            if np.any(per_asset <= 0):
                raise ValueError("a shock of -100% or worse is not supported")
            shock_factor = float(np.dot(w, per_asset))

        asset_growth = np.exp(np.cumsum(sim, axis=1))
        port_values = initial_value * shock_factor * np.einsum("ptn,n->pt", asset_growth, w)
        start_col = np.full((n_paths, 1), initial_value * shock_factor)
        port_values = np.concatenate([start_col, port_values], axis=1)

        result = self._build_result(
            port_values,
            method="regime_aware",
            n_paths=n_paths,
            horizon_days=horizon_days,
            initial_value=initial_value,
        )
        # Share of simulated days spent in each regime, for the UI.
        result.scenario = {
            "key": "regime_aware",
            "name": "Regime-Aware Monte Carlo",
            "description": (
                "Paths switch between fitted regime distributions via the "
                "estimated Markov chain."
            ),
            "horizon_days": horizon_days,
            "shocks": {},
            "start_state": int(start_state),
            "regime_occupancy": {
                str(r): round(float((path_states == r).mean()), 4) for r in range(k)
            },
        }
        return result

    # ------------------------------------------------------------------ #
    # Reverse stress testing
    # ------------------------------------------------------------------ #

    def reverse_stress(
        self,
        weights: np.ndarray,
        target_loss: float,
        horizon_days: int = 21,
    ) -> dict[str, object]:
        """Find the *most likely* joint market move that loses ``target_loss``.

        Solves ``min_s  s' Sigma^-1 s  subject to  w's = -L`` — the smallest
        Mahalanobis-distance (i.e. highest-likelihood) shock vector under the
        historical covariance scaled to the horizon. Closed form:

            ``s* = -L * (Sigma w) / (w' Sigma w)``

        so each asset's shock is proportional to its covariance with the
        portfolio: diversifiers (e.g. bonds) can legitimately move *up* in
        the most plausible loss scenario.

        Args:
            weights: Portfolio weights aligned to ``self.tickers``.
            target_loss: Loss fraction of portfolio value in (0, 0.9].
            horizon_days: Window the move happens over (sets the sigma scale).

        Returns:
            Per-asset shock vector (simple returns), the Mahalanobis sigma
            distance, and the horizon.
        """
        if not 0.0 < target_loss <= 0.9:
            raise ValueError("target_loss must be within (0, 0.9]")
        if not 1 <= horizon_days <= 504:
            raise ValueError("horizon_days must be within [1, 504]")

        w = _validate_weights(weights, len(self.tickers))
        # Simple-return covariance scaled to the horizon.
        simple = np.expm1(self._returns)
        cov_h = np.cov(simple, rowvar=False) * horizon_days

        sigma_w = cov_h @ w
        port_var = float(w @ sigma_w)
        if port_var <= 0:
            raise ValueError("portfolio variance is non-positive")

        shock = -target_loss * sigma_w / port_var          # (N,)
        sigma_distance = target_loss / float(np.sqrt(port_var))

        achieved = float(w @ shock)                         # == -target_loss
        return {
            "target_loss": target_loss,
            "horizon_days": horizon_days,
            "shocks": {
                t: round(float(s), 6) for t, s in zip(self.tickers, shock)
            },
            "achieved_loss": round(-achieved, 6),
            "sigma_distance": round(sigma_distance, 3),
            "plausible": bool(sigma_distance <= 3.0),
        }

    # ------------------------------------------------------------------ #
    # Historical scenario replay
    # ------------------------------------------------------------------ #

    def historical_scenario(
        self,
        weights: np.ndarray,
        scenario_key: str,
        n_paths: int = 2_000,
        initial_value: float = 1_000_000.0,
        seed: int = 123,
    ) -> StressTestResult:
        """Replay a crisis template as a deterministic trajectory plus noise.

        Each asset follows a geometric path hitting its class's total shock
        at the horizon, with Gaussian noise scaled by its historical vol
        times the scenario's volatility multiplier.
        """
        if scenario_key not in HISTORICAL_SCENARIOS:
            raise KeyError(
                f"unknown scenario {scenario_key!r}; valid: {sorted(HISTORICAL_SCENARIOS)}"
            )
        scenario = HISTORICAL_SCENARIOS[scenario_key]
        w = _validate_weights(weights, len(self.tickers))
        rng = np.random.default_rng(seed)
        horizon = scenario.horizon_days
        n_assets = len(self.tickers)

        total_shock = np.array(
            [scenario.shocks.get(self.asset_classes.get(t, ""), 0.0) for t in self.tickers]
        )
        daily_vol = self._returns.std(axis=0) * scenario.vol_multiplier    # (N,)
        # Drift that compounds to the target shock in expectation: the
        # -0.5*sigma^2 Ito correction cancels the lognormal convexity of the
        # noise, so E[terminal] = initial * (1 + total_shock) per asset.
        daily_drift = np.log1p(total_shock) / horizon - 0.5 * daily_vol**2  # (N,)

        # Unit-variance correlated normals scaled by stressed daily vol.
        z = rng.standard_normal((n_paths, horizon, n_assets)) @ self._chol_corr.T
        sim = daily_drift[None, None, :] + z * daily_vol[None, None, :]

        asset_growth = np.exp(np.cumsum(sim, axis=1))
        port_values = initial_value * np.einsum("ptn,n->pt", asset_growth, w)
        start_col = np.full((n_paths, 1), initial_value)
        port_values = np.concatenate([start_col, port_values], axis=1)

        result = self._build_result(
            port_values,
            method="historical_replay",
            n_paths=n_paths,
            horizon_days=horizon,
            initial_value=initial_value,
        )
        result.scenario = {
            "key": scenario.key,
            "name": scenario.name,
            "description": scenario.description,
            "horizon_days": scenario.horizon_days,
            "shocks": scenario.shocks,
        }
        return result

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def _build_result(
        self,
        port_values: np.ndarray,
        method: str,
        n_paths: int,
        horizon_days: int,
        initial_value: float,
    ) -> StressTestResult:
        terminal = port_values[:, -1]
        pnl = terminal - initial_value

        var_95 = float(-np.percentile(pnl, 5.0))
        var_99 = float(-np.percentile(pnl, 1.0))
        cvar_95 = float(-pnl[pnl <= -var_95].mean()) if np.any(pnl <= -var_95) else var_95
        cvar_99 = float(-pnl[pnl <= -var_99].mean()) if np.any(pnl <= -var_99) else var_99

        mdd = max_drawdowns(port_values)

        # Downsample the time axis for the fan chart (~120 points max).
        t_total = port_values.shape[1]
        stride = max(1, t_total // 120)
        time_idx = np.arange(0, t_total, stride)
        if time_idx[-1] != t_total - 1:
            time_idx = np.append(time_idx, t_total - 1)
        fan: dict[str, list[float]] = {
            "days": time_idx.astype(int).tolist(),
        }
        percentile_matrix = np.percentile(port_values[:, time_idx], FAN_PERCENTILES, axis=0)
        for p, row in zip(FAN_PERCENTILES, percentile_matrix):
            fan[f"p{int(p)}"] = np.round(row, 2).tolist()

        hist_counts, hist_edges = np.histogram(terminal, bins=40)
        histogram = {
            "bin_edges": np.round(hist_edges, 2).tolist(),
            "counts": hist_counts.astype(float).tolist(),
        }

        return StressTestResult(
            method=method,
            n_paths=n_paths,
            horizon_days=horizon_days,
            initial_value=initial_value,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            expected_terminal=float(terminal.mean()),
            worst_terminal=float(terminal.min()),
            best_terminal=float(terminal.max()),
            max_drawdown_median=float(np.median(mdd)),
            max_drawdown_p95=float(np.percentile(mdd, 5.0)),  # 5th pct = severe tail
            prob_loss=float(np.mean(terminal < initial_value)),
            fan_chart=fan,
            terminal_histogram=histogram,
        )
