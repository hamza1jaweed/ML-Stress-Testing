"""Portfolio analytics: historical performance, risk decomposition,
correlation structure, and mean-variance efficient frontier.

All computations are vectorized NumPy/SciPy; the optimizer uses SLSQP with
long-only, fully-invested constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from engine.data_generator import TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Historical portfolio analytics
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PortfolioAnalytics:
    """Historical performance and risk decomposition for a weight vector."""

    dates: list[str]
    equity_curve: list[float]          # normalized to 100 at start
    drawdown: list[float]              # underwater series, <= 0
    cagr: float
    annual_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    hist_var_95_daily: float           # historical 1-day 95% VaR (as +fraction)
    skew: float
    excess_kurtosis: float
    risk_contributions: dict[str, float]   # fraction of portfolio vol, sums to 1
    correlation: list[list[float]]
    tickers: list[str]

    def to_payload(self) -> dict[str, object]:
        return {
            "dates": self.dates,
            "equity_curve": self.equity_curve,
            "drawdown": self.drawdown,
            "metrics": {
                "cagr": round(self.cagr, 6),
                "annual_vol": round(self.annual_vol, 6),
                "sharpe": round(self.sharpe, 4),
                "sortino": round(self.sortino, 4),
                "calmar": round(self.calmar, 4),
                "max_drawdown": round(self.max_drawdown, 6),
                "hist_var_95_daily": round(self.hist_var_95_daily, 6),
                "skew": round(self.skew, 4),
                "excess_kurtosis": round(self.excess_kurtosis, 4),
            },
            "risk_contributions": {k: round(v, 6) for k, v in self.risk_contributions.items()},
            "correlation": self.correlation,
            "tickers": self.tickers,
        }


def _downsample_indices(n: int, max_points: int = 400) -> np.ndarray:
    stride = max(1, n // max_points)
    idx = np.arange(0, n, stride)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def compute_portfolio_analytics(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = 0.0,
) -> PortfolioAnalytics:
    """Full historical analytics for a (renormalized) weight vector.

    The portfolio is treated as daily-rebalanced to the target weights, so
    the daily portfolio simple return is the weighted sum of per-asset
    simple returns.
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != log_returns.shape[1]:
        raise ValueError(f"expected {log_returns.shape[1]} weights, got {w.shape[0]}")
    if np.any(w < -1e-9):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    w = w / total

    simple = np.expm1(log_returns.to_numpy(dtype=float))     # (T, N)
    port_ret = simple @ w                                     # (T,)
    equity = np.concatenate([[1.0], np.cumprod(1.0 + port_ret)]) * 100.0
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0

    n_days = len(port_ret)
    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    daily_mean = float(port_ret.mean())
    daily_std = float(port_ret.std(ddof=1))
    annual_vol = daily_std * float(np.sqrt(TRADING_DAYS_PER_YEAR))
    excess_daily = daily_mean - risk_free_rate / TRADING_DAYS_PER_YEAR
    sharpe = (
        excess_daily / daily_std * float(np.sqrt(TRADING_DAYS_PER_YEAR))
        if daily_std > 0
        else 0.0
    )
    downside = port_ret[port_ret < 0]
    downside_std = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    sortino = (
        excess_daily / downside_std * float(np.sqrt(TRADING_DAYS_PER_YEAR))
        if downside_std > 0
        else 0.0
    )
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    hist_var = float(-np.percentile(port_ret, 5.0))

    centered = port_ret - daily_mean
    m2 = float(np.mean(centered**2))
    skew = float(np.mean(centered**3) / m2**1.5) if m2 > 0 else 0.0
    kurt = float(np.mean(centered**4) / m2**2 - 3.0) if m2 > 0 else 0.0

    # Risk contributions: RC_i = w_i * (C w)_i / (w' C w); sums to 1.
    cov = np.cov(simple, rowvar=False)
    port_var = float(w @ cov @ w)
    if port_var > 0:
        contrib = w * (cov @ w) / port_var
    else:
        contrib = np.full_like(w, 1.0 / len(w))

    std = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    # Downsample series for the wire; equity/drawdown have T+1 points.
    dates = ["start", *[str(d) for d in log_returns.index]]
    idx = _downsample_indices(len(equity))

    tickers = list(log_returns.columns)
    return PortfolioAnalytics(
        dates=[dates[i] for i in idx],
        equity_curve=[round(float(equity[i]), 4) for i in idx],
        drawdown=[round(float(drawdown[i]), 6) for i in idx],
        cagr=cagr,
        annual_vol=annual_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        hist_var_95_daily=hist_var,
        skew=skew,
        excess_kurtosis=kurt,
        risk_contributions={t: float(c) for t, c in zip(tickers, contrib)},
        correlation=np.round(corr, 4).tolist(),
        tickers=tickers,
    )


# --------------------------------------------------------------------------- #
# Efficient frontier
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FrontierResult:
    """Random-portfolio cloud plus optimized portfolios and frontier curve."""

    cloud: list[dict[str, float]]              # {ret, vol, sharpe} per sample
    frontier: list[dict[str, float]]           # min-vol curve {ret, vol}
    max_sharpe: dict[str, object] = field(default_factory=dict)
    min_vol: dict[str, object] = field(default_factory=dict)
    tickers: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        return {
            "cloud": self.cloud,
            "frontier": self.frontier,
            "max_sharpe": self.max_sharpe,
            "min_vol": self.min_vol,
            "tickers": self.tickers,
        }


def _annualized_stats(
    w: np.ndarray, mu: np.ndarray, cov: np.ndarray
) -> tuple[float, float]:
    ret = float(w @ mu) * TRADING_DAYS_PER_YEAR
    vol = float(np.sqrt(max(w @ cov @ w, 0.0))) * float(np.sqrt(TRADING_DAYS_PER_YEAR))
    return ret, vol


def compute_efficient_frontier(
    log_returns: pd.DataFrame,
    n_samples: int = 3000,
    seed: int = 11,
    risk_free_rate: float = 0.0,
    max_cloud_points: int = 800,
    n_frontier_points: int = 25,
) -> FrontierResult:
    """Random long-only portfolio cloud + SLSQP-optimized frontier.

    Returns annualized (return, vol, Sharpe) triples for a Dirichlet-sampled
    cloud, the minimum-variance frontier between the min-vol and max-return
    corner portfolios, and the max-Sharpe / min-vol weight vectors.
    """
    if n_samples < 100:
        raise ValueError("n_samples must be >= 100")

    simple = np.expm1(log_returns.to_numpy(dtype=float))
    mu = simple.mean(axis=0)
    cov = np.cov(simple, rowvar=False)
    n_assets = len(mu)
    tickers = list(log_returns.columns)
    rng = np.random.default_rng(seed)

    # ---- random cloud (vectorized) ----
    cloud_w = rng.dirichlet(np.ones(n_assets), size=n_samples)        # (S, N)
    rets = cloud_w @ mu * TRADING_DAYS_PER_YEAR
    vols = np.sqrt(np.einsum("sn,nm,sm->s", cloud_w, cov, cloud_w)) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    sharpes = np.where(vols > 0, (rets - risk_free_rate) / vols, 0.0)

    keep = _downsample_indices(n_samples, max_cloud_points)
    cloud = [
        {
            "ret": round(float(rets[i]), 5),
            "vol": round(float(vols[i]), 5),
            "sharpe": round(float(sharpes[i]), 4),
        }
        for i in keep
    ]

    # ---- optimizers ----
    # Annualized covariance keeps the objective O(0.01-1): daily variances
    # (~1e-4) make SLSQP's gradient so small it stops at the first iterate.
    cov_ann = cov * TRADING_DAYS_PER_YEAR
    bounds = [(0.0, 1.0)] * n_assets
    budget = {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}
    options = {"maxiter": 500, "ftol": 1e-12}

    def neg_sharpe(w: np.ndarray) -> float:
        ret, vol = _annualized_stats(w, mu, cov)
        return -(ret - risk_free_rate) / vol if vol > 0 else 0.0

    def variance(w: np.ndarray) -> float:
        return float(w @ cov_ann @ w)

    # Warm-start each optimizer from the best random sample it must beat.
    x0_sharpe = cloud_w[int(np.argmax(sharpes))]
    x0_minvol = cloud_w[int(np.argmin(vols))]

    opt_sharpe = minimize(
        neg_sharpe, x0_sharpe, method="SLSQP", bounds=bounds, constraints=[budget], options=options
    )
    opt_minvol = minimize(
        variance, x0_minvol, method="SLSQP", bounds=bounds, constraints=[budget], options=options
    )
    if not opt_minvol.success:
        logger.warning("min-vol optimization did not converge: %s", opt_minvol.message)
    if not opt_sharpe.success:
        logger.warning("max-sharpe optimization did not converge: %s", opt_sharpe.message)

    def portfolio_payload(w: np.ndarray) -> dict[str, object]:
        w = np.clip(w, 0.0, 1.0)
        w = w / w.sum()
        ret, vol = _annualized_stats(w, mu, cov)
        return {
            "weights": {t: round(float(x), 4) for t, x in zip(tickers, w)},
            "ret": round(ret, 5),
            "vol": round(vol, 5),
            "sharpe": round((ret - risk_free_rate) / vol, 4) if vol > 0 else 0.0,
        }

    max_sharpe = portfolio_payload(opt_sharpe.x)
    min_vol = portfolio_payload(opt_minvol.x)

    # ---- frontier curve: min vol at a grid of target returns ----
    ret_lo = float(min_vol["ret"])  # type: ignore[arg-type]
    ret_hi = float(mu.max()) * TRADING_DAYS_PER_YEAR
    frontier: list[dict[str, float]] = []
    if ret_hi > ret_lo:
        x_prev = np.clip(opt_minvol.x, 0.0, 1.0)
        x_prev = x_prev / x_prev.sum()
        for target in np.linspace(ret_lo, ret_hi, n_frontier_points):
            cons = [
                budget,
                {
                    "type": "eq",
                    "fun": lambda w, t=float(target): float(w @ mu) * TRADING_DAYS_PER_YEAR - t,
                },
            ]
            # Warm-start from the previous frontier point for a smooth curve.
            res = minimize(
                variance, x_prev, method="SLSQP", bounds=bounds, constraints=cons, options=options
            )
            if res.success:
                _, vol = _annualized_stats(res.x, mu, cov)
                frontier.append({"ret": round(float(target), 5), "vol": round(vol, 5)})
                x_prev = res.x

    return FrontierResult(
        cloud=cloud,
        frontier=frontier,
        max_sharpe=max_sharpe,
        min_vol=min_vol,
        tickers=tickers,
    )


# --------------------------------------------------------------------------- #
# Risk parity
# --------------------------------------------------------------------------- #


def compute_risk_parity(
    log_returns: pd.DataFrame, risk_free_rate: float = 0.0
) -> dict[str, object]:
    """Equal-risk-contribution portfolio via SLSQP.

    Minimizes the dispersion of risk contributions RC_i = w_i (Σw)_i / w'Σw
    around 1/N under long-only, fully-invested constraints.
    """
    simple = np.expm1(log_returns.to_numpy(dtype=float))
    mu = simple.mean(axis=0)
    cov = np.cov(simple, rowvar=False)
    cov_ann = cov * TRADING_DAYS_PER_YEAR
    n = len(mu)
    tickers = list(log_returns.columns)

    def rc_dispersion(w: np.ndarray) -> float:
        port_var = float(w @ cov_ann @ w)
        if port_var <= 0:
            return 1e6
        rc = w * (cov_ann @ w) / port_var
        return float(np.sum((rc - 1.0 / n) ** 2)) * 1e4  # scale for SLSQP

    inv_vol = 1.0 / np.sqrt(np.maximum(np.diag(cov_ann), 1e-18))
    x0 = inv_vol / inv_vol.sum()
    result = minimize(
        rc_dispersion,
        x0,
        method="SLSQP",
        bounds=[(1e-6, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}],
        options={"maxiter": 500, "ftol": 1e-14},
    )
    w = np.clip(result.x if result.success else x0, 0.0, 1.0)
    w = w / w.sum()
    ret, vol = _annualized_stats(w, mu, cov)
    return {
        "weights": {t: round(float(x), 4) for t, x in zip(tickers, w)},
        "ret": round(ret, 5),
        "vol": round(vol, 5),
        "sharpe": round((ret - risk_free_rate) / vol, 4) if vol > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# VaR backtest (Kupiec proportion-of-failures test)
# --------------------------------------------------------------------------- #


def compute_var_backtest(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
    window: int = 250,
) -> dict[str, object]:
    """Backtest rolling historical-simulation VaR with the Kupiec POF test.

    For each day after the warm-up window, the previous ``window`` portfolio
    returns set the VaR quantile; a breach is a realized return below it.
    Kupiec's likelihood-ratio statistic tests whether the observed breach
    frequency is consistent with the nominal rate (chi-squared, 1 dof).
    """
    from scipy.stats import chi2

    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be within (0.5, 1)")
    w = np.asarray(weights, dtype=float).ravel()
    w = w / w.sum()
    simple = np.expm1(log_returns.to_numpy(dtype=float))
    port_ret = simple @ w
    t_total = len(port_ret)
    if t_total < window + 50:
        raise ValueError(f"need at least {window + 50} observations, have {t_total}")

    alpha = 1.0 - confidence
    # Rolling quantile via a strided sliding window (vectorized).
    windows = np.lib.stride_tricks.sliding_window_view(port_ret, window)[:-1]  # (T-w, w)
    var_series = -np.quantile(windows, alpha, axis=1)                          # (T-w,)
    realized = port_ret[window:]
    breaches = realized < -var_series

    n_obs = int(realized.size)
    n_breach = int(breaches.sum())
    rate = n_breach / n_obs

    # Kupiec POF likelihood ratio (guard the degenerate 0/100% cases).
    if n_breach in (0, n_obs):
        lr = (
            -2.0 * (n_obs - n_breach) * np.log(1 - alpha)
            - 2.0 * n_breach * np.log(alpha)
        )
    else:
        lr = -2.0 * (
            (n_obs - n_breach) * np.log((1 - alpha) / (1 - rate))
            + n_breach * np.log(alpha / rate)
        )
    p_value = float(chi2.sf(lr, df=1))

    return {
        "confidence": confidence,
        "window": window,
        "observations": n_obs,
        "breaches": n_breach,
        "expected_breaches": round(alpha * n_obs, 1),
        "breach_rate": round(rate, 5),
        "kupiec_lr": round(float(lr), 4),
        "p_value": round(p_value, 5),
        "model_ok": bool(p_value > 0.05),
    }


# --------------------------------------------------------------------------- #
# Regime-conditional portfolio performance
# --------------------------------------------------------------------------- #


def compute_regime_performance(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    states: np.ndarray,
    state_names: dict[int, str],
) -> list[dict[str, object]]:
    """Portfolio return/vol/Sharpe/worst-day per detected regime.

    ``states`` aligns to the tail of ``log_returns`` (rolling-feature warm-up
    rows are unlabeled and skipped).
    """
    w = np.asarray(weights, dtype=float).ravel()
    w = w / w.sum()
    simple = np.expm1(log_returns.to_numpy(dtype=float))
    port_ret = simple @ w

    states = np.asarray(states, dtype=np.int64).ravel()
    if states.size > port_ret.size:
        raise ValueError("more regime labels than return rows")
    aligned = port_ret[-states.size:]

    rows: list[dict[str, object]] = []
    for regime in sorted(state_names):
        sub = aligned[states == regime]
        if sub.size < 5:
            continue
        mean, std = float(sub.mean()), float(sub.std(ddof=1))
        rows.append(
            {
                "regime": int(regime),
                "name": state_names[regime],
                "days": int(sub.size),
                "ann_return": round(mean * TRADING_DAYS_PER_YEAR, 5),
                "ann_vol": round(std * np.sqrt(TRADING_DAYS_PER_YEAR), 5),
                "sharpe": round(
                    mean / std * np.sqrt(TRADING_DAYS_PER_YEAR), 4
                ) if std > 0 else 0.0,
                "worst_day": round(float(sub.min()), 5),
                "hit_rate": round(float((sub > 0).mean()), 4),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Crisis early warning (absorbing-chain hitting statistics)
# --------------------------------------------------------------------------- #


def bear_hitting_stats(
    transition: np.ndarray,
    current_state: int,
    bear_state: int,
    horizons: tuple[int, ...] = (5, 21, 63),
) -> dict[str, object]:
    """P(first visit to the bear regime within h days) and expected hitting time.

    Probabilities come from the chain with the bear state made absorbing;
    the expected hitting time solves ``(I - Q) t = 1`` on the transient
    (non-bear) states.
    """
    t_matrix = np.asarray(transition, dtype=float)
    k = t_matrix.shape[0]
    if t_matrix.shape != (k, k):
        raise ValueError("transition must be square")
    for idx, label in ((current_state, "current_state"), (bear_state, "bear_state")):
        if not 0 <= idx < k:
            raise ValueError(f"{label} must be in [0, {k})")
    if any(h < 1 for h in horizons):
        raise ValueError("horizons must be positive")

    if current_state == bear_state:
        return {
            "already_in_bear": True,
            "prob_within": {str(h): 1.0 for h in horizons},
            "expected_days_to_bear": 0.0,
        }

    absorbing = t_matrix.copy()
    absorbing[bear_state] = 0.0
    absorbing[bear_state, bear_state] = 1.0

    probs: dict[str, float] = {}
    p = np.zeros(k)
    p[current_state] = 1.0
    max_h = max(horizons)
    for day in range(1, max_h + 1):
        p = p @ absorbing
        if day in horizons:
            probs[str(day)] = round(float(p[bear_state]), 6)

    transient = [i for i in range(k) if i != bear_state]
    q = t_matrix[np.ix_(transient, transient)]
    hitting = np.linalg.solve(np.eye(len(transient)) - q, np.ones(len(transient)))
    expected = float(hitting[transient.index(current_state)])

    return {
        "already_in_bear": False,
        "prob_within": probs,
        "expected_days_to_bear": round(expected, 1),
    }


# --------------------------------------------------------------------------- #
# Regime forecast
# --------------------------------------------------------------------------- #


def forecast_regime_probabilities(
    transition: np.ndarray,
    current_state: int,
    horizon_days: int = 126,
) -> np.ndarray:
    """Probability of each regime over the next ``horizon_days``.

    Returns an (H, K) matrix whose row ``t`` is ``e_s @ T^(t+1)``. Rows sum
    to 1 up to floating-point error.
    """
    t_matrix = np.asarray(transition, dtype=float)
    k = t_matrix.shape[0]
    if t_matrix.shape != (k, k):
        raise ValueError("transition must be square")
    if not 0 <= current_state < k:
        raise ValueError(f"current_state must be in [0, {k})")
    if not 1 <= horizon_days <= 2520:
        raise ValueError("horizon_days must be within [1, 2520]")

    probs = np.empty((horizon_days, k))
    p = np.zeros(k)
    p[current_state] = 1.0
    for t in range(horizon_days):
        p = p @ t_matrix
        probs[t] = p
    return probs
