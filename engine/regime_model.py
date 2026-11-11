"""Unsupervised market regime detection.

Fits a Gaussian Mixture Model on a cross-asset feature matrix (mean log
return, rolling realized volatility, normalized ATR) and estimates an
empirical Markov transition matrix from the decoded state sequence —
functionally a Gaussian HMM decoded with hard assignments, without the
hmmlearn dependency.

States are relabeled post-fit by their (return, volatility) signature so
regime 0/1/2 always map to semantically stable names:
Bull / High Volatility Bear / Consolidation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from engine.data_generator import (
    REGIME_BEAR_HIGH_VOL,
    REGIME_BULL,
    REGIME_CONSOLIDATION,
    REGIME_NAMES,
    TRADING_DAYS_PER_YEAR,
    GeneratedMarket,
)

logger = logging.getLogger(__name__)

FEATURE_NAMES: tuple[str, ...] = ("mean_log_return", "rolling_vol", "norm_atr")


@dataclass(slots=True)
class RegimeFitResult:
    """Everything the API needs after fitting the regime model."""

    dates: list[str]
    states: np.ndarray                 # (T,) semantic regime labels
    state_names: dict[int, str]
    transition_matrix: np.ndarray      # (K, K) row-stochastic
    state_means: dict[int, dict[str, float]]      # per-regime feature means (raw scale)
    state_covariances: dict[int, list[list[float]]]
    stationary_distribution: np.ndarray
    log_likelihood: float
    bic: float
    accuracy_vs_true: float | None     # only when generator truth is available

    def to_payload(self) -> dict[str, object]:
        """JSON-safe dict for the API layer."""
        return {
            "dates": self.dates,
            "states": self.states.astype(int).tolist(),
            "state_names": {str(k): v for k, v in self.state_names.items()},
            "transition_matrix": np.round(self.transition_matrix, 6).tolist(),
            "state_means": {str(k): v for k, v in self.state_means.items()},
            "stationary_distribution": np.round(self.stationary_distribution, 6).tolist(),
            "log_likelihood": float(self.log_likelihood),
            "bic": float(self.bic),
            "accuracy_vs_true": self.accuracy_vs_true,
            "feature_names": list(FEATURE_NAMES),
        }


def build_features(
    close: pd.DataFrame,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    vol_window: int = 21,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Build the (T', 3) regime feature matrix from multi-asset prices.

    Features are cross-asset means so the model sees the market, not one
    ticker:
      * ``mean_log_return`` — rolling mean of the equal-weighted cross-asset
        daily log return.
      * ``rolling_vol`` — relative volatility: each asset's rolling std is
        normalized by that asset's own sample mean vol, then averaged across
        assets (~1.0 in calm markets, >2 in stress).
      * ``norm_atr`` — relative ATR%: ATR/close per asset, normalized by its
        sample mean, averaged across assets; falls back to a close-to-close
        proxy when OHLC is unavailable (the live-fetch path).

    Rows with any NaN (warm-up window) are dropped.
    """
    if close.shape[0] < max(vol_window, atr_window) + 5:
        raise ValueError("not enough rows to build rolling features")

    log_ret = np.log(close).diff()
    # Rolling mean smooths daily noise so drift separates bull from bear.
    mean_ret = log_ret.mean(axis=1).rolling(vol_window).mean()

    # Per-asset rolling vol, normalized by each asset's own sample mean vol
    # before averaging — otherwise the highest-vol asset (e.g. crypto)
    # dominates the cross-asset mean and drowns out regime shifts.
    per_asset_vol = log_ret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rel_vol = per_asset_vol / per_asset_vol.mean(axis=0)
    rolling_vol = rel_vol.mean(axis=1)

    if high is not None and low is not None:
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=0,
        ).groupby(level=0).max()
        true_range = true_range.reindex(close.index)
    else:
        # Proxy: absolute close-to-close move.
        true_range = (close - close.shift(1)).abs()

    atr = true_range.rolling(atr_window).mean()
    atr_pct = atr / close
    norm_atr = (atr_pct / atr_pct.mean(axis=0)).mean(axis=1)

    features = pd.DataFrame(
        {
            "mean_log_return": mean_ret,
            "rolling_vol": rolling_vol,
            "norm_atr": norm_atr,
        }
    ).dropna()
    if features.empty:
        raise ValueError("feature matrix is empty after dropping warm-up rows")
    return features


def _stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Left eigenvector of the transition matrix with eigenvalue 1."""
    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    vec = np.real(eigenvectors[:, idx])
    vec = np.abs(vec)
    return vec / vec.sum()


class RegimeDetector:
    """GMM-based market regime classifier with Markov transition estimation."""

    def __init__(self, n_regimes: int = 3, seed: int = 7) -> None:
        if n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")
        self.n_regimes = n_regimes
        self.seed = seed
        self._gmm: GaussianMixture | None = None
        self._scaler: StandardScaler | None = None
        self._label_map: dict[int, int] = {}

    def fit(
        self,
        features: pd.DataFrame,
        true_regimes: np.ndarray | None = None,
    ) -> RegimeFitResult:
        """Fit the mixture, decode states, and estimate transitions.

        Args:
            features: Output of :func:`build_features`.
            true_regimes: Optional generator truth aligned to the *full*
                price index; used only to report decode accuracy.
        """
        if features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} feature columns")

        self._scaler = StandardScaler()
        x = self._scaler.fit_transform(features.to_numpy(dtype=float))

        self._gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            n_init=8,
            max_iter=500,
            random_state=self.seed,
            reg_covar=1e-6,
        )
        raw_states = self._gmm.fit_predict(x)
        self._label_map = self._semantic_label_map(features, raw_states)
        states = np.array([self._label_map[int(s)] for s in raw_states], dtype=np.int64)

        transition = self._estimate_transition(states)
        state_means, state_covs = self._raw_scale_params(features, states)

        accuracy: float | None = None
        if true_regimes is not None and self.n_regimes == 3:
            aligned_truth = true_regimes[-len(states):]
            accuracy = float(np.mean(aligned_truth == states))

        result = RegimeFitResult(
            dates=[str(d) for d in features.index],
            states=states,
            state_names={k: REGIME_NAMES.get(k, f"Regime {k}") for k in range(self.n_regimes)},
            transition_matrix=transition,
            state_means=state_means,
            state_covariances=state_covs,
            stationary_distribution=_stationary_distribution(transition),
            log_likelihood=float(self._gmm.score(x) * len(x)),
            bic=float(self._gmm.bic(x)),
            accuracy_vs_true=accuracy,
        )
        logger.info(
            "Regime fit complete: BIC=%.1f, accuracy_vs_true=%s",
            result.bic,
            f"{accuracy:.2%}" if accuracy is not None else "n/a",
        )
        return result

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Classify new feature rows with the fitted model."""
        if self._gmm is None or self._scaler is None:
            raise RuntimeError("RegimeDetector.fit must be called before predict")
        x = self._scaler.transform(features.to_numpy(dtype=float))
        raw = self._gmm.predict(x)
        return np.array([self._label_map[int(s)] for s in raw], dtype=np.int64)

    def _semantic_label_map(
        self, features: pd.DataFrame, raw_states: np.ndarray
    ) -> dict[int, int]:
        """Map arbitrary GMM component ids to stable semantic regime ids.

        Labels are assigned by volatility rank, which is far more stable
        than mean return: highest vol -> High Volatility Bear, middle ->
        Bull, lowest -> Consolidation. With more than 3 components they are
        simply ordered by volatility.
        """
        stats: list[tuple[int, float, float]] = []
        for comp in range(self.n_regimes):
            mask = raw_states == comp
            if not mask.any():
                stats.append((comp, 0.0, 0.0))
                continue
            sub = features.loc[mask]
            stats.append(
                (comp, float(sub["mean_log_return"].mean()), float(sub["rolling_vol"].mean()))
            )

        mapping: dict[int, int] = {}
        if self.n_regimes == 3:
            by_vol = sorted(stats, key=lambda s: s[2])  # ascending volatility
            mapping[by_vol[0][0]] = REGIME_CONSOLIDATION
            mapping[by_vol[1][0]] = REGIME_BULL
            mapping[by_vol[2][0]] = REGIME_BEAR_HIGH_VOL
        else:
            by_vol = sorted(stats, key=lambda s: s[2])
            for rank, (comp, _, _) in enumerate(by_vol):
                mapping[comp] = rank
        return mapping

    def _estimate_transition(self, states: np.ndarray) -> np.ndarray:
        """Empirical row-stochastic transition matrix with Laplace smoothing."""
        k = self.n_regimes
        counts = np.ones((k, k)) * 0.5  # smoothing avoids zero rows
        for a, b in zip(states[:-1], states[1:]):
            counts[a, b] += 1.0
        return counts / counts.sum(axis=1, keepdims=True)

    def _raw_scale_params(
        self, features: pd.DataFrame, states: np.ndarray
    ) -> tuple[dict[int, dict[str, float]], dict[int, list[list[float]]]]:
        """Per-regime feature means and covariances in original units."""
        means: dict[int, dict[str, float]] = {}
        covs: dict[int, list[list[float]]] = {}
        for regime in range(self.n_regimes):
            mask = states == regime
            if mask.sum() < 2:
                means[regime] = {name: 0.0 for name in FEATURE_NAMES}
                covs[regime] = np.zeros((len(FEATURE_NAMES),) * 2).tolist()
                continue
            sub = features.loc[mask]
            means[regime] = {name: float(sub[name].mean()) for name in FEATURE_NAMES}
            covs[regime] = np.cov(sub.to_numpy(dtype=float), rowvar=False).tolist()
        return means, covs


def fit_market_regimes(
    market: GeneratedMarket, n_regimes: int = 3, seed: int = 7
) -> RegimeFitResult:
    """Convenience one-shot: features + fit from a generated market."""
    features = build_features(market.close, market.high, market.low)
    detector = RegimeDetector(n_regimes=n_regimes, seed=seed)
    return detector.fit(features, true_regimes=market.true_regimes)
