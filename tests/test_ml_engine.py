"""Statistical sanity tests for the ML engine."""

from __future__ import annotations

import numpy as np
import pytest

from engine.data_generator import (
    REGIME_BEAR_HIGH_VOL,
    REGIME_BULL,
    REGIME_NAMES,
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
from engine.regime_model import RegimeDetector, build_features
from engine.stress_engine import (
    HISTORICAL_SCENARIOS,
    StressTester,
    max_drawdowns,
)


@pytest.fixture(scope="module")
def market() -> GeneratedMarket:
    return SyntheticMarketGenerator().generate(n_days=1500, seed=42)


@pytest.fixture(scope="module")
def tester(market: GeneratedMarket) -> StressTester:
    return StressTester(market.log_returns, market.asset_classes)


EQUAL_WEIGHTS = np.ones(5) / 5


# --------------------------------------------------------------------------- #
# Data generator
# --------------------------------------------------------------------------- #


class TestDataGenerator:
    def test_shapes_and_alignment(self, market: GeneratedMarket) -> None:
        assert market.close.shape == (1500, 5)
        assert market.log_returns.shape == (1499, 5)
        assert len(market.dates) == 1500
        assert market.true_regimes.shape == (1500,)

    def test_reproducibility(self) -> None:
        gen = SyntheticMarketGenerator()
        a = gen.generate(n_days=300, seed=7)
        b = gen.generate(n_days=300, seed=7)
        np.testing.assert_array_equal(a.close.to_numpy(), b.close.to_numpy())
        np.testing.assert_array_equal(a.true_regimes, b.true_regimes)

    def test_different_seeds_differ(self) -> None:
        gen = SyntheticMarketGenerator()
        a = gen.generate(n_days=300, seed=1)
        b = gen.generate(n_days=300, seed=2)
        assert not np.allclose(a.close.to_numpy(), b.close.to_numpy())

    def test_prices_positive_and_bars_valid(self, market: GeneratedMarket) -> None:
        close = market.close.to_numpy()
        assert np.all(close > 0)
        assert np.all(market.high.to_numpy() >= close - 1e-9)
        assert np.all(market.low.to_numpy() <= close + 1e-9)

    def test_all_regimes_visited(self, market: GeneratedMarket) -> None:
        visited = set(np.unique(market.true_regimes).tolist())
        assert visited == set(REGIME_NAMES)

    def test_bear_regime_has_higher_vol(self, market: GeneratedMarket) -> None:
        returns = market.log_returns.to_numpy()
        regimes = market.true_regimes[1:]
        bull_vol = returns[regimes == REGIME_BULL].std()
        bear_vol = returns[regimes == REGIME_BEAR_HIGH_VOL].std()
        assert bear_vol > bull_vol * 1.3, "bear regime should be clearly more volatile"

    def test_correlation_structure_recovered(self, market: GeneratedMarket) -> None:
        # Equity pairs should correlate far more than equity vs. bonds.
        corr = np.corrcoef(market.log_returns.to_numpy(), rowvar=False)
        eq_us, eq_em, bond = 0, 1, 2
        assert corr[eq_us, eq_em] > 0.4
        assert corr[eq_us, bond] < corr[eq_us, eq_em]

    def test_rejects_bad_inputs(self) -> None:
        gen = SyntheticMarketGenerator()
        with pytest.raises(ValueError):
            gen.generate(n_days=10)
        with pytest.raises(ValueError):
            SyntheticMarketGenerator(garch_alpha=0.6, garch_beta=0.6)


# --------------------------------------------------------------------------- #
# Regime model
# --------------------------------------------------------------------------- #


class TestRegimeModel:
    def test_feature_matrix(self, market: GeneratedMarket) -> None:
        features = build_features(market.close, market.high, market.low)
        assert list(features.columns) == ["mean_log_return", "rolling_vol", "norm_atr"]
        assert not features.isna().any().any()
        assert (features["rolling_vol"] > 0).all()
        assert (features["norm_atr"] > 0).all()

    def test_fit_recovers_regimes(self, market: GeneratedMarket) -> None:
        features = build_features(market.close, market.high, market.low)
        detector = RegimeDetector(n_regimes=3, seed=7)
        result = detector.fit(features, true_regimes=market.true_regimes)

        assert set(np.unique(result.states).tolist()) <= {0, 1, 2}
        assert result.accuracy_vs_true is not None
        # Unsupervised decode should beat chance (1/3) by a wide margin.
        assert result.accuracy_vs_true > 0.55

    def test_transition_matrix_is_row_stochastic(self, market: GeneratedMarket) -> None:
        features = build_features(market.close, market.high, market.low)
        result = RegimeDetector(n_regimes=3, seed=7).fit(features)
        rows = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(rows, 1.0, atol=1e-9)
        # Regimes are persistent: diagonal dominates every row.
        assert np.all(np.diag(result.transition_matrix) > 0.5)

    def test_stationary_distribution_valid(self, market: GeneratedMarket) -> None:
        features = build_features(market.close, market.high, market.low)
        result = RegimeDetector(n_regimes=3, seed=7).fit(features)
        pi = result.stationary_distribution
        assert pytest.approx(pi.sum(), abs=1e-9) == 1.0
        assert np.all(pi >= 0)

    def test_semantic_labels_ordered_by_vol(self, market: GeneratedMarket) -> None:
        features = build_features(market.close, market.high, market.low)
        result = RegimeDetector(n_regimes=3, seed=7).fit(features)
        bear_vol = result.state_means[REGIME_BEAR_HIGH_VOL]["rolling_vol"]
        bull_vol = result.state_means[REGIME_BULL]["rolling_vol"]
        assert bear_vol > bull_vol

    def test_predict_requires_fit(self, market: GeneratedMarket) -> None:
        features = build_features(market.close, market.high, market.low)
        with pytest.raises(RuntimeError):
            RegimeDetector().predict(features)

    def test_fit_is_deterministic(self, market: GeneratedMarket) -> None:
        features = build_features(market.close, market.high, market.low)
        r1 = RegimeDetector(n_regimes=3, seed=7).fit(features)
        r2 = RegimeDetector(n_regimes=3, seed=7).fit(features)
        np.testing.assert_array_equal(r1.states, r2.states)


# --------------------------------------------------------------------------- #
# Stress engine
# --------------------------------------------------------------------------- #


class TestStressEngine:
    def test_var_ordering(self, tester: StressTester) -> None:
        result = tester.monte_carlo(EQUAL_WEIGHTS, n_paths=5000, horizon_days=126, seed=1)
        # 99% VaR is at least as severe as 95%; CVaR at least as severe as VaR.
        assert result.var_99 >= result.var_95
        assert result.cvar_95 >= result.var_95 - 1e-6
        assert result.cvar_99 >= result.var_99 - 1e-6

    def test_seed_locking(self, tester: StressTester) -> None:
        a = tester.monte_carlo(EQUAL_WEIGHTS, n_paths=2000, horizon_days=63, seed=99)
        b = tester.monte_carlo(EQUAL_WEIGHTS, n_paths=2000, horizon_days=63, seed=99)
        assert a.var_95 == b.var_95
        assert a.fan_chart == b.fan_chart

    def test_fan_chart_percentiles_ordered(self, tester: StressTester) -> None:
        result = tester.monte_carlo(EQUAL_WEIGHTS, n_paths=3000, horizon_days=126, seed=3)
        fan = result.fan_chart
        for i in range(len(fan["days"])):
            assert fan["p5"][i] <= fan["p25"][i] <= fan["p50"][i] <= fan["p75"][i] <= fan["p95"][i]

    def test_bootstrap_method(self, tester: StressTester) -> None:
        result = tester.monte_carlo(
            EQUAL_WEIGHTS, n_paths=2000, horizon_days=63, method="bootstrap", seed=5
        )
        assert result.method == "bootstrap"
        assert result.n_paths == 2000
        assert 0.0 <= result.prob_loss <= 1.0

    def test_negative_shock_reduces_value(self, tester: StressTester) -> None:
        base = tester.monte_carlo(EQUAL_WEIGHTS, n_paths=2000, horizon_days=63, seed=11)
        shocked = tester.monte_carlo(
            EQUAL_WEIGHTS,
            n_paths=2000,
            horizon_days=63,
            seed=11,
            shock_overrides={"equity": -0.30, "crypto": -0.40},
        )
        assert shocked.expected_terminal < base.expected_terminal
        assert shocked.var_95 > base.var_95

    def test_historical_scenarios(self, tester: StressTester) -> None:
        for key in HISTORICAL_SCENARIOS:
            result = tester.historical_scenario(EQUAL_WEIGHTS, key, n_paths=1000, seed=2)
            assert result.scenario is not None
            assert result.scenario["key"] == key
            assert result.horizon_days == HISTORICAL_SCENARIOS[key].horizon_days

    def test_gfc_scenario_is_severe(self, tester: StressTester) -> None:
        result = tester.historical_scenario(EQUAL_WEIGHTS, "gfc_2008", n_paths=2000, seed=2)
        # An equal-weight book in a GFC replay should lose money in expectation.
        assert result.expected_terminal < result.initial_value
        assert result.prob_loss > 0.8

    def test_max_drawdown_math(self) -> None:
        paths = np.array([[100.0, 120.0, 60.0, 90.0], [100.0, 100.0, 100.0, 100.0]])
        mdd = max_drawdowns(paths)
        assert pytest.approx(mdd[0], abs=1e-12) == 60.0 / 120.0 - 1.0  # -50%
        assert mdd[1] == 0.0

    def test_weight_validation(self, tester: StressTester) -> None:
        with pytest.raises(ValueError):
            tester.monte_carlo(np.array([0.5, 0.5]), n_paths=500)  # wrong length
        with pytest.raises(ValueError):
            tester.monte_carlo(np.array([-0.2, 0.4, 0.4, 0.2, 0.2]), n_paths=500)
        with pytest.raises(ValueError):
            tester.monte_carlo(EQUAL_WEIGHTS, n_paths=10)  # too few paths

    def test_unknown_scenario_raises(self, tester: StressTester) -> None:
        with pytest.raises(KeyError):
            tester.historical_scenario(EQUAL_WEIGHTS, "dotcom_1999")

    def test_weights_are_renormalized(self, tester: StressTester) -> None:
        raw = np.array([2.0, 2.0, 2.0, 2.0, 2.0])  # sums to 10, renormalized to 0.2 each
        a = tester.monte_carlo(raw, n_paths=1000, horizon_days=21, seed=4)
        b = tester.monte_carlo(EQUAL_WEIGHTS, n_paths=1000, horizon_days=21, seed=4)
        assert a.var_95 == pytest.approx(b.var_95, rel=1e-9)


# --------------------------------------------------------------------------- #
# Regime-aware Monte Carlo
# --------------------------------------------------------------------------- #


class TestRegimeAwareMC:
    @pytest.fixture(scope="class")
    def regime_fit(self, market: GeneratedMarket):
        features = build_features(market.close, market.high, market.low)
        return RegimeDetector(n_regimes=3, seed=7).fit(features)

    def test_runs_and_is_reproducible(self, tester: StressTester, regime_fit) -> None:
        kwargs = dict(
            weights=EQUAL_WEIGHTS,
            regime_states=regime_fit.states,
            transition=regime_fit.transition_matrix,
            start_state=int(regime_fit.states[-1]),
            n_paths=2000,
            horizon_days=126,
            seed=77,
        )
        a = tester.monte_carlo_regime_aware(**kwargs)
        b = tester.monte_carlo_regime_aware(**kwargs)
        assert a.method == "regime_aware"
        assert a.var_95 == b.var_95
        assert a.fan_chart == b.fan_chart

    def test_occupancy_sums_to_one(self, tester: StressTester, regime_fit) -> None:
        result = tester.monte_carlo_regime_aware(
            weights=EQUAL_WEIGHTS,
            regime_states=regime_fit.states,
            transition=regime_fit.transition_matrix,
            start_state=0,
            n_paths=1000,
            horizon_days=252,
            seed=5,
        )
        assert result.scenario is not None
        occupancy = result.scenario["regime_occupancy"]
        assert sum(occupancy.values()) == pytest.approx(1.0, abs=0.01)

    def test_bear_start_is_riskier(self, tester: StressTester, regime_fit) -> None:
        common = dict(
            weights=EQUAL_WEIGHTS,
            regime_states=regime_fit.states,
            transition=regime_fit.transition_matrix,
            n_paths=4000,
            horizon_days=63,
            seed=9,
        )
        from engine.data_generator import REGIME_BEAR_HIGH_VOL, REGIME_CONSOLIDATION

        bear = tester.monte_carlo_regime_aware(start_state=REGIME_BEAR_HIGH_VOL, **common)
        calm = tester.monte_carlo_regime_aware(start_state=REGIME_CONSOLIDATION, **common)
        assert bear.var_95 > calm.var_95

    def test_invalid_start_state(self, tester: StressTester, regime_fit) -> None:
        with pytest.raises(ValueError):
            tester.monte_carlo_regime_aware(
                weights=EQUAL_WEIGHTS,
                regime_states=regime_fit.states,
                transition=regime_fit.transition_matrix,
                start_state=9,
                n_paths=500,
            )


# --------------------------------------------------------------------------- #
# Portfolio analytics & frontier
# --------------------------------------------------------------------------- #


class TestPortfolioAnalytics:
    def test_basic_metrics(self, market: GeneratedMarket) -> None:
        result = compute_portfolio_analytics(market.log_returns, EQUAL_WEIGHTS)
        assert result.equity_curve[0] == pytest.approx(100.0)
        assert result.max_drawdown <= 0.0
        assert result.annual_vol > 0.0
        assert all(d <= 1e-12 for d in result.drawdown)
        assert len(result.dates) == len(result.equity_curve) == len(result.drawdown)

    def test_risk_contributions_sum_to_one(self, market: GeneratedMarket) -> None:
        w = np.array([0.4, 0.1, 0.3, 0.15, 0.05])
        result = compute_portfolio_analytics(market.log_returns, w)
        assert sum(result.risk_contributions.values()) == pytest.approx(1.0, abs=1e-6)

    def test_correlation_matrix_valid(self, market: GeneratedMarket) -> None:
        result = compute_portfolio_analytics(market.log_returns, EQUAL_WEIGHTS)
        corr = np.array(result.correlation)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-9)
        np.testing.assert_allclose(corr, corr.T, atol=1e-9)
        assert np.all(np.abs(corr) <= 1.0 + 1e-9)

    def test_rejects_bad_weights(self, market: GeneratedMarket) -> None:
        with pytest.raises(ValueError):
            compute_portfolio_analytics(market.log_returns, np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            compute_portfolio_analytics(
                market.log_returns, np.array([-0.1, 0.3, 0.3, 0.3, 0.2])
            )


class TestEfficientFrontier:
    def test_optimum_dominates_cloud(self, market: GeneratedMarket) -> None:
        result = compute_efficient_frontier(market.log_returns, n_samples=1500, seed=3)
        best_cloud_sharpe = max(p["sharpe"] for p in result.cloud)
        assert result.max_sharpe["sharpe"] >= best_cloud_sharpe - 1e-6
        best_cloud_vol = min(p["vol"] for p in result.cloud)
        assert result.min_vol["vol"] <= best_cloud_vol + 1e-6

    def test_weights_valid(self, market: GeneratedMarket) -> None:
        result = compute_efficient_frontier(market.log_returns, n_samples=500, seed=3)
        for opt in (result.max_sharpe, result.min_vol):
            weights = opt["weights"]
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
            assert all(w >= -1e-9 for w in weights.values())

    def test_frontier_vol_increases_with_return(self, market: GeneratedMarket) -> None:
        result = compute_efficient_frontier(market.log_returns, n_samples=500, seed=3)
        vols = [p["vol"] for p in result.frontier]
        # Above the min-vol point the frontier is non-decreasing in vol.
        assert all(b >= a - 1e-6 for a, b in zip(vols, vols[1:]))

    def test_deterministic(self, market: GeneratedMarket) -> None:
        a = compute_efficient_frontier(market.log_returns, n_samples=500, seed=3)
        b = compute_efficient_frontier(market.log_returns, n_samples=500, seed=3)
        assert a.cloud == b.cloud
        assert a.max_sharpe == b.max_sharpe


class TestReverseStress:
    def test_achieves_target_loss(self, tester: StressTester) -> None:
        result = tester.reverse_stress(EQUAL_WEIGHTS, target_loss=0.25, horizon_days=21)
        assert result["achieved_loss"] == pytest.approx(0.25, abs=1e-6)

    def test_longer_horizon_is_more_plausible(self, tester: StressTester) -> None:
        short = tester.reverse_stress(EQUAL_WEIGHTS, target_loss=0.25, horizon_days=5)
        long = tester.reverse_stress(EQUAL_WEIGHTS, target_loss=0.25, horizon_days=252)
        # Same loss over more time = fewer sigmas.
        assert long["sigma_distance"] < short["sigma_distance"]

    def test_shock_direction_follows_covariance(self, tester: StressTester) -> None:
        # In a long-only book dominated by equity/crypto risk, those assets
        # must be hit hardest in the most-likely loss scenario.
        result = tester.reverse_stress(EQUAL_WEIGHTS, target_loss=0.2, horizon_days=21)
        shocks = result["shocks"]
        assert shocks["CRYPTO_BTC"] < shocks["BOND_10Y"]
        assert shocks["EQ_US"] < 0

    def test_validation(self, tester: StressTester) -> None:
        with pytest.raises(ValueError):
            tester.reverse_stress(EQUAL_WEIGHTS, target_loss=0.0)
        with pytest.raises(ValueError):
            tester.reverse_stress(EQUAL_WEIGHTS, target_loss=0.95)
        with pytest.raises(ValueError):
            tester.reverse_stress(EQUAL_WEIGHTS, target_loss=0.2, horizon_days=0)


class TestCorrelationStress:
    def test_full_panic_raises_var_for_diversified_book(self, tester: StressTester) -> None:
        base = tester.monte_carlo(EQUAL_WEIGHTS, n_paths=4000, horizon_days=126, seed=21)
        panic = tester.monte_carlo(
            EQUAL_WEIGHTS, n_paths=4000, horizon_days=126, seed=21, correlation_stress=1.0
        )
        assert panic.var_95 > base.var_95

    def test_zero_stress_is_identity(self, tester: StressTester) -> None:
        a = tester.monte_carlo(EQUAL_WEIGHTS, n_paths=1000, horizon_days=21, seed=2)
        b = tester.monte_carlo(
            EQUAL_WEIGHTS, n_paths=1000, horizon_days=21, seed=2, correlation_stress=0.0
        )
        assert a.var_95 == b.var_95

    def test_bootstrap_rejects_corr_stress(self, tester: StressTester) -> None:
        with pytest.raises(ValueError):
            tester.monte_carlo(
                EQUAL_WEIGHTS, n_paths=500, method="bootstrap", correlation_stress=0.5
            )

    def test_out_of_range(self, tester: StressTester) -> None:
        with pytest.raises(ValueError):
            tester.monte_carlo(EQUAL_WEIGHTS, n_paths=500, correlation_stress=1.5)


class TestRiskParity:
    def test_equal_risk_contributions(self, market: GeneratedMarket) -> None:
        result = compute_risk_parity(market.log_returns)
        w = np.array([result["weights"][t] for t in market.log_returns.columns])
        simple = np.expm1(market.log_returns.to_numpy())
        cov = np.cov(simple, rowvar=False)
        rc = w * (cov @ w) / float(w @ cov @ w)
        np.testing.assert_allclose(rc, 1.0 / len(w), atol=0.02)

    def test_weights_sum_to_one(self, market: GeneratedMarket) -> None:
        result = compute_risk_parity(market.log_returns)
        assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-3)


class TestVarBacktest:
    def test_sane_output(self, market: GeneratedMarket) -> None:
        result = compute_var_backtest(market.log_returns, EQUAL_WEIGHTS)
        assert result["observations"] > 0
        assert 0 <= result["breaches"] <= result["observations"]
        assert 0.0 <= result["p_value"] <= 1.0
        # A Gaussian-ish synthetic market should be roughly consistent at 95%.
        assert result["breach_rate"] < 0.15

    def test_requires_enough_data(self, market: GeneratedMarket) -> None:
        with pytest.raises(ValueError):
            compute_var_backtest(market.log_returns.iloc[:200], EQUAL_WEIGHTS)


class TestRegimePerformance:
    def test_rows_cover_regimes(self, market: GeneratedMarket) -> None:
        features = build_features(market.close, market.high, market.low)
        fit = RegimeDetector(n_regimes=3, seed=7).fit(features)
        rows = compute_regime_performance(
            market.log_returns, EQUAL_WEIGHTS, fit.states, fit.state_names
        )
        assert len(rows) == 3
        assert sum(r["days"] for r in rows) == len(fit.states)
        bear = next(r for r in rows if r["name"] == "High Volatility Bear")
        bull = next(r for r in rows if r["name"] == "Bull")
        assert bear["ann_vol"] > bull["ann_vol"]


class TestBearHitting:
    TRANSITION = np.array([[0.95, 0.03, 0.02], [0.05, 0.9, 0.05], [0.1, 0.05, 0.85]])

    def test_probabilities_monotone_in_horizon(self) -> None:
        stats = bear_hitting_stats(self.TRANSITION, 0, 1, horizons=(5, 21, 63))
        probs = [stats["prob_within"][k] for k in ("5", "21", "63")]
        assert probs[0] <= probs[1] <= probs[2]
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_already_in_bear(self) -> None:
        stats = bear_hitting_stats(self.TRANSITION, 1, 1)
        assert stats["already_in_bear"] is True
        assert stats["expected_days_to_bear"] == 0.0

    def test_expected_time_positive(self) -> None:
        stats = bear_hitting_stats(self.TRANSITION, 0, 1)
        assert stats["expected_days_to_bear"] > 0

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            bear_hitting_stats(self.TRANSITION, 5, 1)


class TestRegimeForecast:
    TRANSITION = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])

    def test_rows_sum_to_one(self) -> None:
        probs = forecast_regime_probabilities(self.TRANSITION, 0, 50)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)

    def test_day_one_matches_transition_row(self) -> None:
        probs = forecast_regime_probabilities(self.TRANSITION, 1, 10)
        np.testing.assert_allclose(probs[0], self.TRANSITION[1], atol=1e-12)

    def test_converges_to_stationary(self) -> None:
        probs = forecast_regime_probabilities(self.TRANSITION, 2, 500)
        from_other_start = forecast_regime_probabilities(self.TRANSITION, 0, 500)
        np.testing.assert_allclose(probs[-1], from_other_start[-1], atol=1e-6)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            forecast_regime_probabilities(self.TRANSITION, 5, 10)
        with pytest.raises(ValueError):
            forecast_regime_probabilities(self.TRANSITION, 0, 0)
