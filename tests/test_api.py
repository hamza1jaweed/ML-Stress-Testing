"""API schema-compliance and workflow tests (FastAPI TestClient)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.routes import STATE


@pytest.fixture()
def client() -> TestClient:
    """Fresh client with cleared state per test."""
    with STATE.lock:
        STATE.market = None
        STATE.regime_result = None
    return TestClient(app)


@pytest.fixture()
def seeded_client(client: TestClient) -> TestClient:
    response = client.post("/api/generate-data", json={"n_days": 700, "seed": 42})
    assert response.status_code == 200
    return client


class TestHealth:
    def test_health_shape(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["cpu_count"] >= 1
        assert body["data_loaded"] is False
        assert body["model_trained"] is False
        assert body["uptime_seconds"] >= 0

    def test_health_reflects_state(self, seeded_client: TestClient) -> None:
        body = seeded_client.get("/api/health").json()
        assert body["data_loaded"] is True


class TestGenerateData:
    def test_generate_schema(self, client: TestClient) -> None:
        response = client.post("/api/generate-data", json={"n_days": 500, "seed": 1})
        assert response.status_code == 200
        body = response.json()
        assert body["n_days"] == 500
        assert len(body["tickers"]) == 5
        assert len(body["assets"]) == 5
        assert {a["asset_class"] for a in body["assets"]} <= {
            "equity",
            "bond",
            "commodity",
            "crypto",
        }
        assert len(body["prices"]) > 0
        first = body["prices"][0]
        assert set(first["values"].keys()) == set(body["tickers"])
        assert sum(body["true_regime_counts"].values()) == 500

    def test_generate_validation(self, client: TestClient) -> None:
        assert client.post("/api/generate-data", json={"n_days": 5}).status_code == 422
        assert client.post("/api/generate-data", json={"n_days": 500, "seed": -1}).status_code == 422
        assert (
            client.post("/api/generate-data", json={"n_days": 500, "bogus": 1}).status_code
            == 422
        )


class TestTrainRegimes:
    def test_requires_data(self, client: TestClient) -> None:
        response = client.post("/api/train-regimes", json={})
        assert response.status_code == 409

    def test_train_schema(self, seeded_client: TestClient) -> None:
        response = seeded_client.post("/api/train-regimes", json={"n_regimes": 3, "seed": 7})
        assert response.status_code == 200
        body = response.json()

        n = len(body["dates"])
        assert n > 0
        assert len(body["states"]) == n
        assert len(body["market_index"]) == n
        assert set(body["states"]) <= {0, 1, 2}
        assert set(body["state_names"].values()) == {
            "Bull",
            "High Volatility Bear",
            "Consolidation",
        }

        tm = body["transition_matrix"]
        assert len(tm) == 3 and all(len(row) == 3 for row in tm)
        for row in tm:
            assert sum(row) == pytest.approx(1.0, abs=1e-4)
            assert all(p >= 0 for p in row)

        assert sum(body["stationary_distribution"]) == pytest.approx(1.0, abs=1e-4)
        assert body["accuracy_vs_true"] is None or 0.0 <= body["accuracy_vs_true"] <= 1.0


class TestStressTest:
    WEIGHTS = {
        "EQ_US": 0.35,
        "EQ_EM": 0.15,
        "BOND_10Y": 0.30,
        "GOLD": 0.15,
        "CRYPTO_BTC": 0.05,
    }

    def test_requires_data(self, client: TestClient) -> None:
        response = client.post(
            "/api/stress-test", json={"weights": self.WEIGHTS, "n_paths": 500}
        )
        assert response.status_code == 409

    def test_monte_carlo_schema(self, seeded_client: TestClient) -> None:
        response = seeded_client.post(
            "/api/stress-test",
            json={
                "weights": self.WEIGHTS,
                "n_paths": 2000,
                "horizon_days": 126,
                "method": "parametric",
                "seed": 123,
            },
        )
        assert response.status_code == 200
        body = response.json()
        metrics = body["metrics"]
        assert metrics["var_99"] >= metrics["var_95"]
        assert metrics["cvar_95"] >= metrics["var_95"] - 1e-6
        assert 0.0 <= metrics["prob_loss"] <= 1.0
        assert -1.0 <= metrics["max_drawdown_median"] <= 0.0

        fan = body["fan_chart"]
        for key in ("days", "p5", "p25", "p50", "p75", "p95"):
            assert key in fan
        assert len(fan["p50"]) == len(fan["days"])

        hist = body["terminal_histogram"]
        assert len(hist["bin_edges"]) == len(hist["counts"]) + 1

    def test_reproducible_across_calls(self, seeded_client: TestClient) -> None:
        payload = {"weights": self.WEIGHTS, "n_paths": 1000, "horizon_days": 63, "seed": 9}
        a = seeded_client.post("/api/stress-test", json=payload).json()
        b = seeded_client.post("/api/stress-test", json=payload).json()
        assert a["metrics"] == b["metrics"]

    def test_scenario_replay(self, seeded_client: TestClient) -> None:
        response = seeded_client.post(
            "/api/stress-test",
            json={"weights": self.WEIGHTS, "n_paths": 1000, "scenario": "covid_2020"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["method"] == "historical_replay"
        assert body["scenario"]["key"] == "covid_2020"
        assert body["horizon_days"] == body["scenario"]["horizon_days"]

    def test_shock_overrides(self, seeded_client: TestClient) -> None:
        base = seeded_client.post(
            "/api/stress-test",
            json={"weights": self.WEIGHTS, "n_paths": 1000, "horizon_days": 63, "seed": 5},
        ).json()
        shocked = seeded_client.post(
            "/api/stress-test",
            json={
                "weights": self.WEIGHTS,
                "n_paths": 1000,
                "horizon_days": 63,
                "seed": 5,
                "shocks": {"equity": -0.25},
            },
        ).json()
        assert shocked["metrics"]["expected_terminal"] < base["metrics"]["expected_terminal"]

    def test_validation_errors(self, seeded_client: TestClient) -> None:
        # Unknown ticker
        response = seeded_client.post(
            "/api/stress-test", json={"weights": {"NOT_A_TICKER": 1.0}, "n_paths": 500}
        )
        assert response.status_code == 422
        # Unknown scenario
        response = seeded_client.post(
            "/api/stress-test",
            json={"weights": self.WEIGHTS, "n_paths": 500, "scenario": "nope"},
        )
        assert response.status_code == 422
        # Negative weight
        response = seeded_client.post(
            "/api/stress-test",
            json={"weights": {"EQ_US": -0.5, "GOLD": 1.5}, "n_paths": 500},
        )
        assert response.status_code == 422
        # Shock out of range
        response = seeded_client.post(
            "/api/stress-test",
            json={"weights": self.WEIGHTS, "n_paths": 500, "shocks": {"equity": -1.5}},
        )
        assert response.status_code == 422


class TestPortfolioAnalytics:
    WEIGHTS = TestStressTest.WEIGHTS

    def test_requires_data(self, client: TestClient) -> None:
        response = client.post("/api/portfolio-analytics", json={"weights": self.WEIGHTS})
        assert response.status_code == 409

    def test_schema(self, seeded_client: TestClient) -> None:
        response = seeded_client.post(
            "/api/portfolio-analytics", json={"weights": self.WEIGHTS}
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["dates"]) == len(body["equity_curve"]) == len(body["drawdown"])
        assert body["equity_curve"][0] == pytest.approx(100.0)
        assert sum(body["risk_contributions"].values()) == pytest.approx(1.0, abs=1e-4)
        n = len(body["tickers"])
        assert len(body["correlation"]) == n
        assert all(len(row) == n for row in body["correlation"])
        assert body["metrics"]["max_drawdown"] <= 0

    def test_unknown_ticker(self, seeded_client: TestClient) -> None:
        response = seeded_client.post(
            "/api/portfolio-analytics", json={"weights": {"BOGUS": 1.0}}
        )
        assert response.status_code == 422


class TestEfficientFrontier:
    def test_requires_data(self, client: TestClient) -> None:
        assert client.post("/api/efficient-frontier", json={}).status_code == 409

    def test_schema(self, seeded_client: TestClient) -> None:
        response = seeded_client.post(
            "/api/efficient-frontier", json={"n_samples": 500, "seed": 3}
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["cloud"]) > 0
        assert len(body["frontier"]) > 0
        for opt_key in ("max_sharpe", "min_vol"):
            opt = body[opt_key]
            assert sum(opt["weights"].values()) == pytest.approx(1.0, abs=1e-3)
            assert opt["vol"] > 0

    def test_validation(self, seeded_client: TestClient) -> None:
        assert (
            seeded_client.post("/api/efficient-frontier", json={"n_samples": 5}).status_code
            == 422
        )


class TestRegimeForecast:
    def test_requires_trained_model(self, seeded_client: TestClient) -> None:
        assert seeded_client.get("/api/regime-forecast").status_code == 409

    def test_schema(self, seeded_client: TestClient) -> None:
        seeded_client.post("/api/train-regimes", json={})
        response = seeded_client.get("/api/regime-forecast?horizon_days=60")
        assert response.status_code == 200
        body = response.json()
        assert body["horizon_days"] == 60
        assert len(body["probabilities"]) == 60
        for row in body["probabilities"]:
            assert sum(row) == pytest.approx(1.0, abs=1e-4)
        assert body["current_state_name"] == body["state_names"][str(body["current_state"])]

    def test_horizon_validation(self, seeded_client: TestClient) -> None:
        seeded_client.post("/api/train-regimes", json={})
        assert seeded_client.get("/api/regime-forecast?horizon_days=0").status_code == 422


class TestRegimeAwareStress:
    WEIGHTS = TestStressTest.WEIGHTS

    def test_requires_trained_model(self, seeded_client: TestClient) -> None:
        response = seeded_client.post(
            "/api/stress-test",
            json={"weights": self.WEIGHTS, "n_paths": 500, "regime_aware": True},
        )
        assert response.status_code == 409

    def test_runs_after_training(self, seeded_client: TestClient) -> None:
        seeded_client.post("/api/train-regimes", json={})
        response = seeded_client.post(
            "/api/stress-test",
            json={
                "weights": self.WEIGHTS,
                "n_paths": 1000,
                "horizon_days": 63,
                "regime_aware": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["method"] == "regime_aware"
        assert body["scenario"]["key"] == "regime_aware"
        occupancy = body["scenario"]["regime_occupancy"]
        assert sum(occupancy.values()) == pytest.approx(1.0, abs=0.01)


class TestReverseStressApi:
    WEIGHTS = TestStressTest.WEIGHTS

    def test_requires_data(self, client: TestClient) -> None:
        response = client.post("/api/reverse-stress", json={"weights": self.WEIGHTS})
        assert response.status_code == 409

    def test_schema(self, seeded_client: TestClient) -> None:
        response = seeded_client.post(
            "/api/reverse-stress",
            json={"weights": self.WEIGHTS, "target_loss": 0.2, "horizon_days": 21},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["achieved_loss"] == pytest.approx(0.2, abs=1e-4)
        assert set(body["shocks"].keys()) == set(self.WEIGHTS.keys())
        assert body["sigma_distance"] > 0

    def test_validation(self, seeded_client: TestClient) -> None:
        response = seeded_client.post(
            "/api/reverse-stress", json={"weights": self.WEIGHTS, "target_loss": 0.95}
        )
        assert response.status_code == 422


class TestEarlyWarningApi:
    def test_requires_trained_model(self, seeded_client: TestClient) -> None:
        assert seeded_client.get("/api/early-warning").status_code == 409

    def test_schema(self, seeded_client: TestClient) -> None:
        seeded_client.post("/api/train-regimes", json={})
        response = seeded_client.get("/api/early-warning")
        assert response.status_code == 200
        body = response.json()
        assert set(body["prob_within"].keys()) == {"5", "21", "63"}
        probs = [body["prob_within"][k] for k in ("5", "21", "63")]
        assert probs[0] <= probs[1] <= probs[2]
        if not body["already_in_bear"]:
            assert body["expected_days_to_bear"] > 0


class TestAnalyticsEnrichment:
    WEIGHTS = TestStressTest.WEIGHTS

    def test_var_backtest_present(self, seeded_client: TestClient) -> None:
        body = seeded_client.post(
            "/api/portfolio-analytics", json={"weights": self.WEIGHTS}
        ).json()
        backtest = body["var_backtest"]
        assert backtest is not None
        assert 0.0 <= backtest["p_value"] <= 1.0

    def test_regime_performance_after_training(self, seeded_client: TestClient) -> None:
        before = seeded_client.post(
            "/api/portfolio-analytics", json={"weights": self.WEIGHTS}
        ).json()
        assert before["regime_performance"] is None
        seeded_client.post("/api/train-regimes", json={})
        after = seeded_client.post(
            "/api/portfolio-analytics", json={"weights": self.WEIGHTS}
        ).json()
        assert after["regime_performance"] is not None
        assert len(after["regime_performance"]) == 3

    def test_frontier_includes_risk_parity(self, seeded_client: TestClient) -> None:
        body = seeded_client.post(
            "/api/efficient-frontier", json={"n_samples": 500, "seed": 3}
        ).json()
        rp = body["risk_parity"]
        assert sum(rp["weights"].values()) == pytest.approx(1.0, abs=1e-3)

    def test_correlation_stress_param(self, seeded_client: TestClient) -> None:
        base = seeded_client.post(
            "/api/stress-test",
            json={"weights": self.WEIGHTS, "n_paths": 2000, "horizon_days": 63, "seed": 1},
        ).json()["metrics"]["var_95"]
        panic = seeded_client.post(
            "/api/stress-test",
            json={
                "weights": self.WEIGHTS,
                "n_paths": 2000,
                "horizon_days": 63,
                "seed": 1,
                "correlation_stress": 1.0,
            },
        ).json()["metrics"]["var_95"]
        assert panic > base
        # Bootstrap + correlation stress is rejected.
        response = seeded_client.post(
            "/api/stress-test",
            json={
                "weights": self.WEIGHTS,
                "n_paths": 500,
                "method": "bootstrap",
                "correlation_stress": 0.5,
            },
        )
        assert response.status_code == 422


class TestScenarios:
    def test_scenario_catalog(self, client: TestClient) -> None:
        response = client.get("/api/scenarios")
        assert response.status_code == 200
        scenarios = response.json()["scenarios"]
        keys = {s["key"] for s in scenarios}
        assert {"gfc_2008", "covid_2020"} <= keys
        for s in scenarios:
            assert s["horizon_days"] > 0
            assert isinstance(s["shocks"], dict)
