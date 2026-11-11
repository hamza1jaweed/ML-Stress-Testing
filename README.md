# Regime Terminal — Market Regime Detection & Portfolio Stress-Testing

Real-time market regime detection (unsupervised GMM + Markov transition
estimation) and dynamic portfolio stress testing (Monte Carlo + historical
crisis replay) with a dark financial-terminal React frontend.

Runs fully offline: the synthetic data generator produces correlated,
regime-switching geometric Brownian motion with volatility clustering, so no
external API or market-data key is required. A cached `yfinance` wrapper
(`engine.data_generator.LiveDataFetcher`) is available for real tickers.

## Architecture

```
├── engine/                  # ML core (pure Python, no web deps)
│   ├── data_generator.py    #   correlated regime-switching GBM + GARCH vol, OHLC
│   ├── regime_model.py      #   GMM regime detection, transition matrix, labeling
│   └── stress_engine.py     #   Monte Carlo (parametric + block bootstrap),
│                            #   historical replay, VaR / CVaR / max drawdown
├── backend/                 # FastAPI (Pydantic v2, async, threadpooled NumPy)
│   ├── main.py              #   app factory, CORS
│   ├── routes.py            #   /api/generate-data, /train-regimes, /stress-test,
│   │                        #   /scenarios, /health
│   └── schemas.py           #   request/response models
├── frontend/                # React 18 + Vite + TypeScript + Tailwind + Recharts
│   └── src/components/      #   AllocationMatrix, RegimeVisualizer (+ heatmap),
│                            #   StressCockpit (fan chart, shock sliders, VaR cards)
├── tests/                   # pytest: statistical sanity + API schema compliance
├── docker-compose.yml
├── run.sh / run.ps1         # local dev (bash / Windows PowerShell)
```

## Quick start (local)

Prereqs: Python 3.11+, Node 18+.

```bash
./run.sh          # macOS / Linux / Git Bash
.\run.ps1         # Windows PowerShell
```

Then open http://localhost:5173 — click **Generate data**, then **Train
regimes**, and drag the allocation / shock sliders. The FastAPI docs are at
http://localhost:8000/docs.

## Quick start (Docker)

```bash
docker compose up --build
```

## Tests

```bash
pip install -r backend/requirements.txt pytest httpx
pytest
```

- `tests/test_ml_engine.py` — reproducibility under seed locking, regime
  recovery accuracy vs. generator truth, VaR/CVaR ordering, fan-chart
  monotonicity, drawdown math, scenario severity.
- `tests/test_api.py` — endpoint schemas, validation errors (422/409),
  cross-call reproducibility.

## API sketch

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/generate-data` | POST | Seed synthetic multi-asset history (`n_days`, `seed`) |
| `/api/train-regimes` | POST | Fit GMM regimes; returns states, transition matrix, signatures |
| `/api/stress-test` | POST | Monte Carlo (`parametric` / `bootstrap` / `regime_aware`) or `scenario` replay against custom `weights`; returns VaR/CVaR/MaxDD + percentile fan |
| `/api/portfolio-analytics` | POST | Historical equity curve, drawdown, Sharpe/Sortino/Calmar, risk contributions, correlation matrix |
| `/api/efficient-frontier` | POST | Random-portfolio cloud + SLSQP max-Sharpe / min-vol portfolios and frontier curve |
| `/api/regime-forecast` | GET | Markov-chain regime probability forecast from the latest decoded state |
| `/api/reverse-stress` | POST | Most likely joint market move producing a target loss (min-Mahalanobis closed form) |
| `/api/early-warning` | GET | P(entering bear regime within 5/21/63 days) + expected hitting time (absorbing chain) |
| `/api/scenarios` | GET | Built-in crisis templates (2008 GFC, 2020 COVID, taper tantrum, stagflation) |
| `/api/health` | GET | Status + hardware metrics |

### Highlight: regime-aware Monte Carlo

`POST /api/stress-test` with `"regime_aware": true` (after training) simulates
each path's own regime sequence from the fitted Markov transition matrix and
draws daily returns from that regime's historical distribution — so fat tails
and volatility clustering are endogenous, and risk depends on the regime the
market is in *right now*. The response reports the share of simulated days
spent in each regime.

### Highlight: reverse stress testing

`POST /api/reverse-stress` inverts the usual question: given a target loss L,
it solves ``min s'Σ⁻¹s subject to w's = -L`` in closed form
(``s* = -L·Σw / w'Σw``) — the highest-likelihood joint market move that
produces exactly that loss. Hedging assets can move *up* in the answer, and
the Mahalanobis distance reports how many sigmas the move is.

### Highlight: correlation breakdown

`correlation_stress` on `/api/stress-test` blends the correlation matrix
toward all-ones ("in a crisis, correlations go to 1") with marginal variances
fixed — a direct way to watch diversification benefits evaporate. The
analytics endpoint also ships a rolling-VaR **Kupiec backtest**, per-regime
portfolio performance, and a **risk-parity** optimizer on the frontier.

All simulations are vectorized NumPy with explicit seeds — identical requests
return identical results.
