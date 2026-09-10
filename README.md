# Monte Carlo Portfolio Risk Engine

A Python engine for portfolio-level market risk analytics: correlated
Geometric Brownian Motion simulation, three Value-at-Risk methodologies,
Expected Shortfall, drawdown analysis, and predefined stress scenarios —
exposed both as a CLI and an interactive [Dash](https://dash.plotly.com/)
dashboard.

## What it does

1. Pulls historical adjusted-close prices for a set of tickers (`yfinance`,
   cached locally after the first fetch).
2. Estimates annualised drift, volatility, and the covariance / correlation
   structure from log returns.
3. Simulates thousands of correlated portfolio value paths with discrete GBM,
   correlating the Brownian shocks via a Cholesky factor of the covariance
   matrix.
4. Computes risk metrics on the simulated terminal P&L:
   - **Value at Risk** — Monte Carlo, Historical, and Parametric (normal),
     reported as positive loss amounts over the chosen horizon so the three
     methods are directly comparable.
   - **Expected Shortfall (CVaR)** at 95%.
   - **Maximum drawdown** across paths and **probability of loss**.
5. Re-runs the simulation under five stress scenarios (market crash, volatility
   shock, rate shock, correlation shock, recession).
6. Renders interactive Plotly charts and exports an Excel report.

## Requirements

- Python 3.10+
- Internet access on the first run for each ticker set (prices are then cached
  in `cache/`).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Interactive dashboard

```bash
python app.py
```

Open <http://127.0.0.1:8050>. Set the tickers, investment, horizon, path count
and history window in the sidebar, then **Run risk simulation**. The dashboard
shows KPI tiles, a portfolio-path fan chart with a 5–95% band, the terminal
P&L distribution with VaR / CVaR markers, a VaR-by-method comparison, the asset
correlation matrix, an optional stress-scenario panel, and an Excel export.

### Command line

```bash
python -m cli.main --tickers AAPL MSFT JPM --investment 100000 \
    --horizon 1.0 --sims 10000 --start 2019-01-01 --end 2025-01-01
```

Prints the VaR methodology comparison to stdout.

| Flag | Default | Meaning |
|---|---|---|
| `--tickers` | *(required)* | Space-separated ticker symbols; equal weights are assigned |
| `--investment` | `100000` | Starting portfolio value (USD) |
| `--horizon` | `1.0` | Simulation horizon in years |
| `--sims` | `10000` | Number of Monte Carlo paths |
| `--start` / `--end` | `2021-01-01` / `2026-01-01` | Historical data window |

## Project layout

```
app.py                 Dash dashboard (thin wiring over the engine)
cli/main.py            Command-line entry point
data/
  fetcher.py           yfinance download + CSV cache
  processor.py         log returns, annualisation, covariance/correlation, Cholesky
models/
  gbm.py               single-asset GBM path generator
  portfolio_sim.py     correlated multi-asset portfolio simulation
risk/
  metrics.py           VaR (MC / Historical / Parametric), ES, drawdown, prob. of loss
  stress.py            five predefined stress scenarios
charts/figures.py      Plotly figure builders
reports/generator.py   Markdown and Excel report generation
cache/                 cached price CSVs (git-ignored)
outputs/               generated reports (git-ignored)
```

## Model assumptions & limitations

The simulation uses **Geometric Brownian Motion**, which is a reasonable first
model for an equity portfolio — prices stay positive, the process is easy to
reason about, and Monte Carlo VaR can be sanity-checked against the parametric
estimate. Its assumptions are material and should be understood before relying
on the numbers:

- **Log returns are assumed normal.** Real equity returns have fatter tails, so
  extreme losses are understated.
- **Volatility is constant** over the forecast horizon. There is no volatility
  clustering (no GARCH / stochastic-vol dynamics).
- **Drift is the historical mean** and is held constant. When the lookback
  window is a bull market the forward simulation is optimistic — Monte Carlo
  VaR can legitimately come out at **$0** (the model sees no loss at that
  confidence level) even while Historical VaR, which still contains past
  drawdowns, is large. That divergence is expected, not a bug.
- **No jumps.** Gap risk (earnings surprises, macro shocks) is not modelled.
- **Portfolio weights are static** and equal. Rebalancing, transaction costs,
  taxes, liquidity, and market impact are ignored.
- **Cholesky captures linear correlation only**, estimated over the lookback
  window and held constant (except where a stress scenario overrides it). Tail
  dependence — assets crashing together — is not represented.
- **Historical VaR** is sensitive to the lookback window: it takes the
  empirical one-day loss quantile of the weighted portfolio and scales it to
  the horizon with the square-root-of-time rule. **Parametric VaR** relies on
  the normality assumption. **Monte Carlo VaR** is only as realistic as the GBM
  process above.
- **Data** comes from `yfinance` (Yahoo Finance) adjusted/close prices and
  inherits any gaps, corporate-action handling, or survivorship bias in that
  source.

Expected Shortfall is reported alongside VaR precisely because VaR gives a
threshold, not the severity of losses beyond it.

### Possible extensions

GARCH or stochastic-volatility dynamics; jump-diffusion for gap risk;
copula-based dependency for tail co-movement; fixed-income processes (Vasicek,
CIR, Hull–White); Monte Carlo option pricing.

## License

[MIT](LICENSE)
