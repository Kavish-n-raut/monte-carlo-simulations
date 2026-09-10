"""Dash dashboard for the Monte Carlo financial risk engine.

Run locally:

    python app.py

then open http://127.0.0.1:8050. This file is a thin wiring layer: all the
quantitative work lives in ``data/``, ``models/``, ``risk/`` and is unchanged
from the CLI (``python -m cli.main``).
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc

from data.fetcher import fetch_prices
from data.processor import (
    annualised_return,
    correlation_matrix,
    covariance_matrix,
    log_returns,
)
from models.portfolio_sim import simulate_portfolio
from risk.metrics import (
    compare_var_methods,
    expected_shortfall,
    maximum_drawdown,
    probability_of_loss,
)
from risk.stress import run_stress_tests
from reports.generator import generate_excel_report
from charts.figures import (
    correlation_heatmap_figure,
    pnl_distribution_figure,
    portfolio_paths_figure,
    stress_figure,
    var_comparison_figure,
)

SIM_SEED = 42
STRESS_MAX_SIMS = 10_000
EXCEL_PATH = os.path.join("outputs", "risk_report.xlsx")

app = Dash(
    __name__,
    title="Monte Carlo Risk Engine",
    external_stylesheets=[dbc.themes.DARKLY],
)
server = app.server  # exposed for gunicorn / production servers

# DARKLY renders native inputs and dcc.Dropdown light, which makes their text
# invisible on the dark page. Ship the contrast fixes inline so the app is
# self-contained (no assets/ dependency).
app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        body { font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
        .form-control, .form-select {
            background-color: #2b3035 !important; color: #e9ecef !important;
            border-color: #495057 !important;
        }
        .form-control:focus, .form-select:focus {
            background-color: #2b3035 !important; color: #e9ecef !important;
            border-color: #6ea8fe !important;
            box-shadow: 0 0 0 0.2rem rgba(110,168,254,.25) !important;
        }
        .form-control::placeholder { color: #868e96 !important; }
        .form-control::-webkit-calendar-picker-indicator { filter: invert(.85); cursor: pointer; }
        input:-webkit-autofill, input:-webkit-autofill:hover, input:-webkit-autofill:focus {
            -webkit-text-fill-color: #e9ecef !important;
            -webkit-box-shadow: 0 0 0 1000px #2b3035 inset !important;
        }
        /* dcc.Dropdown (Dash >= 3 markup) */
        .dash-dropdown, .mcp-dropdown {
            background-color: #2b3035 !important; border-color: #495057 !important; color: #e9ecef !important;
        }
        .dash-dropdown span, .dash-dropdown .dash-dropdown-value,
        .dash-dropdown .dash-dropdown-value-item { color: #e9ecef !important; }
        .dash-dropdown svg path { fill: #e9ecef !important; }
        .dash-dropdown-menu, [class*="dash-dropdown-menu"], [class*="dash-dropdown-option"] {
            background-color: #2b3035 !important; color: #e9ecef !important;
        }
        [class*="dash-dropdown-option"]:hover,
        [class*="dash-dropdown-option"][aria-selected="true"] {
            background-color: #375a7f !important; color: #fff !important;
        }
        .card { border-color: rgba(148,163,184,.18); }
        .dash-graph { min-height: 380px; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""


# --------------------------------------------------------------------------- UI


def _blank_figure(message: str = "Run a simulation to populate this panel.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=message, showarrow=False, font=dict(size=14, color="#94a3b8"))],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
        height=360,
    )
    return fig


def control_panel() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Portfolio & Simulation", className="mb-3"),
                dbc.Label("Tickers (comma separated)"),
                dbc.Input(id="tickers", value="AAPL, MSFT, JPM", type="text"),
                html.Div("Equal weights are assigned automatically.", className="text-muted small mb-3"),
                dbc.Label("Initial investment ($)"),
                dbc.Input(id="investment", value=100_000, type="number", min=1_000, step=1_000),
                html.Br(),
                dbc.Label("Time horizon (years)"),
                dcc.Slider(
                    id="horizon", min=0.25, max=5, step=0.25, value=1,
                    marks={i: str(i) for i in range(1, 6)},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Br(),
                dbc.Label("Monte Carlo paths"),
                dcc.Dropdown(
                    id="nsims",
                    options=[{"label": f"{n:,}", "value": n} for n in (1_000, 5_000, 10_000, 25_000, 50_000)],
                    value=10_000,
                    clearable=False,
                    className="mcp-dropdown",
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("History start"),
                                dbc.Input(id="start", type="date", value="2019-01-01"),
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Label("History end"),
                                dbc.Input(id="end", type="date", value=date.today().isoformat()),
                            ]
                        ),
                    ],
                    className="g-2",
                ),
                html.Br(),
                dbc.Checklist(
                    id="opts",
                    options=[{"label": "  Include stress scenarios (slower)", "value": "stress"}],
                    value=[], switch=True,
                ),
                html.Br(),
                dbc.Button("Run risk simulation", id="run", color="primary", className="w-100"),
                html.Div(id="status", className="text-muted small mt-2"),
            ]
        ),
        className="shadow-sm",
    )


def kpi_card(title: str, value: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted small text-uppercase"),
                html.H4(value, className="mb-0 mt-1"),
            ]
        ),
        className="text-center shadow-sm h-100",
    )


def graph(fig_id: str) -> dcc.Graph:
    return dcc.Graph(id=fig_id, figure=_blank_figure(), config={"displayModeBar": False})


app.layout = dbc.Container(
    [
        html.Div(
            [
                html.H3("Monte Carlo Financial Risk Analytics Engine", className="mb-0"),
                html.Div(
                    "Correlated Geometric Brownian Motion · VaR / Expected Shortfall · stress testing",
                    className="text-muted",
                ),
            ],
            className="py-3",
        ),
        dbc.Row(
            [
                dbc.Col(control_panel(), lg=3, md=4),
                dbc.Col(
                    dcc.Loading(
                        html.Div(
                            [
                                html.Div(id="kpis", className="mb-3"),
                                dbc.Row(
                                    [
                                        dbc.Col(graph("fig-paths"), lg=6),
                                        dbc.Col(graph("fig-pnl"), lg=6),
                                    ],
                                    className="g-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(graph("fig-var"), lg=6),
                                        dbc.Col(graph("fig-corr"), lg=6),
                                    ],
                                    className="g-3 mt-1",
                                ),
                                html.Div(
                                    dcc.Graph(id="fig-stress", figure=_blank_figure(), config={"displayModeBar": False}),
                                    id="stress-wrap",
                                    style={"display": "none"},
                                    className="mt-3",
                                ),
                                html.Div(id="var-note", className="text-muted small mt-3"),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Download Excel report", id="dl-btn",
                                            color="secondary", outline=True, size="sm",
                                            disabled=True,
                                        ),
                                        dcc.Download(id="dl"),
                                    ],
                                    className="mt-3",
                                ),
                            ]
                        ),
                        type="default",
                    ),
                    lg=9, md=8,
                ),
            ],
            className="g-4",
        ),
        dcc.Store(id="store"),
        html.Footer(
            "Model assumptions (constant volatility, normal shocks, static weights) are "
            "documented in LIMITATIONS.md.",
            className="text-muted small text-center py-4",
        ),
    ],
    fluid=True,
    className="px-4",
)


# ---------------------------------------------------------------------- callbacks


def _parse_tickers(raw: str) -> list[str]:
    cleaned = (raw or "").replace(";", ",").replace(" ", ",")
    return [t.strip().upper() for t in cleaned.split(",") if t.strip()]


def _var_note(prob_loss: float) -> str:
    return (
        f"Probability of ending below the initial investment: {prob_loss:.1%}. "
        "VaR and Expected Shortfall are positive loss amounts over the selected horizon; "
        "$0 means the model sees no loss at that confidence level, which is common for "
        "Monte Carlo VaR when the historical drift is strongly positive while Historical "
        "VaR still reflects past drawdowns."
    )


def _error(message: str) -> tuple:
    """Return tuple for the run_simulation callback that only sets the status line."""
    return (*(no_update,) * 9, dbc.Alert(message, color="danger"), no_update)


@app.callback(
    Output("kpis", "children"),
    Output("fig-paths", "figure"),
    Output("fig-pnl", "figure"),
    Output("fig-var", "figure"),
    Output("fig-corr", "figure"),
    Output("fig-stress", "figure"),
    Output("stress-wrap", "style"),
    Output("var-note", "children"),
    Output("store", "data"),
    Output("status", "children"),
    Output("dl-btn", "disabled"),
    Input("run", "n_clicks"),
    State("tickers", "value"),
    State("investment", "value"),
    State("horizon", "value"),
    State("nsims", "value"),
    State("start", "value"),
    State("end", "value"),
    State("opts", "value"),
    prevent_initial_call=True,
)
def run_simulation(_clicks, tickers_raw, investment, horizon, nsims, start, end, opts):
    tickers = _parse_tickers(tickers_raw)
    if not tickers:
        return _error("Enter at least one ticker.")

    try:
        investment = float(investment)
        horizon = float(horizon)
        nsims = int(nsims)
        if investment <= 0 or horizon <= 0:
            raise ValueError("Investment and horizon must be positive.")
        prices = fetch_prices(tickers, str(start), str(end))
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        return _error(f"Could not run: {exc}")

    # fetch_prices can drop tickers with no data; realign weights to what loaded.
    tickers = list(prices.columns)
    weights = np.repeat(1.0 / len(tickers), len(tickers))

    sim = simulate_portfolio(prices, weights, investment, horizon, nsims, seed=SIM_SEED)
    returns = log_returns(prices)
    cov = covariance_matrix(returns)
    mu_port = float(weights @ annualised_return(returns).to_numpy())
    sigma_port = float(np.sqrt(weights @ cov.to_numpy() @ weights))

    var_comp = compare_var_methods(sim, prices, weights, mu_port, sigma_port, investment, horizon)
    cvar_95 = expected_shortfall(sim.pnl, 0.95)
    max_dd = maximum_drawdown(sim.portfolio_paths)
    prob_loss = probability_of_loss(sim.final_values, investment)
    corr = correlation_matrix(returns)

    kpis = dbc.Row(
        [
            dbc.Col(kpi_card("Expected value", f"${sim.final_values.mean():,.0f}"), md=3),
            dbc.Col(kpi_card("Monte Carlo VaR 95%", f"${var_comp['monte_carlo'][0]:,.0f}"), md=3),
            dbc.Col(kpi_card("Expected shortfall 95%", f"${cvar_95:,.0f}"), md=3),
            dbc.Col(kpi_card("Max drawdown", f"{max_dd:.1%}"), md=3),
        ],
        className="g-2",
    )

    stress_fig = _blank_figure()
    stress_style = {"display": "none"}
    if "stress" in (opts or []):
        stress = run_stress_tests(prices, weights, investment, horizon, min(nsims, STRESS_MAX_SIMS))
        stress_fig = stress_figure(stress, investment)
        stress_style = {"display": "block"}

    store = {
        "tickers": tickers,
        "weights": weights.tolist(),
        "investment": investment,
        "horizon": horizon,
        "nsims": nsims,
        "var_comparison": var_comp,
        "covariance": cov.to_dict(),
    }
    status = f"Ran {nsims:,} paths for {', '.join(tickers)}."

    return (
        kpis,
        portfolio_paths_figure(sim.portfolio_paths, investment),
        pnl_distribution_figure(sim.pnl, var_comp["monte_carlo"][0], var_comp["monte_carlo"][1], cvar_95),
        var_comparison_figure(var_comp),
        correlation_heatmap_figure(corr),
        stress_fig,
        stress_style,
        _var_note(prob_loss),
        store,
        status,
        False,
    )


@app.callback(
    Output("dl", "data"),
    Input("dl-btn", "n_clicks"),
    State("store", "data"),
    prevent_initial_call=True,
)
def download_excel(_clicks, data):
    if not data:
        return no_update
    cov = pd.DataFrame(data["covariance"])
    os.makedirs("outputs", exist_ok=True)
    generate_excel_report(
        tickers=data["tickers"],
        weights=np.array(data["weights"]),
        investment=data["investment"],
        horizon=data["horizon"],
        n_sims=data["nsims"],
        var_comparison=data["var_comparison"],
        sim_result=None,
        stress_results={},
        cov_matrix=cov,
        output_path=EXCEL_PATH,
    )
    return dcc.send_file(EXCEL_PATH)


if __name__ == "__main__":
    app.run(debug=True)
