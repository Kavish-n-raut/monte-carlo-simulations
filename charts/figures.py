"""Plotly figure builders for the Monte Carlo risk dashboard.

Every function returns a ``plotly.graph_objects.Figure`` styled for a dark
layout. Keeping figure construction here means the Dash app file stays a thin
wiring layer over the risk engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TEAL = "#2dd4bf"
BLUE = "#3b82f6"
RED = "#ef4444"
DARK_RED = "#b91c1c"
AMBER = "#f59e0b"
GRID = "rgba(148, 163, 184, 0.18)"

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=64, r=28, t=64, b=48),
    font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=13),
    title=dict(x=0, xanchor="left", y=0.97, font=dict(size=15)),
    legend=dict(
        yanchor="top", y=0.99, xanchor="right", x=0.99,
        bgcolor="rgba(15,23,42,0.6)", bordercolor="rgba(148,163,184,0.25)", borderwidth=1,
    ),
    height=380,
)


def _apply(fig: go.Figure, title: str, xtitle: str, ytitle: str) -> go.Figure:
    layout = dict(_LAYOUT)
    layout["title"] = {**_LAYOUT["title"], "text": title}
    fig.update_layout(xaxis_title=xtitle, yaxis_title=ytitle, **layout)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def portfolio_paths_figure(
    portfolio_paths: np.ndarray, investment: float, n_display: int = 120
) -> go.Figure:
    """Fan chart of simulated portfolio value paths with a 5-95% band."""
    sample = portfolio_paths[:n_display]
    n_steps = sample.shape[1]
    steps = np.arange(n_steps)

    xs: list = []
    ys: list = []
    for path in sample:
        xs.extend(steps.tolist())
        xs.append(None)
        ys.extend(path.tolist())
        ys.append(None)

    median = np.median(portfolio_paths, axis=0)
    p05 = np.percentile(portfolio_paths, 5, axis=0)
    p95 = np.percentile(portfolio_paths, 95, axis=0)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=xs, y=ys, mode="lines", name="Simulated paths",
            line=dict(color=TEAL, width=1), opacity=0.16, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(x=steps, y=p95, mode="lines", line=dict(width=0),
                   showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=steps, y=p05, mode="lines", line=dict(width=0), name="5-95% band",
            fill="tonexty", fillcolor="rgba(59, 130, 246, 0.16)", hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(x=steps, y=median, mode="lines", name="Median",
                   line=dict(color=BLUE, width=2.5))
    )
    fig.add_hline(
        y=investment, line=dict(color="#e5e7eb", width=1.5, dash="dash"),
        annotation_text="Initial investment", annotation_position="bottom right",
    )
    _apply(fig, "Simulated Portfolio Value Paths", "Trading day", "Portfolio value ($)")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def pnl_distribution_figure(
    pnl: np.ndarray, var_95: float, var_99: float, cvar_95: float
) -> go.Figure:
    """Histogram of terminal P&L with VaR / CVaR markers on the loss side.

    ``var_95``, ``var_99`` and ``cvar_95`` are positive loss magnitudes.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=pnl, nbinsx=90, marker_color=BLUE, opacity=0.75, name="P&L")
    )
    markers = [
        (-var_95, RED, f"VaR 95%  ${var_95:,.0f}", "top"),
        (-var_99, DARK_RED, f"VaR 99%  ${var_99:,.0f}", "top left"),
        (-cvar_95, AMBER, f"CVaR 95%  ${cvar_95:,.0f}", "bottom"),
    ]
    for x, color, label, pos in markers:
        fig.add_vline(
            x=x, line=dict(color=color, width=2, dash="dash"),
            annotation_text=label, annotation_position=pos,
            annotation_font_color=color,
        )
    fig.add_vline(x=0, line=dict(color="#94a3b8", width=1))
    _apply(fig, "Terminal Profit & Loss Distribution", "Profit / loss ($)", "Simulations")
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    return fig


def var_comparison_figure(var_comp: dict) -> go.Figure:
    """Grouped bar chart comparing VaR methods at each confidence level."""
    conf = [f"{c:.0%}" for c in var_comp["confidence"]]
    fig = go.Figure()
    for name, key, color in [
        ("Monte Carlo", "monte_carlo", TEAL),
        ("Historical", "historical", BLUE),
        ("Parametric", "parametric", AMBER),
    ]:
        fig.add_trace(
            go.Bar(
                name=name, x=conf, y=var_comp[key], marker_color=color,
                text=[f"${v:,.0f}" for v in var_comp[key]], textposition="outside",
            )
        )
    fig.update_layout(barmode="group")
    _apply(fig, "Value-at-Risk by Method", "Confidence level", "Loss over horizon ($)")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def correlation_heatmap_figure(corr: pd.DataFrame) -> go.Figure:
    """Annotated asset correlation heatmap."""
    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            colorscale="RdBu", zmid=0.0, zmin=-1.0, zmax=1.0,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=12), colorbar=dict(title="Corr"),
        )
    )
    _apply(fig, "Asset Return Correlation", "", "")
    fig.update_yaxes(autorange="reversed")
    return fig


def stress_figure(stress_results: dict, investment: float) -> go.Figure:
    """Expected portfolio impact (% of investment) for each stress scenario."""
    labels = [name.replace("_", " ").title() for name in stress_results]
    impact_pct = [
        (investment - float(sr.final_values.mean())) / investment * 100.0
        for sr in stress_results.values()
    ]
    colors = [RED if v > 0 else TEAL for v in impact_pct]
    fig = go.Figure(
        go.Bar(
            x=labels, y=impact_pct, marker_color=colors,
            text=[f"{v:+.1f}%" for v in impact_pct], textposition="outside",
        )
    )
    _apply(
        fig,
        "Stress Scenarios - Expected Portfolio Impact",
        "",
        "Expected loss (% of investment)",
    )
    return fig
