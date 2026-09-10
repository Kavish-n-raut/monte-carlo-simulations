import os
import pandas as pd
import numpy as np
from datetime import datetime
from risk.metrics import expected_shortfall, maximum_drawdown, probability_of_loss


def _cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix (D^-1 . Cov . D^-1)."""
    std = np.sqrt(np.diag(cov.values))
    std[std == 0] = 1.0
    corr = cov.values / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)

def generate_markdown_report(
    tickers: list[str],
    weights: np.ndarray,
    investment: float,
    horizon: float,
    n_sims: int,
    var_comparison: dict,
    sim_result,
    stress_results: dict,
    cov_matrix: pd.DataFrame,
    output_path: str = "outputs/risk_report.md"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pnl = sim_result.pnl
    cvar_95 = expected_shortfall(pnl, 0.95)
    max_dd = maximum_drawdown(sim_result.portfolio_paths)
    prob_loss = probability_of_loss(sim_result.final_values, investment)
    
    md_content = f"""# Monte Carlo Financial Risk Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Portfolio Configuration
* **Assets:** {', '.join(tickers)}
* **Weights:** {', '.join([f"{w:.2%}" for w in weights])}
* **Initial Investment:** ${investment:,.2f}
* **Time Horizon:** {horizon} Years
* **Simulation Paths:** {n_sims:,}

## 2. Value-at-Risk (VaR) Comparison

All figures are positive loss amounts over the {horizon:g}-year horizon. $0.00 means
the method sees no loss at that confidence level.

| Confidence | Monte Carlo | Historical | Parametric |
|------------|-------------|------------|------------|
| 95%        | ${var_comparison['monte_carlo'][0]:,.2f} | ${var_comparison['historical'][0]:,.2f} | ${var_comparison['parametric'][0]:,.2f} |
| 99%        | ${var_comparison['monte_carlo'][1]:,.2f} | ${var_comparison['historical'][1]:,.2f} | ${var_comparison['parametric'][1]:,.2f} |

**Conditional VaR (Expected Shortfall) at 95%:** ${cvar_95:,.2f}

## 3. Advanced Risk Metrics
* **Maximum Drawdown:** {max_dd:.2%}
* **Probability of Loss:** {prob_loss:.2%}
"""
    with open(output_path, "w") as f:
        f.write(md_content)
    return output_path

def generate_excel_report(
    tickers: list[str],
    weights: np.ndarray,
    investment: float,
    horizon: float,
    n_sims: int,
    var_comparison: dict,
    sim_result,
    stress_results: dict,
    cov_matrix: pd.DataFrame,
    output_path: str = "outputs/risk_report.xlsx"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_data = {
            "Metric": ["Assets", "Initial Investment", "Time Horizon (Years)", "Simulations Run"],
            "Value": [", ".join(tickers), investment, horizon, n_sims]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(var_comparison).to_excel(writer, sheet_name="VaR_Comparison", index=False)
        _cov_to_corr(cov_matrix).to_excel(writer, sheet_name="Correlation_Matrix")
    return output_path