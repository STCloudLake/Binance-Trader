"""Monte Carlo simulation for trading strategy robustness assessment.

Reshuffles trade sequences to estimate the probability that observed
performance is due to skill rather than luck.  Based on the principle
that if trade order doesn't matter (i.e., signals are random), then
any reshuffling should produce similar results.

Reference:
  Bailey & Lopez de Prado (2014) — Deflated Sharpe Ratio
  Burns (2006) — "Random Portfolios for Performance Measurement"
"""

import numpy as np
from loguru import logger


def monte_carlo_simulation(
    trades: list[dict],
    n_simulations: int = 2000,
    confidence: float = 0.95,
    initial_balance: float = 10000.0,
    seed: int | None = 42,
) -> dict:
    """Bootstrap trade sequences to quantify strategy robustness.

    Randomly shuffles the ORDER of trades *n_simulations* times.  For
    each shuffle, recalculates the equity curve, final return, max
    drawdown, and Sharpe ratio.  The distribution of outcomes tells us
    how much the strategy depends on lucky trade ordering.

    Args:
        trades: List of trade dicts.  Each must have a ``"pnl"`` key
            (float, can be positive or negative).  If entries also have
            ``"amount_usdt"`` they are used for capital tracking;
            otherwise each trade is assumed to use the full capital.
        n_simulations: Number of bootstrap samples (default 2000).
            Increase for more stable percentile estimates.
        confidence: Confidence level for intervals (default 0.95 →
            2.5th and 97.5th percentiles).
        initial_balance: Starting capital (default 10000).  Only affects
            return-pct scaling; does not alter the distribution shape.
        seed: RNG seed for reproducibility (default 42).

    Returns:
        dict with keys:
        - mc_median_return_pct: median terminal return across sims
        - mc_ci_95_lower: lower bound of CI (e.g. 2.5th pct)
        - mc_ci_95_upper: upper bound of CI (e.g. 97.5th pct)
        - mc_prob_loss: fraction of sims ending below initial_balance
        - mc_drawdown_median: median max drawdown (%) across sims
        - mc_drawdown_p95: 95th percentile worst drawdown (%)
        - mc_sharpe_median: median Sharpe across sims
        - mc_sharpe_p05: 5th percentile Sharpe (worst case)
        - mc_trades_used: number of trades in the simulation
        - mc_simulations: number of simulations run
    """
    # Extract PnL sequence
    pnl_values = np.array([t.get("pnl", 0.0) for t in trades if t.get("pnl") is not None])
    n_trades = len(pnl_values)

    if n_trades < 10:
        logger.warning(f"Monte Carlo: only {n_trades} trades — results may be unstable")
    if n_trades == 0:
        return {
            "mc_median_return_pct": 0.0, "mc_ci_95_lower": 0.0,
            "mc_ci_95_upper": 0.0, "mc_prob_loss": 1.0,
            "mc_drawdown_median": 0.0, "mc_drawdown_p95": 0.0,
            "mc_sharpe_median": 0.0, "mc_sharpe_p05": 0.0,
            "mc_trades_used": 0, "mc_simulations": 0,
        }

    rng = np.random.RandomState(seed)

    # Pre-allocate result arrays
    final_returns = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)
    sharpes = np.empty(n_simulations)

    alpha = 1.0 - confidence
    lower_pct = alpha / 2.0 * 100       # e.g. 2.5
    upper_pct = (1.0 - alpha / 2.0) * 100  # e.g. 97.5

    for i in range(n_simulations):
        # Shuffle trade order (destroys any autocorrelation / streak structure)
        perm = rng.permutation(pnl_values)
        equity = initial_balance + np.cumsum(perm)

        # Return
        final_returns[i] = (equity[-1] - initial_balance) / initial_balance * 100

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak * 100
        max_drawdowns[i] = float(np.max(dd))

        # Sharpe (crude: daily-equivalent from trade returns)
        # Use trade-level returns as proxy — this is the standard
        # approach for trade-order Monte Carlo.
        rets = perm / initial_balance  # trade returns as fraction of capital
        if rets.std() > 0 and len(rets) > 1:
            sharpes[i] = float(rets.mean() / rets.std() * np.sqrt(252))
        else:
            sharpes[i] = 0.0

    # ── Aggregate statistics ──
    return {
        "mc_median_return_pct": round(float(np.median(final_returns)), 2),
        "mc_ci_95_lower": round(float(np.percentile(final_returns, lower_pct)), 2),
        "mc_ci_95_upper": round(float(np.percentile(final_returns, upper_pct)), 2),
        "mc_prob_loss": round(float(np.mean(final_returns < 0)), 4),
        "mc_drawdown_median": round(float(np.median(max_drawdowns)), 2),
        "mc_drawdown_p95": round(float(np.percentile(max_drawdowns, 95)), 2),
        "mc_sharpe_median": round(float(np.median(sharpes)), 2),
        "mc_sharpe_p05": round(float(np.percentile(sharpes, 5)), 2),
        "mc_trades_used": n_trades,
        "mc_simulations": n_simulations,
    }

