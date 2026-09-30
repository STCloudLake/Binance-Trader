"""Conditional-volatility forecasting (Phase P3 of the algorithm upgrade plan).

Why this module exists (scientific rationale)
---------------------------------------------
The P2 measurement says directional prediction on returns is **not** a usable
edge here: every ML candidate was refused by the credibility gate
(OOS AUC 0.52 / 0.53 on BTC / ETH 1h, negative net-of-cost expectancy — see
``docs/core-algorithms/08-ml-triple-barrier.md``).  What *is* predictable in
financial time series is the **conditional variance**: the ARCH/GARCH family
(Engle 1982; Bollerslev 1986; Tsay, *Analysis of Financial Time Series*,
ch. 3) shows that squared returns are autocorrelated — a large move today
raises the variance of tomorrow.  Volatility also clusters, mean-reverts and is
far more persistent than the sign of the return.

So the highest-value use of prediction is **risk, not direction**:

1. ``core/risk/position_sizer.py`` — volatility-targeted position sizing,
2. ``core/risk/position_guard.py`` + ``core/executor/executor.py`` — stop /
   trailing widths that widen in turbulent regimes and tighten in calm ones,
3. ``core/ml/labels.py`` — triple-barrier widths driven by the same forecast,
4. risk / circuit-breaker thresholds (regime percentile, see
   :func:`vol_percentile`).

Units (read this before using a number)
---------------------------------------
Every quantity here is a **fraction of price per bar**, never a percent:

* ``log_returns`` is dimensionless (``log(P_t / P_{t-1})``),
* ``realized_vol(..., unit="per_bar")`` is a standard deviation of those log
  returns, i.e. ``0.004`` = 0.4 % per bar,
* ``unit="annual"`` multiplies by ``sqrt(periods_per_year)`` — annualisation
  **assumes i.i.d. returns**, so it is a reporting convention, not something to
  price a single bar with,
* :func:`annualize` / :func:`deannualize` convert between the two,
* :func:`to_pct` is the *only* place a percent appears (the existing config
  knobs are percent-typed).

Cost: every estimator in the live path is O(window) numpy with no allocation of
an (n × n) matrix, and :class:`VolForecaster` memoises per bar.  The measured
per-bar budget is asserted in ``tests/test_volatility_targeting.py``.

GARCH dependency note
---------------------
``arch`` (Kevin Sheppard's package) is **not** installed in this deployment and
was deliberately *not* added to ``requirements.txt``: a heavy compiled
dependency for one estimator on a per-bar path.  :func:`garch11_forecast`
therefore has two documented paths:

1. ``import arch`` succeeds → the reference MLE fit (``arch_model``);
2. otherwise → our own GARCH(1,1) Gaussian MLE with ``scipy.optimize`` (always
   available, since scipy is already a dependency), with the long-run variance
   pinned to the sample variance (variance targeting, a standard trick that
   removes one parameter and keeps the fit stable).

:func:`garch_backend` reports which one ran, and the research/reporting
scripts print it, so a number can never be silently attributed to the wrong
fitter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── defaults ─────────────────────────────────────────────────────────────

#: RiskMetrics decay factor for the EWMA conditional variance (J.P. Morgan,
#: RiskMetrics Technical Document, 4th ed., 1996, ch. 5.3).  The documented
#: default rather than a fitted parameter: it is the robust, no-tuning choice.
DEFAULT_LAMBDA = 0.94

#: Bars used by the estimator when the caller does not say.  1 h crypto data
#: over ~500 bars ≈ 3 weeks: long enough for the EWMA to forget its start,
#: short enough that a regime shift is visible.
DEFAULT_WINDOW = 500

#: Bars per year used for annualisation, by interval label.
PERIODS_PER_YEAR: dict[str, float] = {
    "1m": 365.0 * 24.0 * 60.0,
    "3m": 365.0 * 24.0 * 20.0,
    "5m": 365.0 * 24.0 * 12.0,
    "15m": 365.0 * 24.0 * 4.0,
    "30m": 365.0 * 24.0 * 2.0,
    "1h": 365.0 * 24.0,
    "2h": 365.0 * 12.0,
    "4h": 365.0 * 6.0,
    "6h": 365.0 * 4.0,
    "8h": 365.0 * 3.0,
    "12h": 365.0 * 2.0,
    "1d": 365.0,
    "1w": 365.0 / 7.0,
}

#: Return-clipping threshold in robust sigmas (``k * 1.4826 * MAD``) applied
#: before any squared return enters an estimator.  Justification is measured,
#: not theoretical: the shipped cache carries a **data splice** — the BTC 1h
#: frame jumps 2026-07-29 → 2026-09-29 inside one "bar", i.e. a single
#: ``+27.63 %`` log return that is a calendar gap, not a one-hour move.  An
#: unclipped RiskMetrics recursion (effective window ``1/(1-λ) = 16.7`` bars)
#: then reports 5.47 %/bar instead of 0.44 %, a 12× overstatement that would
#: shrink every position by the same factor.  ``0`` disables clipping.
DEFAULT_OUTLIER_SIGMA = 6.0

#: ``1 / Phi^{-1}(0.75)`` — makes the MAD a consistent scale estimate for a
#: normal sample (the usual 1.4826 constant).
_MAD_TO_SIGMA = 1.4826

#: Method names accepted by :func:`forecast_vol`.
METHODS = (
    "ewma",             # RiskMetrics EWMA conditional variance (default)
    "realized_cc",      # close-to-close sample std
    "realized_parkinson",
    "realized_garman_klass",
    "garch11",          # MLE fit (see the module docstring for the two paths)
)

#: Methods that need the full OHLC frame rather than a single return series.
OHLC_METHODS = ("realized_parkinson", "realized_garman_klass")

#: Per-bar compute budget (seconds) for one :func:`forecast_vol` call on a
#: 500-bar window.  Asserted in the test suite; the live predictor's async path
#: computes indicators once per kline, so a forecast must be negligible next to
#: that.  Measured ≈0.5 ms worst case (GARCH MLE ≈30 ms) — see the doc's table.
PER_BAR_BUDGET_SEC = 2.0e-3


# ── returns ──────────────────────────────────────────────────────────────

def simple_returns(close) -> np.ndarray:
    """Arithmetic returns ``P_t / P_{t-1} - 1`` (NaN-free float array)."""
    arr = np.asarray(close, dtype=np.float64).ravel()
    if arr.size < 2:
        return np.empty(0, dtype=np.float64)
    out = arr[1:] / arr[:-1] - 1.0
    return out[np.isfinite(out)]


def log_returns(close) -> np.ndarray:
    """Log returns ``log(P_t / P_{t-1})`` — the estimator input of record."""
    arr = np.asarray(close, dtype=np.float64).ravel()
    if arr.size < 2:
        return np.empty(0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(arr[1:] / arr[:-1])
    return out[np.isfinite(out)]


def _last(series, window: int) -> np.ndarray:
    """Last ``window`` finite values of ``series`` as a float array."""
    arr = np.asarray(series, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if window and window > 0:
        arr = arr[-int(window):]
    return arr


def clip_outliers(returns, *, sigma: float = DEFAULT_OUTLIER_SIGMA) -> np.ndarray:
    """Winsorise returns at ``±sigma`` robust sigmas (``1.4826 * MAD``).

    A stale cache, a calendar gap or a fat finger produces a return that no
    one-hour move could produce; squaring it makes it dominate every estimator
    below.  The MAD is used instead of the standard deviation because the
    quantity being guarded against is exactly what inflates the latter.
    ``sigma <= 0`` returns the input unchanged.  The MAD is computed over the
    same window that is clipped, so this stays causal and O(n).
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if sigma is None or float(sigma) <= 0 or r.size < 8:
        return r
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    scale = mad * _MAD_TO_SIGMA
    if not math.isfinite(scale) or scale <= 0.0:
        return r
    limit = float(sigma) * scale
    return np.clip(r, med - limit, med + limit)


# ── realised-volatility estimators ───────────────────────────────────────

def realized_vol(returns, *, window: int = DEFAULT_WINDOW,
                 unit: str = "per_bar",
                 periods_per_year: float = 8760.0) -> float:
    """Close-to-close realised volatility = sample std (ddof=1) of returns.

    The plainest estimator and the honest baseline: it uses only closing
    prices, treats every bar as equally informative, and is what
    :func:`forecast_vol` is compared against.  Returns 0.0 (not NaN) when
    there are fewer than 2 observations, so callers can branch on a number.
    """
    r = _last(returns, window)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return annualize(sd, periods_per_year) if unit == "annual" else sd


def _ohlc_arrays(df) -> tuple[np.ndarray, ...]:
    """(open, high, low, close) float arrays from a DataFrame, NaNs dropped."""
    out = []
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise KeyError(f"volatility estimator needs the '{col}' column")
        out.append(np.asarray(df[col], dtype=np.float64).ravel())
    return tuple(out)


def parkinson_vol(df, *, window: int = DEFAULT_WINDOW, unit: str = "per_bar",
                  periods_per_year: float = 8760.0) -> float:
    """Parkinson (1980) high-low range volatility.

    ``sigma^2 = mean(ln(H/L)^2) / (4 ln 2)``.  Uses the intrabar range, so it
    is ≈5× more efficient than close-to-close when the true process is a
    driftless diffusion — but it ignores gaps and any close-inside-the-range
    information, and a stale/one-sided bar biases it downward.
    """
    _, high, low, _ = _ohlc_arrays(df)
    n = min(high.size, low.size)
    if n < 2:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rng = np.log(high[-int(window):] / low[-int(window):])
    rng = rng[np.isfinite(rng)]
    if rng.size < 2:
        return 0.0
    var = float(np.mean(rng ** 2)) / (4.0 * math.log(2.0))
    sd = math.sqrt(max(var, 0.0))
    return annualize(sd, periods_per_year) if unit == "annual" else sd


def garman_klass_vol(df, *, window: int = DEFAULT_WINDOW, unit: str = "per_bar",
                     periods_per_year: float = 8760.0) -> float:
    """Garman-Klass (1980) OHLC volatility.

    ``sigma^2 = mean(0.5*ln(H/L)^2 - (2ln2 - 1)*ln(C/O)^2)``.  Uses all four
    prices and is ≈7× more efficient than close-to-close on a diffusion, at the
    cost of being more sensitive to bad prints and to opening gaps.  The mean is
    clamped at 0 because with few bars the second term can dominate.
    """
    op, high, low, close = _ohlc_arrays(df)
    n = min(op.size, high.size, low.size, close.size)
    if n < 2:
        return 0.0
    sl = slice(-int(window), None)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(high[sl] / low[sl])
        co = np.log(close[sl] / op[sl])
    ok = np.isfinite(hl) & np.isfinite(co)
    if ok.sum() < 2:
        return 0.0
    var = float(np.mean(0.5 * hl[ok] ** 2 - (2.0 * math.log(2.0) - 1.0) * co[ok] ** 2))
    sd = math.sqrt(max(var, 0.0))
    return annualize(sd, periods_per_year) if unit == "annual" else sd


# ── EWMA (RiskMetrics) ───────────────────────────────────────────────────

def ewma_variance(returns, *, lam: float = DEFAULT_LAMBDA,
                  window: int = DEFAULT_WINDOW,
                  outlier_sigma: float = DEFAULT_OUTLIER_SIGMA) -> float:
    """RiskMetrics conditional variance for the **next** bar.

    ``sigma^2_{t+1} = (1 - lambda) * sum_{i>=0} lambda^i * r_{t-i}^2``.

    Recursive form (the one computed here): ``v <- lambda*v + (1-lambda)*r^2``
    seeded with the sample variance of the first two observations, which makes
    the recursion forget its seed geometrically — with ``lambda = 0.94`` and a
    500-bar window the seed carries weight ``0.94^500 ≈ 3e-14``, i.e. none.
    Only past returns enter, so the forecast is causal (no look-ahead).

    Note the **effective** memory: ``1/(1-lambda) ≈ 16.7`` bars at the default
    decay, so ``window`` is only the code-side cap on the recursion, not the
    horizon that matters.  Returns are winsorised first
    (:func:`clip_outliers`) because a squared outlier otherwise *is* the
    forecast for the next ~17 bars.
    """
    r = _last(returns, window)
    if r.size < 2:
        return 0.0
    r = clip_outliers(r, sigma=outlier_sigma)
    lam = float(min(max(lam, 0.0), 0.9999))
    # Seed on the first TWO observations, never on the whole sample: seeding
    # with the full-sample variance would leak future information into the
    # oldest recursion step.
    v = float(np.var(r[:2], ddof=0))
    for x in r:
        v = lam * v + (1.0 - lam) * float(x) * float(x)
    return max(v, 0.0)


def ewma_vol(returns, *, lam: float = DEFAULT_LAMBDA, window: int = DEFAULT_WINDOW,
             unit: str = "per_bar", periods_per_year: float = 8760.0,
             outlier_sigma: float = DEFAULT_OUTLIER_SIGMA) -> float:
    """Square root of :func:`ewma_variance`, per bar or annualised."""
    sd = math.sqrt(ewma_variance(returns, lam=lam, window=window,
                                 outlier_sigma=outlier_sigma))
    return annualize(sd, periods_per_year) if unit == "annual" else sd


def ewma_vol_series(returns, *, lam: float = DEFAULT_LAMBDA, window: int = 0,
                    periods_per_year: float = 8760.0,
                    index=None,
                    outlier_sigma: float = DEFAULT_OUTLIER_SIGMA) -> pd.Series:
    """Rolling one-step-ahead EWMA volatility (fraction per bar).

    ``out[i]`` uses only ``r[:i+1]`` — the value a live system would have had
    at bar ``i``.  The first entry is the full-sample-free seed (std of the
    first two returns).  ``window=0`` means "use everything" (the series is O(n)
    and cheap); a positive window restricts the recursion to the tail.

    Used by the research/reporting path (high- vs low-vol windows) and by the
    tests; the live path calls the scalar :func:`ewma_vol` through
    :class:`VolForecaster`.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if window and window > 0:
        r = r[-int(window):]
    n = r.size
    if n == 0:
        return pd.Series(dtype=float, index=index)
    r = clip_outliers(r, sigma=outlier_sigma)
    n = r.size
    lam = float(min(max(lam, 0.0), 0.9999))
    out = np.empty(n, dtype=np.float64)
    v = float(np.var(r[:2], ddof=0)) if n >= 2 else float(r[0] ** 2)
    for i, x in enumerate(r):
        v = lam * v + (1.0 - lam) * float(x) * float(x)
        out[i] = math.sqrt(max(v, 0.0))
    idx = index if index is not None else pd.RangeIndex(n)
    return pd.Series(out, index=idx, name="ewma_vol_per_bar")


# ── GARCH(1,1) ───────────────────────────────────────────────────────────

def garch_backend() -> str:
    """``"arch"`` when the reference package is importable, else ``"scipy"``.

    Probed at call time (never imported at module import) so a machine without
    ``arch`` still imports this module — a hard import would take the whole
    predictor down for an optional estimator.
    """
    try:  # pragma: no cover - depends on the deployment
        import arch  # noqa: F401
    except Exception:
        return "scipy"
    return "arch"


def garch11_loglik_grad(x2: np.ndarray, var_s: float, theta) -> tuple[float, np.ndarray]:
    """``(0.5 * sum(log v + x^2/v), d/d theta)`` for untargeted GARCH(1,1).

    ``x2`` are squared returns in the fit's own (percent) units, ``var_s`` the
    seed for ``v_0``, ``theta = (omega, alpha, beta)``.  This is the standard
    free-``omega`` likelihood of :func:`_garch11_scipy_mle`'s item 1 — the model
    whose *unboundedness* is the reason the shipped estimator is the
    unit-persistence one.  It is kept (and finite-difference tested) so that
    claim stays checkable rather than being folklore.

    The three partials obey their own recursions, updated **after** ``v_t`` so
    that ``d v_t/d beta = v_t + beta * d v_{t-1}/d beta`` is exact::

        v_t      = omega + alpha * x2_{t-1} + beta * v_{t-1}
        dv_t/dw  = 1     + beta * dv_{t-1}/dw
        dv_t/da  = x2_{t-1} + beta * dv_{t-1}/da
        dv_t/db  = v_t   + beta * dv_{t-1}/db
        d(0.5 ll)/dp = 0.5 * (1 - x2_t/v_t) * dv_t/dp

    Kept as a module-level function, not a closure, precisely because the first
    version *was* a closure whose accumulators leaked between calls: its gradient
    disagreed with a finite difference in sign and magnitude, and every optimiser
    then walked to the degenerate ``omega = 0, beta = 0, alpha -> 1`` corner.
    ``tests/test_volatility_targeting.py`` asserts this against
    ``scipy.optimize.approx_fprime``.
    """
    w, a, b = float(theta[0]), float(theta[1]), float(theta[2])
    v = var_s                 # seed the filter at the sample level
    dv_dw = dv_da = dv_db = 0.0
    g_w = g_a = g_b = 0.0
    ll = 0.0
    for i in range(x2.size):
        # v_i takes x2_{i-1}; at i = 0 there is no previous shock.
        lag = x2[i - 1] if i > 0 else 0.0
        v = w + a * lag + b * v
        if not np.isfinite(v):
            return 1e12, np.zeros(3)
        if v <= 1e-6:
            # Smooth barrier instead of a step: a discontinuous 1e12 makes the
            # objective non-differentiable right where a finite difference would
            # probe it (`test_garch_loglik_gradient_matches_finite_difference`).
            short = 1e-6 - v
            return 1e12 + short * short, np.zeros(3)
        dv_dw = 1.0 + b * dv_dw
        dv_da = lag + b * dv_da
        dv_db = v + b * dv_db
        ll += math.log(v) + x2[i] / v
        resid = 0.5 * (1.0 - x2[i] / v)
        g_w += resid * dv_dw
        g_a += resid * dv_da
        g_b += resid * dv_db
    return 0.5 * ll, np.array([g_w, g_a, g_b], dtype=float)


def garch11_filter(x2: np.ndarray, var_s: float, a: float, b: float,
                   v0: float | None = None) -> np.ndarray:
    """Conditional variances for ``v_t = alpha x_{t-1}^2 + beta v_{t-1}``.

    ``v[t]`` is the variance of return ``t`` and uses the **previous** squared
    return, never the current one: a filter run on the contemporaneous ``x_t^2``
    cannot line up with the model it is scoring.  ``var_s`` seeds ``v[0]``.

    Variances are floored at ``var_s * 1e-4`` (a scale-relative floor) so the
    Gaussian likelihood stays finite.  An **absolute** floor was tried and is
    wrong: with ``v`` pinned near 0 the ``x^2/v`` term explodes and the average
    likelihood of a degenerate parameter pair measures ``3.6e5``, i.e. the floor
    itself becomes the optimum and every optimiser walks there.
    """
    floor = float(var_s) * 1e-4 if var_s > 0 else 1e-12
    prev = float(var_s) if v0 is None else float(v0)
    v = np.empty(x2.size, dtype=np.float64)
    v[0] = prev if prev > floor else floor
    for t in range(1, x2.size):
        prev = a * x2[t - 1] + b * prev
        v[t] = prev if prev > floor else floor
    return v


def garch11_filter_grid(x2: np.ndarray, var_s: float, betas) -> np.ndarray:
    """The same recursion for a **grid** of ``beta`` values, vectorised.

    ``alpha = 1 - beta`` (unit persistence), so ``betas`` alone describes the
    grid.  Returns a ``(len(betas), len(x2))`` array of conditional variances.

    The scalar version in a Python loop measured ~434 ms per fit on the full
    8 846-bar cache (100 grid points x 8 846 steps), which is not a path the
    doc's budget table can report; the vectorised form advances the whole grid
    one bar at a time and is ~1 ms per grid point.
    """
    x2 = np.asarray(x2, dtype=np.float64)
    n = x2.size
    betas_arr = np.asarray(betas, dtype=np.float64).ravel()
    alpha = (1.0 - betas_arr).reshape(-1, 1)
    beta = betas_arr.reshape(-1, 1)
    floor = float(var_s) * 1e-4 if var_s > 0 else 1e-12
    lag = np.empty(n, dtype=np.float64)
    lag[0] = 0.0
    lag[1:] = x2[:-1]
    out = np.empty((betas_arr.size, n), dtype=np.float64)
    v = np.full((betas_arr.size, 1), max(float(var_s), floor), dtype=np.float64)
    out[:, 0:1] = v
    for t in range(1, n):
        v = alpha * lag[t] + beta * v
        np.maximum(v, floor, out=v)
        out[:, t:t + 1] = v
    return out


#: ``beta`` grid for the IGARCH(1,1) fit: ``0.00 … 0.99`` in steps of 0.01.
#: Deliberately starts at the constant-variance corner: the fit is allowed to
#: say "there is no persistence here" instead of being forced to look like EWMA.
_GARCH_BETA_GRID = tuple(round(0.01 * i, 2) for i in range(100))


def _garch11_avg_ll(x2: np.ndarray, var_s: float, a: float, b: float) -> float:
    """Mean ``0.5 * (log v + x²/v)`` for EWMA-family parameters (percent² units)."""
    v = garch11_filter(x2, var_s, a, b)
    return float(np.mean(0.5 * (np.log(v) + x2 / v)))


def _garch11_best_beta(x2: np.ndarray, var_s: float,
                       betas: tuple = _GARCH_BETA_GRID) -> tuple[float, float]:
    """``(beta, mean log-likelihood)`` maximising the Gaussian likelihood on a grid.

    ``alpha = 1 - beta`` is substituted (unit persistence), so the grid is exactly
    :data:`_GARCH_BETA_GRID`.
    """
    v = garch11_filter_grid(x2, var_s, betas)              # (n_beta, n_bars)
    scores = np.mean(0.5 * (np.log(v) + x2[None, :] / v), axis=1)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    idx = int(np.argmax(scores))
    return float(betas[idx]), float(scores[idx])


def _garch11_scipy_mle(r: np.ndarray, *, outlier_sigma: float = 8.0
                       ) -> tuple[float, float, float, bool]:
    """IGARCH(1,1) / RiskMetrics fit -> ``(omega, alpha, beta, ok)``.

    The fitted model is the **unit-persistence** GARCH, ``omega = 0`` and
    ``alpha + beta = 1``, i.e. ``v_t = alpha x_{t-1}^2 + beta v_{t-1}`` with
    ``alpha = 1 - beta`` substituted.  Only ``beta`` is searched (on a fixed
    0.01 grid, by the Gaussian likelihood), which is why this is a fit rather
    than a fixed ``lambda = 0.94``.

    Three cheaper-looking alternatives were implemented, measured, and rejected
    on this repo's data — the reasons are worth keeping because each one *looked*
    right:

    1. **Untargeted GARCH(1,1) MLE** (free ``omega``, analytic gradient verified
       against ``scipy.optimize.approx_fprime`` to 1e-4).  The Gaussian
       likelihood is **unbounded**: at ``omega = 0, beta = 0``,
       ``v_t = alpha x_{t-1}^2``, so a near-zero return drives ``v_t -> 0`` and
       ``log v_t -> -inf``.  Nelder-Mead, L-BFGS-B and SLSQP — with analytic and
       numeric gradients — all converged to that corner and rejected the
       generating parameters of a synthetic GARCH(1,1).
    2. **Variance targeting without unit persistence** (``omega = V(1-a-b)``,
       grid over ``a`` and ``b``).  It does not remove the pathology, because
       ``a + b`` may approach 1 while the *filter* still collapses whenever
       ``x_{t-1}`` is tiny: on synthetic data the likelihood preferred
       ``(alpha, beta) = (0.001, 0)`` with an average of ``1.8e4``.
    3. **``arch``**: not installed, and deliberately not added to
       ``requirements.txt`` — a heavy compiled dependency for one optional
       estimator on a per-bar path.

    Pinning ``alpha + beta = 1`` is what makes the problem well posed: with
    ``omega = 0`` the variance floor is proportional to ``alpha`` times a squared
    return, so it is always a legitimate positive number rather than an
    arbitrary epsilon the optimiser can exploit.  The cost is the honest one —
    this estimator can express **no** mean reversion in volatility, so its
    long-horizon forecasts equal the current level.  That is acceptable here
    because the forecast is consumed one bar ahead (sizing, stop width), and the
    EWMA default has the same property.

    ``ok=False`` (fewer than 50 usable returns, or a flat/zero series) makes the
    caller degrade to EWMA; a number is never silently pathological.
    ``outlier_sigma`` is looser than the module default (8 vs 6) because the
    likelihood is *supposed* to see the large moves whose clustering it models.
    """
    x = clip_outliers(np.asarray(r, dtype=np.float64), sigma=outlier_sigma) * 100.0
    n = x.size
    if n < 50:
        return 0.0, 0.0, 0.0, False
    x2 = x * x
    var_s = float(np.var(x, ddof=1))
    if not np.isfinite(var_s) or var_s <= 0.0:
        return 0.0, 0.0, 0.0, False

    best_b, best_ll = _garch11_best_beta(x2, var_s)
    if best_b is None or not np.isfinite(best_ll):
        return 0.0, 0.0, 0.0, False
    beta = float(best_b)
    return 0.0, 1.0 - beta, beta, True


def garch11_params(returns, *, window: int = DEFAULT_WINDOW) -> dict:
    """Fit GARCH(1,1) and return ``{omega, alpha, beta, backend, ok, n}``.

    ``alpha + beta`` is persistence (how slowly a shock decays); the
    unconditional variance is ``omega / (1 - alpha - beta)``.  Returns are
    winsorised before the fit (:func:`clip_outliers`) for the data-splice reason
    documented on :data:`DEFAULT_OUTLIER_SIGMA`.
    """
    r = clip_outliers(_last(returns, window), sigma=DEFAULT_OUTLIER_SIGMA)
    backend = garch_backend()
    if r.size < 50:
        return {"omega": 0.0, "alpha": 0.0, "beta": 0.0,
                "backend": backend, "ok": False, "n": int(r.size)}
    if backend == "arch":  # pragma: no cover - not installed in this deployment
        try:
            from arch import arch_model
            res = arch_model(r * 100.0, vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
            p = res.params
            omega = float(p["omega"]) / 1e4  # back to fraction^2
            return {"omega": omega, "alpha": float(p["alpha"]),
                    "beta": float(p["beta"]), "backend": "arch", "ok": True,
                    "n": int(r.size)}
        except Exception:
            backend = "scipy"  # fall through to our own MLE
    omega, alpha, beta, ok = _garch11_scipy_mle(r)
    return {"omega": omega, "alpha": alpha, "beta": beta,
            "backend": "scipy", "ok": ok, "n": int(r.size)}


#: Weight of the **fitted** one-step variance in :func:`garch11_forecast`; the
#: rest is the EWMA level.  The ``arch`` package does the same by default
#: (``res.forecast(horizon=1)`` returns ``0.5 * sigma2_{t+1} + 0.5 * sigma2_t``),
#: and it matters here: the IGARCH grid fit on the shipped BTC 1h window lands on
#: the constant-variance corner (``alpha = 1, beta = 0``), so a *pure* one-step
#: forecast would be the last squared return divided by 10000 —
#: ``0.00076 %/bar``, 5.5× below the sample's ``0.42 %/bar``.  Blending bounds
#: the forecast between the fitted conditional variance and the EWMA level, so a
#: degenerate ``beta`` cannot produce a degenerate *forecast*.
GARCH_FIT_WEIGHT = 0.5


def garch11_forecast(returns, *, window: int = DEFAULT_WINDOW, unit: str = "per_bar",
                     periods_per_year: float = 8760.0,
                     lam: float = DEFAULT_LAMBDA,
                     fit_weight: float = GARCH_FIT_WEIGHT) -> float:
    """One-step-ahead volatility forecast from the fitted GARCH/IGARCH model.

    ``v_hat = w * v_{t+1} + (1 - w) * v_t`` with ``w = fit_weight``, where
    ``v_{t+1} = omega + alpha r_t^2 + beta v_t`` is the model's one-step variance
    and ``v_t`` the filtered current variance (equivalently the EWMA level for the
    unit-persistence fit).  See :data:`GARCH_FIT_WEIGHT` for why the blend is not
    optional in practice, and :func:`_garch11_scipy_mle` for the fit itself.

    Returns ``0.0`` when the fit is unusable (fewer than 50 bars) — the caller
    then falls back to EWMA; :func:`forecast_vol` does that for you.
    """
    r = _last(returns, window)
    # The *same* window feeds the fit and the filter: fitting on the full history
    # and filtering a 500-bar tail (or vice versa) mixes two sample levels, and it
    # is also measurably slower — a full-history grid over the 8 846-bar cache
    # cost 33 ms/call against 2.1 ms for the default 500-bar window.
    p = garch11_params(r, window=0)
    if not p["ok"]:
        return 0.0
    v = _garch11_variance(r, p, lam=lam, fit_weight=fit_weight)
    sd = math.sqrt(max(v, 0.0))
    return annualize(sd, periods_per_year) if unit == "annual" else sd


def _garch11_variance(r: np.ndarray, p: dict, *, lam: float = DEFAULT_LAMBDA,
                      fit_weight: float = GARCH_FIT_WEIGHT) -> float:
    """Blended one-step conditional variance (fraction²) for a fitted model."""
    if r.size == 0:
        return 0.0
    v = float(np.var(r, ddof=1)) if r.size >= 2 else float(r[0] ** 2)
    for x in r:
        v = p["omega"] + p["alpha"] * float(x) * float(x) + p["beta"] * v
    v_t = max(v, 0.0)
    last = float(r[-1])
    v_next = max(p["omega"] + p["alpha"] * last * last + p["beta"] * v_t, 0.0)
    w = min(max(float(fit_weight), 0.0), 1.0)
    # ``ewma_variance`` is the same quantity for the unit-persistence family and
    # is well conditioned even when the grid fit degenerates.
    v_ewma = ewma_variance(r, lam=lam, window=0)
    base = v_t if v_ewma <= 0.0 else v_ewma
    return w * v_next + (1.0 - w) * base


# ── units ────────────────────────────────────────────────────────────────

def periods_per_year_for(interval: str, default: float = 8760.0) -> float:
    """Bars per year for an interval label (``'1h'`` → 8760, ``'1d'`` → 365)."""
    if interval is None:
        return float(default)
    return float(PERIODS_PER_YEAR.get(str(interval).strip().lower(), default))


def annualize(vol_per_bar: float, periods_per_year: float = 8760.0) -> float:
    """Per-bar vol → annualised vol (``sqrt``-time scaling, i.i.d. assumption)."""
    v = float(vol_per_bar)
    if not math.isfinite(v) or v <= 0.0:
        return 0.0
    return v * math.sqrt(max(float(periods_per_year), 0.0))


def deannualize(vol_annual: float, periods_per_year: float = 8760.0) -> float:
    """Annualised vol → per-bar vol (inverse of :func:`annualize`)."""
    v = float(vol_annual)
    ppy = max(float(periods_per_year), 0.0)
    if not math.isfinite(v) or v <= 0.0 or ppy <= 0.0:
        return 0.0
    return v / math.sqrt(ppy)


def to_pct(vol_fraction: float) -> float:
    """Fraction → percent (the unit of the config knobs)."""
    return float(vol_fraction) * 100.0


def from_pct(vol_pct: float) -> float:
    """Percent → fraction."""
    return float(vol_pct) / 100.0


# ── regime helpers ───────────────────────────────────────────────────────

def vol_percentile(returns, current: float | None = None, *,
                   window: int = DEFAULT_WINDOW, lam: float = DEFAULT_LAMBDA,
                   lookback: int = 0) -> float:
    """Rank of ``current`` vol inside its own recent history, in ``[0, 1]``.

    ``current=None`` → the latest EWMA value (the live case).  The history is
    the **causal** EWMA series over the last ``lookback`` bars (0 = all).
    ``0.9`` means "today's conditional vol is above 90 % of the recent past" —
    a regime flag cheap enough for a risk threshold, and *scale free*, which a
    raw vol number is not (BTC's 2025 level is not BTC's 2021 level).
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size < 4:
        return 0.5
    series = ewma_vol_series(r, lam=lam, window=0, index=None)
    if lookback and lookback > 0:
        series = series.iloc[-int(lookback):]
    cur = float(series.iloc[-1]) if current is None else float(current)
    if not math.isfinite(cur):
        return 0.5
    arr = series.values
    return float(np.mean(arr <= cur))


# ── single interface ─────────────────────────────────────────────────────

def _as_returns(data, *, window: int = DEFAULT_WINDOW) -> np.ndarray:
    """Coerce the accepted inputs to a finite log-return array, window applied.

    * ``Series``/``ndarray`` → treated **as returns** (documented, and what the
      sizer/guard have in hand),
    * ``DataFrame`` with a ``close`` column → ``log_returns(df['close'])``
      (an OHLC frame never has to be pre-processed by the caller).

    ``window`` is applied to **both** branches on purpose.  It used to apply only
    to the return-array branch, so ``forecast_vol(df)`` and
    ``forecast_vol(log_returns(df))`` silently disagreed — measured, the GARCH
    path cost **33.6 ms** and returned ``0.003860`` for the frame against
    **2.2 ms** and ``0.003745`` for the same returns as an array, because the
    frame branch fed the fit the full 8 845-bar history instead of the window.
    """
    if isinstance(data, pd.DataFrame):
        if "close" not in data.columns:
            raise KeyError("a DataFrame input needs a 'close' column "
                           "(pass a return series instead)")
        return _last(log_returns(data["close"].values), window)
    return _last(data, window)


def forecast_vol(data, *, method: str = "ewma", window: int = DEFAULT_WINDOW,
                 lam: float = DEFAULT_LAMBDA, unit: str = "per_bar",
                 periods_per_year: float = 8760.0, interval: str | None = None,
                 allow_garch: bool = False) -> float:
    """Conditional volatility for the **next** bar — the single interface.

    Parameters
    ----------
    data : Series/ndarray of returns, or an OHLCV DataFrame (uses ``close``).
    method : str
        One of :data:`METHODS`.  Default ``"ewma"`` — RiskMetrics ``lambda=0.94``,
        chosen because it has no fitted parameter, cannot fail to converge, is
        causal by construction, and needs only a return series (P2's own
        conclusion: the robust estimator, not the best-in-class fit, is what a
        live per-bar path should run).
    window : int
        Bars of history used (0 = all).  Default 500.
    lam : float
        EWMA decay; 0.94 = RiskMetrics daily standard, applied per bar here.
    unit : str
        ``"per_bar"`` (fraction of price, default) or ``"annual"``.
    periods_per_year : float
        Annualisation factor; ignored for ``unit="per_bar"``.  ``interval``
        (e.g. ``"1h"``) overrides it via :func:`periods_per_year_for`.
    allow_garch : bool
        ``method="garch11"`` with ``allow_garch=False`` (the live default)
        silently degrades to EWMA when the fit is unusable, so a caller can
        never receive a 0.0 that looks like "zero volatility".  Set ``True``
        for research, where an explicit 0.0 must stay visible.

    Returns
    -------
    float
        Volatility in the requested unit; ``0.0`` only when there is not enough
        data (fewer than 2 returns), never NaN.
    """
    ppy = periods_per_year_for(interval) if interval else float(periods_per_year)
    m = str(method or "ewma").strip().lower()
    if m not in METHODS:
        raise ValueError(f"unknown volatility method {method!r}; expected one of {METHODS}")

    r = _as_returns(data, window=window)
    if m in OHLC_METHODS:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"method {m!r} needs an OHLC DataFrame, not a return series")
        fn = parkinson_vol if m == "realized_parkinson" else garman_klass_vol
        return fn(data, window=window, unit=unit, periods_per_year=ppy)
    if m == "realized_cc":
        return realized_vol(r, window=0, unit=unit, periods_per_year=ppy)
    if m == "garch11":
        val = garch11_forecast(r, window=0, unit=unit, periods_per_year=ppy)
        if val > 0.0 or allow_garch:
            return val
        return ewma_vol(r, lam=lam, window=0, unit=unit, periods_per_year=ppy)
    return ewma_vol(r, lam=lam, window=0, unit=unit, periods_per_year=ppy)


def forecast_realized_ratio(data, *, method: str = "ewma",
                            window: int = DEFAULT_WINDOW,
                            lam: float = DEFAULT_LAMBDA,
                            baseline_window: int = 500) -> float:
    """Forecast vol ÷ trailing realised vol — scale-free "is vol high now?".

    ``> 1`` means the conditional model expects *more* turbulence than the
    plain trailing realised average; the sizer uses this ratio so a target set
    in volatility points does not silently become "always levered" or "always
    flat" when the symbol's volatility level drifts over months.
    """
    r = _as_returns(data, window=window)
    base = realized_vol(r, window=baseline_window or 0)
    if base <= 0.0:
        return 1.0
    return float(forecast_vol(r, method=method, window=0, lam=lam) / base)


@dataclass
class VolForecast:
    """One forecast plus the provenance a log line / report needs."""
    vol_per_bar: float
    method: str
    window: int
    n_returns: int
    backend: str = ""

    @property
    def vol_annual(self) -> float:
        return annualize(self.vol_per_bar)

    def as_dict(self, *, periods_per_year: float = 8760.0) -> dict:
        return {
            "vol_per_bar": self.vol_per_bar,
            "vol_pct_per_bar": to_pct(self.vol_per_bar),
            "vol_annual": annualize(self.vol_per_bar, periods_per_year),
            "method": self.method,
            "window": self.window,
            "n_returns": self.n_returns,
            "garch_backend": self.backend,
        }


class VolForecaster:
    """Memoising wrapper for the live per-bar path.

    ``_on_kline`` fires many times per bar (every tick rebuilds the frame), so
    the same bar is recomputed over and over.  This caches the last forecast
    **per key** (symbol, interval) and invalidates it when the newest bar's
    timestamp changes — the same trick ``MLPredictor`` uses for its feature
    matrix.  Cheap EWMA is recomputed only once per bar, and a GARCH fit is
    never repeated for a bar that has not closed.
    """

    def __init__(self, *, method: str = "ewma", window: int = DEFAULT_WINDOW,
                 lam: float = DEFAULT_LAMBDA, interval: str | None = None,
                 allow_garch: bool = False):
        self.method = method
        self.window = int(window)
        self.lam = float(lam)
        self.interval = interval
        self.allow_garch = bool(allow_garch)
        self._cache: dict[tuple, tuple] = {}
        self.compute_count = 0

    def forecast(self, key, data, *, interval: str | None = None) -> VolForecast:
        """Forecast for ``key``, recomputed only when the newest bar changes."""
        bar = None
        if isinstance(data, pd.DataFrame) and len(data.index):
            bar = data.index[-1]
        sig = (self.method, interval or self.interval, self.window, self.lam)
        cached = self._cache.get(key)
        if cached is not None and cached[0] == bar and cached[1] == sig:
            return cached[2]
        vol = forecast_vol(data, method=self.method, window=self.window,
                           lam=self.lam, interval=interval or self.interval,
                           allow_garch=self.allow_garch)
        if isinstance(data, pd.DataFrame):
            n = max(len(data) - 1, 0)
        else:
            n = len(_last(data, 0))
        out = VolForecast(vol_per_bar=vol, method=self.method, window=self.window,
                          n_returns=n,
                          backend=garch_backend() if self.method == "garch11" else "")
        self._cache[key] = (bar, sig, out)
        self.compute_count += 1
        return out

    def clear(self) -> None:
        self._cache.clear()
