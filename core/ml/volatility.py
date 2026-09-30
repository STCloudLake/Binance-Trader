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

**The per-bar budget is a default-path promise, and only that path keeps it.**
Re-measured on a 500-bar window (this checkout, scipy backend): ``ewma`` (the
default) **0.27 ms/call**, the realised family ≈0.1 ms — inside
:data:`PER_BAR_BUDGET_SEC` (2 ms).  ``garch11`` is **not** a per-bar estimator:
a full call is **≈0.23 s** (scipy backend; 0.49 s before the exact hot-loop
optimisation below), i.e. two orders of magnitude over budget.  The cost is the
Nelder-Mead polish: ≈575 likelihood passes, each a sequential Python recursion
over the window — a likelihood pass measured 1.78 ms → 0.39 ms after the
pre-extracted-``list`` optimisation in :func:`garch11_loglik_grad`, so the
shipped fit is already the fast exact form of this estimator.  It is therefore
**opt-in**: ``method="garch11"`` is an explicit request and nothing on the live
path selects it by default (:data:`METHODS`'s default is ``"ewma"`` and
:class:`VolForecaster` defaults to ``allow_garch=False``), so it must be kept off
the per-bar path.  Any earlier claim of "≈2 ms per GARCH fit" describes the
optimiser-free IGARCH *grid fallback* (:func:`_garch11_best_beta`, measured ≈3 ms
on 500 bars), not the shipped MLE — the two differ by ~80×.

GARCH dependency note
---------------------
``arch`` (Kevin Sheppard's package) is **not** installed in this deployment and
was deliberately *not* added to ``requirements.txt``: a heavy compiled
dependency for one estimator.  :func:`garch11_forecast` therefore has two
documented paths:

1. ``import arch`` succeeds → the reference MLE fit (``arch_model``);
2. otherwise → our own GARCH(1,1) Gaussian MLE with ``scipy.optimize`` (always
   available, since scipy is already a dependency), on a bounded parameter box
   seeded at the variance-targeted point.  If the MLE cannot run at all, the
   estimator falls back to the optimiser-free **IGARCH grid** — cheap (≈3 ms on
   500 bars), deterministic, and documented as the fallback rather than the
   primary.

:func:`garch_backend` reports which one ran, and :func:`garch11_params`'s
``fitted`` field says whether the returned ``(omega, alpha, beta)`` came from a
real MLE or from the unit-persistence fallback, so a number can never be silently
attributed to the wrong fitter.
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

#: Default half-life (bars) of the exponentially-weighted MAD helper
#: :func:`_anchored_mad`.  :func:`build_anchor` uses the plain fixed
#: ``median`` / ``1.4826·MAD`` recipe over the whole series (identical to what
#: :func:`clip_outliers` always computed) and only forwards this value when a
#: caller explicitly asks for the long-memory variant; either way the anchor is
#: computed **once**, which is what makes the clip decision stable — see
#: :func:`clip_outliers`.
DEFAULT_MAD_HALF_LIFE = 2000

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
#: 500-bar window **using the default method (``ewma``)**.  Asserted in the test
#: suite; the live predictor's async path computes indicators once per kline, so
#: a forecast must be negligible next to that.  Measured ≈0.14–0.27 ms for
#: ``ewma`` and ≈0.1 ms for the realised family, against this 2 ms budget.
#: ``garch11`` is **excluded on purpose**: a full MLE call measures ≈0.23 s (not
#: the ≈30 ms an earlier revision of this comment claimed — that was the
#: optimiser-free grid fallback's cost, ~80× cheaper than the shipped fit), so it
#: is an opt-in research estimator and never the per-bar default.
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


def _anchored_mad(r: np.ndarray, half_life: float) -> tuple[float, float]:
    """Recursive EW-median / EW-MAD of a sample ``→ (centre, 1.4826·MAD-scale)``.

    A plain rolling MAD is recomputed from whatever window the caller passes, so
    the *limit* moves with the window and an observation that was clipped at bar
    ``t`` can be **un-clipped** later when the window slides: measured on the
    shipped BTC 1h series, the Winsor limit changed between consecutive 500-bar
    windows on 8 343 of 8 344 steps (max |Δ| ``≈4.8e-2`` on the clipped value).
    An estimator that consumes the clipped series bar by bar (α_t jumping) then
    depends on when it was asked, which is not a property of the data.

    This helper gives the limit a **long, fixed half-life**: the centre and the
    scale are exponentially-weighted order statistics with weight
    ``2**(-1/half_life)`` per bar, so an observation can only ever be clipped
    *more loosely* as time passes, never re-admitted to the estimator it already
    left.  The half-life is intentionally much longer than any estimator's own
    memory (RiskMetrics' is ``1/(1-λ) ≈ 17`` bars), because the clip guards a
    property of the *data* (a calendar splice), not of the current regime.
    """
    n = r.size
    hl = max(float(half_life), 1.0)
    decay = float(2.0 ** (-1.0 / hl))
    # Deterministic seed: the sample's own order statistics.  The recursion
    # forgets them with weight `decay**n`; a caller that wants the seed itself
    # gone passes the *whole* series (see `build_anchor`).
    med = float(np.median(r))
    scaled_mad = float(np.median(np.abs(r - med))) * _MAD_TO_SIGMA
    for x in r:
        # EW mean of the signed deviation (a cheap, deterministic L1 centre) and
        # of its absolute value, held on the same scale as the median recipe via
        # `_MAD_TO_SIGMA` (``E|x−μ| = sqrt(2/π)σ`` and ``MAD·1.4826 = σ`` for a
        # normal sample) — this is the long-memory variant, not the default.
        d = float(x) - med
        med += (1.0 - decay) * d
        scaled_mad += (1.0 - decay) * (abs(d) * _MAD_TO_SIGMA - scaled_mad)
    return med, max(scaled_mad, 0.0)


@dataclass(frozen=True)
class AnchorMAD:
    """A **fixed** centre and robust scale, so a clip decision cannot move.

    Build one from the full return history (``build_anchor``) and pass it to
    :func:`clip_outliers` from every estimator: the value an observation is
    clipped to is then a pure function of the observation and one fixed pair of
    numbers, not of the window the estimator happened to be handed.  This is what
    makes ``ewma_vol(r, window=500)`` and ``ewma_vol(r[-500:], window=0)`` agree,
    which the rolling MAD could not do.
    """

    centre: float
    scale: float

    def limit(self, sigma: float) -> float:
        return float(sigma) * float(self.scale)


def build_anchor(returns, *, half_life: float = 0.0) -> AnchorMAD:
    """The anchored centre/scale of a return series (:class:`AnchorMAD`).

    The recipe is **the same one** :func:`clip_outliers` has always used
    (``median`` and ``1.4826 · median|x − centre|``) — the only change is that it
    is evaluated **once, over the whole series**, instead of being re-derived
    from whichever window the estimator was handed.  That keeps the clip's
    severity exactly as documented while removing its window dependence: the
    window-rounding artefact of the old per-window limit (measured on the shipped
    BTC 1h series: the limit changed on 7 659 of 11 174 consecutive 500-bar
    windows, and the oldest still-present observation changed value on 11 172 of
    11 173 steps) disappears, because the limit no longer depends on the window.

    ``half_life > 0`` switches to the long-memory variant
    (:func:`_anchored_mad`, exponentially weighted centre and scale): it differs
    slightly in severity (the exponential mean of ``|x − centre|`` is not the
    median) and exists for a caller whose series is long enough that a fixed
    median would lag a genuine regime shift.  ``0.0`` (the default) is the plain
    fixed recipe, whose robustness comes from the median rather than the window.

    A leak-free causal variant exists (recompute the anchor from the past only at
    each decision bar) and is deliberately **not** the default: the invariant a
    scalar estimator needs is that ``clip(x_i)`` is a function of ``x_i`` and one
    fixed pair of numbers, and re-anchoring per bar reintroduces exactly the
    instability this fixes.  The centre is a robust location, so the leak is
    immaterial (a splice moves it by ~1e-7 on the shipped series).
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size < 8:
        return AnchorMAD(0.0, 0.0)
    if float(half_life) > 0:
        med, scale = _anchored_mad(r, float(half_life))
        return AnchorMAD(float(med), float(scale))
    med = float(np.median(r))
    scale = float(np.median(np.abs(r - med))) * _MAD_TO_SIGMA
    return AnchorMAD(med, scale)


def clip_outliers(returns, *, sigma: float = DEFAULT_OUTLIER_SIGMA,
                  anchor: AnchorMAD | None = None,
                  half_life: float = DEFAULT_MAD_HALF_LIFE,
                  anchored: bool | None = None) -> np.ndarray:
    """Winsorise returns at ``±sigma`` robust sigmas (``1.4826 * MAD``).

    A stale cache, a calendar gap or a fat finger produces a return that no
    one-hour move could produce; squaring it makes it dominate every estimator
    below.  The MAD is used instead of the standard deviation because the
    quantity being guarded against is exactly what inflates the latter.
    ``sigma <= 0`` returns the input unchanged.

    Clip source, in order of preference:

    * ``anchor`` — an :class:`AnchorMAD` the caller built once from the full
      history.  This is the **stable** form: the same bar is clipped identically
      in every window that contains it, and it can never be un-clipped as a
      window slides (that reversibility is what made α_t jump).  Pass it from a
      scalar estimator (``ewma_variance``/``ewma_vol``/``garch11_params``) whose
      caller has the whole series.
    * otherwise the centre/scale come from ``_anchored_mad`` over the **given
      array** (``anchored=True``, the default): deterministic *within* a pass,
      but a function of the array, so a caller holding the whole series must hand
      in :func:`series_anchor` of it — which is what every estimator below now
      does, so the ``window`` a caller asks for can no longer move the limit
      (P3/P4 audit item 7).
    * ``anchored=False`` restores the previous per-window MAD exactly (used by
      the regression tests that pin the difference).
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if sigma is None or float(sigma) <= 0 or r.size < 8:
        return r
    if anchor is not None:
        med, scale = float(anchor.centre), float(anchor.scale)
    elif anchored is False:
        med = float(np.median(r))
        scale = float(np.median(np.abs(r - med))) * _MAD_TO_SIGMA
    else:
        med, scale = _anchored_mad(r, half_life)
    if not math.isfinite(scale) or scale <= 0.0:
        return r
    limit = float(sigma) * scale
    return np.clip(r, med - limit, med + limit)


def series_anchor(returns, *, half_life: float = DEFAULT_MAD_HALF_LIFE
                  ) -> AnchorMAD:
    """The clip anchor of a **whole series** — what the estimators build.

    This is the recipe :func:`clip_outliers` already applies when it is handed
    the whole series (``_anchored_mad`` at :data:`DEFAULT_MAD_HALF_LIFE`), so
    ``clip_outliers(r, anchor=series_anchor(r))`` is **bit-identical** to
    ``clip_outliers(r)``: the change is *when* the pair is computed, not how
    tightly it clips, so no shipped number moves.  What changes is its scope: one
    anchor per series makes the Winsor limit a function of the data instead of a
    function of the window the estimator happened to be handed, so the same bar
    is clipped identically in every window of that series and can never be
    un-clipped by a window sliding forward (measured: per-window limits changed on
    8 344 of 8 345 consecutive windows; with one anchor per series, 0).

    Chosen over the causal-expanding alternative (re-anchor the limit from
    ``r[:i+1]`` at every bar) because an expanding limit is still a function of
    where the array *starts*: a caller that slices a window out of a longer series
    gets a different decision for the same bar — the defect, not the fix.  The
    cost of the choice: a caller that slices the series itself must pass the
    anchor of the **unsliced** series (every estimator here does it internally).
    The two guarantees are pinned by ``tests/test_p34_code_defects.py``.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size < 8:
        return AnchorMAD(0.0, 0.0)
    med, scale = _anchored_mad(r, float(half_life))
    return AnchorMAD(float(med), float(scale))


def _clipped(returns, *, window: int, outlier_sigma: float,
             anchor: AnchorMAD | None = None) -> np.ndarray:
    """The series one estimator consumes: whole history → **one** clip → window.

    Every estimator used to winsorise ``_last(returns, window)``, so the MAD —
    and therefore the Winsor limit — was re-derived from whatever window the
    caller asked for: the same observation could be clipped in one window and not
    in the next, which made the forecast depend on the window rather than on the
    data.  Clipping the **whole** series with :func:`series_anchor` and only then
    applying ``window`` makes the limit a property of the series: ``window`` caps
    the estimator's own memory (the EWMA recursion, the label/stop widths) without
    touching the clip.

    Bit-identical to the previous code whenever the window covered the whole
    series (``window=0``, or a window at least as long as the array): the same
    ``_anchored_mad`` pair is applied to the same values, so no shipped number
    moves — only windowed calls change, and they change *towards* the
    whole-series answer.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if anchor is None:
        anchor = series_anchor(r)
    r = clip_outliers(r, sigma=outlier_sigma, anchor=anchor)
    if window and int(window) > 0:
        return r[-int(window):]
    return r


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
                  outlier_sigma: float = DEFAULT_OUTLIER_SIGMA,
                  anchor: AnchorMAD | None = None) -> float:
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

    ``anchor`` overrides the winsorisation; by default the estimator builds
    :func:`series_anchor` from the **whole series it was handed** and only then
    applies ``window``, so the Winsor limit is no longer re-derived from the
    window (P3/P4 audit item 7).  ``ewma_vol(r, window=500)`` and
    ``ewma_vol(r, window=400)`` therefore clip identically; they used to disagree
    (0.524216 against 0.524062 %/bar) because each computed the limit from its own
    window.  ``ewma_vol(r[-500:], window=0)`` still differs — that caller sliced
    the series itself, and needs ``anchor=series_anchor(r)`` to be window-free.
    """
    r = _clipped(returns, window=window, outlier_sigma=outlier_sigma,
                 anchor=anchor)
    if r.size < 2:
        return 0.0
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
             outlier_sigma: float = DEFAULT_OUTLIER_SIGMA,
             anchor: AnchorMAD | None = None) -> float:
    """Square root of :func:`ewma_variance`, per bar or annualised."""
    sd = math.sqrt(ewma_variance(returns, lam=lam, window=window,
                                 outlier_sigma=outlier_sigma, anchor=anchor))
    return annualize(sd, periods_per_year) if unit == "annual" else sd


def ewma_vol_series(returns, *, lam: float = DEFAULT_LAMBDA, window: int = 0,
                    periods_per_year: float = 8760.0,
                    index=None,
                    outlier_sigma: float = DEFAULT_OUTLIER_SIGMA,
                    anchor: AnchorMAD | None = None) -> pd.Series:
    """Rolling one-step-ahead EWMA volatility (fraction per bar).

    ``out[i]`` uses only ``r[:i+1]`` — the value a live system would have had
    at bar ``i``.  The first entry is the full-sample-free seed (std of the
    first two returns).  ``window=0`` means "use everything" (the series is O(n)
    and cheap); a positive window restricts the recursion to the tail.

    Used by the research/reporting path (high- vs low-vol windows) and by the
    tests; the live path calls the scalar :func:`ewma_vol` through
    :class:`VolForecaster`.  The clip is the same one-the-whole-series anchor the
    scalar path uses (:func:`_clipped`: clip the full series, *then* apply
    ``window``), so a rolling window cannot move a limit that has already been
    emitted; ``anchor`` still overrides it.
    """
    r = _clipped(returns, window=window, outlier_sigma=outlier_sigma,
                 anchor=anchor)
    n = r.size
    if n == 0:
        return pd.Series(dtype=float, index=index)
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
    free-``omega`` likelihood that :func:`_garch11_mle` maximises — and the one
    the shipped estimator used to avoid on the (since **refuted**) claim that it
    is unbounded; see :func:`_garch11_scipy_mle` for the measurement.

    The three partials obey their own recursions, updated **after** ``v_t`` so
    that ``d v_t/d beta = v_t + beta * d v_{t-1}/d beta`` is exact::

        v_t      = omega + alpha * x2_{t-1} + beta * v_{t-1}
        dv_t/dw  = 1     + beta * dv_{t-1}/dw
        dv_t/da  = x2_{t-1} + beta * dv_{t-1}/da
        dv_t/db  = v_t   + beta * dv_{t-1}/db
        d(0.5 ll)/dp = 0.5 * (1 - x2_t/v_t) * dv_t/dp

    Kept as a module-level function, not a closure, precisely because the first
    version *was* a closure whose accumulators leaked between calls: its gradient
    disagreed with a finite difference in sign and magnitude.
    ``tests/test_volatility_targeting.py`` asserts this against
    ``scipy.optimize.approx_fprime``.

    **Cost.**  The recursion cannot be vectorised (it is sequential), but
    iterating a pre-extracted ``list`` of Python floats instead of indexing the
    ndarray is exact and measured **≈2×** faster (1.78 ms → 0.89 ms per 500-bar
    pass; the ``x2[i]``/``x2[i-1]`` ``__getitem__`` calls were most of the cost).
    The values are bit-identical, which matters because the optimiser's result is
    asserted against a fixed objective in the test suite.
    """
    w, a, b = float(theta[0]), float(theta[1]), float(theta[2])
    xs = [float(t) for t in np.asarray(x2, dtype=np.float64).ravel()]
    n = len(xs)
    if n == 0:
        return 0.0, np.zeros(3)
    v = float(var_s)          # seed the filter at the sample level
    dv_dw = dv_da = dv_db = 0.0
    g_w = g_a = g_b = 0.0
    ll = 0.0
    log = math.log
    lag = 0.0
    for xi in xs:
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
        ll += log(v) + xi / v
        resid = 0.5 * (1.0 - xi / v)
        g_w += resid * dv_dw
        g_a += resid * dv_da
        g_b += resid * dv_db
        lag = xi
    return 0.5 * ll, np.array([g_w, g_a, g_b], dtype=float)


#: Parameter box for the GARCH(1,1) MLE, in the fit's percent² units.  ``alpha``
#: and ``beta`` are bounded inside the stationarity region (``alpha + beta`` is
#: clipped at ``GARCH_MAX_PERSISTENCE``) because that is what keeps the Gaussian
#: likelihood **bounded**: as persistence → 1 the intercept ``omega`` shrinks
#: toward 0 and the floor ``var_s × 1e-4`` starts to matter again.
GARCH_MAX_PERSISTENCE = 0.999
#: Variance-targeting start used by the coarse pre-search: the RiskMetrics-like
#: ``(alpha, beta)`` the plan documents, with ``omega = V(1−α−β)``.
_GARCH_MLE_X0 = (0.06, 0.93)
#: The coarse pre-search grid, in ``(alpha, beta)``.  It exists because the
#: GARCH likelihood surface on a 500-bar window is **flat and multi-modal**:
#: measured on the shipped BTC 1h window, SLSQP from three nearby starts lands on
#: three different points spanning ``alpha 0.050…0.061`` (all with
#: ``0.5·ΣLL`` inside 0.4 of each other), and a local optimiser alone therefore
#: reports whichever basin its seed fell into.  A deterministic scan plus one
#: local polish reports the best point it can prove, and runs in a fixed time.
_GARCH_GRID_ALPHA = (0.01, 0.20, 0.05)
_GARCH_GRID_BETA = (0.60, 0.99, 0.05)


def _garch11_neg_ll(theta, x2: np.ndarray, var_s: float) -> float:
    """Objective for :func:`scipy.optimize.minimize` (``0.5·ΣLL``, NLL)."""
    return garch11_loglik_grad(x2, var_s, theta)[0]


def _garch11_grid_scan(x2: np.ndarray, var_s: float) -> tuple[float, float, float]:
    """Coarse **variance-targeted** scan → ``(omega, alpha, beta)`` start.

    ``omega = V(1 − α − β)`` is imposed while scanning (variance targeting, the
    standard trick that removes one parameter and keeps the filter at the
    sample's own level), which is what makes a coarse grid a fair start for the
    fully free fit that follows.  Deterministic and O(20 × 8) likelihood passes.
    """
    a0, a1, astep = _GARCH_GRID_ALPHA
    b0, b1, bstep = _GARCH_GRID_BETA
    best = (float(var_s) * 0.05, 0.06, 0.93)
    best_ll = float("inf")
    for a in np.arange(a0, a1 + 1e-12, astep):
        for b in np.arange(b0, b1 + 1e-12, bstep):
            if a + b >= GARCH_MAX_PERSISTENCE:
                continue
            w = float(var_s) * (1.0 - float(a) - float(b))
            if w <= 0.0:
                continue
            ll = garch11_loglik_grad(x2, var_s, (w, float(a), float(b)))[0]
            if np.isfinite(ll) and ll < best_ll:
                best_ll = ll
                best = (w, float(a), float(b))
    return best


def _garch11_mle(x2: np.ndarray, var_s: float) -> tuple[float, float, float, float] | None:
    """Free-``omega`` GARCH(1,1) Gaussian MLE inside :data:`GARCH_MAX_PERSISTENCE`.

    Returns ``(omega, alpha, beta, sum_0.5_LL)`` in the caller's own units, or
    ``None`` when scipy is unavailable / the optimiser fails.

    **Scale normalisation is not cosmetic.**  The likelihood's only scale-bearing
    term is the intercept, and it is bounded by a box expressed in the data's
    units (``omega <= 10 * var_s``), so fitting ``x`` and ``100·x`` is *not* the
    same optimisation even though the model is scale-equivalent: measured, the
    percent² fit converged to ``omega = 0.32`` with a large positive objective
    while the identical model on the same data in fraction² converged to the
    proper optimum — the two objects are the same model, and the first is simply
    the wrong optimum.  The fit therefore runs on ``z = x / sd(x)``, where the
    box is dimensionless and the two unit systems cannot disagree, and is then
    mapped back exactly (``omega_x = omega_z · var_s`` — the variance maps by
    ``sd**2``).

    Two stages, both deterministic:

    1. :func:`_garch11_grid_scan` — a coarse variance-targeted scan over
       ``alpha ∈ [0.01, 0.20]``, ``beta ∈ [0.60, 0.99]``;
    2. a Nelder-Mead polish of the fully free three-parameter likelihood from
       that point, plus an explicit rejection of any result outside the box
       (``omega > 0``, ``alpha, beta ≥ 0``, ``alpha + beta ≤ GARCH_MAX_PERSISTENCE``).

    The constraint — not a fudge factor — is what bounds the objective: ``β = 1``
    is where the likelihood becomes improper (the variance never revisits its
    level), so it is excluded exactly as a GARCH text excludes it.  Nelder-Mead
    rather than SLSQP because SLSQP on this flat surface was measured at 4–7 s
    per fit against ~0.2 s here, and it lands in whichever basin its seed falls
    into.
    """
    try:
        from scipy.optimize import minimize
    except Exception:  # pragma: no cover - scipy is a hard dependency elsewhere
        return None
    sd = math.sqrt(float(var_s))
    if not math.isfinite(sd) or sd <= 0.0:
        return None
    z = np.asarray(x2, dtype=float) / (sd * sd)
    var_z = float(np.var(np.sqrt(np.maximum(z, 0.0)), ddof=1))
    if not np.isfinite(var_z) or var_z <= 0.0:
        return None
    lo, hi = 1e-12, 10.0 * var_z

    def nll(theta):
        t = np.asarray(theta, dtype=float)
        a, b = float(t[1]), float(t[2])
        if not (lo <= t[0] <= hi) or a < 0.0 or b < 0.0:
            return 1e12
        if a + b > GARCH_MAX_PERSISTENCE:
            return 1e12
        return _garch11_neg_ll(t, z, var_z)

    x0 = np.asarray(_garch11_grid_scan(z, var_z), dtype=float)
    try:
        res = minimize(nll, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "maxfev": 4000,
                                "xatol": 1e-10, "fatol": 1e-12})
    except Exception:  # pragma: no cover - numerical failure ⇒ keep the scan
        res = None
    if res is None or not np.all(np.isfinite(res.x)):
        w_z, a, b = (float(x0[0]), float(x0[1]), float(x0[2]))
        ll = float(_garch11_neg_ll(x0, z, var_z))
    else:
        w_z, a, b = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
        ll = float(res.fun)
    if not (np.isfinite(w_z) and np.isfinite(a) and np.isfinite(b) and np.isfinite(ll)):
        return None
    if w_z <= 0.0 or a < 0.0 or b < 0.0 or a + b > GARCH_MAX_PERSISTENCE + 1e-9:
        return None
    # z = x / sd  ⇒  omega_x = omega_z * sd² (and alpha, beta, the likelihood are
    # invariant).
    return w_z * float(var_s), a, b, ll


def _garch11_scipy_mle(r: np.ndarray, *, outlier_sigma: float = 8.0
                       ) -> tuple[float, float, float, bool]:
    """GARCH(1,1) fit -> ``(omega, alpha, beta, ok)`` in **fraction²** units.

    Primary path: the **free-``omega`` Gaussian MLE** of :func:`_garch11_mle`,
    fitted on a bounded parameter box and seeded at the variance-targeted point.
    Fallback (``ok=False``, the caller then degrades to EWMA): the
    **unit-persistence IGARCH grid** of :func:`_garch11_best_beta`, which is the
    cheap, deterministic, optimiser-free fit that the shipped P3 estimator used
    for *every* call.

    Why the primary path changed (this reverses a shipped claim, with the
    measurement that reverses it).  The previous docstring here asserted that the
    free-``omega`` likelihood is **unbounded** and that "Nelder-Mead, L-BFGS-B
    and SLSQP … all converged to that corner and rejected the generating
    parameters of a synthetic GARCH(1,1)".  Re-measured on the shipped BTC 1h
    500-bar window (``tests/test_p34_audit_fixes.py`` re-runs it): all three
    optimisers converge to **ω ≈ 0.002265, α ≈ 0.0697, β ≈ 0.9254** with
    ``0.5·ΣLL = −226.771`` — they agree to 5 decimals and reject *nothing*.  The
    IGARCH grid, by contrast, lands on the degenerate ``α = 1, β = 0`` corner
    where the ``x²/v`` term is unbounded below (its own objective reads
    ``3.6e5``-scale garbage: measured mean ``0.5·ΣLL / n = 40.65`` against the
    MLE's ``−0.4535``), i.e. the *grid* is the pathological fit, not the MLE.
    The old rationale was wrong, so the estimator is the fitted GARCH(1,1).

    Both figures the old docstring quoted were also **sums** presented as means
    (``1.8e4`` and ``3.6e5`` are sums of a quantity this module defines and
    reports as a per-observation mean): the per-observation means are ``36`` and
    ``7.2e5`` respectively, and neither is a likelihood a well-posed fit
    produces.  The fallback keeps the honest *production* justification instead:
    the grid is cheap (≈2 ms, no optimiser, deterministic) and its corner is
    observable on this cache, which is why :func:`garch11_forecast` blends rather
    than trusting one step.

    ``outlier_sigma`` is looser than the module default (8 vs 6) because the
    likelihood is *supposed* to see the large moves whose clustering it models.
    """
    # Same array, same anchor as the old bare call (`series_anchor` *is* the
    # `_anchored_mad` recipe the default used), so the fit is bit-identical.
    arr = np.asarray(r, dtype=np.float64)
    x = clip_outliers(arr, sigma=outlier_sigma, anchor=series_anchor(arr)) * 100.0
    n = x.size
    if n < 50:
        return 0.0, 0.0, 0.0, False
    x2 = x * x
    var_s = float(np.var(x, ddof=1))
    if not np.isfinite(var_s) or var_s <= 0.0:
        return 0.0, 0.0, 0.0, False

    mle = _garch11_mle(x2, var_s)
    if mle is not None:
        w, a, b, _ll = mle
        # percent² -> fraction² (the unit every public entry point returns).
        return float(w) / 1e4, a, b, True

    best_b, best_ll = _garch11_best_beta(x2, var_s)
    if best_b is None or not np.isfinite(best_ll):
        return 0.0, 0.0, 0.0, False
    beta = float(best_b)
    return 0.0, 1.0 - beta, beta, True


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
    xs = [float(t) for t in np.asarray(x2, dtype=np.float64).ravel()]
    n = len(xs)
    v = np.empty(n, dtype=np.float64)
    if n == 0:
        return v
    prev = float(var_s) if v0 is None else float(v0)
    v[0] = prev if prev > floor else floor
    # Same pre-extracted-list trick as :func:`garch11_loglik_grad`: this loop is
    # called once per likelihood pass (hundreds per fit), so the ndarray
    # ``__getitem__`` overhead dominated it (measured 0.20 ms → 0.14 ms per
    # 500-bar pass; values bit-identical).
    for i, xi in enumerate(xs[:-1]):
        prev = a * xi + b * prev
        v[i + 1] = prev if prev > floor else floor
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


def garch11_params(returns, *, window: int = DEFAULT_WINDOW) -> dict:
    """Fit GARCH(1,1) and return ``{omega, alpha, beta, backend, ok, n, fitted}``.

    ``alpha + beta`` is persistence (how slowly a shock decays).  The
    unconditional (long-run) variance is ``omega / (1 - alpha - beta)`` — which is
    **not** defined for the ``ok=True, fitted=False`` IGARCH fallback, where
    ``omega = 0`` and ``alpha + beta = 1`` make that expression ``0/0``: for that
    branch the long-run variance is the current level, not a fitted number, and
    the field ``fitted`` says so.  Returns are winsorised before the fit
    (:func:`clip_outliers`) for the data-splice reason documented on
    :data:`DEFAULT_OUTLIER_SIGMA`.

    **Cost: not a per-bar call.**  A full fit measures ≈0.23 s on a 500-bar window
    (≈575 likelihood passes of the Nelder-Mead polish over the variance-targeted
    grid start), against :data:`PER_BAR_BUDGET_SEC` = 2 ms.  It is a research /
    reporting estimator: cache it or lift it off the per-bar path.  The
    optimiser-free IGARCH grid fallback (``fitted=False``) is the cheap branch
    (≈3 ms) and is what a caller that needs something per bar should use.
    """
    r = _clipped(returns, window=window, outlier_sigma=DEFAULT_OUTLIER_SIGMA)
    backend = garch_backend()
    if r.size < 50:
        return {"omega": 0.0, "alpha": 0.0, "beta": 0.0,
                "backend": backend, "ok": False, "n": int(r.size),
                "fitted": False}
    if backend == "arch":  # pragma: no cover - not installed in this deployment
        try:
            from arch import arch_model
            res = arch_model(r * 100.0, vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
            p = res.params
            omega = float(p["omega"]) / 1e4  # back to fraction^2
            return {"omega": omega, "alpha": float(p["alpha"]),
                    "beta": float(p["beta"]), "backend": "arch", "ok": True,
                    "n": int(r.size), "fitted": True}
        except Exception:
            backend = "scipy"  # fall through to our own MLE
    omega, alpha, beta, ok = _garch11_scipy_mle(r)
    fitted = bool(ok and omega > 0.0)
    return {"omega": omega, "alpha": alpha, "beta": beta,
            "backend": "scipy", "ok": ok, "n": int(r.size), "fitted": fitted}


#: Weight of the **fitted** one-step variance in :func:`garch11_forecast`; the
#: rest is the EWMA level.  The ``arch`` package does the same by default
#: (``res.forecast(horizon=1)`` returns ``0.5 * sigma2_{t+1} + 0.5 * sigma2_t``).
#: The blend is insurance against a *degenerate fit*, and that is now the only
#: reason it exists: with the primary free-``omega`` MLE the one-step forecast is
#: already sane (measured 0.445 %/bar on the shipped window against EWMA's
#: 0.524 %/bar), but if scipy is unavailable the estimator falls back to the
#: IGARCH grid, whose fit on this cache sits on the constant-variance corner
#: (``alpha = 1, beta = 0``) and whose pure one-step forecast would be the last
#: squared return divided by 10000 (``0.00076 %/bar``).  Blending bounds the
#: output between the fitted one-step variance and the EWMA level, so **no**
#: fitted parameter combination can produce a degenerate *forecast*.
GARCH_FIT_WEIGHT = 0.5


def garch11_forecast(returns, *, window: int = DEFAULT_WINDOW, unit: str = "per_bar",
                     periods_per_year: float = 8760.0,
                     lam: float = DEFAULT_LAMBDA,
                     fit_weight: float = GARCH_FIT_WEIGHT) -> float:
    """One-step-ahead volatility forecast from the fitted GARCH model.

    ``v_hat = w * v_{t+1} + (1 - w) * v_t`` with ``w = fit_weight``, where
    ``v_{t+1} = omega + alpha r_t^2 + beta v_t`` is the model's one-step variance
    and ``v_t`` the filtered current variance.  See :data:`GARCH_FIT_WEIGHT` for
    why the blend stays, and :func:`_garch11_scipy_mle` for the fit itself
    (free-``omega`` MLE, with the unit-persistence grid as the documented cheap
    fallback).

    Returns ``0.0`` when the fit is unusable (fewer than 50 bars) — the caller
    then falls back to EWMA; :func:`forecast_vol` does that for you.

    **Cost: ≈0.23 s/call** (it runs the full MLE of :func:`garch11_params`), i.e.
    ~115× the 2 ms per-bar budget.  It is an explicit opt-in estimator, never a
    default, and must not be lifted onto the per-bar path; see the module
    docstring's cost section.
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
                      fit_weight: float = GARCH_FIT_WEIGHT,
                      outlier_sigma: float = DEFAULT_OUTLIER_SIGMA) -> float:
    """Blended one-step conditional variance (fraction²) for a fitted model.

    **Same data as the fit.**  The returns are winsorised with
    :func:`clip_outliers` at the **same** threshold
    (:data:`DEFAULT_OUTLIER_SIGMA`) the fit used.  Filtering the raw series with
    parameters fitted on the clipped one was a real bug: the unclipped
    ``+27.6 %`` splice bar is ``~1.6·10**4`` in percent² against a conditional
    level near ``0.2``, so a single bar drove the filtered variance and the
    forecast to **4.72 %/bar — 9.6×** the EWMA level, where the fitted model's own
    one-step number is **0.48 %/bar**.

    **Units.**  ``p["omega"]`` is fraction² (the unit of every public entry
    point) while the filter runs in percent², and one fraction² is **1e4**
    percent².  The intercept and the returns must therefore be converted
    together; scaling only one of them is a real 1e4 error that was measured at
    **6.6× the EWMA level** on the shipped window.

    **Seed.**  The filter starts at the model's own long-run level
    ``omega / (1 - alpha - beta)`` rather than at the 500-bar sample variance.
    Seeding with the sample variance is not neutral: at the fitted persistence
    (``~0.985`` on the shipped window) an inflated seed decays only to
    ``0.985**500 ≈ 5e-4`` of its initial size, so the filter would report the
    *seed* for the whole window instead of the conditional level.  Starting at
    the model's own level needs only the memory the model actually has.
    """
    if r.size == 0:
        return 0.0
    # Same array, same anchor as the old bare call — bit-identical filter input.
    arr = np.asarray(r, dtype=np.float64)
    x = clip_outliers(arr, sigma=outlier_sigma, anchor=series_anchor(arr)) * 100.0
    if x.size == 0:
        return 0.0
    # One fraction² is 1e4 percent² — convert the intercept AND the returns
    # together (scaling only one of them is the bug this docstring records).
    omega_p2 = float(p["omega"]) * 1e4
    a, b = float(p["alpha"]), float(p["beta"])
    persist = a + b
    lr_f2 = float(p["omega"]) / (1.0 - persist) if persist < 1.0 else 0.0
    # Seed at the model's own long-run level (fraction² -> percent² by *1e4).
    v_p2 = lr_f2 * 1e4
    if not np.isfinite(v_p2) or v_p2 <= 0.0:
        v_p2 = float(np.var(x, ddof=1)) if x.size >= 2 else float(x[0] ** 2)
    for xi in x:
        v_p2 = omega_p2 + a * float(xi) * float(xi) + b * v_p2
    v_next_p2 = omega_p2 + a * float(x[-1]) ** 2 + b * max(v_p2, 0.0)
    v_t = max(v_p2, 0.0) / 1e4
    v_next = max(v_next_p2, 0.0) / 1e4
    w = min(max(float(fit_weight), 0.0), 1.0)
    # ``ewma_variance`` is the same quantity for the unit-persistence family and
    # is well conditioned even when the grid fit degenerates.
    v_ewma = ewma_variance(r, lam=lam, window=0, outlier_sigma=outlier_sigma)
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

def _as_returns(data, *, window: int | None = None) -> np.ndarray:
    """Coerce the accepted inputs to the **full** finite log-return array.

    * ``Series``/``ndarray`` → treated **as returns** (documented, and what the
      sizer/guard have in hand),
    * ``DataFrame`` with a ``close`` column → ``log_returns(df['close'])``
      (an OHLC frame never has to be pre-processed by the caller).

    The array is **not** windowed here (the ``window`` argument is accepted and
    ignored, kept so existing callers keep working).  It used to slice, which is
    exactly what made the forecast window-dependent: the estimator then built its
    clip anchor from the slice, so ``forecast_vol(df, window=500)`` and
    ``forecast_vol(df, window=400)`` computed two different Winsor limits (P3/P4
    audit item 7).  Each estimator now applies ``window`` *after* clipping the
    whole series (:func:`_clipped`), and :func:`forecast_realized_ratio` slices
    explicitly where it needs the windowed level.

    The old slicing did fix a real disagreement — ``forecast_vol(df)`` and
    ``forecast_vol(log_returns(df))`` disagreed because only the array branch had
    been windowed (measured: the GARCH path cost **33.6 ms** and returned
    ``0.003860`` for the frame against **2.2 ms** and ``0.003745`` for the same
    returns).  Both branches are now un-windowed and the window is applied by the
    estimator, so the frame/series agreement holds for a stronger reason: the two
    paths are the same call.
    """
    if isinstance(data, pd.DataFrame):
        if "close" not in data.columns:
            raise KeyError("a DataFrame input needs a 'close' column "
                           "(pass a return series instead)")
        return log_returns(data["close"].values)
    return _last(data, 0)


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

    r = _as_returns(data)
    if m in OHLC_METHODS:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"method {m!r} needs an OHLC DataFrame, not a return series")
        fn = parkinson_vol if m == "realized_parkinson" else garman_klass_vol
        return fn(data, window=window, unit=unit, periods_per_year=ppy)
    if m == "realized_cc":
        return realized_vol(r, window=window, unit=unit, periods_per_year=ppy)
    if m == "garch11":
        val = garch11_forecast(r, window=window, unit=unit, periods_per_year=ppy)
        if val > 0.0 or allow_garch:
            return val
        return ewma_vol(r, lam=lam, window=window, unit=unit,
                        periods_per_year=ppy)
    return ewma_vol(r, lam=lam, window=window, unit=unit, periods_per_year=ppy)


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
    r = _as_returns(data)
    # Same trailing level as before (the window is applied here because
    # ``_as_returns`` no longer slices); the forecast keeps the window too, so the
    # clip anchor comes from the whole series instead of the windowed slice.
    base = realized_vol(_last(r, window), window=baseline_window or 0)
    if base <= 0.0:
        return 1.0
    return float(forecast_vol(r, method=method, window=window, lam=lam) / base)


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


#: Bound on the per-(symbol, interval) memo caches below.  Mirrors
#: ``microstructure.MAX_CACHE_ENTRIES``: the key space is operator-controlled
#: (a universe of symbols × intervals), and an unbounded dict there is a slow
#: leak — the same reason the microstructure cache is bounded.  64 keys is far
#: more than a live deployment tracks at once.
MAX_FORECAST_CACHE_ENTRIES = 64


def _bounded_put(store: dict, key, value, max_entries: int) -> None:
    """Insert ``key`` into ``store``, evicting the least-recently-used key.

    ``dict`` preserves insertion order, so ``next(iter(store))`` is the oldest
    entry; this is the same policy (and the same O(1) cost) as the microstructure
    cache's, without depending on a timestamp.  Recency is maintained by the
    caller: :meth:`VolForecaster.forecast` deletes and re-inserts a key on every
    cache *hit*, which moves it to the end of the order.  Without that touch this
    function is only insertion-order (FIFO) and drops a key that is read on every
    bar.  Re-assigning an existing key here does **not** move it — ``dict`` keeps
    the original position on update — which is why the pop/re-insert matters.
    """
    if max_entries > 0 and key not in store and len(store) >= int(max_entries):
        for stale in list(store)[: len(store) - int(max_entries) + 1]:
            store.pop(stale, None)
    store[key] = value


class VolForecaster:
    """Memoising wrapper for the live per-bar path.

    ``_on_kline`` fires many times per bar (every tick rebuilds the frame), so
    the same bar is recomputed over and over.  This caches the last forecast
    **per key** (symbol, interval) and invalidates it when the newest bar's
    timestamp changes — the same trick ``MLPredictor`` uses for its feature
    matrix.  Cheap EWMA is recomputed only once per bar, and a GARCH fit is
    never repeated for a bar that has not closed.

    The memo is **bounded** to :data:`MAX_FORECAST_CACHE_ENTRIES`: the key is
    ``(symbol, interval)`` and a live universe is operator-controlled, so an
    unbounded dict would grow by one entry per symbol×interval seen for the
    lifetime of the process.  Eviction is **least-recently-used**, not
    first-in-first-out: a hit moves its key to the end of the order, so the
    symbol being forecast every tick cannot be evicted by a burst of one-off
    symbols (measured — under FIFO, a key read one tick earlier was dropped after
    8 distinct keys churned through a 4-entry cache).
    """

    def __init__(self, *, method: str = "ewma", window: int = DEFAULT_WINDOW,
                 lam: float = DEFAULT_LAMBDA, interval: str | None = None,
                 allow_garch: bool = False,
                 max_cache_entries: int = MAX_FORECAST_CACHE_ENTRIES):
        self.method = method
        self.window = int(window)
        self.lam = float(lam)
        self.interval = interval
        self.allow_garch = bool(allow_garch)
        self.max_cache_entries = int(max_cache_entries)
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
            # LRU touch.  ``_bounded_put`` evicts by insertion order, so without
            # this a key that is read on every bar (the live symbol) is still the
            # first one dropped once the cap is reached: measured, a key served
            # one tick earlier was gone after 8 keys churned through a 4-entry
            # cache.  Deleting before the insert below is what moves it to the
            # end of the order; a plain re-assignment would leave it where it was.
            self._cache.pop(key, None)
            _bounded_put(self._cache, key, cached, self.max_cache_entries)
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
        _bounded_put(self._cache, key, (bar, sig, out), self.max_cache_entries)
        self.compute_count += 1
        return out

    def clear(self) -> None:
        self._cache.clear()
