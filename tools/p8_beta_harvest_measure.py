"""P8 evidence — does **volatility targeting** improve a beta-harvesting basket?

Why this tool exists (P8, the operator's pivot)
-----------------------------------------------
Five searches for short-horizon **directional** alpha failed on this repository's
real cache (ML gates, pairs 0/30, meta-labelling, 52 GA generations, P7's four
stages): none reached a positive out-of-sample deflated Sharpe.  In the same
windows **buy-and-hold was strongly positive** (2026-09 ``+10.06 %``;
2026-07..09 ``+51.36 %``) while the strategies lost a little and captured none of
it.  The question this tool answers is therefore no longer "is there alpha?" but
**"is a risk-managed beta harvest better than a plain beta harvest?"** — a
deterministic question that needs no forecasting skill, because conditional
*volatility* is predictable (ARCH/GARCH) even though the *sign* of the next
return is not.

The machinery already exists and is shipped **off**:
``risk.vol_targeting`` (``enabled: false``) with five estimators, consumed by
:mod:`core.risk.position_sizer` (``vol_scale``) and
:class:`core.risk.manager.RiskManager` (``forecast_vol_pct``).

Arms (this is a *comparison*, not a search)
-------------------------------------------
``H``   **hold** — equal-weight buy-and-hold of the basket, the **parameter-free
        baseline**.  Rule (stated, because "equal weight" is ambiguous): split
        the account 1/N at the window's first 1h close, buy each symbol at that
        close, **never rebalance** (weights drift), liquidate at the window's last
        close.  Costs: one per-side charge per leg at entry, one at liquidation.
``V``   **vol-targeted hold** — the *same book, same weights, same entry and
        liquidation convention*; the only thing added is a scalar exposure path.
        Once per UTC day the overlay sets total gross notional to
        ``equity × clip(target_vol_pct / forecast_vol_pct, min_scale, max_scale)``
        by scaling **every leg by the same factor**, so the overlay cannot (and
        does not) change the relative weights — otherwise "H versus V" would
        measure the rebalancing rule rather than the overlay, and on this cache it
        does: in ``W1`` ``ZECUSDT`` returns ``+581.8 %`` while the other six lose
        15-59 %, so a daily-**rebalanced** equal-weight index returns ``-1.9 %``
        while the drifting-weight book returns ``+54.4 %``.  The forecast is the
        equal-weight mean of the per-symbol :func:`core.ml.volatility.forecast_vol`
        readings and the scale is
        :meth:`core.risk.position_sizer.PositionSizer.vol_scale` — **nothing is
        reimplemented**; ``forecast_vol`` sees the trailing
        ``RiskManager._VOL_HISTORY_BARS`` (600) bars, exactly like the live path.
``Hreb``**diagnostic** — the same basket **rebalanced to equal notional every UTC
        day** with the overlay pinned at scale 1.0 (so it is *not* drifted).  It
        isolates how much of an ``H``↔``V`` difference is the rebalancing rule
        rather than the overlay; on this cache that gap is large (``W1``:
        ``Hreb -1.9 %`` against ``H +54.4 %``), which is exactly why ``V`` is
        built on ``H``'s drifting book and not on a rebalanced one.
``V<=1`` ``V`` with ``max_scale=1.0`` — the same rule with no leverage, because a
        cash backtest cannot go above 100 % gross.
``Vleg`` ``V`` applied **per leg** (each symbol's notional scaled by its own
        ``vol_scale``, so the book is reweighted to equal *risk* each day) — the
        literal production sizing rule, and a genuinely different portfolio.
``S``   **the existing strategies** — every ``strategies/ga_champion_*.yaml``
        (read-only) run through the production engine on 1h bars and composited at
        equal capital (1/7 each).  A **reference**, never the yardstick: the
        operator's rule is that ``V`` must beat ``H``, not ``S``.

Windows
-------
``W0`` 2025-05-01→2025-08-01, ``W1`` 2025-10-01→2026-01-01,
``W2`` 2026-02-01→2026-05-01, ``W3`` 2026-06-01→2026-09-01.

All four are inside the ``2025-04-01..2026-10-01`` 1h cache with ≥700 bars of
estimator warm-up before the start; they are disjoint ~92-day blocks (≈92 daily
observations each, the basis ``core.ga.fitness.deflated_sharpe_ratio`` wants), and
they differ in regime (the tool prints each window's basket buy-and-hold return
and realised volatility so the claim is measured, not asserted).  ``W0``..``W2``
are also **before the shipped champions' training windows** (2026-03-01 onward),
so arm ``S`` is out of sample there; ``W3`` overlaps them — said plainly, because
it is one more reason ``S`` is a reference only.

Pre-stated acceptance threshold (fixed before any number was computed)
---------------------------------------------------------------------
``V`` (the documented default: ``method="ewma"``, ``lam=0.94``, ``window=500``,
``target_vol_pct=0.45``, ``min_scale=0.25``, ``max_scale=2.0``) is worth
**enabling** only if all three hold:

``T1`` Calmar(V) > Calmar(H) in **every** window;
``T2`` at matched risk (V's daily returns scaled to H's realised volatility) V
       keeps ≥ **80 %** of H's total return in **every** window;
``T3`` DSR(V) > 0 with the honest trial count in **≥ 2** of the 4 windows.

Otherwise the recommendation is to leave ``risk.vol_targeting.enabled: false``.
The tool **never writes config/config.yaml** (it drives the overlay through an
in-memory ``VolTargetingConfig`` copy) and never writes ``strategies/**`` or
``data/binance_trader.db``.

Honest multi-testing accounting
------------------------------
The primary arm ``V`` is one pre-registered setting, so it is **one** trial; the
sensitivity grid (4 estimators × 3 target vols × 2 windows = **24** points) is
swept too, so **every** point is counted: the reported DSR for the ``V`` family
uses ``n_trials = 1 + 24 = 25``.  Arm ``S`` is additionally deflated by the
champions' own published GA trial counts (``provenance.n_trials``).

Usage::

    python tools/p8_beta_harvest_measure.py
    python tools/p8_beta_harvest_measure.py --no-arm-s --grid none   # H vs V only
    python tools/p8_beta_harvest_measure.py --out %TEMP%\\p8.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

DEFAULT_OUT = Path(tempfile.gettempdir()) / "p8_beta_harvest.json"
INITIAL_BALANCE = 10_000.0
INTERVAL = "1h"

#: ``(label, start, end)``; the window is ``[start, end)`` on the 1h close grid.
WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("W0", "2025-05-01", "2025-08-01"),
    ("W1", "2025-10-01", "2026-01-01"),
    ("W2", "2026-02-01", "2026-05-01"),
    ("W3", "2026-06-01", "2026-09-01"),
)

#: Bars of history loaded **before** a window so the estimator is warm at bar 0.
WARMUP_HOURS = 700

#: Cash pairs are not beta: a USD-stable pair has ~0 volatility and would only
#: dilute the basket.  Excluded by name pattern, stated in the report.
STABLE_EXCLUDE = ("USDCUSDT", "USDTUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "DAIUSDT")

#: The sensitivity grid swept for arm ``V`` (every point is a counted trial).
GRID_ESTIMATORS = ("ewma", "realized_cc", "realized_parkinson", "realized_garman_klass")
GRID_TARGETS = (0.30, 0.45, 0.60)
GRID_WINDOWS = (250, 500)

#: Arm ``S`` is run on 1h bars.  Reason: the engine evaluates a strategy's
#: **shortest** declared timeframe, and four shipped champions declare ``1m``/``5m``
#: (789 k / 158 k bars over the cache); a native-timeframe multi-month run is
#: hours, not minutes.  Every arm is therefore on the same 1h grid and the same
#: cost model.  ``--arm-s-timeframes native`` restores the champions' own frames.
ARM_S_TIMEFRAMES = ("1h",)

#: Pre-stated threshold (see the module docstring).  Printed before the numbers.
THRESHOLD = {
    "T1_calmar_beats_hold_every_window": True,
    "T2_matched_risk_return_retention": 0.80,
    "T3_dsr_positive_windows": 2,
}


# ── tiny helpers ───────────────────────────────────────────────────────

def sha256_16(path: Path) -> str:
    """First 16 hex chars of a file's sha256 (``"--"`` when absent)."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return "--"


def _finite(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _jsonable(obj):
    """Recursively make *obj* JSON-safe (numpy scalars, NaN/inf → None)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ── data ───────────────────────────────────────────────────────────────

def market_dir() -> Path:
    return ROOT / "data" / "market"


def basket_symbols(data_dir: Path | None = None) -> tuple[list[str], dict]:
    """Every cached symbol that covers the **whole** cache span on 1h, minus stables.

    The rule is data-driven and stated rather than hand-picked: a symbol is in the
    basket when its ``1h.parquet`` starts within a day of the earliest start and
    ends within a day of the latest end across all cached symbols.  Anything else
    (``ENAUSDT``/``VTHOUSDT`` start 2026-09-23; ``MOVRUSDT`` has no 1h file) cannot
    support four windows and is excluded with its reason recorded.
    """
    root = data_dir or market_dir()
    coverage: dict[str, dict] = {}
    starts, ends = [], []
    for symbol_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        path = symbol_dir / f"{INTERVAL}.parquet"
        if not path.exists():
            coverage[symbol_dir.name] = {"reason": f"no {INTERVAL}.parquet"}
            continue
        index = pd.read_parquet(path).index
        if len(index) == 0:
            coverage[symbol_dir.name] = {"reason": "empty"}
            continue
        coverage[symbol_dir.name] = {
            "bars": int(len(index)),
            "start": str(index.min())[:19],
            "end": str(index.max())[:19],
        }
        starts.append(index.min())
        ends.append(index.max())
    if not starts:
        raise SystemExit(f"no {INTERVAL} parquet under {root}")
    span_start, span_end = min(starts), max(ends)
    keep, excluded = [], {}
    for symbol, info in coverage.items():
        if "start" not in info:
            excluded[symbol] = info["reason"]
            continue
        if symbol in STABLE_EXCLUDE:
            excluded[symbol] = "USD-stable pair (no beta exposure)"
            continue
        late = pd.Timestamp(info["start"]) - span_start > pd.Timedelta(days=1)
        early = span_end - pd.Timestamp(info["end"]) > pd.Timedelta(days=1)
        if late or early:
            excluded[symbol] = (f"partial {INTERVAL} coverage "
                                f"({info['start']}..{info['end']})")
            continue
        keep.append(symbol)
    meta = {"span": [str(span_start)[:19], str(span_end)[:19]],
            "basket_rule": ("full 1h coverage of the cache span (within 1 day at "
                            "each end), minus USD-stable pairs"),
            "excluded": excluded}
    return keep, meta


def load_panel(symbols, start: str, end: str, warmup_hours: int = WARMUP_HOURS
               ) -> dict[str, pd.DataFrame]:
    """Per-symbol OHLC frames covering ``[start - warmup, end)`` on the 1h grid."""
    lo = pd.Timestamp(start) - pd.Timedelta(hours=warmup_hours)
    hi = pd.Timestamp(end)
    panel: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = pd.read_parquet(market_dir() / symbol / f"{INTERVAL}.parquet")
        frame = frame.loc[(frame.index >= lo) & (frame.index < hi)]
        frame = frame[["open", "high", "low", "close"]].astype(float)
        panel[symbol] = frame
    return panel


def closes_frame(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The common hourly close grid — the intersection of every symbol's index."""
    return pd.DataFrame({sym: frame["close"] for sym, frame in panel.items()}).dropna()


def basket_levels(closes: pd.DataFrame) -> pd.Series:
    """Equal-weight **rebalanced-every-bar** basket level (starts at 1.0).

    ``r_t = mean_i r_{i,t}`` (equal weight at every bar) and ``L_t = prod(1 + r)``.
    This is the series the overlay forecasts, because it *is* the portfolio the
    overlay would hold; using ``mean_i P_i,t/P_i,0`` instead would describe a
    drifting-weight book that the daily rebalance does not produce.
    """
    rets = closes.pct_change()
    basket_r = rets.mean(axis=1)
    basket_r.iloc[0] = 0.0
    return (1.0 + basket_r).cumprod()


# ── the overlay: forecast_vol + PositionSizer.vol_scale (nothing reimplemented) ──

@dataclass
class Overlay:
    """One ``risk.vol_targeting`` configuration, driven through the production code."""

    method: str = "ewma"
    lam: float = 0.94
    window: int = 500
    target_vol_pct: float = 0.45
    min_scale: float = 0.25
    max_scale: float = 2.0

    def __post_init__(self):
        from app.config import HardRiskLimits, SoftRiskParams, VolTargetingConfig
        from core.risk.position_sizer import PositionSizer

        #: An in-memory COPY of the shipped block with ``enabled`` turned on.
        #: ``config/config.yaml`` is never opened for writing.
        self.config = VolTargetingConfig(
            enabled=True, method=self.method, lam=self.lam, window=self.window,
            target_vol_pct=self.target_vol_pct, min_scale=self.min_scale,
            max_scale=self.max_scale)
        self.sizer = PositionSizer(HardRiskLimits(), SoftRiskParams(), 0.7, 0.3,
                                  self.config)

    def forecast_pct(self, frames: dict[str, pd.DataFrame]) -> float:
        """Equal-weight mean of the per-symbol ``forecast_vol`` readings, in %/bar.

        The frames handed in are the trailing ``_VOL_HISTORY_BARS`` bars **strictly
        before** the decision bar, which is what ``RiskManager._history_frame``
        gives the live path.
        """
        from core.ml.volatility import forecast_vol, to_pct

        values = []
        for frame in frames.values():
            if len(frame) < 3:
                continue
            try:
                value = to_pct(forecast_vol(
                    frame, method=self.method, window=self.window, lam=self.lam,
                    interval=INTERVAL))
            except Exception:
                value = 0.0
            if value > 0.0:
                values.append(float(value))
        if not values:
            return 0.0
        return float(np.mean(values))

    def scale(self, forecast_pct: float) -> float:
        """``PositionSizer.vol_scale`` — the production arithmetic, not a copy of it."""
        return float(self.sizer.vol_scale(forecast_pct))

    def per_symbol_scales(self, frames: dict[str, pd.DataFrame]) -> dict[str, float]:
        """One ``vol_scale`` per symbol (the literal production sizing rule)."""
        from core.ml.volatility import forecast_vol, to_pct

        out = {}
        for symbol, frame in frames.items():
            if len(frame) < 3:
                continue
            try:
                pct = to_pct(forecast_vol(frame, method=self.method,
                                          window=self.window, lam=self.lam,
                                          interval=INTERVAL))
            except Exception:
                pct = 0.0
            out[symbol] = float(self.sizer.vol_scale(pct))
        return out


class _UnitOverlay(Overlay):
    """``Overlay`` with the scale pinned at exactly 1.0 (the ``Hreb`` diagnostic).

    ``target_vol_pct == forecast`` makes ``PositionSizer.vol_scale`` return
    ``clip(1.0, min_scale, max_scale) == 1.0``, so the book is the same
    equal-notional basket rebalanced daily and nothing else changes.
    """

    def forecast_pct(self, frames):                      # noqa: D102
        return float(self.target_vol_pct)


# ── cost model plumbing (the repo's own function, halved for one side) ──

class Costs:
    """Per-symbol one-side cost of a traded notional, via ``apply_trading_costs``.

    ``apply_trading_costs(p, p, qty, sym, config)`` charges **both** sides
    (fee + half-spread each), so half of it is exactly one side — the same
    convention the GA's engine pays, without re-deriving the formula here.
    """

    def __init__(self, config, symbols):
        from core.backtest.cost_model import resolve_spreads

        resolved = resolve_spreads(symbols, config, use_live=False)
        self.config = config
        self.spreads = {sym: entry["spread_pct"] for sym, entry in resolved.items()}
        self.sources = {sym: entry["source"] for sym, entry in resolved.items()}
        self.fee_pct = _finite(getattr(config, "backtest_taker_fee_pct", 0.04), 0.04)

    def one_side(self, symbol: str, notional: float) -> float:
        from core.backtest.cost_model import apply_trading_costs

        notional = float(notional)
        if notional <= 0.0:
            return 0.0
        total = apply_trading_costs(1.0, 1.0, notional, symbol, self.config,
                                    spread_pct=self.spreads.get(symbol))
        return float(total) / 2.0

    def rate(self, symbol: str) -> float:
        """Per-side cost as a fraction of traded notional (for the report)."""
        return (self.fee_pct + _finite(self.spreads.get(symbol)) / 2.0) / 100.0


# ── simulations ────────────────────────────────────────────────────────

def _curve(equity: pd.Series) -> list[dict]:
    return [{"time": ts, "equity": float(value)} for ts, value in equity.items()]


def simulate_hold(prices: pd.DataFrame, costs: Costs,
                  initial: float = INITIAL_BALANCE) -> dict:
    """Arm ``H`` — equal weight at the first close, never rebalanced, liquidated last."""
    symbols = list(prices.columns)
    share = initial / len(symbols)
    first = prices.iloc[0]
    units = {sym: share / float(first[sym]) for sym in symbols}
    entry_cost = sum(costs.one_side(sym, share) for sym in symbols)
    cash = initial - sum(units[sym] * float(first[sym]) for sym in symbols) - entry_cost

    gross = sum(prices[sym] * units[sym] for sym in symbols)
    equity = cash + gross
    last_notional = {sym: units[sym] * float(prices[sym].iloc[-1]) for sym in symbols}
    liquidation = sum(costs.one_side(sym, value) for sym, value in last_notional.items())
    equity.iloc[-1] = equity.iloc[-1] - liquidation

    turnover = sum(share for _ in symbols) + sum(last_notional.values())
    exposure = (gross / equity).clip(lower=0.0)
    return {"equity": equity, "exposure": exposure, "costs": entry_cost + liquidation,
            "turnover_notional": turnover, "rebalances": 2,
            "tim": pd.Series(1.0, index=prices.index)}


def simulate_vol_target(prices: pd.DataFrame, panel: dict[str, pd.DataFrame],
                        overlay: Overlay, costs: Costs, *, per_leg: bool = False,
                        rebalance_to_equal: bool = False, daily: bool = True,
                        initial: float = INITIAL_BALANCE) -> dict:
    """Arms ``Hreb`` / ``V`` / ``V<=1`` / ``Vleg`` — the same book, scaled exposure.

    The decision at bar *t* uses the close of *t* (prices) and the estimator's
    view of the bars **strictly before** *t* (``panel`` carries the 700 hourly bars
    loaded before the window, so the first decision already sees a full
    ``_VOL_HISTORY_BARS`` history), so the exposure applied to the ``t → t+1``
    return uses no future information.  Costs are charged at every rebalance on
    the traded notional, and at liquidation.

    Three bookkeeping rules, and the difference matters:

    * default (arms ``V``/``V<=1``) — total gross notional is set to
      ``equity × scale`` and **every leg is scaled by the same factor**, so the
      drifting weight vector of arm ``H`` is preserved exactly.  The overlay is
      then the only difference from ``H``, which is what makes the comparison an
      overlay comparison rather than a rebalancing comparison.
    * ``rebalance_to_equal=True`` (arm ``Hreb``) — every leg is set to
      ``equity × scale / N``, i.e. back to **equal notional** every day.  With a
      unit overlay this isolates the rebalancing rule.
    * ``per_leg=True`` (arm ``Vleg``) — each leg's notional is set to
      ``equity × scale_i / N``, i.e. the book is **reweighted to equal risk** every
      day.  That is a different portfolio rule as well as a different exposure,
      and it is labelled as such.
    """
    from core.risk.manager import _VOL_HISTORY_BARS

    symbols = list(prices.columns)
    index = prices.index
    day = pd.Series(index.normalize(), index=index)

    cash = initial
    units = {sym: 0.0 for sym in symbols}
    equity_values, exposures, tim_values = [], [], []
    total_cost = 0.0
    turnover = 0.0
    rebalances = 0
    last_day = None

    for position, stamp in enumerate(index):
        price_row = prices.iloc[position]
        gross = sum(units[sym] * float(price_row[sym]) for sym in symbols)
        equity = cash + gross
        is_last = position == len(index) - 1
        # ``last_day`` marks the last day a decision was actually TAKEN, not the
        # last day seen: a bar with no usable history leaves it unset so the next
        # bar retries instead of silently skipping the whole day in cash.
        is_new_day = daily and last_day != day.iloc[position]

        if is_last:
            liquidation = sum(costs.one_side(sym, units[sym] * float(price_row[sym]))
                              for sym in symbols)
            equity -= liquidation
            total_cost += liquidation
            turnover += sum(units[sym] * float(price_row[sym]) for sym in symbols)
            rebalances += 1
        elif is_new_day:
            decision_frames = {
                sym: panel[sym].loc[panel[sym].index < stamp].iloc[-_VOL_HISTORY_BARS:]
                for sym in symbols}
            ready = sum(1 for frame in decision_frames.values() if len(frame) >= 3)
            if ready:
                if per_leg:
                    scales = overlay.per_symbol_scales(decision_frames)
                    targets = {sym: equity * scales.get(sym, 1.0) / len(symbols)
                               for sym in symbols}
                elif rebalance_to_equal:
                    scale = overlay.scale(overlay.forecast_pct(decision_frames))
                    targets = {sym: equity * scale / len(symbols) for sym in symbols}
                else:
                    scale = overlay.scale(overlay.forecast_pct(decision_frames))
                    if gross > 0.0:
                        factor = equity * scale / gross
                        targets = {sym: units[sym] * float(price_row[sym]) * factor
                                   for sym in symbols}
                    else:
                        targets = {sym: equity * scale / len(symbols) for sym in symbols}
                for sym in symbols:
                    price = float(price_row[sym])
                    if price <= 0:
                        continue
                    held = units[sym] * price
                    delta = targets[sym] - held
                    if abs(delta) < 1e-9:
                        continue
                    cost = costs.one_side(sym, abs(delta))
                    cash -= delta + cost
                    total_cost += cost
                    turnover += abs(delta)
                    units[sym] = targets[sym] / price
                rebalances += 1
                last_day = day.iloc[position]

        gross = sum(units[sym] * float(price_row[sym]) for sym in symbols)
        equity = cash + gross
        equity_values.append(equity)
        exposures.append(gross / equity if equity > 0 else 0.0)
        tim_values.append(1.0 if gross > 0 else 0.0)

    return {
        "equity": pd.Series(equity_values, index=index),
        "exposure": pd.Series(exposures, index=index),
        "costs": total_cost,
        "turnover_notional": turnover,
        "rebalances": rebalances,
        "tim": pd.Series(tim_values, index=index),
    }


# ── metrics (same basis as core.ga.fitness.stats_from_trades) ──────────

def rescaled_daily_returns(equity: pd.Series, initial: float = INITIAL_BALANCE
                           ) -> np.ndarray:
    """Daily returns that **compound back to the curve's own total return**.

    ``core.ga.fitness.daily_returns`` resamples to the last equity of each day, so
    the first day's intraday move is never measured and ``prod(1 + r)`` misses it
    (measured on arm ``H``/``W0``: 29.34 % against a 31.64 % total).  Seeding the
    series with the initial balance one day before the first resampled close makes
    ``k = 1`` reproduce the arm's own total return exactly, which is what the
    matched-risk comparison needs.  ``metrics`` still reports Sharpe on the repo's
    own basis, so nothing here changes a reported risk statistic.
    """
    if len(equity) < 2:
        return np.array([])
    daily = equity.resample("1D").last().dropna()
    if len(daily) < 1:
        return np.array([])
    seeded = pd.concat([pd.Series([float(initial)],
                                  index=[daily.index[0] - pd.Timedelta(days=1)]),
                        daily])
    return seeded.pct_change().dropna().values


def metrics(equity: pd.Series, exposure: pd.Series, tim: pd.Series, costs: float,
            turnover_notional: float, rebalances: int,
            initial: float = INITIAL_BALANCE) -> dict:
    """Return/risk metrics on the **same daily basis** the GA scorer uses."""
    from core.ga.fitness import (daily_returns, max_drawdown_pct, per_period_sharpe)

    curve = _curve(equity)
    rets = daily_returns(curve)
    n = int(rets.size)
    span_days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400.0, 1.0)
    periods = max(n, 1) * (365.0 / span_days)
    mean = float(np.mean(rets)) if n else 0.0
    sd = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    downside = np.minimum(rets, 0.0) if n else np.array([0.0])
    dd_dev = float(np.sqrt(np.mean(downside ** 2))) if n else 0.0
    final = float(equity.iloc[-1])
    total = (final - initial) / initial * 100.0
    growth = final / initial
    cagr = (growth ** (365.0 / span_days) - 1.0) * 100.0 if growth > 0 else -100.0
    max_dd = max_drawdown_pct(curve)
    mean_equity = float(equity.mean())
    net_pnl = final - initial
    gross = net_pnl + costs
    return {
        "total_return_pct": round(total, 4),
        "cagr_pct": round(cagr, 4),
        "ann_vol_pct": round(sd * math.sqrt(periods) * 100.0, 4),
        "sharpe": round(per_period_sharpe(rets) * math.sqrt(periods), 4),
        "sortino": round((mean / dd_dev) * math.sqrt(periods), 4) if dd_dev > 0 else 0.0,
        "max_dd_pct": round(max_dd, 4),
        "calmar": round(cagr / max_dd, 4) if max_dd > 0 else None,
        "time_in_market_pct": round(float(tim.mean()) * 100.0, 4),
        "avg_gross_exposure_pct": round(float(exposure.mean()) * 100.0, 4),
        "max_gross_exposure_pct": round(float(exposure.max()) * 100.0, 4),
        "turnover_x_per_year": round(turnover_notional / mean_equity
                                     / (span_days / 365.0), 4) if mean_equity else None,
        "costs_usdt": round(float(costs), 4),
        "net_pnl_usdt": round(net_pnl, 4),
        "cost_share_of_gross_pct": (round(costs / gross * 100.0, 4)
                                    if gross > 0 else None),
        "rebalances": int(rebalances),
        "bars": int(len(equity)),
        "daily_observations": n,
        "span_days": round(span_days, 2),
        "skew": round(float(pd.Series(rets).skew()), 6) if n > 2 else 0.0,
        "kurtosis": round(float(pd.Series(rets).kurtosis() + 3.0), 6) if n > 3 else 3.0,
        "_daily_returns": rets,
        "_matched_returns": rescaled_daily_returns(equity, initial),
    }


def matched_risk(arm: dict, reference: dict) -> dict:
    """Arm's daily returns scaled to the *reference* arm's realised volatility.

    ``k = sigma_reference / sigma_arm``; the scaled series is **compounded**
    (``prod(1 + k*r) - 1``) on :func:`rescaled_daily_returns`, which for ``k = 1``
    reproduces the arm's own total return exactly.  The repo's own ``risk_matched``
    convention — scale the *total* return linearly (``core.ga.benchmark``) — is
    reported next to it as ``linear_pct`` so the two cannot be confused.  Both
    ignore that a scaled book pays proportionally scaled costs; at ``k > 1`` that
    makes them marginally optimistic.  ``k_extrapolation`` flags a scaling beyond
    5x, which leaves the cash model's range of validity (no leverage, no margin).
    """
    rets_arm = np.asarray(arm.get("_matched_returns", []), dtype=float)
    rets_ref = np.asarray(reference.get("_matched_returns", []), dtype=float)
    sd_arm = float(np.std(rets_arm, ddof=1)) if rets_arm.size > 1 else 0.0
    sd_ref = float(np.std(rets_ref, ddof=1)) if rets_ref.size > 1 else 0.0
    if sd_arm <= 0.0 or sd_ref <= 0.0:
        return {"k": 1.0, "compounded_pct": arm.get("total_return_pct"),
                "linear_pct": arm.get("total_return_pct"), "fallback": True,
                "k_extrapolation": False}
    k = sd_ref / sd_arm
    compounded = (float(np.prod(1.0 + k * rets_arm)) - 1.0) * 100.0
    return {"k": round(k, 6), "compounded_pct": round(compounded, 4),
            "linear_pct": round(_finite(arm.get("total_return_pct")) * k, 4),
            "fallback": False, "k_extrapolation": bool(k > 5.0)}


def dsr_for(arm: dict, n_trials: int) -> dict:
    """``core.ga.fitness.deflated_sharpe_ratio`` on the arm's own daily returns."""
    from core.ga.fitness import deflated_sharpe_ratio

    return deflated_sharpe_ratio(
        arm.get("sharpe", 0.0), n_trials=max(int(n_trials), 1),
        observation_periods=max(int(arm.get("daily_observations", 1)), 1),
        sharpe_is_annualized=True, skew=arm.get("skew", 0.0),
        kurtosis=arm.get("kurtosis", 3.0))


# ── arm S: the shipped champions, production engine, read-only ─────────

def champion_setups() -> list[dict]:
    """Every ``strategies/ga_champion_*.yaml`` as ``(name, StrategyConfig, prov)``."""
    import yaml
    from core.strategy.loader import StrategyConfig

    out = []
    for path in sorted((ROOT / "strategies").glob("ga_champion_*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config = StrategyConfig(**{k: v for k, v in raw.items() if k != "provenance"})
        out.append({"name": path.stem, "path": str(path), "config": config,
                    "provenance": raw.get("provenance") or {},
                    "sha256_16": sha256_16(path)})
    return out


def engine_stack():
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager

    Config._instance = None
    config = Config.load("sim")
    # The GA evaluation contract: legacy engine, no ML, no historical fill priced
    # from today's order book.
    config.backtest_engine_mode = "legacy"
    config.backtest_ml_enabled = False
    config.backtest_live_spread_enabled = False
    bus = EventBus()
    return config, BacktestEngine(config, None, RiskManager(config, bus),
                                  OrderExecutor(config, bus))


def run_arm_s(symbols, start: str, end: str, setups, timeframes) -> dict:
    """Run every champion through the production engine and composite at equal capital."""
    from core.ga.fitness import isolated_eval_kwargs

    config, engine = engine_stack()
    curves: dict[str, pd.Series] = {}
    per_champion: dict[str, dict] = {}
    trades_total = 0
    costs_total = 0.0
    for setup in setups:
        update = {"name": setup["name"]}
        if timeframes:
            update["timeframes"] = list(timeframes)
        strategy = setup["config"].model_copy(update=update)
        result = engine.run_with_exit_evaluation(
            strategies=[strategy], symbols=list(symbols), date_start=start,
            date_end=end, initial_balance=INITIAL_BALANCE, mode="full",
            simulate_ai_weights=False, use_live_spread=False,
            **isolated_eval_kwargs())
        per = result.get("per_strategy_equity") or {}
        block = per.get(strategy.name) or (next(iter(per.values())) if per else {})
        equity = block.get("equity_curve") or []
        trades = block.get("trades") or []
        if not equity:
            per_champion[setup["name"]] = {"error": result.get("error") or "no equity curve"}
            continue
        series = pd.Series([float(p["equity"]) for p in equity],
                           index=pd.DatetimeIndex([pd.Timestamp(p["time"]) for p in equity]))
        curves[setup["name"]] = series
        gains = sum(max(_finite(t.get("pnl")), 0.0) for t in trades)
        losses = sum(max(-_finite(t.get("pnl")), 0.0) for t in trades)
        per_champion[setup["name"]] = {
            "trades": len(trades),
            "total_return_pct": round((float(series.iloc[-1]) / INITIAL_BALANCE - 1) * 100, 4),
            "gross_win_usdt": round(gains, 4),
            "gross_loss_usdt": round(losses, 4),
            "n_trials": setup["provenance"].get("n_trials"),
            "prior_trials": setup["provenance"].get("prior_trials"),
            "window": (setup["provenance"].get("window") or {}).get("key"),
            "symbols": setup["provenance"].get("symbols"),
            "timeframes_native": list(setup["config"].timeframes or []),
            "sha256_16": setup["sha256_16"],
        }
        trades_total += len(trades)
        costs_total += 0.0
    if not curves:
        return {"error": "no champion produced an equity curve"}
    panel = pd.DataFrame(curves).sort_index()
    # Every champion's ledger covers the whole window in this engine, so the
    # forward/back fill is only a guard against a ragged first/last stamp; the
    # composite is then the **equal-capital** (1/K) portfolio of the champions.
    panel = panel.ffill().bfill().dropna(how="all")
    composite = panel.mean(axis=1)
    exposure = pd.Series(1.0, index=composite.index)
    tim = pd.Series(1.0, index=composite.index)
    return {"equity": composite, "exposure": exposure, "costs": costs_total,
            "turnover_notional": 0.0, "rebalances": 0, "tim": tim,
            "per_champion": per_champion, "trades_total": trades_total,
            "champions": len(curves)}


# ── grid ───────────────────────────────────────────────────────────────

def run_grid(panel: dict[str, pd.DataFrame], costs: Costs, start: str, end: str
             ) -> list[dict]:
    """The 4 × 3 × 2 sensitivity sweep — every point is a counted trial."""
    rows = []
    for estimator in GRID_ESTIMATORS:
        for target in GRID_TARGETS:
            for lookback in GRID_WINDOWS:
                overlay = Overlay(method=estimator, target_vol_pct=target,
                                  window=lookback)
                result = simulate_vol_target(panel["prices"], panel["frames"], overlay,
                                             costs)
                m = metrics(result["equity"], result["exposure"], result["tim"],
                            result["costs"], result["turnover_notional"],
                            result["rebalances"])
                rows.append({
                    "method": estimator, "target_vol_pct": target,
                    "lookback": lookback,
                    "total_return_pct": m["total_return_pct"],
                    "sharpe": m["sharpe"], "max_dd_pct": m["max_dd_pct"],
                    "calmar": m["calmar"],
                    "ann_vol_pct": m["ann_vol_pct"],
                    "avg_gross_exposure_pct": m["avg_gross_exposure_pct"],
                    "cost_share_of_gross_pct": m["cost_share_of_gross_pct"],
                })
    return rows


# ── reporting ──────────────────────────────────────────────────────────

ARM_ORDER = ("H", "Hreb", "V", "V<=1", "Vleg", "S")


def _fmt(value, width=11, digits=3):
    if value is None:
        return "n/a".rjust(width)
    if isinstance(value, str):
        return value.rjust(width)
    return f"{value:>{width}.{digits}f}"


def print_table(rows: list[dict]) -> None:
    print("\n=== per arm / per window ===")
    head = ("arm  window  tot_ret%   ann_vol%     sharpe    sortino"
            "      maxDD%     calmar     TIM%     gross%   turn/yr  costshare%")
    print(head)
    for row in rows:
        print(f"{row['arm']:<4} {row['window']:<6} "
              f"{_fmt(row['total_return_pct'], 9, 2)} {_fmt(row['ann_vol_pct'], 10, 2)}"
              f" {_fmt(row['sharpe'], 10, 3)} {_fmt(row['sortino'], 10, 3)}"
              f" {_fmt(row['max_dd_pct'], 10, 2)} {_fmt(row['calmar'], 10, 3)}"
              f" {_fmt(row['time_in_market_pct'], 8, 2)}"
              f" {_fmt(row['avg_gross_exposure_pct'], 9, 2)}"
              f" {_fmt(row['turnover_x_per_year'], 9, 2)}"
              f" {_fmt(row['cost_share_of_gross_pct'], 11, 2)}")
    print("\n=== matched risk (each arm's daily returns scaled to H's realised vol) ===")
    print("arm  window         k   compounded%   linear%   H_tot_ret%  "
          "retention%   dsr(trials)")
    for row in rows:
        mr = row["matched_to_H"]
        retention = (None if not row["H_total_return_pct"]
                     else mr["compounded_pct"] / row["H_total_return_pct"] * 100.0)
        flag = "  <- k>5x, outside the cash model" if mr.get("k_extrapolation") else ""
        print(f"{row['arm']:<4} {row['window']:<6} {_fmt(mr['k'], 9, 3)}"
              f" {_fmt(mr['compounded_pct'], 12, 2)} {_fmt(mr['linear_pct'], 9, 2)}"
              f" {_fmt(row['H_total_return_pct'], 11, 2)}"
              f" {_fmt(retention, 10, 1)}"
              f"   {_fmt((row.get('dsr') or {}).get('dsr'), 6, 4)} ({row['n_trials']}){flag}")


def verdict(rows: list[dict]) -> dict:
    """Apply the pre-stated threshold to the **primary** arm V.

    ``T2`` is stated as "V keeps >= 80 % of H's return at matched risk".  A ratio
    only expresses "loses more / less" while ``H > 0``; when ``H <= 0`` the
    is-direction of the test is "V must not do WORSE than H" (a V that loses 4.9 %
    where H loses 10.7 % has lost *less*, which is the condition being tested, not
    a failure).  The edge case is spelled out here rather than left to a division
    by a negative number; it does not change this run's verdict either way
    (``T1`` and ``T3`` both fail independently).
    """
    by_window: dict[str, dict] = {}
    for row in rows:
        by_window.setdefault(row["window"], {})[row["arm"]] = row
    t1, t2, t3 = [], [], []
    detail = []
    retention_floor = THRESHOLD["T2_matched_risk_return_retention"]
    for window, arms in sorted(by_window.items()):
        hold, volt = arms.get("H"), arms.get("V")
        if hold is None or volt is None:
            continue
        calmar_h, calmar_v = hold.get("calmar"), volt.get("calmar")
        t1.append(bool(calmar_h is not None and calmar_v is not None
                       and calmar_v > calmar_h))
        matched_v = volt["matched_to_H"]["compounded_pct"]
        hold_total = hold.get("total_return_pct")
        retention = (matched_v / hold_total * 100.0) if hold_total else None
        if hold_total is None:
            t2.append(False)
        elif hold_total > 0.0:
            t2.append(bool(matched_v >= retention_floor * hold_total))
        else:
            t2.append(bool(matched_v >= hold_total))
        t3.append(bool(_finite((volt.get("dsr") or {}).get("dsr")) > 0.0))
        detail.append({"window": window, "calmar_H": calmar_h, "calmar_V": calmar_v,
                       "matched_V_pct": matched_v, "H_total_return_pct": hold_total,
                       "retention_pct": (None if retention is None
                                         else round(retention, 2)),
                       "dsr_V": (volt.get("dsr") or {}).get("dsr")})
    passes = {
        "T1": all(t1) if t1 else False,
        "T2": all(t2) if t2 else False,
        "T3": sum(1 for flag in t3 if flag) >= THRESHOLD["T3_dsr_positive_windows"],
    }
    return {
        "threshold": THRESHOLD,
        "per_window": {"T1_calmar": t1, "T2_retention": t2, "T3_dsr": t3},
        "detail": detail,
        "passes": passes,
        "recommend_enable": all(passes.values()),
        "recommendation": ("recommend enabling risk.vol_targeting (operator decides)"
                           if all(passes.values())
                           else "leave risk.vol_targeting.enabled: false"),
    }


# ── main ───────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--windows", default=",".join(label for label, _, _ in WINDOWS),
                        help="comma-separated labels from W0,W1,W2,W3")
    parser.add_argument("--grid", choices=("full", "none"), default="full",
                        help="'none' skips the 24-point sensitivity sweep")
    parser.add_argument("--no-arm-s", action="store_true",
                        help="skip the champion reference arm (H/V only, fast)")
    parser.add_argument("--arm-s-timeframes", default=",".join(ARM_S_TIMEFRAMES),
                        help="timeframes arm S is evaluated on; 'native' keeps the "
                             "champions' own (slow: 1m/5m caches)")
    parser.add_argument("--no-leverage-variant", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    config_hash_before = sha256_16(ROOT / "config" / "config.yaml")

    symbols, basket_meta = basket_symbols()
    config, _engine = engine_stack()
    costs = Costs(config, symbols)

    requested = {label.strip().upper() for label in args.windows.split(",") if label.strip()}
    windows = [w for w in WINDOWS if w[0] in requested]

    setups = [] if args.no_arm_s else champion_setups()
    if args.arm_s_timeframes.strip().lower() == "native":
        arm_s_timeframes = None            # keep each champion's own frames
    else:
        arm_s_timeframes = tuple(tf.strip() for tf in args.arm_s_timeframes.split(",")
                                 if tf.strip())

    print("=" * 78)
    print("P8 — does volatility targeting improve a beta-harvesting basket?")
    print("=" * 78)
    print(f"basket ({len(symbols)}): {' '.join(symbols)}")
    print(f"  rule: {basket_meta['basket_rule']}")
    print(f"  cache span: {basket_meta['span'][0]} .. {basket_meta['span'][1]}")
    for symbol, reason in sorted(basket_meta["excluded"].items()):
        print(f"  excluded {symbol}: {reason}")
    print(f"costs: taker_fee={costs.fee_pct}%/side, spread/2 per side, sources="
          f"{sorted(set(costs.sources.values()))}")
    print("  per-symbol spread%: " + ", ".join(
        f"{sym}={costs.spreads[sym]}" for sym in symbols))
    print(f"arms: H (equal-weight drifting hold) | Hreb (diagnostic: same basket "
          f"rebalanced daily) | V (H's book, exposure scaled to target vol, defaults)"
          f"{'' if args.no_leverage_variant else ' | V<=1 (max_scale=1.0)'}"
          f" | Vleg (per-leg vol_scale, equal-risk reweighting)"
          f"{'' if args.no_arm_s else ' | S (champions, reference)'}")
    print(f"arm V driven by: core.ml.volatility.forecast_vol + "
          f"core.risk.position_sizer.PositionSizer.vol_scale "
          f"(in-memory VolTargetingConfig copy; config.yaml is never written)")
    print("PRE-STATED THRESHOLD (fixed before any number below):")
    print(f"  T1 Calmar(V) > Calmar(H) in EVERY window")
    print(f"  T2 matched-risk V return >= {THRESHOLD['T2_matched_risk_return_retention']:.2f}"
          f" x H return in EVERY window")
    print(f"  T3 DSR(V) > 0 in >= {THRESHOLD['T3_dsr_positive_windows']} of the windows")
    print(f"  recommended iff T1 and T2 and T3; otherwise leave the switch off")

    rows: list[dict] = []
    per_window_meta: list[dict] = []
    grid_rows: list[dict] = []
    arm_s_detail: dict = {}
    probe = Overlay()          # the documented default — the primary arm V
    #: Honest trial count for the whole V family: the one pre-registered default
    #: setting **plus every grid point actually swept**.
    n_trials_v = 1 + (len(GRID_ESTIMATORS) * len(GRID_TARGETS) * len(GRID_WINDOWS)
                      if args.grid == "full" else 0)

    for label, start, end in windows:
        panel = load_panel(symbols, start, end)
        closes = closes_frame(panel)
        warm = closes.loc[closes.index < pd.Timestamp(start)]
        window_closes = closes.loc[closes.index >= pd.Timestamp(start)]
        warm = warm.iloc[-WARMUP_HOURS:]
        frames = {sym: panel[sym].loc[panel[sym].index < pd.Timestamp(end)]
                  for sym in symbols}
        data = {"prices": window_closes, "frames": frames}

        hold = simulate_hold(window_closes, costs)
        volat = simulate_vol_target(window_closes, frames, probe, costs)
        arms = {"H": hold, "V": volat}
        if not args.no_leverage_variant:
            capped = Overlay(method=probe.method, target_vol_pct=probe.target_vol_pct,
                             window=probe.window, min_scale=probe.min_scale,
                             max_scale=1.0)
            arms["V<=1"] = simulate_vol_target(window_closes, frames, capped, costs)
        arms["Vleg"] = simulate_vol_target(window_closes, frames, probe, costs,
                                           per_leg=True)
        # Diagnostic: the same book rebuilt to **equal notional** every UTC day
        # with the overlay pinned at scale 1.0 — isolates the rebalancing rule.
        arms["Hreb"] = simulate_vol_target(
            window_closes, frames,
            _UnitOverlay(target_vol_pct=probe.target_vol_pct), costs,
            rebalance_to_equal=True)

        baseline = basket_levels(window_closes)
        regime = {
            "window": label, "start": start, "end": end,
            "bars": int(len(window_closes)),
            "basket_total_return_pct": round(float(baseline.iloc[-1] - 1.0) * 100.0, 4),
            "basket_ann_vol_pct": round(
                float(baseline.pct_change().dropna().std() * math.sqrt(8760) * 100.0), 4),
            "warmup_bars": int(len(warm)),
            # Per-symbol dispersion: it is what makes the *rebalancing* rule, not
            # the overlay, the dominant term in some windows (W1's ZECUSDT).
            "per_symbol_return_pct": {
                sym: round(float(window_closes[sym].iloc[-1]
                                 / window_closes[sym].iloc[0] - 1.0) * 100.0, 4)
                for sym in symbols},
        }

        computed: dict[str, dict] = {}
        for name, result in arms.items():
            computed[name] = metrics(result["equity"], result["exposure"],
                                     result["tim"], result["costs"],
                                     result["turnover_notional"], result["rebalances"])

        if setups:
            arm_s = run_arm_s(symbols, start, end, setups, arm_s_timeframes or None)
            if "error" not in arm_s:
                computed["S"] = metrics(arm_s["equity"], arm_s["exposure"],
                                        arm_s["tim"], arm_s["costs"],
                                        arm_s["turnover_notional"], arm_s["rebalances"])
                computed["S"]["trades_total"] = arm_s["trades_total"]
                arm_s_detail[label] = arm_s["per_champion"]

        # Arm S has no per-bar exposure series; time-in-market and turnover come
        # from the engine's own trades, so they are reported as None rather than 0.
        if "S" in computed:
            computed["S"]["time_in_market_pct"] = None
            computed["S"]["avg_gross_exposure_pct"] = None
            computed["S"]["max_gross_exposure_pct"] = None
            computed["S"]["turnover_x_per_year"] = None

        # Arm S is deflated by the champions' own published GA trial counts.  The
        # four newest carry a cumulative chain (prior_trials + n_trials), so the
        # chain's own total is its last value — ``max`` — not the sum.  The three
        # oldest artifacts carry no provenance at all, so this is a **lower bound**
        # on how many genomes were really tried before these were published.
        n_trials_s = max((int(s["provenance"].get("n_trials") or 0)
                          for s in setups), default=0) or 1

        for name in ARM_ORDER:
            if name not in computed:
                continue
            arm = computed[name]
            if name == "S":
                arm["matched_to_H"] = matched_risk(arm, computed["H"])
                arm["n_trials"] = n_trials_s
            else:
                arm["matched_to_H"] = matched_risk(arm, computed["H"])
                arm["n_trials"] = n_trials_v if name.startswith("V") else 1
            arm["dsr"] = dsr_for(arm, arm["n_trials"])
            arm["arm"] = name
            arm["window"] = label
            arm["H_total_return_pct"] = computed["H"]["total_return_pct"]
            rows.append(arm)

        if args.grid == "full":
            grid_rows.extend([dict(row, window=label)
                              for row in run_grid(data, costs, start, end)])

        per_window_meta.append({**regime, "arm_s": arm_s_detail.get(label)})
        dispersion = " ".join(f"{sym[:3]}={value:+.0f}"
                              for sym, value in regime["per_symbol_return_pct"].items())
        print(f"[{label}] {start}..{end} bars={regime['bars']} "
              f"warmup={regime['warmup_bars']} "
              f"basket(rebalanced)={regime['basket_total_return_pct']:+.2f}% "
              f"basket_vol={regime['basket_ann_vol_pct']:.1f}% ann\n"
              f"      per-symbol return% (H's drifting book rides winners; the "
              f"rebalanced index trims them): {dispersion}", flush=True)

    # ── strip the internal return arrays before printing / serialising ──
    for row in rows:
        row.pop("_daily_returns", None)
        row.pop("_matched_returns", None)
    print_table(rows)
    result_verdict = verdict(rows)

    print("\n=== trial accounting ===")
    print(f"arm V primary setting: method=ewma lam=0.94 window=500 target=0.45 "
          f"min_scale=0.25 max_scale=2.0  (one pre-registered trial)")
    print(f"sensitivity grid: {len(GRID_ESTIMATORS)} estimators x {len(GRID_TARGETS)} "
          f"targets x {len(GRID_WINDOWS)} windows = "
          f"{len(GRID_ESTIMATORS)*len(GRID_TARGETS)*len(GRID_WINDOWS)} points "
          f"({'swept' if args.grid == 'full' else 'SKIPPED'})")
    print(f"-> n_trials used for every V-family DSR: {n_trials_v}"
          f"{' (n_trials=1 leaves deflated_sharpe_ratio undefined -> 0.0)' if n_trials_v <= 1 else ''}")
    if setups:
        print(f"arm S deflated by the champions' own published GA trials: "
              f"{max((int(s['provenance'].get('n_trials') or 0) for s in setups), default=0) or 1}"
              f" (the newest champion's cumulative chain; a LOWER bound — three "
              f"shipped artifacts carry no provenance)")
    print("note: the 6-arm menu and the H/Hreb/V/V<=1/Vleg choice are themselves "
          "selections; deflating for them too would only lower every DSR below")
    print("garch11 excluded from the grid: measured "
          f"{_garch_seconds():.3f}s per call x 600 bars x 9 symbol-windows ≈ "
          f"{_garch_seconds()*600*9/60:.1f} min just for the forecasts — outside the "
          f"bounded-minutes budget (the repo documents it as research-only)")

    if grid_rows:
        print("\n=== sensitivity grid (arm V family, same book as H) ===")
        print(f"points={len(grid_rows)}  per window, best by Calmar:")
        for label in sorted({row["window"] for row in grid_rows}):
            rows_w = [row for row in grid_rows if row["window"] == label]
            best = max(rows_w, key=lambda r: _finite(r["calmar"]))
            default = [r for r in rows_w if r["method"] == "ewma"
                       and r["target_vol_pct"] == 0.45 and r["lookback"] == 500]
            print(f"  {label}: best method={best['method']:<22} "
                  f"target={best['target_vol_pct']} lookback={best['lookback']} "
                  f"calmar={best['calmar']} tot={best['total_return_pct']}% "
                  f"maxDD={best['max_dd_pct']}%   |   default(ewma/0.45/500) "
                  f"calmar={default[0]['calmar'] if default else 'n/a'} "
                  f"sharpe={default[0]['sharpe'] if default else 'n/a'}")

    print("\n=== verdict (pre-stated threshold) ===")
    print(json.dumps(result_verdict, indent=2))
    print(f"\nRECOMMENDATION: {result_verdict['recommendation']}")

    config_hash_after = sha256_16(ROOT / "config" / "config.yaml")
    strategy_hashes = {s["name"]: sha256_16(Path(s["path"])) for s in setups}
    hashes = {
        "config/config.yaml": {"before": config_hash_before, "after": config_hash_after,
                               "unchanged": config_hash_before == config_hash_after},
        "strategies": strategy_hashes,
        "tools/p8_beta_harvest_measure.py": sha256_16(Path(__file__)),
    }
    print("\n=== hashes (sha256_16) ===")
    print(json.dumps(hashes, indent=2))

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - started, 1),
        "basket": {"symbols": symbols, **basket_meta},
        "costs": {"taker_fee_pct": costs.fee_pct, "spreads_pct": costs.spreads,
                  "spread_sources": costs.sources},
        "windows": per_window_meta,
        "rows": rows,
        "grid": grid_rows,
        "grid_size": (len(GRID_ESTIMATORS) * len(GRID_TARGETS) * len(GRID_WINDOWS)
                      if args.grid == "full" else 0),
        "verdict": result_verdict,
        "overlay_source": ["core.ml.volatility.forecast_vol",
                           "core.risk.position_sizer.PositionSizer.vol_scale"],
        "hashes": hashes,
        "arm_s_timeframes": list(arm_s_timeframes) if arm_s_timeframes else "native",
        "arm_s_detail": arm_s_detail,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({round(time.time() - started, 1)}s)")
    return 0


_GARCH_CACHE: list[float] = []


def _garch_seconds() -> float:
    """Measured cost of one ``garch11`` forecast (once per process)."""
    if _GARCH_CACHE:
        return _GARCH_CACHE[0]
    from core.ml.volatility import forecast_vol

    close = pd.read_parquet(market_dir() / "BTCUSDT" / f"{INTERVAL}.parquet")["close"]
    frame = pd.DataFrame({"close": close.iloc[-600:]})
    t0 = time.time()
    forecast_vol(frame, method="garch11", window=500, interval=INTERVAL)
    _GARCH_CACHE.append(max(time.time() - t0, 1e-6))
    return _GARCH_CACHE[0]


if __name__ == "__main__":
    raise SystemExit(main())
