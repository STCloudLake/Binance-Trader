"""P7-S4 — the **composite** (orchestrator + sub-strategies) out-of-sample contract.

WHY THIS MODULE EXISTS
----------------------
P7-S1 and P7-S3 measured *single* strategies and an *upper layer* separately, and
both were negative (``docs/overhaul/P7_REGIME_EVIDENCE.md`` §5/§5c: out-of-sample
``dsr > 0`` in **0 of 11** unconditioned and **0 of 55** conditioned cells; the
orchestrator's bounded preview lowered drawdown 2.51 % → 2.26 % but also return
−1.76 % → −2.00 % because it refused exposure worth **+24.30 USDT** net).  S4 is
the defining stage: evaluate "upper layer + state-specialised sub-strategies" as
**one** strategy, with controls and matched benchmarks, and report the real
numbers even though the expected verdict is negative
(``P7_REGIME_PLAN.md`` §3 S4).

THE CONTRACT (everything here is a *rule*, not a fitted parameter)
------------------------------------------------------------------

**Weighting rule (fixed on the training window).**  For each selected
sub-strategy ``s``::

    w_s = clip( mean(amount_usdt over s's TRAIN-window trades) / initial_balance,
                0, 1 )

— the *deployed-capital share*, the same relative exposure
:func:`core.ga.benchmark.build_benchmark` already measures from a strategy's own
fills (``weighting='capital'``).  The composite weights are the normalised
``w_s / Σw``, so they sum to 1 and the composite is always fully allocated across
its sub-strategies; cash is what the sub-strategies themselves hold (their own
equity already includes it).  Strategies with no train trade fall back to an
equal share.  **The weights are never re-read or re-normalised on the evaluation
window** (``build_composite_spec`` takes the train trade lists only).

**Composite curve (recomputable to < 1e-9 relative error).**  On the union of the
sub-strategies' bar stamps, with a sub-strategy ``deployed`` on the bars of a
position **and on one extra bar after it closes** (the bar on which the trade's
realized cash first shows up in its own equity series — without it a child's
realized PnL would never enter the composite)::

    r_s(t)      = equity_s(t) / equity_s(t−1) − 1        (its own return)
    exposure(t) = Σ_s weight_s · deployed_s(t)
    ret(t)      = Σ_s weight_s · deployed_s(t) · r_s(t) / exposure(t)   if exposure > 0
    ret(t)      = 0                                                     otherwise
    C(t)        = C(t−1) · (1 + ret(t)),   C(0) = initial_balance

An undeployed child contributes **0 %** — it is in cash — so the idle weight keeps
its value and the composite earns nothing on a bar where no child is deployed (the
plan's "idle periods return 0 %", the same convention the exposure-matched
benchmark uses).  Wealth is never renormalised away: when every child closes, the
composite keeps everything it made.  ``ret`` is exactly the fixed-weight portfolio
return of the sub-strategies, reproducible from the children's own curves and the
deployment flags *alone* (pinned to < 1e-9 relative error).

**Matched benchmarks.**  Both are computed over the **composite's own in-market
intervals** (the union of ``[opened_at, closed_at]`` of every sub-strategy trade,
clipped to the window) and both **reuse** :mod:`core.ga.benchmark` — nothing is
forked:

* ``exposure_matched`` — ``build_benchmark('exposure_matched', ...)`` on the
  composite trade list: the equal-weight basket held only while the composite
  held a position, weighted by the same measured capital shares;
* ``buy_hold`` — :func:`buy_hold_over_intervals`, the same basket at the same
  weights held over the same intervals, but **fully invested inside them**
  (``exposure_matched`` divided by nothing — i.e. every symbol's interval return
  counted at full weight).  The plain fully-invested whole-window return is
  reported next to it, so the difference is visible.

**Trial counting (DST honesty).**  :class:`CompositeTrials` counts **every**
variant this stage evaluated — sub-strategy candidates on both windows, every
arm (always-on, orchestrated, each random draw) and every orchestrator rule set
fixed on the training window.  The composite DSR is deflated by that number via
:func:`core.ga.fitness.deflated_sharpe_ratio`, with ``observation_periods`` the
composite's own daily-return count.

**Usability bar.**  :func:`usability_verdict` refuses to call a composite usable
below **100** out-of-sample composite trades (``ml.gate_min_trades``) — the plan
requires the bar to be stated in the output, and it is deliberately *stricter*
than the GA's ``min_champion_trades: 30``.

Nothing here is enabled by default and nothing here touches the live path: the
module is pure arithmetic over trade lists and equity curves a caller already
has.
"""
from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.ga.benchmark import (
    EXPOSURE_MATCHED,
    annualized_sharpe,
    build_benchmark,
    covered_bars,
    in_market_intervals,
    intervals_return,
    max_drawdown_of_returns,
    merge_intervals,
)

__all__ = [
    "CompositeContractError", "DEFAULT_USABILITY_MIN_TRADES",
    "CompositeWeight", "CompositeSpec", "CompositeFund", "CompositeTrials",
    "deployed_shares", "normalise_weights", "build_composite_spec",
    "composite_return", "composite_fund", "position_intervals",
    "buy_hold_over_intervals", "composite_in_market_intervals",
    "composite_time_in_market_pct", "deployed_time_in_market_pct",
    "union_time_in_market_pct",
    "matched_benchmarks", "composite_metrics", "usability_verdict",
    "regime_contribution", "random_enabled_stamps", "enable_fractions",
    "daily_returns_from_equity", "EQUITY_TOLERANCE",
]

#: The plan's usability bar: ``ml.gate_min_trades`` (``config/config.yaml``).
DEFAULT_USABILITY_MIN_TRADES = 100
#: Relative-error budget of the "recompute the curve" contract (plan §3 S4).
EQUITY_TOLERANCE = 1e-9


class CompositeContractError(ValueError):
    """A composite was asked for something the contract cannot express."""


# ══════════════════════════════════════════════════════════════════════════
# The fixed weights (training window only)
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CompositeWeight:
    """One sub-strategy's fixed share of the composite, and where it came from."""

    name: str
    weight: float          # normalised: Σ over sub-strategies == 1
    deployed_share: float  # the raw measured share (mean notional / balance)
    source: str            # "capital" | "equal_share"
    symbols: tuple = ()

    def as_dict(self) -> dict:
        return {"name": str(self.name), "weight": float(self.weight),
                "deployed_share": float(self.deployed_share),
                "source": str(self.source),
                "symbols": [str(s) for s in self.symbols]}


def deployed_shares(train_trades: Mapping[str, Sequence[Mapping]],
                    initial_balance: float = 10_000.0,
                    symbols: Sequence[str] | None = None) -> dict:
    """``{strategy: (share, source, symbols)}`` from **train-window** trades.

    The share is the exposure the strategy *actually deployed* — mean
    ``amount_usdt`` per trade over ``initial_balance``, clipped to ``[0, 1]`` —
    which is the ``capital`` weighting :func:`core.ga.benchmark.build_benchmark`
    uses for each sub-strategy.  A strategy with no usable notional (or no
    trades) falls back to ``1 / len(symbols)`` of the traded symbol set, and
    ``source`` says which rule produced the number.
    """
    balance = float(initial_balance)
    if not (balance > 0):
        raise CompositeContractError(
            f"initial_balance must be > 0, got {initial_balance!r}")
    pool = [str(s) for s in (symbols or [])]
    out: dict = {}
    for name, trades in (train_trades or {}).items():
        rows = list(trades or [])
        traded: list[str] = []
        for trade in rows:
            symbol = str(trade.get("symbol") or "")
            if symbol and symbol not in traded:
                traded.append(symbol)
        notionals = []
        for trade in rows:
            try:
                amount = float(trade.get("amount_usdt"))
            except (TypeError, ValueError):
                continue
            if amount > 0 and math.isfinite(amount):
                notionals.append(amount)
        if notionals:
            share = min(max(float(np.mean(notionals)) / balance, 0.0), 1.0)
            source = "capital"
        else:
            universe = traded or pool
            share = 1.0 / max(len(universe), 1)
            source = "equal_share"
        out[str(name)] = (share, source, tuple(traded))
    return out


def normalise_weights(shares: Mapping[str, float]) -> dict:
    """``{name: share}`` → ``{name: weight}`` summing to exactly 1.

    All-zero (or negative) input ⇒ equal weights; this is a *mechanical*
    normalisation, never a re-fit on the evaluation window.
    """
    clean = {str(k): max(float(v), 0.0) for k, v in (shares or {}).items()}
    if not clean:
        raise CompositeContractError("a composite needs at least one strategy")
    total = float(sum(clean.values()))
    if total <= 0:
        even = 1.0 / len(clean)
        return {name: even for name in clean}
    return {name: value / total for name, value in clean.items()}


@dataclass(frozen=True)
class CompositeSpec:
    """A composite **frozen on the training window**: names + fixed weights.

    ``weights`` are the normalised shares (Σ = 1) and ``deployed`` the raw
    measured shares they came from — both recorded so the curve can be recomputed
    and the weighting rule audited after the fact.
    """

    weights: Mapping[str, float]
    deployed: Mapping[str, float] = field(default_factory=dict)
    sources: Mapping[str, str] = field(default_factory=dict)
    symbols: tuple = ()
    train_window: tuple | None = None
    weighting_rule: str = "capital"

    def weight_of(self, name: str) -> float:
        return float(self.weights.get(str(name), 0.0))

    @property
    def names(self) -> list:
        return sorted(self.weights)

    @property
    def exposure(self) -> float:
        """``Σ w`` — always 1.0 by construction (cash lives inside the children)."""
        return float(sum(self.weights.values()))

    def as_dict(self) -> dict:
        return {"weighting_rule": str(self.weighting_rule),
                "weights": {k: float(v) for k, v in sorted(self.weights.items())},
                "deployed_shares": {k: float(v)
                                    for k, v in sorted(self.deployed.items())},
                "weight_sources": {k: str(v)
                                   for k, v in sorted(self.sources.items())},
                "symbols": [str(s) for s in self.symbols],
                "train_window": (None if self.train_window is None
                                 else [str(self.train_window[0]),
                                       str(self.train_window[1])])}


def build_composite_spec(train_trades: Mapping[str, Sequence[Mapping]], *,
                         initial_balance: float = 10_000.0,
                         symbols: Sequence[str] | None = None,
                         train_window: tuple | None = None) -> CompositeSpec:
    """The fixed-weight composite contract, built from **train-window** trades."""
    raw = deployed_shares(train_trades, initial_balance, symbols)
    if not raw:
        raise CompositeContractError(
            "no sub-strategy trades on the training window — a composite cannot "
            "be weighted")
    shares = {name: entry[0] for name, entry in raw.items()}
    weights = normalise_weights(shares)
    deployed = {name: entry[0] for name, entry in raw.items()}
    sources = {name: entry[1] for name, entry in raw.items()}
    traded: list[str] = []
    for _share, _source, symbols_for in raw.values():
        for symbol in symbols_for:
            if symbol not in traded:
                traded.append(symbol)
    for symbol in (symbols or []):
        if str(symbol) not in traded:
            traded.append(str(symbol))
    return CompositeSpec(weights=weights, deployed=deployed, sources=sources,
                         symbols=tuple(traded), train_window=train_window,
                         weighting_rule="capital")


# ══════════════════════════════════════════════════════════════════════════
# Positions, the curve, and its recomputation
# ══════════════════════════════════════════════════════════════════════════

def position_intervals(trades: Sequence[Mapping]) -> list:
    """Merged ``[opened_at, closed_at]`` union of one strategy's trades.

    The same reconstruction :func:`core.ga.benchmark.in_market_intervals` uses,
    with the window bounds left open (the caller clips).  A trade still open at
    the end has no ``closed_at`` and is dropped here rather than invented.
    """
    pairs = []
    for trade in trades or []:
        opened = _stamp(trade.get("opened_at"))
        closed = _stamp(trade.get("closed_at"))
        if opened is None or closed is None or closed < opened:
            continue
        pairs.append((opened, closed))
    return merge_intervals(pairs)


def _stamp(value):
    if value is None or value == "":
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    return None if pd.isna(ts) else ts


def _positions_at(intervals, stamp) -> bool:
    for start, end in intervals or ():
        if start <= stamp <= end:
            return True
    return False


def _deployed_at(intervals, grid, stamp) -> bool:
    """``True`` while *stamp* is a deployment bar for this child.

    A deployment bar is any bar of a position, **plus the first bar of the grid
    strictly after the position closed** — the bar on which the trade's realized
    cash first shows up in the child's own equity series.  Without that extra bar
    a child's realized PnL would never enter the composite (its equity is marked
    during the position and then the child is flat again).
    """
    for start, end in intervals or ():
        if start <= stamp <= end:
            return True
        if stamp > end and _is_first_after(grid, stamp, end):
            return True
    return False


def _is_first_after(grid, stamp, boundary) -> bool:
    """``True`` when no grid stamp lies strictly between *boundary* and *stamp*."""
    cut = bisect.bisect_right(grid, boundary)
    return cut < len(grid) and grid[cut] == stamp


@dataclass(frozen=True)
class CompositeFund:
    """The composite funding curve plus everything needed to recompute it."""

    spec: CompositeSpec
    initial_balance: float
    equity_curve: list           # [{"time", "equity"}] on the union stamp grid
    allocations: dict            # {strategy: share of the composite at t} = w_s
    deployed: dict               # {strategy: [0.0/1.0 per point]} (incl. cash bar)
    exposure: list               # [Σ w_s · deployed_s(t)]
    stamps: list
    positions: dict = field(default_factory=dict)   # {strategy: merged intervals}
    in_market: list = field(default_factory=list)   # [0.0/1.0] any child holding

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve[-1]["equity"]) if self.equity_curve \
            else float(self.initial_balance)

    @property
    def total_return_pct(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return (self.final_equity - self.initial_balance) / self.initial_balance * 100.0

    def as_dict(self, head: int = 0) -> dict:
        curve = self.equity_curve if not head else self.equity_curve[:head]
        return {"spec": self.spec.as_dict(),
                "initial_balance": float(self.initial_balance),
                "points": len(self.equity_curve),
                "final_equity": round(self.final_equity, 6),
                "total_return_pct": round(self.total_return_pct, 6),
                "deployed_time_share": round(
                    float(np.mean(self.exposure)) if self.exposure else 0.0, 6),
                "equity_curve_head": curve}


def composite_return(spec: CompositeSpec, equities: Mapping[str, Sequence[Mapping]],
                     deployed: Mapping[str, Sequence[float]],
                     initial_balance: float = 10_000.0) -> list:
    """The composite's per-bar returns — the **recomputation** half of the contract.

    ``ret(t) = Σ_s w_s · flag_s(t) · r_s(t) / Σ_s w_s · flag_s(t)`` where ``r_s``
    is the child's own bar return (``0`` for an undeployed child, i.e. cash), and
    the series is empty for the first bar (nothing to compare against).  Feeding
    these through ``C(t) = C(t−1)·(1 + ret(t))`` reproduces
    :func:`composite_fund`'s curve point for point (< 1e-9 relative error, pinned
    by ``tests/test_p7_composite.py``); ``initial_balance`` is accepted for
    signature symmetry and never scales the returns.
    """
    names = spec.names
    for name in names:
        if name not in equities:
            raise CompositeContractError(
                f"composite sub-strategy '{name}' has no equity curve")
        if name not in deployed:
            raise CompositeContractError(
                f"composite sub-strategy '{name}' has no deployment flags")
    length = min(len(equities[name]) for name in names)
    if length < 2:
        return []
    out = []
    for index in range(1, length):
        weighted = 0.0
        exposure = 0.0
        for name in names:
            weight = spec.weight_of(name)
            flag = float(deployed[name][index])
            if not weight or not flag:
                continue
            before = _equity_of(equities[name][index - 1])
            after = _equity_of(equities[name][index])
            child_return = (after / before - 1.0) if before > 0 else 0.0
            weighted += weight * child_return
            exposure += weight
        out.append(weighted / exposure if exposure > 0 else 0.0)
    return out


def _equity_of(point) -> float:
    if isinstance(point, Mapping):
        value = point.get("equity")
    else:
        value = point
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


class _CurveView:
    """Forward-filled view of one sub-strategy's equity curve (bisect lookup)."""

    __slots__ = ("times", "values", "initial")

    def __init__(self, curve, initial_balance: float):
        times, values = [], []
        for point in curve or ():
            stamp = _stamp(point.get("time") if isinstance(point, Mapping) else None)
            if stamp is None:
                continue
            times.append(stamp)
            values.append(_equity_of(point))
        if not times:
            self.times, self.values, self.initial = [], [], float(initial_balance)
            return
        order = sorted(range(len(times)), key=lambda i: times[i])
        self.times = [times[i] for i in order]
        self.values = [values[i] for i in order]
        self.initial = float(self.values[0])

    def at(self, stamp):
        """``(equity, exact)`` — forward-filled value and whether it is exact."""
        if not self.times or stamp is None:
            return None, False
        cut = bisect.bisect_right(self.times, stamp)
        if cut <= 0:
            return None, False
        return float(self.values[cut - 1]), bool(self.times[cut - 1] == stamp)

    @property
    def last(self):
        return self.times[-1] if self.times else None


def composite_fund(spec: CompositeSpec,
                   trades: Mapping[str, Sequence[Mapping]],
                   equities: Mapping[str, Sequence[Mapping]], *,
                   initial_balance: float = 10_000.0,
                   stamps: Iterable | None = None,
                   clip: tuple | None = None) -> CompositeFund:
    """The composite funding curve at bar resolution — **deterministic arithmetic**.

    Every sub-strategy's equity is forward-filled onto a shared stamp grid (the
    union of the supplied stamps, or of all the sub-strategies' own curve stamps,
    clipped to *clip* when given), each sub-strategy is marked in/out of the
    market from its **own trades**, and the curve is folded with the spec's fixed
    weights.  No RNG, no clock, no re-fitting: the same inputs give byte-equal
    points.
    """
    names = spec.names
    if not names:
        raise CompositeContractError("a composite needs at least one strategy")
    views = {}
    for name in names:
        if name not in equities:
            raise CompositeContractError(
                f"composite sub-strategy '{name}' has no equity curve")
        views[name] = _CurveView(equities[name], initial_balance)
    grid = []
    if stamps is not None:
        for stamp in stamps:
            parsed = _stamp(stamp)
            if parsed is not None:
                grid.append(parsed)
    else:
        for name in names:
            grid.extend(views[name].times)
    if clip is not None:
        start, end = _stamp(clip[0]), _stamp(clip[1])
        grid = [stamp for stamp in grid
                if (start is None or stamp >= start) and (end is None or stamp <= end)]
    grid = sorted(set(grid))
    if len(grid) < 2:
        return CompositeFund(spec=spec, initial_balance=float(initial_balance),
                             equity_curve=[], allocations=dict(spec.weights),
                             deployed={name: [] for name in names}, exposure=[],
                             stamps=grid, positions={}, in_market=[])

    intervals = {name: position_intervals(trades.get(name) or []) for name in names}
    equity_curves = {name: [] for name in names}
    flags = {name: [] for name in names}
    in_market_flags: list = []
    curve = []
    exposure_series = []
    balance = float(initial_balance)
    equity = balance
    for stamp in grid:
        exposure = 0.0
        weighted = 0.0
        holding = False
        for name in names:
            weight = spec.weight_of(name)
            value, exact = views[name].at(stamp)
            if value is None:
                value = views[name].initial
            position = _positions_at(intervals[name], stamp)
            held = _deployed_at(intervals[name], grid, stamp)
            # A bar after the strategy's last equity point has no data: keep the
            # strategy flat there rather than freezing a stale position.
            last = views[name].last
            if last is not None and stamp > last:
                held = False
                position = False
            holding = holding or position
            equity_curves[name].append({"time": stamp, "equity": value})
            flags[name].append(1.0 if held else 0.0)
            before = _previous_at(views[name].values, views[name].times, stamp)
            child_return = (value / before - 1.0) if before and before > 0 else 0.0
            if weight and held:
                weighted += weight * child_return
                exposure += weight
        in_market_flags.append(1.0 if holding else 0.0)
        exposure_series.append(exposure)
        if exposure > 0:
            equity *= (1.0 + weighted / exposure)
        curve.append({"time": stamp, "equity": equity})
    return CompositeFund(spec=spec, initial_balance=balance,
                         equity_curve=curve, allocations=dict(spec.weights),
                         deployed=flags, exposure=exposure_series, stamps=grid,
                         positions=intervals, in_market=in_market_flags)


def _previous_at(values, times, stamp):
    """The sub-strategy's equity at the grid bar **before** *stamp*, or ``None``."""
    if not times:
        return None
    cut = bisect.bisect_left(times, stamp)
    if cut <= 0:
        return None
    return float(values[cut - 1])


# ══════════════════════════════════════════════════════════════════════════
# Matched benchmarks (the composite's own intervals; core.ga.benchmark reused)
# ══════════════════════════════════════════════════════════════════════════

def composite_in_market_intervals(trades: Mapping[str, Sequence[Mapping]],
                                  symbols: Sequence[str],
                                  window_start, window_end) -> dict:
    """``{symbol: merged intervals}`` of the **composite's** own in-market bars.

    The union over *every* sub-strategy's trades on that symbol — exactly how a
    caller of :func:`core.ga.benchmark.in_market_intervals` would see a single
    combined strategy, so the benchmark helper is reused rather than forked.
    """
    out = {}
    for symbol in symbols or []:
        merged = in_market_intervals(flatten_trades(trades), str(symbol),
                                     window_start, window_end)
        if merged:
            out[str(symbol)] = merged
    return out


def flatten_trades(trades: Mapping[str, Sequence[Mapping]] | Sequence[Mapping]) -> list:
    """``{strategy: [trades]}`` (or a plain list) → one chronological trade list."""
    if isinstance(trades, Mapping):
        rows = [trade for group in trades.values() for trade in (group or [])]
    else:
        rows = list(trades or [])
    return sorted(rows, key=lambda trade: str(trade.get("opened_at") or ""))


def composite_time_in_market_pct(trades: Mapping[str, Sequence[Mapping]],
                                 symbols: Sequence[str],
                                 window_start, window_end,
                                 stamps: Sequence | None = None) -> float | None:
    """``%`` of bars on which **any** sub-strategy held a position.

    The union-of-intervals definition: the composite is "in the market" when any
    of its children is, and overlapping children are never double-counted.  The
    denominator is *stamps* when given (the bar grid the composite curve was built
    on), else the trade open/close stamps themselves.
    """
    intervals = composite_in_market_intervals(trades, symbols, window_start,
                                              window_end)
    start, end = _stamp(window_start), _stamp(window_end)
    if stamps is not None:
        grid = [parsed for parsed in (_stamp(s) for s in stamps)
                if parsed is not None]
    else:
        grid = _trade_stamps(trades, start, end)
    if not grid:
        return 0.0
    all_intervals = [iv for merged in intervals.values() for iv in merged]
    return float(covered_bars(pd.DatetimeIndex(grid), all_intervals)
                 / float(len(grid)) * 100.0)


def _trade_stamps(trades, start, end) -> list:
    """The trade open/close stamps inside the window (the fallback grid)."""
    grid = set()
    for group in (trades or {}).values():
        for trade in group or ():
            for key in ("opened_at", "closed_at"):
                stamp = _stamp(trade.get(key))
                if stamp is None:
                    continue
                if (start is None or stamp >= start) and (end is None or stamp <= end):
                    grid.add(stamp)
    return sorted(grid)


def deployed_time_in_market_pct(fund: CompositeFund) -> float:
    """``%`` of the composite's bar grid with non-zero exposure (mean of ``Σw·flag``).

    This is the composite's own exposure: bars on which at least one child is
    deployed (a position bar **or** the cash-settle bar after a close).  It is
    reported next to the union-of-positions share
    (:func:`composite_time_in_market_pct`) so the one-bar accounting difference is
    visible rather than hidden in a single number.
    """
    if not fund.exposure:
        return 0.0
    return float(np.mean(fund.exposure) * 100.0)


def union_time_in_market_pct(fund: CompositeFund) -> float:
    """``%`` of the grid on which any child held a **position** (no cash-settle bar)."""
    if not fund.stamps or not fund.in_market:
        return 0.0
    return float(np.mean(fund.in_market) * 100.0)


def buy_hold_over_intervals(frames: Mapping[str, pd.DataFrame],
                            weights: Mapping[str, float],
                            intervals: Mapping[str, Sequence],
                            window_start, window_end) -> dict:
    """Fully-invested buy & hold of the composite's basket **inside its intervals**.

    :func:`core.ga.benchmark.intervals_return` supplies each symbol's return over
    its merged in-market intervals; here the weights are used **raw** (not scaled
    by any deployment share), so the basket is held at full weight for exactly the
    bars the composite held *something*.  With equal weights this is the plain
    "hold the basket whenever the composite was in the market" benchmark the plan
    asks for; the per-symbol returns are reported so the number can be checked.
    """
    usable = {}
    per_symbol = {}
    for symbol, weight in (weights or {}).items():
        intervals_for = (intervals or {}).get(symbol) or []
        frame = (frames or {}).get(symbol)
        if frame is None or len(frame) == 0 or not intervals_for:
            continue
        start, end = _stamp(window_start), _stamp(window_end)
        window = frame
        if start is not None:
            window = window[window.index >= start]
        if end is not None:
            window = window[window.index <= end]
        if len(window) < 2:
            continue
        ret = intervals_return(window["close"].astype(float), intervals_for)
        if ret is None:
            continue
        usable[str(symbol)] = float(weight)
        per_symbol[str(symbol)] = float(ret)
    total = float(sum(usable.values()))
    if total <= 0:
        return {"benchmark_pct": None, "benchmark_available": False,
                "symbol_weights": {}, "per_symbol_return": {},
                "notes": "no usable in-market bars for the composite's symbols"}
    return {
        "benchmark_pct": float(sum(usable[s] * per_symbol[s] for s in usable)
                               / total * 100.0),
        "benchmark_available": True,
        "symbol_weights": {s: float(usable[s] / total) for s in sorted(usable)},
        "per_symbol_return": {s: float(per_symbol[s]) for s in sorted(per_symbol)},
        "notes": ("fully-invested equal-weight buy & hold of the composite's own "
                  "basket, held only inside the composite's in-market intervals "
                  "(same intervals as exposure_matched, no exposure scaling)"),
    }


def matched_benchmarks(composite_trades: Sequence[Mapping],
                       spec: CompositeSpec,
                       frames: Mapping[str, pd.DataFrame], *,
                       window_start, window_end,
                       initial_balance: float = 10_000.0,
                       buy_hold_pct=None) -> dict:
    """The two benchmarks the plan requires, both over the composite's intervals.

    ``exposure_matched`` is :func:`core.ga.benchmark.build_benchmark` **reused**
    (imported, never forked) on the composite's flattened trade list; ``buy_hold``
    is :func:`buy_hold_over_intervals` on the same intervals and weights.  The
    engine's whole-window fully-invested number travels along as
    ``window_buy_hold_pct`` so the three are distinguishable.
    """
    rows = list(composite_trades or [])
    symbols = [s for s in (spec.symbols or ())
               if s in {str(t.get("symbol")) for t in rows}]
    if not symbols:
        symbols = list(spec.symbols or ())
    report = build_benchmark(EXPOSURE_MATCHED, trades=rows, symbols=symbols,
                             frames=frames, window_start=window_start,
                             window_end=window_end,
                             initial_balance=initial_balance)
    intervals = composite_in_market_intervals({"_": rows}, symbols, window_start,
                                              window_end)
    weights = {s: float(spec.deployed.get(s, 0.0)) for s in symbols}
    if not any(w > 0 for w in weights.values()):
        weights = {s: 1.0 / max(len(symbols), 1) for s in symbols}
    hold = buy_hold_over_intervals(frames, weights, intervals, window_start,
                                   window_end)
    return {
        "exposure_matched": report,
        "buy_hold_matched": hold,
        "window_buy_hold_pct": (None if buy_hold_pct is None
                                else float(buy_hold_pct)),
        "in_market_intervals": {
            s: [[str(a), str(b)] for a, b in merged]
            for s, merged in sorted(intervals.items())},
    }


# ══════════════════════════════════════════════════════════════════════════
# Metrics, trials and the usability bar
# ══════════════════════════════════════════════════════════════════════════

def daily_returns_from_equity(equity_curve: Sequence[Mapping]) -> pd.Series:
    """Daily returns of a composite curve — the ``core.ga.benchmark`` construction.

    Same resample (``1D`` last → ``pct_change`` → dropna) the benchmark and the
    fitness statistics use, so the composite's Sharpe and DSR are on the same
    footing as every other number in this repository.
    """
    if not equity_curve or len(equity_curve) < 2:
        return pd.Series(dtype=float)
    series = pd.Series(
        [float(p["equity"]) for p in equity_curve],
        index=pd.DatetimeIndex([pd.Timestamp(p["time"]) for p in equity_curve]))
    daily = series.resample("1D").last().dropna()
    if len(daily) < 2:
        return pd.Series(dtype=float)
    return daily.pct_change().dropna()


@dataclass(frozen=True)
class CompositeTrials:
    """Every variant/trial this evaluation spent, so the DSR can be deflated.

    ``total`` is exactly the sum of the parts — the number the composite DSR is
    deflated by.  Nothing is silently omitted: the random controls count, the
    orchestrator rule sets fixed on the training window count, and the
    sub-strategy candidates count on **both** windows (the train-window ranking
    is a selection too).
    """

    sub_strategy_candidates: int = 0
    windows: int = 1
    arm_variants: int = 0
    orchestrator_configs: int = 0
    notes: str = ""

    @property
    def total(self) -> int:
        return int(max(
            int(self.sub_strategy_candidates) * int(self.windows)
            + int(self.arm_variants)
            + int(self.orchestrator_configs), 1))

    def as_dict(self) -> dict:
        return {"sub_strategy_candidates": int(self.sub_strategy_candidates),
                "windows": int(self.windows),
                "arm_variants": int(self.arm_variants),
                "orchestrator_configs": int(self.orchestrator_configs),
                "total": self.total,
                "notes": str(self.notes)}


def composite_metrics(fund: CompositeFund,
                      composite_trades: Sequence[Mapping], *,
                      trials: CompositeTrials,
                      window_start, window_end,
                      initial_balance: float = 10_000.0,
                      benchmarks: Mapping | None = None,
                      min_trades: int = DEFAULT_USABILITY_MIN_TRADES) -> dict:
    """Trades / return / max DD / Sharpe / time-in-market / DSR for one variant.

    The DSR is :func:`core.ga.fitness.deflated_sharpe_ratio` — one implementation
    for the whole repository — deflated by ``trials.total`` and estimated on the
    composite's own daily-return count.  ``alpha_vs_exposure_matched_pct`` and
    ``alpha_vs_buy_hold_pct`` are in **percentage points of the window's return**
    (not annualised), like every other alpha in this project.
    """
    from core.ga.fitness import MIN_OBSERVATIONS, deflated_sharpe_ratio

    rows = list(composite_trades or [])
    curve = list(fund.equity_curve or [])
    returns = daily_returns_from_equity(curve)
    total_return = fund.total_return_pct
    sharpe = annualized_sharpe(returns.values)
    max_dd = max_drawdown_of_returns(returns.values) if len(returns) else 0.0
    if not len(returns):
        max_dd = _curve_max_dd_pct(curve)
    skew = float(returns.skew()) if len(returns) > 2 else 0.0
    kurtosis = float(returns.kurtosis() + 3.0) if len(returns) > 3 else 3.0
    dsr = deflated_sharpe_ratio(
        sharpe, trials.total, observation_periods=int(len(returns)),
        sharpe_is_annualized=True, skew=skew, kurtosis=kurtosis)

    bench = dict(benchmarks or {})
    matched = bench.get("exposure_matched") or {}
    hold = bench.get("buy_hold_matched") or {}
    bench_pct = matched.get("benchmark_pct")
    alpha = (None if bench_pct is None else total_return - float(bench_pct))
    hold_pct = hold.get("benchmark_pct")
    alpha_hold = (None if hold_pct is None else total_return - float(hold_pct))
    verdict = usability_verdict(int(len(rows)), float(dsr["dsr"]),
                                min_trades=min_trades)
    observations = int(len(returns))
    # Honest DSR semantics: ``deflated_sharpe_ratio`` is DEFINED as 0.0 when the
    # observed Sharpe is <= 0 or there are fewer than MIN_OBSERVATIONS of them —
    # that is "not estimated", not "estimated at zero".  The flag keeps the two
    # apart in the table (the P7 evidence file already paid for this distinction).
    dsr_estimated = bool(float(sharpe) > 0.0 and observations >= MIN_OBSERVATIONS)
    return {
        "trades": int(len(rows)),
        "total_return_pct": round(float(total_return), 4),
        "final_equity": round(float(fund.final_equity), 4),
        "max_drawdown_pct": round(float(max_dd), 4),
        "sharpe": round(float(sharpe), 4),
        "sharpe_per_period": round(
            float(dsr.get("sharpe_per_period") or 0.0), 6),
        "observations": observations,
        "time_in_market_pct": round(
            float(_finite_or(composite_time_in_market_pct(
                {"_": rows}, fund.spec.symbols, window_start, window_end,
                stamps=fund.stamps), 0.0)), 4),
        "deployed_time_in_market_pct": round(
            deployed_time_in_market_pct(fund), 4),
        "union_time_in_market_pct": round(
            union_time_in_market_pct(fund), 4),
        "exposure_matched_pct": (None if bench_pct is None
                                 else round(float(bench_pct), 4)),
        "alpha_vs_exposure_matched_pct": (None if alpha is None
                                          else round(float(alpha), 4)),
        "buy_hold_matched_pct": (None if hold_pct is None
                                 else round(float(hold_pct), 4)),
        "alpha_vs_buy_hold_pct": (None if alpha_hold is None
                                  else round(float(alpha_hold), 4)),
        "window_buy_hold_pct": (None if bench.get("window_buy_hold_pct") is None
                                else round(float(bench["window_buy_hold_pct"]), 4)),
        "benchmark_weighting": matched.get("weighting"),
        "benchmark_symbol_weights": matched.get("symbol_weights") or {},
        "benchmark_available": bool(matched.get("benchmark_available")),
        "deflated_sharpe": round(float(dsr["dsr"]), 6),
        "dsr_detail": dsr,
        "dsr_estimated": dsr_estimated,
        "dsr_note": ("estimated" if dsr_estimated else
                     "not estimated ("
                     + " and ".join(
                         reason for reason in (
                             (f"Sharpe {round(float(sharpe), 4)} <= 0"
                              if not float(sharpe) > 0.0 else ""),
                             (f"observations {observations} < {MIN_OBSERVATIONS}"
                              if observations < MIN_OBSERVATIONS else ""))
                         if reason)
                     + "); the reported 0.0 means 'not distinguishable', "
                       "not 'measured zero'"),
        "expected_max_random": dsr.get("expected_max_random"),
        "skew": round(float(skew), 4),
        "kurtosis": round(float(kurtosis), 4),
        "insufficient_data": bool(int(len(rows)) < int(min_trades)),
        "usability": verdict,
    }


def _curve_max_dd_pct(equity_curve) -> float:
    """Peak-to-trough drawdown of a composite curve (used when it has < 2 days)."""
    equities = [_equity_of(point) for point in (equity_curve or [])]
    if not equities:
        return 0.0
    peak = equities[0]
    worst = 0.0
    for value in equities:
        if value > peak:
            peak = value
        if peak > 0:
            worst = max(worst, (peak - value) / peak * 100.0)
    return float(worst)


def _finite_or(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def usability_verdict(trades: int, dsr: float, *,
                      min_trades: int = DEFAULT_USABILITY_MIN_TRADES) -> dict:
    """The plan's usability bar, stated: usable ⇔ ``trades ≥ 100`` **and** ``DSR > 0``.

    ``false`` means "no claim of usability is allowed", and ``reason`` names the
    failing criterion (``insufficient_trades`` / ``dsr_not_positive`` /
    ``both``) so a low-trade, high-Sharpe composite is never read as evidence.
    """
    enough = int(trades) >= int(min_trades)
    positive = float(dsr) > 0.0
    if enough and positive:
        reason = ""
    elif not enough and not positive:
        reason = "both"
    elif not enough:
        reason = "insufficient_trades"
    else:
        reason = "dsr_not_positive"
    return {"usable": bool(enough and positive), "reason": reason,
            "min_composite_trades": int(min_trades),
            "composite_trades": int(trades),
            "trades_ok": bool(enough), "dsr_ok": bool(positive),
            "statement": (f"usable requires >= {int(min_trades)} out-of-sample "
                          f"composite trades AND DSR > 0")}


# ══════════════════════════════════════════════════════════════════════════
# Timeline helpers: the orchestrator's verdicts and the random control
# ══════════════════════════════════════════════════════════════════════════

def enable_fractions(enabled: Mapping[str, Iterable[bool]]) -> dict:
    """``{strategy: fraction of stamps enabled}`` — the exposure the gate allowed."""
    out = {}
    for name, flags in (enabled or {}).items():
        values = [bool(flag) for flag in flags]
        out[str(name)] = (sum(values) / len(values)) if values else 0.0
    return out


def random_enabled_stamps(stamps: Sequence, names: Sequence[str],
                          fractions: Mapping[str, float], *,
                          seed: int) -> dict:
    """A seeded, exposure-matched random gate: ``{strategy: set(stamps)}``.

    For every strategy, each stamp is enabled independently with probability equal
    to that strategy's **orchestrated** enable fraction, so the random control
    refits nothing and holds the same average exposure as the orchestrator while
    throwing away the regime selection.  ``random.Random(seed)`` (Mersenne
    Twister, fixed seed) makes the draw reproducible across runs and machines.
    """
    rng = random.Random(int(seed))
    ordered = list(stamps)
    out: dict = {}
    for name in names:
        probability = min(max(float(fractions.get(str(name), 0.0)), 0.0), 1.0)
        out[str(name)] = {stamp for stamp in ordered if rng.random() < probability}
    return out


def _pick(names, enabled_by_stamp, stamp):
    if enabled_by_stamp is None:
        return set(names)
    entry = enabled_by_stamp.get(stamp)
    if entry is None:
        return set()
    if isinstance(entry, Mapping):
        return {name for name in names if entry.get(name)}
    return {name for name in names if name in entry}


def filter_trades_by_timeline(trades: Mapping[str, Sequence[Mapping]],
                              enabled_by_stamp: Mapping) -> dict:
    """Keep only the trades whose **entry bar** the gate enabled.

    This is the S3/S4 gating semantics: a refusal stops an *entry*, so a trade the
    gate refused simply never happens; everything else about the child strategy
    (sizing, costs, exits) is untouched.
    """
    out = {}
    for name, rows in (trades or {}).items():
        kept = []
        for trade in rows or ():
            stamp = _stamp(trade.get("opened_at"))
            allowed = (_pick([str(name)], enabled_by_stamp, stamp)
                       if enabled_by_stamp is not None else {str(name)})
            if str(name) in allowed:
                kept.append(trade)
        out[str(name)] = kept
    return out


def regime_contribution(composite_trades: Sequence[Mapping], labels: Mapping,
                        initial_balance: float = 10_000.0) -> dict:
    """Per-regime contribution of the composite's own trades.

    ``labels`` maps a timestamp to its causal label (or is callable).  A trade is
    attributed to the label of its **entry bar**; the block reports trade count,
    net PnL, the share of the composite's trades and the PnL as a percentage of
    the initial balance — a decomposition of *where* the result came from, never a
    claim about which regime is "good".
    """
    pnl_by: dict = {}
    count_by: dict = {}
    for trade in composite_trades or []:
        stamp = _stamp(trade.get("opened_at"))
        if callable(labels):
            label = labels(stamp)
        else:
            label = labels.get(stamp) if hasattr(labels, "get") else None
        label = "unknown" if label is None else str(label)
        count_by[label] = count_by.get(label, 0) + 1
        pnl_by[label] = pnl_by.get(label, 0.0) + _finite_or(trade.get("pnl"), 0.0)
    total = max(sum(count_by.values()), 1)
    balance = float(initial_balance) if initial_balance else 1.0
    return {label: {"trades": int(count_by[label]),
                    "trade_share_pct": round(count_by[label] / total * 100.0, 4),
                    "pnl": round(float(pnl_by.get(label, 0.0)), 4),
                    "pnl_pct_of_initial": round(
                        float(pnl_by.get(label, 0.0)) / balance * 100.0, 4)}
            for label in sorted(count_by)}
