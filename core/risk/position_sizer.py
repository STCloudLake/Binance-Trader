"""Volatility-targeted position sizing (Phase P3).

What this module does and why
-----------------------------
Directional prediction on returns measured as *unpredictable* in this project
(``docs/core-algorithms/08-ml-triple-barrier.md``: OOS AUC 0.52 / 0.53, negative
net-of-cost expectancy), while **conditional volatility is predictable**
(ARCH/GARCH — Engle 1982, Bollerslev 1986; Tsay, *Analysis of Financial Time
Series*, ch. 3).  The forecast is therefore spent on risk:

* :meth:`PositionSizer.vol_scale` — how much of the fixed-fraction notional to
  take, given the forecast volatility,
* :meth:`PositionSizer.stop_distance_pct` — a stop/trailing width that widens in
  turbulent regimes and tightens in calm ones,
* :meth:`PositionSizer.barrier_widths_pct` — the triple-barrier widths used by
  ``core/ml/labels.py``.

Formulas and units
------------------
``scale = clip(target_vol_pct / forecast_vol_pct, min_scale, max_scale)``, where
both volatilities are **percent of price per bar** (0.45 = 0.45 %/bar) and the
scale is dimensionless: ``1.0`` reproduces the existing fixed-fraction position
exactly.  Because ``scale ∝ 1/forecast``, a symbol whose forecast volatility
doubles gets **half** the notional — the defining property of volatility
targeting, and the one the integration test asserts.

``stop = clip(stop_vol_multiple × forecast_vol_pct, stop_min_pct, stop_max_pct)``
is also a percent of price.  The multiple is the "how many bars of typical
movement" the stop tolerates: 3.0 at the default means a stop is roughly a
3-sigma move on the forecast horizon, which is the same shape as a Chandelier
exit ``k × ATR`` (LeBeau 1995) with the ATR replaced by a variance-based
forecast.

Default safety
--------------
Every entry point short-circuits when ``risk.vol_targeting.enabled`` is false
(the shipped default) **or** when the caller has no forecast.  In that case:

* ``vol_scale`` returns ``1.0`` and the size arithmetic below consumes it, so
  the result is bit-identical to the pre-P3 fixed-fraction calculation;
* ``stop_distance_pct`` returns exactly ``soft_params.stop_loss_pct`` (floored at
  ``hard_limits.min_stop_loss_distance_pct``);
* ``barrier_widths_pct`` returns ``None`` so ``core/ml/labels.py`` keeps its
  ATR-scaled widths.

This is deliberate: P3 must not change live behaviour until an operator opts in,
and ``tests/test_volatility_targeting.py`` pins the identity.

Volume-aware capacity (P6-A)
----------------------------
The same "spend the information on risk, not direction" rule applies to volume.
:meth:`PositionSizer.apply_participation_cap` refuses or shrinks a notional that
exceeds ``risk.liquidity.max_participation_pct`` of the recent traded quote
volume (``core.risk.liquidity``), and it is called from
:meth:`calculate_position_size` **only** when ``risk.liquidity.enabled`` is true —
the shipped default is ``false``, so the sizing arithmetic is bit-identical and
``tests/test_liquidity.py::test_participation_disabled_is_bit_identical`` pins
it.  The cap can only ever *shrink*: it is applied after ``max_position_size_pct``
/ the capital-pool split / the vol-targeting ceiling, so those hard caps stay
authoritative.
"""

from __future__ import annotations

import logging
import math

from app.config import SoftRiskParams, HardRiskLimits
from core.risk import liquidity as _liquidity

_log = logging.getLogger(__name__)


def _f(value, default: float = 0.0) -> float:
    """Coerce to a finite float (a ``None``/NaN forecast must not reach the maths)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


class PositionSizer:
    def __init__(self, hard_limits: HardRiskLimits, soft_params: SoftRiskParams,
                 core_capital_pct: float = 0.7, satellite_capital_pct: float = 0.3,
                 vol_targeting=None, liquidity=None):
        self.hard = hard_limits
        self.soft = soft_params
        self.core_capital_pct = core_capital_pct
        self.satellite_capital_pct = satellite_capital_pct
        # ``None`` is the documented fallback: with no config object the sizer
        # behaves exactly like the pre-P3 class (every vol hook returns "off").
        self.vol_targeting = vol_targeting
        # P6-A: ``risk.liquidity``.  ``None`` → resolved from ``vol_targeting``'s
        # owner when that looks like a config, else the shipped block (disabled);
        # ``resolve_liquidity_config`` never raises, so a missing block behaves
        # like ``enabled: false``.
        self.liquidity = liquidity

    # ── volatility hooks ────────────────────────────────────────────────

    def vol_targeting_enabled(self) -> bool:
        """True only when a ``risk.vol_targeting`` block is present **and** enabled."""
        vt = self.vol_targeting
        return bool(vt is not None and getattr(vt, "enabled", False))

    def vol_scale(self, forecast_vol_pct: float | None) -> float:
        """``clip(target / forecast, min_scale, max_scale)``; ``1.0`` when off.

        ``forecast_vol_pct`` is a **percent of price per bar** (use
        ``core.ml.volatility.to_pct`` on a fraction).  A missing, non-finite or
        non-positive forecast — the documented "volatility unavailable" case —
        falls back to the fixed fraction (``1.0``) rather than to zero: refusing
        to size a trade is a bigger behavioural change than sizing it the old way.
        """
        vt = self.vol_targeting
        if vt is None or not getattr(vt, "enabled", False):
            return 1.0
        vol = _f(forecast_vol_pct, 0.0)
        target = _f(getattr(vt, "target_vol_pct", 0.0), 0.0)
        if vol <= 0.0 or target <= 0.0:
            return 1.0
        lo = _f(getattr(vt, "min_scale", 0.0), 0.0)
        hi = _f(getattr(vt, "max_scale", 1.0), 1.0)
        return float(min(max(target / vol, lo), hi))

    def stop_distance_pct(self, forecast_vol_pct: float | None = None,
                          volatility_expanding: bool = False) -> float:
        """Stop / trailing distance in percent, scaled by the forecast volatility.

        Pre-P3 behaviour (returned whenever vol targeting is off) is
        ``max(soft.stop_loss_pct, hard.min_stop_loss_distance_pct)`` with the
        legacy ``volatility_expanding`` ×1.3 widening kept intact — that boolean
        path is *not* changed, so existing callers see identical numbers.
        """
        base = max(_f(self.soft.stop_loss_pct) , _f(self.hard.min_stop_loss_distance_pct))
        if not self.vol_targeting_enabled():
            return base * (1.3 if volatility_expanding else 1.0)
        vol = _f(forecast_vol_pct, 0.0)
        if vol <= 0.0:
            return base * (1.3 if volatility_expanding else 1.0)
        vt = self.vol_targeting
        dist = _f(getattr(vt, "stop_vol_multiple", 0.0)) * vol
        lo = max(_f(getattr(vt, "stop_min_pct", 0.0)), _f(self.hard.min_stop_loss_distance_pct))
        hi = _f(getattr(vt, "stop_max_pct", 0.0), 0.0)
        if hi <= 0.0:
            hi = lo
        return float(min(max(dist, lo), max(hi, lo)))

    def barrier_widths_pct(self, forecast_vol_pct: float | None = None) -> tuple[float, float] | None:
        """Triple-barrier ``(upper_pct, lower_pct)`` as **fractions**, or ``None``.

        ``None`` is the "keep the ATR-scaled widths" signal consumed by
        :func:`core.ml.labels.barrier_widths` (its ``vol_pct`` parameter), so this
        hook cannot change label construction while the switch is off.

        Unit conversion is explicit because the two vocabularies differ: the
        forecast arrives in **percent per bar** (``0.45``) while ``barrier_*``
        are fractions of price (``0.004`` = 0.4 %), matching ``labels.py``.  So
        ``width = barrier_vol_multiple × forecast_pct / 100``, and the default
        multiple of 1.0 at BTC's 0.45 %/bar gives a 0.45 % barrier — i.e. roughly
        one forecast sigma, inside the ATR-scaled default's clamp.

        .. warning::
           **This hook has no production caller, so
           ``risk.vol_targeting.barrier_*`` is currently inert** (P3/P4 audit
           defect 5).  The live label path builds its widths from
           ``ml.barrier_atr_period`` / ``ml.barrier_atr_multiple`` /
           ``ml.barrier_min_pct`` / ``ml.barrier_max_pct`` via
           ``core.ml.predictor.MLPredictor._barrier_params`` and never calls
           this method, so an operator who sets ``barrier_vol_multiple`` today
           changes nothing.  Wiring it means passing the resolved forecast into
           the predictor's ``barrier_widths(vol_pct=…)`` call (that file is
           outside this change's scope).  ``tests/test_p34_audit_fixes.py::
           test_barrier_widths_hook_has_no_production_caller`` is the tripwire:
           it fails the moment a caller appears, so this comment cannot go stale
           silently.

           The three keys are therefore **reserved**, and a non-default value is
           no longer silent: ``app.config.inert_barrier_key_warnings`` makes
           ``Config.load`` log a startup WARNING naming the key and the
           ``ml.barrier_*`` keys that do take effect
           (``tests/test_p34_code_defects.py`` pins both halves).
        """
        if not self.vol_targeting_enabled():
            return None
        vol = _f(forecast_vol_pct, 0.0)
        if vol <= 0.0:
            return None
        vt = self.vol_targeting
        width = _f(getattr(vt, "barrier_vol_multiple", 1.0), 1.0) * vol / 100.0
        lo = _f(getattr(vt, "barrier_min_pct", 0.0), 0.0)
        hi = _f(getattr(vt, "barrier_max_pct", 0.0), 0.0)
        if hi <= 0.0:
            hi = lo
        w = float(min(max(width, lo), max(hi, lo)))
        return w, w

    # ── volume/liquidity capacity hooks (P6-A) ──────────────────────────

    def liquidity_config(self, symbol: str = ""):
        """Effective ``risk.liquidity`` block for ``symbol`` (``None`` = not configured).

        Resolution is delegated to :func:`core.risk.liquidity.resolve_liquidity_config`
        so a caller can pass a config, a bare ``risk.liquidity`` block, or nothing
        at all: the explicit ``liquidity=`` constructor argument wins, then a
        ``risk_liquidity`` attribute on ``vol_targeting`` (which in production is
        ``app.config.Config.risk_vol_targeting``, so the sizer finds the live
        block without any new plumbing), then the shipped defaults.
        """
        block = self.liquidity
        if block is None:
            block = _liquidity.resolve_liquidity_config(self.vol_targeting)
        if block is None:
            return None
        return _liquidity.liquidity_for_symbol(block, symbol)

    def participation_cap_enabled(self, symbol: str = "") -> bool:
        """True only when a ``risk.liquidity`` block is present **and** enabled."""
        block = self.liquidity_config(symbol)
        return bool(block is not None and getattr(block, "enabled", False))

    def apply_participation_cap(self, risk_amount: float, recent_quote_volume,
                                symbol: str = "") -> tuple[float, str]:
        """Shrink ``risk_amount`` to the participation ceiling; never grow it.

        Returns ``(capped_risk_amount, reason)``.  While
        ``risk.liquidity.enabled`` is false this is a pure pass-through that does
        **not** touch the number (``("disabled: …", risk_amount)`` returned
        unchanged), which is what makes the off-path bit-identical.  When it is
        on, the notional is capped at ``max_participation_pct`` % of the window
        and the reason (``ok`` / ``capped`` / ``no_volume``) is logged at **debug**
        level so an operator can see *why* a size was reduced without a log line
        per signal at info level.

        ``recent_quote_volume`` may be a number, a bar frame, or a callable — it
        is passed straight to :func:`core.risk.liquidity.recent_quote_volume`.
        """
        block = self.liquidity_config(symbol)
        if block is None or not getattr(block, "enabled", False):
            return risk_amount, "disabled: risk.liquidity.enabled is false"
        volume = recent_quote_volume
        if callable(volume):
            volume = _liquidity.recent_quote_volume(
                lookback_bars=int(getattr(block, "lookback_bars",
                                          _liquidity.DEFAULT_LOOKBACK_BARS)),
                provider=volume)
        elif not isinstance(volume, (int, float)) or isinstance(volume, bool):
            volume = _liquidity.recent_quote_volume(
                volume,
                lookback_bars=int(getattr(block, "lookback_bars",
                                          _liquidity.DEFAULT_LOOKBACK_BARS)))
        allowed, reason = _liquidity.cap_notional(
            risk_amount, volume,
            float(getattr(block, "max_participation_pct",
                          _liquidity.DEFAULT_MAX_PARTICIPATION_PCT)))
        # The cap may only shrink.  A mis-sized call must not be able to *raise*
        # the notional above what the hard caps already allowed.
        if allowed > risk_amount:
            return risk_amount, f"{reason} (ignored: cap would GROW the notional)"
        if allowed != risk_amount:
            _log.debug(
                "participation cap %s: notional %.4f -> %.4f (%s)",
                symbol or "-", risk_amount, allowed, reason)
        return allowed, reason

    # ── sizing ──────────────────────────────────────────────────────────

    def calculate_position_size(self, account_balance: float, current_price: float,
                                 position_type: str = "satellite",
                                 volatility_expanding: bool = False,
                                 forecast_vol_pct: float | None = None,
                                 recent_quote_volume=None,
                                 symbol: str = ""
                                 ) -> tuple[float, float]:
        """Calculate position size with optional volatility-based adjustment.

        When volatility is predicted to expand:
        - Position size reduced to 70% (tighten risk during turbulent periods)
        - This is grounded in the GARCH volatility clustering literature

        Volatility targeting (Phase P3) additionally scales the notional by
        ``target_vol / forecast_vol`` — see :meth:`vol_scale` — and is applied
        **before** the hard notional ceiling, so ``max_position_size_pct`` and the
        capital-pool split always bound the result no matter how low the forecast
        volatility is.  With no forecast (or with the switch off) the arithmetic
        below is byte-for-byte the legacy calculation.

        Volume-aware capacity (P6-A) is the **last** step: when
        ``risk.liquidity.enabled`` is true and ``recent_quote_volume`` is
        supplied, :meth:`apply_participation_cap` shrinks the notional to at most
        ``max_participation_pct`` % of the recent traded quote volume.  It can
        only shrink, and with the shipped default (``enabled: false``) the hook
        is never called at all — passing ``recent_quote_volume`` while the switch
        is off changes nothing (pinned by ``tests/test_liquidity.py``).

        Args:
            account_balance: Current account balance in USDT.
            current_price: Entry price of the asset.
            position_type: "core" or "satellite" (capital pool allocation).
            volatility_expanding: If True, ML predicts vol will expand →
                reduce position size.
            forecast_vol_pct: Forecast conditional volatility, percent of price
                per bar (e.g. ``0.45``).  None → fixed-fraction fallback.
            recent_quote_volume: Recent traded quote (USDT) notional — a number,
                a bar frame, or a callable.  None → participation cap is skipped
                (documented: an unmeasured window cannot cap anything).
            symbol: Pair name, used for per-symbol ``risk.liquidity`` overrides
                and for the debug log.

        Returns:
            (quantity, risk_amount_usdt) tuple.
        """
        if position_type == "core":
            capital_pool = account_balance * self.core_capital_pct
        else:
            capital_pool = account_balance * self.satellite_capital_pct

        effective_pct = max(self.soft.position_size_pct, 0.1)
        risk_per_trade = capital_pool * (effective_pct / 100)
        max_risk = account_balance * (self.hard.max_position_size_pct / 100)
        risk_per_trade = min(risk_per_trade, max_risk)

        # ── Volatility-based adjustment ──
        if volatility_expanding:
            risk_per_trade *= 0.7  # reduce position during high-vol regimes

        # ── Volatility targeting (P3) ──
        scale = self.vol_scale(forecast_vol_pct)
        if scale != 1.0:
            risk_per_trade *= scale
            # Re-apply the caps: the scale may have pushed the notional above the
            # hard ceiling (it can only be > 1) and it must never exceed it.
            risk_per_trade = min(risk_per_trade, max_risk)
            vt_cap = self._vol_notional_cap(account_balance)
            if vt_cap is not None:
                risk_per_trade = min(risk_per_trade, vt_cap)

        # ── Participation cap (P6-A) — last, and shrink-only ──
        if recent_quote_volume is not None and self.participation_cap_enabled(symbol):
            risk_per_trade, _reason = self.apply_participation_cap(
                risk_per_trade, recent_quote_volume, symbol)

        quantity = risk_per_trade / current_price if current_price > 0 else 0
        return quantity, risk_per_trade

    def _vol_notional_cap(self, account_balance: float) -> float | None:
        """``risk.vol_targeting.max_position_notional_pct`` of balance, or None."""
        if not self.vol_targeting_enabled():
            return None
        pct = _f(getattr(self.vol_targeting, "max_position_notional_pct", 0.0), 0.0)
        if pct <= 0.0:
            return None
        return float(account_balance) * pct / 100.0

    def calculate_stop_loss(self, entry_price: float, side: str,
                            volatility_expanding: bool = False,
                            forecast_vol_pct: float | None = None) -> float:
        """Calculate stop-loss price with optional volatility-based widening.

        When volatility is predicted to expand, SL is widened to 130%
        to give the trade more room and avoid being prematurely stopped out.

        With ``forecast_vol_pct`` (and vol targeting on) the distance comes from
        :meth:`stop_distance_pct` instead — a forecast-scaled width clamped to
        ``[stop_min_pct, stop_max_pct]``.  Fees, spread and slippage are *not*
        part of this distance; callers that need a cost-aware stop should widen it
        themselves, as ``core/executor`` does.
        """
        if self.vol_targeting_enabled() and _f(forecast_vol_pct, 0.0) > 0.0:
            sl_pct = self.stop_distance_pct(forecast_vol_pct) / 100.0
        else:
            sl_pct = max(self.soft.stop_loss_pct / 100,
                         self.hard.min_stop_loss_distance_pct / 100)
            if volatility_expanding:
                sl_pct *= 1.3  # wider stop during high-vol regimes
        if side == "long":
            return entry_price * (1 - sl_pct)
        else:
            return entry_price * (1 + sl_pct)

    def trailing_stop_distance_pct(self, strategy_risk_exit=None,
                                   forecast_vol_pct: float | None = None) -> float:
        """Trailing-stop distance in percent — single source of truth.

        Live trading, the legacy backtest engine and the hybrid engine all call
        this so a position is managed with identical trailing semantics.
        Order of precedence: per-strategy ``risk_exit.trailing_stop_pct`` →
        (vol targeting, when enabled and a forecast is available) →
        ``hard_limits.trailing_stop_distance_pct`` (when enabled) → disabled (0).

        A per-strategy override still wins: a strategy that asked for its own
        trailing distance is not silently overridden by the global vol model.
        """
        if strategy_risk_exit is not None:
            return float(getattr(strategy_risk_exit, "trailing_stop_pct", 0.0) or 0.0)
        enabled = getattr(self.hard, "trailing_stop_enabled", False)
        if not enabled:
            return 0.0
        if self.vol_targeting_enabled() and _f(forecast_vol_pct, 0.0) > 0.0:
            return self.stop_distance_pct(forecast_vol_pct)
        return float(getattr(self.hard, "trailing_stop_distance_pct", 0.0) or 0.0)

    def calculate_take_profits(self, entry_price: float, side: str) -> list[tuple[float, float]]:
        levels = [
            self.soft.take_profit_1_pct / 100,
            self.soft.take_profit_2_pct / 100,
            self.soft.take_profit_3_pct / 100,
        ]
        tps = []
        for pct in levels:
            if side == "long":
                tp_price = entry_price * (1 + pct)
            else:
                tp_price = entry_price * (1 - pct)
            tps.append((tp_price, pct))
        return tps
