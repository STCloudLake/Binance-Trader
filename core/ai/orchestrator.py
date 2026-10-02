"""P7-S3 — the upper-layer :class:`RegimeOrchestrator` (a pure decision function
plus an explicit, replayable state machine; **no hidden globals**).

WHY THIS MODULE EXISTS
----------------------
P7-S1 measured what happens when *each strategy* declares the causal regimes it
may trade in: on the real cached data conditioning produced **no risk-adjusted
alpha** (out-of-sample ``dsr > 0`` in 0 of 8 unconditioned and 0 of 40
conditioned cells), but it did cut exposure consistently (median trades
141.5 → 19.5, time-in-market 24.0 % → 2.1 %; ``docs/overhaul/P7_REGIME_PLAN.md``
§3.1).  The operator's reading of that result is that the decision "what may run
when" belongs to **one upper layer** that can use higher-level information than a
single strategy's own filter — the causal regime *plus* the volatility state
*plus* market breadth *plus* each strategy's own recent track record.

This module is that layer.  It is deliberately **one object with one decision
point** (the plan's "编排对象单一" criterion: ``grep`` finds no second decision
site), it is **deterministic and replayable**, and with the shipped
configuration it is **disabled** (``ai.orchestrator.enabled: false``), so nothing
in the live or backtest path changes.

THE RULES (fixed; never tuned on the evaluation window)
-------------------------------------------------------
Everything below is a *rule*, not a fitted parameter.  The numbers live in
``config/config.yaml`` under ``ai.orchestrator`` and the shipped block is inert;
S4 locks them out-of-sample (chosen on the train window only, one shot at the
test window).  **These thresholds must never be tuned on the evaluation window**
— doing so would turn S4's out-of-sample numbers into in-sample ones, which is
the exact failure P7-S1 already paid for.

Notation: at bar/timestamp ``t``, strategy ``s`` has causal regime label
``reg(t)`` ∈ {``trend_up``, ``trend_down``, ``range_low``, ``range_mid``,
``range_high``} ∪ {``unknown``} (:mod:`core.strategy.regime_causal`; an in-sample
HMM label is *refused by name*, never silently allowed).

1. **Regime mapping** (``ai.orchestrator.regime.allowed``): a block
   ``{strategy: [labels]}`` declares which causal regimes strategy ``s`` may be
   enabled in; ``["all"]`` means every regime.  A strategy **not listed** follows
   ``regime.default_action`` (``allow`` | ``deny``).  Two *different* "I do not
   know" cases are kept apart, because they are different failures:

   * **no regime data at all** (``label is None``: the symbol/interval has no
     usable cached frame) ⇒ ``regime.missing_regime_action``;
   * the classifier ran but the bar is ``range_unknown`` (the expanding
     volatility terciles need ``min_periods`` observations, so the first 31 bars
     are unlabelled) ⇒ ``regime.unknown_label_action``.  ``range_unknown`` is
     never inside a label list — a label list contains
     :data:`~core.strategy.regime_causal.GATE_REGIME_LABELS` — so without this key
     ``["all"]`` would silently refuse the head of every series.

   An unknown label name in the block (including an in-sample
   ``calm``/``stressed``) is refused at *config load*.

2. **Consecutive-loss kill switch** (``ai.orchestrator.kill_switch``): within one
   **regime stretch** (a maximal run of bars carrying the same label), after
   ``consecutive_losses`` consecutive losing trades the strategy is **disabled
   for the rest of that stretch**.  A trade is a loss when
   ``pnl < loss_threshold`` (``0.0`` ⇒ exact break-even is *not* a loss).  When
   the regime label **changes**, the counter resets and the latch clears —
   ``reset_on_regime_change: true`` (the shipped and only implemented behaviour;
   the key exists so the semantics are visible in config, and ``false`` is
   refused at load rather than silently ignored).  ``consecutive_losses: 0``
   turns the rule off.

3. **Volatility gate** (``ai.orchestrator.vol``): with ``multiple = m > 0`` and a
   reference ``med(t)`` = the **causal** median of the vol samples appended
   *strictly before* ``t`` (over the trailing ``window`` samples), an entry is
   blocked when::

       vol(t) > m * med(t)

   ``deny_on_high_vol: false`` inverts it (trade *only* in the high-vol state).
   Fewer than ``min_samples`` trailing samples ⇒ the gate is *unmeasurable* and
   follows ``vol.missing_action``; so does a non-finite value or a non-positive
   median (no ratio exists — reporting "infinite ratio" would be a fabrication).
   ``multiple: 0`` (the shipped value) turns the rule off.

4. **Market-breadth gate** (``ai.orchestrator.breadth``):
   :mod:`core.market_data.breadth` provides ``up_share`` (advance/decline) and
   ``coverage`` per labelled observation.  The latest observation **at or before**
   ``t`` is used, forward-filled by ``sample_ms`` (``0`` = exact timestamps only).
   It blocks when ``up_share < min_up_share`` or ``coverage < min_coverage``.
   An observation older than ``max_staleness_ms`` — or a series that is **absent
   entirely** — is *stale* / *missing* and follows ``breadth.stale_action`` /
   ``breadth.missing_action`` (both ship ``allow``: breadth is a **forward-only**
   record that cannot be backfilled, so "no breadth data" must not silently
   disable the book; an operator who wants a hard guard sets them to ``deny``,
   which is documented in the YAML next to the keys).  ``min_up_share`` /
   ``min_coverage`` left unset disables that half of the rule; with both unset
   the whole gate is inert.

**Decision order.**  :func:`decide` evaluates the gates in the order
regime → kill switch → vol → breadth and returns the **first** blocking reason
together with every reason that applies.  The order affects only the reported
``reason``, never the boolean: ``enabled`` is exactly ``not blocked_reasons``, so
no rule can ever *enable* something another rule refused.

WHAT IS *NOT* HERE
------------------
* **Not enabled.**  ``ai.orchestrator.enabled: false`` is the shipped value;
  :meth:`RegimeOrchestrator.decide` then returns :data:`ALLOW` **without touching
  the vol cache or the breadth series** — the disabled path costs one boolean.
* **Not a learner.**  No fitting, no online update, no randomness: the same
  configuration plus the same event list gives the same timeline on every run
  (:meth:`RegimeOrchestrator.replay`; pinned by ``tests/test_p7_orchestrator.py``).
* **Not an order path.**  Nothing here sizes, prices or routes.  The object
  answers one question — "may strategy ``s`` be enabled at ``t``?" — and the
  caller asks it at the same seam P7-S1 uses.
* **Not wired by default.**  The live ``StrategyEngine`` reads it only when
  ``experimental.regime_conditioning_live`` is ``true`` **and** a caller has
  registered one (``StrategyEngine.wire_regime_orchestrator``); the backtest
  engine reads it only when a caller passes ``orchestrator=`` explicitly.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from core.strategy.regime_causal import (
    GATE_REGIME_LABELS,
    UNKNOWN_REGIME,
    InSampleRegimeLabelError,
)

__all__ = [
    "OrchestratorConfigError", "ALL_REGIMES", "DEFAULT_ACTION_ALLOW",
    "DEFAULT_ACTION_DENY", "ENTRY_ENABLED", "ENTRY_DISABLED", "ALLOW",
    "BLOCK_REGIME", "BLOCK_KILL_SWITCH", "BLOCK_VOL", "BLOCK_BREADTH",
    "OrchestratorConfig", "RegimePolicy", "KillSwitchPolicy", "VolPolicy",
    "BreadthPolicy", "Decision", "StrategyState", "StrategyDecision",
    "TradeOutcome", "BreadthSample", "TimelineEvent",
    "orchestrator_config_from_raw", "regime_policy_blocks",
    "kill_switch_blocks", "vol_policy_blocks", "breadth_policy_blocks", "decide",
    "RegimeOrchestrator", "rules_fingerprint",
    # Re-exported so a caller needs one import for the vocabulary and the refusal.
    "InSampleRegimeLabelError", "GATE_REGIME_LABELS", "UNKNOWN_REGIME",
]

#: The shorthand a regime block may use instead of listing all five labels.
ALL_REGIMES = "all"

#: ``default_action`` / ``missing_*_action`` vocabulary.
DEFAULT_ACTION_ALLOW = "allow"
DEFAULT_ACTION_DENY = "deny"

#: ``Decision.action`` values.
ENTRY_ENABLED = "enabled"
ENTRY_DISABLED = "disabled"

#: ``Decision.reason`` codes — stable strings a report may key on.
BLOCK_REGIME = "regime_not_allowed"
BLOCK_KILL_SWITCH = "kill_switch_consecutive_losses"
BLOCK_VOL = "vol_above_threshold"
BLOCK_BREADTH = "breadth_below_threshold"

#: The all-allow verdict (what the disabled orchestrator returns).
ALLOW = "allow"


class OrchestratorConfigError(ValueError):
    """The ``ai.orchestrator`` block is unusable (an unknown key or value).

    Raised at **config load**, in the spirit of
    :class:`core.strategy.regime_causal.UnknownRegimeLabelError`: a mis-typed
    orchestrator rule fails where the operator can see it, not halfway through a
    backtest.
    """


# ══════════════════════════════════════════════════════════════════════════
# The rule set — one frozen value object, one place the thresholds live
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RegimePolicy:
    """Rule 1 — the regime → eligible-strategy mapping."""

    allowed: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    default_action: str = DEFAULT_ACTION_ALLOW
    missing_regime_action: str = DEFAULT_ACTION_ALLOW
    unknown_label_action: str = DEFAULT_ACTION_ALLOW

    def labels_for(self, strategy: str):
        """``None`` = "not listed" (⇒ ``default_action``); else a label tuple.

        The **empty tuple** means the strategy is listed as eligible in *no*
        regime — an explicit, deliberate "never enable this one".
        """
        entry = self.allowed.get(str(strategy))
        return None if entry is None else tuple(entry)

    def allows(self, strategy: str, label) -> bool:
        """The mapping's own verdict for one bar (``True`` = the regime is eligible).

        Three distinct cases, in this order:

        * ``label is None`` — the bar has **no regime data at all** (symbol not
          cached, frame shorter than the classifier's minimum).  This is the
          ``missing_regime_action`` case.
        * ``label == UNKNOWN_REGIME`` (``range_unknown``) — the composite
          classifier ran but the volatility terciles were not yet measurable, so
          the *bar* is unlabelled while the symbol's data is present.  This is the
          ``unknown_label_action`` case.  It is kept separate from the first
          because a forward-only breadth/regime record can easily have a labelled
          series that starts with unmeasurable bars, and the two failures deserve
          two decisions.
        * otherwise the label list decides (unlisted ⇒ ``default_action``).
        """
        if label is None:
            return self.missing_regime_action == DEFAULT_ACTION_ALLOW
        text = str(label)
        if text == UNKNOWN_REGIME:
            return self.unknown_label_action == DEFAULT_ACTION_ALLOW
        labels = self.labels_for(strategy)
        if labels is None:
            return self.default_action == DEFAULT_ACTION_ALLOW
        return bool(labels) and text in labels


@dataclass(frozen=True)
class KillSwitchPolicy:
    """Rule 2 — consecutive losses within one regime stretch."""

    consecutive_losses: int = 0
    loss_threshold: float = 0.0
    reset_on_regime_change: bool = True

    @property
    def active(self) -> bool:
        return int(self.consecutive_losses) > 0

    def is_loss(self, pnl) -> bool:
        """``pnl < loss_threshold`` (so exact break-even at the default is a tie)."""
        try:
            return float(pnl) < float(self.loss_threshold)
        except (TypeError, ValueError):
            return False

    def trips(self, consecutive_losses) -> bool:
        return self.active and int(consecutive_losses) >= int(self.consecutive_losses)


@dataclass(frozen=True)
class VolPolicy:
    """Rule 3 — ``vol > multiple * trailing_median`` (invertible, causal)."""

    multiple: float = 0.0
    window: int = 200
    min_samples: int = 30
    deny_on_high_vol: bool = True
    missing_action: str = DEFAULT_ACTION_ALLOW

    @property
    def active(self) -> bool:
        return float(self.multiple) > 0.0


@dataclass(frozen=True)
class BreadthPolicy:
    """Rule 4 — the breadth floors (``up_share`` / ``coverage``)."""

    min_up_share: float | None = None
    min_coverage: float | None = None
    max_staleness_ms: int | None = 1_800_000
    sample_ms: int = 0
    stale_action: str = DEFAULT_ACTION_ALLOW
    missing_action: str = DEFAULT_ACTION_ALLOW
    deny_on_low_up_share: bool = True

    @property
    def active(self) -> bool:
        return self.min_up_share is not None or self.min_coverage is not None


@dataclass(frozen=True)
class OrchestratorConfig:
    """The whole ``ai.orchestrator`` rule set, validated.

    ``enabled`` is the master switch and it ships **False**.  A disabled
    orchestrator returns :data:`ALLOW` for every (strategy, bar) *without reading
    any input series*, which is what makes "default off ⇒ nothing changes"
    checkable rather than asserted.
    """

    enabled: bool = False
    regime: RegimePolicy = field(default_factory=RegimePolicy)
    kill_switch: KillSwitchPolicy = field(default_factory=KillSwitchPolicy)
    vol: VolPolicy = field(default_factory=VolPolicy)
    breadth: BreadthPolicy = field(default_factory=BreadthPolicy)

    def as_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "regime": {
                "allowed": {k: list(v) for k, v in sorted(self.regime.allowed.items())},
                "default_action": self.regime.default_action,
                "missing_regime_action": self.regime.missing_regime_action,
                "unknown_label_action": self.regime.unknown_label_action},
            "kill_switch": {
                "consecutive_losses": int(self.kill_switch.consecutive_losses),
                "loss_threshold": float(self.kill_switch.loss_threshold),
                "reset_on_regime_change": bool(self.kill_switch.reset_on_regime_change)},
            "vol": {"multiple": float(self.vol.multiple),
                    "window": int(self.vol.window),
                    "min_samples": int(self.vol.min_samples),
                    "deny_on_high_vol": bool(self.vol.deny_on_high_vol),
                    "missing_action": self.vol.missing_action},
            "breadth": {"min_up_share": self.breadth.min_up_share,
                        "min_coverage": self.breadth.min_coverage,
                        "max_staleness_ms": self.breadth.max_staleness_ms,
                        "sample_ms": int(self.breadth.sample_ms),
                        "stale_action": self.breadth.stale_action,
                        "missing_action": self.breadth.missing_action,
                        "deny_on_low_up_share":
                            bool(self.breadth.deny_on_low_up_share)},
        }


# ══════════════════════════════════════════════════════════════════════════
# Timeline inputs (frozen values — a timeline never mutates its events)
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TradeOutcome:
    """One closed trade as the kill switch sees it (``pnl`` in quote currency)."""

    strategy: str
    pnl: float
    ts: object = None
    regime: str | None = None


@dataclass(frozen=True)
class BreadthSample:
    """One breadth observation reduced to the two gated numbers.

    :meth:`from_observation` accepts a
    :class:`core.market_data.breadth.BreadthObservation` (or anything with the
    same attributes), so a caller never re-implements the mapping.
    """

    as_of_ms: int
    up_share: float | None = None
    coverage: float | None = None
    is_stale: bool = False
    missing: bool = False

    @classmethod
    def from_observation(cls, obs) -> "BreadthSample | None":
        """``None`` for ``None``; otherwise the two gated numbers, verbatim."""
        if obs is None:
            return None
        return cls(as_of_ms=int(getattr(obs, "as_of_ms")),
                   up_share=(None if getattr(obs, "up_share", None) is None
                             else float(obs.up_share)),
                   coverage=(None if getattr(obs, "coverage", None) is None
                             else float(obs.coverage)),
                   is_stale=bool(getattr(obs, "is_stale", False)),
                   missing=bool(getattr(obs, "missing", False)))

    def as_dict(self) -> dict:
        return {"as_of_ms": int(self.as_of_ms), "up_share": self.up_share,
                "coverage": self.coverage, "is_stale": bool(self.is_stale),
                "missing": bool(self.missing)}


@dataclass(frozen=True)
class Decision:
    """One (strategy, bar) verdict.  ``blocked_reasons`` is ordered by priority."""

    enabled: bool
    label: str
    reason: str
    blocked_reasons: tuple[str, ...] = ()
    detail: Mapping[str, object] = field(default_factory=dict)

    @property
    def action(self) -> str:
        return ENTRY_ENABLED if self.enabled else ENTRY_DISABLED

    def as_dict(self) -> dict:
        return {"enabled": bool(self.enabled), "action": self.action,
                "label": self.label, "reason": self.reason,
                "blocked_reasons": list(self.blocked_reasons),
                "detail": dict(self.detail)}


@dataclass
class StrategyState:
    """Per-strategy state of the state machine (nothing here is global).

    ``killed`` is the kill-switch latch, ``killed_in_regime`` the label stretch
    that latched it, and ``consecutive_losses`` counts losses **within the current
    stretch** (reset on every regime change).
    """

    name: str
    regime: str | None = None
    consecutive_losses: int = 0
    killed: bool = False
    killed_in_regime: str | None = None
    enabled: bool = True
    last_reason: str = ALLOW
    last_label: str | None = None
    trades: int = 0
    losses: int = 0
    regime_changes: int = 0

    def as_dict(self) -> dict:
        return {"name": self.name, "regime": self.regime,
                "consecutive_losses": int(self.consecutive_losses),
                "killed": bool(self.killed),
                "killed_in_regime": self.killed_in_regime,
                "enabled": bool(self.enabled), "last_reason": self.last_reason,
                "last_label": self.last_label, "trades": int(self.trades),
                "losses": int(self.losses),
                "regime_changes": int(self.regime_changes)}


@dataclass(frozen=True)
class StrategyDecision:
    """One strategy's row of a :meth:`RegimeOrchestrator.decide_all` call."""

    strategy: str
    decision: Decision
    state: Mapping[str, object]

    @property
    def enabled(self) -> bool:
        return bool(self.decision.enabled)

    def as_dict(self) -> dict:
        return {"strategy": self.strategy, **self.decision.as_dict(),
                "state": dict(self.state)}


@dataclass(frozen=True)
class TimelineEvent:
    """One event of a replayable timeline.

    ``kind`` is one of:

    ``"regime"``
        ``at`` = the timestamp, ``label`` = the causal label from here on.  The
        label is applied to **every** strategy already known to the machine (the
        regime belongs to the market, not to a strategy); ``key`` names a single
        strategy for the case where none is registered yet.
    ``"vol"``
        ``at``, ``vol`` and ``key`` (the series key, default ``""``): one vol
        sample.  The policy's median is computed from the samples before it.
    ``"breadth"``
        ``at`` and ``sample`` (a :class:`BreadthSample`).
    ``"trade"``
        ``outcome`` (a :class:`TradeOutcome`).
    ``"decide"``
        ``at`` (and optionally ``label``): re-evaluate every known strategy and
        append the rows to ``timeline``.
    """

    kind: str
    at: object = None
    label: str | None = None
    vol: float | None = None
    key: str = ""
    sample: "BreadthSample | None" = None
    outcome: "TradeOutcome | None" = None


# ══════════════════════════════════════════════════════════════════════════
# The pure rule functions (each rule in isolation — the tests call these)
# ══════════════════════════════════════════════════════════════════════════

def regime_policy_blocks(policy: RegimePolicy, strategy: str, label) -> bool:
    """Rule 1 in isolation: ``True`` when the regime mapping forbids the bar."""
    return not policy.allows(strategy, label)


def kill_switch_blocks(policy: KillSwitchPolicy, state: "StrategyState | None",
                       label=None) -> bool:
    """Rule 2 in isolation: ``True`` while the strategy is latched for this stretch."""
    if state is None or not policy.active:
        return False
    return bool(state.killed)


def vol_policy_blocks(policy: VolPolicy, current, median) -> bool:
    """Rule 3 in isolation.  ``current``/``median`` may be ``None`` (unmeasurable).

    Unmeasurable (either side missing, non-finite, or a non-positive median ⇒ no
    ratio exists) follows ``policy.missing_action``.  Otherwise the verdict is
    ``current > multiple * median`` when ``deny_on_high_vol``, inverted when not.
    """
    if not policy.active:
        return False
    unmeasurable = policy.missing_action == DEFAULT_ACTION_DENY
    if current is None or median is None:
        return unmeasurable
    try:
        current = float(current)
        median = float(median)
    except (TypeError, ValueError):
        return unmeasurable
    if median != median or current != current or median <= 0.0:
        return unmeasurable
    high = current > float(policy.multiple) * median
    return high if policy.deny_on_high_vol else not high


def breadth_policy_blocks(policy: BreadthPolicy, sample: "BreadthSample | None",
                          now_ms=None) -> bool:
    """Rule 4 in isolation.  ``sample is None`` ⇒ ``policy.missing_action``.

    Staleness is measured against *now_ms* when both it and the sample's
    ``as_of_ms`` are known; ``max_staleness_ms`` of ``None`` disables that check.
    A ``None`` field the policy asks about is likewise *unmeasurable* ⇒
    ``missing_action`` (never silently "fine").
    """
    if not policy.active:
        return False
    missing = policy.missing_action == DEFAULT_ACTION_DENY
    if sample is None:
        return missing
    if policy.max_staleness_ms is not None and now_ms is not None:
        age = int(now_ms) - int(sample.as_of_ms)
        if age > int(policy.max_staleness_ms):
            return policy.stale_action == DEFAULT_ACTION_DENY
    if policy.min_up_share is not None:
        if sample.up_share is None:
            return missing
        low = float(sample.up_share) < float(policy.min_up_share)
        if low if policy.deny_on_low_up_share else not low:
            return True
    if policy.min_coverage is not None:
        if sample.coverage is None:
            return missing
        if float(sample.coverage) < float(policy.min_coverage):
            return True
    return False


def decide(policy: OrchestratorConfig, strategy: str, label, *,
           state: "StrategyState | None" = None, current_vol=None,
           median_vol=None, breadth: "BreadthSample | None" = None,
           now_ms=None) -> Decision:
    """The **whole** decision for one (strategy, bar) — a pure function.

    Evaluates the four rules in the documented order and returns the first
    blocking reason together with every reason that applies.  ``enabled`` is
    exactly ``not blocked_reasons``: a rule can only ever *stop* a strategy, so
    this function cannot enable something another rule refused.

    An in-sample HMM label raises
    :class:`~core.strategy.regime_causal.InSampleRegimeLabelError` — a refusal,
    never a quiet ``False``.
    """
    text = UNKNOWN_REGIME if label is None else str(label)
    if text in ("calm", "stressed"):
        raise InSampleRegimeLabelError(
            f"refusing to orchestrate on the in-sample HMM label '{text}': the "
            f"causal composite labels are {', '.join(GATE_REGIME_LABELS)}")

    if not policy.enabled:
        return Decision(enabled=True, label=text, reason=ALLOW)

    reasons: list[str] = []
    detail: dict = {}
    if regime_policy_blocks(policy.regime, strategy, text):
        reasons.append(BLOCK_REGIME)
        detail["regime_allowed"] = policy.regime.labels_for(strategy)
        detail["regime_default_action"] = policy.regime.default_action
        detail["regime_missing_action"] = policy.regime.missing_regime_action
        detail["regime_unknown_label_action"] = policy.regime.unknown_label_action
    if kill_switch_blocks(policy.kill_switch, state, text):
        reasons.append(BLOCK_KILL_SWITCH)
        detail["consecutive_losses"] = int(getattr(state, "consecutive_losses", 0))
        detail["kill_limit"] = int(policy.kill_switch.consecutive_losses)
        detail["killed_in_regime"] = getattr(state, "killed_in_regime", None)
    if vol_policy_blocks(policy.vol, current_vol, median_vol):
        reasons.append(BLOCK_VOL)
        detail["vol"] = current_vol
        detail["vol_median"] = median_vol
        detail["vol_multiple"] = float(policy.vol.multiple)
    if breadth_policy_blocks(policy.breadth, breadth, now_ms):
        reasons.append(BLOCK_BREADTH)
        detail["breadth"] = None if breadth is None else breadth.as_dict()
    return Decision(enabled=not reasons, label=text,
                    reason=reasons[0] if reasons else ALLOW,
                    blocked_reasons=tuple(reasons), detail=detail)


# ══════════════════════════════════════════════════════════════════════════
# Config parsing (fail loudly, at load)
# ══════════════════════════════════════════════════════════════════════════

_ACTIONS = (DEFAULT_ACTION_ALLOW, DEFAULT_ACTION_DENY)
_TOP_KEYS = ("enabled", "regime", "kill_switch", "vol", "breadth")


def _require_mapping(raw, name: str) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise OrchestratorConfigError(
            f"ai.orchestrator{'.' + name if name else ''} must be a mapping, got "
            f"{type(raw).__name__}")
    return dict(raw)


def _reject_unknown(raw: Mapping, allowed: Iterable[str], name: str) -> None:
    known = set(allowed)
    unknown = sorted(str(key) for key in raw if str(key) not in known)
    if unknown:
        raise OrchestratorConfigError(
            f"unknown key(s) {unknown} in ai.orchestrator"
            f"{'.' + name if name else ''}; accepted: {sorted(known)}")


def _action(raw, default: str, name: str) -> str:
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text not in _ACTIONS:
        raise OrchestratorConfigError(
            f"ai.orchestrator.{name} must be one of {list(_ACTIONS)}, got {raw!r}")
    return text


def _bool(raw, default: bool, name: str) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise OrchestratorConfigError(
        f"ai.orchestrator.{name} must be a boolean, got {raw!r}")


def _float(raw, default, name: str):
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise OrchestratorConfigError(
            f"ai.orchestrator.{name} must be a number, got {raw!r}") from None


def _int(raw, default, name: str):
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise OrchestratorConfigError(
            f"ai.orchestrator.{name} must be an integer, got {raw!r}") from None


def _regime_policy(raw: Mapping) -> RegimePolicy:
    _reject_unknown(raw, ("allowed", "default_action", "missing_regime_action",
                          "unknown_label_action"), "regime")
    block = _require_mapping(raw.get("allowed"), "regime.allowed")
    allowed: dict[str, tuple[str, ...]] = {}
    for strategy, labels in block.items():
        name = str(strategy).strip()
        if not name:
            raise OrchestratorConfigError(
                "ai.orchestrator.regime.allowed has an empty strategy name")
        if isinstance(labels, str):
            items: Iterable = labels.split(",")
        elif isinstance(labels, (list, tuple, set, frozenset)):
            items = [str(piece) for piece in labels]
        else:
            raise OrchestratorConfigError(
                f"ai.orchestrator.regime.allowed.{name} must be a list of regime "
                f"labels, got {type(labels).__name__}")
        out: list[str] = []
        for item in items:
            label = str(item).strip()
            if not label:
                continue
            if label == ALL_REGIMES:
                out = list(GATE_REGIME_LABELS)
                break
            if label in ("calm", "stressed"):
                raise OrchestratorConfigError(
                    f"ai.orchestrator.regime.allowed.{name} names the in-sample "
                    f"HMM label '{label}' — refuse; the causal composite labels "
                    f"are {', '.join(GATE_REGIME_LABELS)} (or '{ALL_REGIMES}')")
            if label not in GATE_REGIME_LABELS:
                raise OrchestratorConfigError(
                    f"unknown regime label '{label}' in "
                    f"ai.orchestrator.regime.allowed.{name}; accepted: "
                    f"{', '.join(GATE_REGIME_LABELS)} (or '{ALL_REGIMES}')")
            if label not in out:
                out.append(label)
        allowed[name] = tuple(out)
    return RegimePolicy(
        allowed=allowed,
        default_action=_action(raw.get("default_action"), DEFAULT_ACTION_ALLOW,
                               "regime.default_action"),
        missing_regime_action=_action(raw.get("missing_regime_action"),
                                      DEFAULT_ACTION_ALLOW,
                                      "regime.missing_regime_action"),
        unknown_label_action=_action(raw.get("unknown_label_action"),
                                     DEFAULT_ACTION_ALLOW,
                                     "regime.unknown_label_action"))


def _kill_switch_policy(raw: Mapping) -> KillSwitchPolicy:
    _reject_unknown(raw, ("consecutive_losses", "loss_threshold",
                          "reset_on_regime_change"), "kill_switch")
    losses = _int(raw.get("consecutive_losses"), 0,
                  "kill_switch.consecutive_losses")
    if losses < 0:
        raise OrchestratorConfigError(
            "ai.orchestrator.kill_switch.consecutive_losses must be >= 0")
    reset = _bool(raw.get("reset_on_regime_change"), True,
                  "kill_switch.reset_on_regime_change")
    if not reset:
        # An unimplemented option is refused rather than silently ignored: a key
        # that promises behaviour the module does not have is a lie in config.
        raise OrchestratorConfigError(
            "ai.orchestrator.kill_switch.reset_on_regime_change: false is not "
            "implemented (a regime change always re-evaluates the kill switch, "
            "per P7-S3); leave the key absent or true")
    return KillSwitchPolicy(
        consecutive_losses=losses,
        loss_threshold=_float(raw.get("loss_threshold"), 0.0,
                              "kill_switch.loss_threshold"),
        reset_on_regime_change=True)


def _vol_policy(raw: Mapping) -> VolPolicy:
    _reject_unknown(raw, ("multiple", "window", "min_samples", "deny_on_high_vol",
                          "missing_action"), "vol")
    multiple = _float(raw.get("multiple"), 0.0, "vol.multiple")
    if multiple < 0:
        raise OrchestratorConfigError("ai.orchestrator.vol.multiple must be >= 0")
    window = _int(raw.get("window"), 200, "vol.window")
    if window < 1:
        raise OrchestratorConfigError("ai.orchestrator.vol.window must be >= 1")
    min_samples = _int(raw.get("min_samples"), 30, "vol.min_samples")
    if min_samples < 2:
        raise OrchestratorConfigError(
            "ai.orchestrator.vol.min_samples must be >= 2 (a median needs two)")
    return VolPolicy(multiple=multiple, window=window, min_samples=min_samples,
                     deny_on_high_vol=_bool(raw.get("deny_on_high_vol"), True,
                                            "vol.deny_on_high_vol"),
                     missing_action=_action(raw.get("missing_action"),
                                            DEFAULT_ACTION_ALLOW,
                                            "vol.missing_action"))


def _breadth_policy(raw: Mapping) -> BreadthPolicy:
    _reject_unknown(raw, ("min_up_share", "min_coverage", "max_staleness_ms",
                          "sample_ms", "stale_action", "missing_action",
                          "deny_on_low_up_share"), "breadth")
    up = raw.get("min_up_share")
    coverage = raw.get("min_coverage")
    max_stale = raw.get("max_staleness_ms")
    sample_ms = _int(raw.get("sample_ms"), 0, "breadth.sample_ms")
    if sample_ms < 0:
        raise OrchestratorConfigError(
            "ai.orchestrator.breadth.sample_ms must be >= 0")
    stale = _int(max_stale, 1_800_000, "breadth.max_staleness_ms") \
        if max_stale is not None else None
    return BreadthPolicy(
        min_up_share=(None if up is None
                      else _float(up, None, "breadth.min_up_share")),
        min_coverage=(None if coverage is None
                      else _float(coverage, None, "breadth.min_coverage")),
        max_staleness_ms=stale,
        sample_ms=sample_ms,
        stale_action=_action(raw.get("stale_action"), DEFAULT_ACTION_ALLOW,
                             "breadth.stale_action"),
        missing_action=_action(raw.get("missing_action"), DEFAULT_ACTION_ALLOW,
                               "breadth.missing_action"),
        deny_on_low_up_share=_bool(raw.get("deny_on_low_up_share"), True,
                                   "breadth.deny_on_low_up_share"))


def orchestrator_config_from_raw(raw) -> OrchestratorConfig:
    """The raw ``ai.orchestrator`` mapping → a validated :class:`OrchestratorConfig`.

    Absent/``None`` ⇒ the shipped all-off default (``enabled=False``).  Every
    unknown key and every unusable value raises :class:`OrchestratorConfigError`
    **here**, at config load — a typo in a rule must not silently become "rule
    ignored", which would make an orchestrated measurement unreproducible.
    """
    if raw is None:
        return OrchestratorConfig()
    block = _require_mapping(raw, "")
    _reject_unknown(block, _TOP_KEYS, "")
    return OrchestratorConfig(
        enabled=_bool(block.get("enabled"), False, "enabled"),
        regime=_regime_policy(_require_mapping(block.get("regime"), "regime")),
        kill_switch=_kill_switch_policy(
            _require_mapping(block.get("kill_switch"), "kill_switch")),
        vol=_vol_policy(_require_mapping(block.get("vol"), "vol")),
        breadth=_breadth_policy(_require_mapping(block.get("breadth"), "breadth")))


# ══════════════════════════════════════════════════════════════════════════
# The orchestrator
# ══════════════════════════════════════════════════════════════════════════

class RegimeOrchestrator:
    """The upper-layer decision object: pure rules + an explicit state machine.

    One instance owns the state of one timeline (a backtest, a GA evaluation, a
    live session).  It holds **no module-level state**, reads no clock and no
    file, and is therefore replayable: :meth:`replay` builds a fresh instance
    from the same configuration and event list and returns the same timeline.

    The object never picks a timestamp or a strategy set on its own: every
    :meth:`decide` call carries an explicit timestamp from the caller, which is
    what keeps the decision point single and auditable.
    """

    def __init__(self, config: OrchestratorConfig | Mapping | None = None, *,
                 breadth_series: Sequence | None = None,
                 names: Iterable[str] | None = None):
        if config is None or isinstance(config, Mapping):
            config = orchestrator_config_from_raw(config)
        if not isinstance(config, OrchestratorConfig):
            raise OrchestratorConfigError(
                f"expected an OrchestratorConfig or its raw mapping, got "
                f"{type(config).__name__}")
        self.config = config
        #: ``{strategy: StrategyState}`` — the whole mutable state of the machine.
        self.states: dict[str, StrategyState] = {}
        #: ``{key: [(ts, vol)]}`` appended in call order (the vol policy's memory).
        self.vol_series: dict[str, list] = {}
        #: ``[(as_of_ms, BreadthSample)]``, ascending — seeded or observed.
        self.breadth_series: list = []
        #: The decision log: ``[{"at", "label", "rows"}]`` in call order.
        self.timeline: list = []
        if breadth_series is not None:
            self.load_breadth(breadth_series)
        for name in names or ():
            self.state(str(name))

    # ── introspection ──────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        """The master switch, verbatim (``False`` ships)."""
        return bool(self.config.enabled)

    def state(self, strategy: str) -> StrategyState:
        """The (created-on-demand) state of one strategy — never a global."""
        key = str(strategy)
        found = self.states.get(key)
        if found is None:
            found = StrategyState(name=key)
            self.states[key] = found
        return found

    def reset(self) -> None:
        """Forget every latch, series and logged decision (a new run on this object)."""
        self.states.clear()
        self.vol_series.clear()
        self.breadth_series.clear()
        self.timeline.clear()

    # ── inputs ─────────────────────────────────────────────────────────
    def observe_regime(self, strategy: str, label, *, at=None) -> bool:
        """Record the causal label of one bar; ``True`` when the stretch **changed**.

        A change resets the consecutive-loss counter and clears the kill-switch
        latch (rule 2's "re-evaluate after a regime change").  An in-sample label
        raises :class:`~core.strategy.regime_causal.InSampleRegimeLabelError`, the
        same refusal :func:`decide` makes, so no caller can move the state machine
        with a look-ahead label.
        """
        text = UNKNOWN_REGIME if label is None else str(label)
        if text in ("calm", "stressed"):
            raise InSampleRegimeLabelError(
                f"refusing to orchestrate on the in-sample HMM label '{text}'")
        state = self.state(strategy)
        changed = state.regime is not None and state.regime != text
        if changed:
            state.regime_changes += 1
            state.consecutive_losses = 0
            state.killed = False
            state.killed_in_regime = None
        state.regime = text
        state.last_label = text
        return changed

    def observe_vol(self, value, *, key: str = "", at=None) -> None:
        """Append one volatility sample to ``key``'s series (call order = time)."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        self.vol_series.setdefault(str(key), []).append((at, value))

    def load_breadth(self, observations: Sequence) -> int:
        """Seed the breadth series from observations (or samples); returns the count.

        Sorted by ``as_of_ms`` and de-duplicated on the same stamp, so a caller
        may hand in a cache read in any order.
        """
        rows: list[tuple[int, BreadthSample]] = []
        for obs in observations or ():
            sample = (obs if isinstance(obs, BreadthSample)
                      else BreadthSample.from_observation(obs))
            if sample is None:
                continue
            rows.append((int(sample.as_of_ms), sample))
        rows.sort(key=lambda row: row[0])
        deduped: list[tuple[int, BreadthSample]] = []
        for stamp, sample in rows:
            if deduped and deduped[-1][0] == stamp:
                deduped[-1] = (stamp, sample)
            else:
                deduped.append((stamp, sample))
        self.breadth_series = deduped
        return len(deduped)

    def observe_breadth(self, sample) -> None:
        """Append one breadth sample (``as_of_ms`` is its information time)."""
        if sample is None:
            return
        if not isinstance(sample, BreadthSample):
            sample = BreadthSample.from_observation(sample)
            if sample is None:
                return
        self.breadth_series.append((int(sample.as_of_ms), sample))

    def breadth_at(self, now_ms, *, allow_stale: bool = True) -> BreadthSample | None:
        """The latest sample at or before ``now_ms`` (forward-filled by ``sample_ms``).

        ``None`` when the series is empty, when nothing precedes ``now_ms``, or
        when the newest sample is older than ``sample_ms`` (``0`` = exact
        timestamps only).  ``allow_stale=False`` also drops a sample labelled
        ``missing=True`` (the breadth module's documented "treat as unavailable"
        state).
        """
        if not self.breadth_series or now_ms is None:
            return None
        stamps = [row[0] for row in self.breadth_series]
        cut = bisect.bisect_right(stamps, int(now_ms))
        if cut <= 0:
            return None
        stamp, sample = self.breadth_series[cut - 1]
        tolerance = int(self.config.breadth.sample_ms)
        if tolerance > 0:
            if int(now_ms) - stamp > tolerance:
                return None
        elif stamp != int(now_ms):
            return None
        if sample.missing and not allow_stale:
            return None
        return sample

    def vol_median(self, *, key: str = "", window: int | None = None) -> float | None:
        """The **causal** trailing median of ``key``'s samples — or ``None``.

        "Causal" = over every sample appended so far, i.e. over samples whose
        timestamp is strictly before the current one (the caller appends nothing
        future).  ``None`` below ``max(min_samples, 2)`` samples: an unmeasurable
        reference, reported as such rather than as ``0``.
        """
        series = self.vol_series.get(str(key)) or []
        limit = int(window if window is not None else self.config.vol.window)
        floor = max(int(self.config.vol.min_samples), 2)
        if len(series) < floor:
            return None
        values = sorted(value for _at, value in series[-limit:] if value == value)
        if len(values) < 2:
            return None
        mid = len(values) // 2
        if len(values) % 2:
            return float(values[mid])
        return float((values[mid - 1] + values[mid]) / 2.0)

    # ── the decision ───────────────────────────────────────────────────
    def decide(self, strategy: str, *, at=None, label=None, vol=None,
               vol_key=None, breadth_at=None, breadth_series=None) -> Decision:
        """The verdict for one strategy at one timestamp.

        * ``label`` — the causal regime label to record (omitted ⇒ the label
          already recorded; ``label`` is *not* ``None``-able, because "no label"
          is spelled ``UNKNOWN_REGIME``, never ``None``).
        * ``vol`` — the current volatility sample.  It is appended to the series
          **after** the trailing median is computed, so ``med(t)`` never sees
          ``vol(t)``.
        * ``vol_key`` — the series key for ``vol`` (default: the strategy name).
        * ``breadth_at`` — the information time for the breadth lookup (default
          ``at``); ``breadth_series`` — an optional
          :class:`core.market_data.breadth.BreadthCache` (or a list) to seed from
          once.  Its *absence* is handled gracefully: the policy's
          ``missing_action`` decides.

        **Disabled ⇒ one boolean.**  With ``config.enabled`` false this returns
        :data:`ALLOW` without reading the vol cache or the breadth series, so the
        off path cannot depend on either.
        """
        state = self.state(strategy)
        if not self.config.enabled:
            state.enabled = True
            state.last_reason = ALLOW
            return Decision(enabled=True,
                            label=state.last_label or UNKNOWN_REGIME, reason=ALLOW)
        if breadth_series is not None and not self.breadth_series:
            loader = getattr(breadth_series, "load", None)
            self.load_breadth(loader() if callable(loader) else breadth_series)

        if label is not None:
            self.observe_regime(strategy, label, at=at)
        text = state.last_label or UNKNOWN_REGIME

        current_vol = None
        median = None
        if self.config.vol.active:
            key = str(strategy) if vol_key is None else str(vol_key)
            median = self.vol_median(key=key)
            current_vol = vol
            if vol is not None:
                self.observe_vol(vol, key=key, at=at)

        sample = None
        now_ms = _as_ms(breadth_at if breadth_at is not None else at)
        if self.config.breadth.active:
            sample = self.breadth_at(now_ms)

        verdict = decide(self.config, strategy, text, state=state,
                         current_vol=current_vol, median_vol=median,
                         breadth=sample, now_ms=now_ms)
        state.enabled = bool(verdict.enabled)
        state.last_reason = verdict.reason
        return verdict

    def decide_all(self, strategies: Iterable[str], *, at=None, labels=None,
                   vol=None, vol_key=None, breadth_at=None,
                   breadth_series=None) -> list:
        """``decide`` for every strategy in one call, in the given order.

        ``labels`` may be a ``{strategy: label}`` mapping; the usual case is one
        shared scalar label, because the regime belongs to the *market*, not to
        the strategy.
        """
        rows = []
        for name in strategies:
            label = (labels.get(str(name)) if isinstance(labels, Mapping)
                     else labels)
            decision = self.decide(str(name), at=at, label=label, vol=vol,
                                   vol_key=vol_key, breadth_at=breadth_at,
                                   breadth_series=breadth_series)
            rows.append(StrategyDecision(strategy=str(name), decision=decision,
                                         state=self.state(str(name)).as_dict()))
        return rows

    def record_decision(self, rows: Iterable, *, at=None, label=None) -> dict:
        """Append one decision round to :attr:`timeline` (the replay artifact)."""
        row_list = [row.as_dict() if hasattr(row, "as_dict") else dict(row)
                    for row in rows]
        entry = {"at": (None if at is None else str(at)),
                 "label": (None if label is None else str(label)),
                 "rows": row_list}
        self.timeline.append(entry)
        return entry

    # ── the kill switch's input ────────────────────────────────────────
    def record_trade(self, strategy: str, pnl, *, at=None, regime=None) -> bool:
        """Feed one closed trade; ``True`` when it **trips** the kill switch.

        The trade is attributed to the strategy's current stretch: a supplied
        ``regime`` first goes through :meth:`observe_regime`, so an unobserved
        change still resets the counter *before* the loss is counted.  A winning
        (or exactly break-even) trade resets the counter; a losing one increments
        it, and at ``consecutive_losses`` the strategy is latched disabled for the
        remainder of the stretch.
        """
        state = self.state(strategy)
        if regime is not None:
            self.observe_regime(strategy, regime, at=at)
        state.trades += 1
        policy = self.config.kill_switch
        if policy.is_loss(pnl):
            state.losses += 1
            state.consecutive_losses += 1
        else:
            state.consecutive_losses = 0
        if policy.trips(state.consecutive_losses):
            state.killed = True
            state.killed_in_regime = state.last_label
            return True
        return False

    # ── the replay API (used by S4) ────────────────────────────────────
    def step(self, event: TimelineEvent) -> dict | None:
        """Apply one :class:`TimelineEvent`; returns the appended row for ``"decide"``."""
        kind = str(event.kind)
        if kind == "regime":
            targets = sorted(self.states) or ([str(event.key)] if event.key else [])
            for name in targets:
                self.observe_regime(name, event.label, at=event.at)
            return None
        if kind == "vol":
            self.observe_vol(event.vol, key=event.key, at=event.at)
            return None
        if kind == "breadth":
            self.observe_breadth(event.sample)
            return None
        if kind == "trade":
            outcome = event.outcome
            if outcome is None:
                raise OrchestratorConfigError(
                    "a 'trade' timeline event needs an outcome")
            self.record_trade(outcome.strategy, outcome.pnl, at=outcome.ts,
                              regime=outcome.regime)
            return None
        if kind == "decide":
            rows = self.decide_all(sorted(self.states), at=event.at,
                                   labels=event.label)
            return self.record_decision(rows, at=event.at, label=event.label)
        raise OrchestratorConfigError(f"unknown timeline event kind {event.kind!r}")

    def run_timeline(self, events: Sequence[TimelineEvent]) -> list:
        """Apply a whole timeline in order and return :attr:`timeline`."""
        for event in events:
            self.step(event)
        return self.timeline

    @classmethod
    def replay(cls, config: OrchestratorConfig | Mapping | None,
               events: Sequence[TimelineEvent], *,
               names: Iterable[str] | None = None,
               breadth_series: Sequence | None = None) -> list:
        """Build a **fresh** orchestrator and replay *events* — the S4 entry point.

        Determinism is a property of this method: the same configuration and the
        same events give the same timeline, with no clock, no RNG and no module
        state involved (``tests/test_p7_orchestrator.py`` asserts two replays are
        equal *and* equal to a third built through the incremental API).
        """
        machine = cls(config, names=names, breadth_series=breadth_series)
        return machine.run_timeline(events)

    def enable_timeline(self) -> list:
        """``[(at, {strategy: enabled})]`` from :attr:`timeline` — the compact artifact."""
        return [(row["at"], {item["strategy"]: bool(item["enabled"])
                             for item in row["rows"]})
                for row in self.timeline]


# ── helpers ────────────────────────────────────────────────────────────

def _as_ms(value):
    """A timestamp → epoch milliseconds, or ``None`` when it cannot be one."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        import pandas as pd
        return int(pd.Timestamp(value).value // 1_000_000)
    except Exception:
        return None


def rules_fingerprint(config: OrchestratorConfig | Mapping | None) -> str:
    """A short, stable digest of a rule set (S4 records it with its evaluation).

    ``sha256`` of the canonical JSON of :meth:`OrchestratorConfig.as_dict`, first
    16 hex characters — the convention the rest of the overhaul uses, and the
    number that makes "the orchestrator was fixed *before* the test window was
    opened" checkable after the fact.
    """
    if config is None or isinstance(config, Mapping):
        config = orchestrator_config_from_raw(config)
    payload = json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
