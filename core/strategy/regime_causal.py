"""P7-S1 — causal regime as a first-class strategy attribute (the checked seam).

Why this module exists
----------------------
``core/strategy/regime.py`` implements *detection*: it can label a bar from a
whole-sample HMM (whose parameters saw the future) **and** from a strictly causal
forward-only decode, and it exposes :class:`NonCausalRegimeError` for a gate that
would consume the former.  Its two entry points for a gate
(:func:`~core.strategy.regime.gate_regimes`,
:meth:`~core.strategy.regime.RegimeGate.gate_row`) answer one row at a time from
a ``RegimeGate`` whose ``allowed`` map is keyed by strategy *kind*.

P7-S1 needs the other half: a *strategy* declares which regimes it may trade in,
and every bar of an evaluation is checked against that declaration — with the
causal-only rule enforced by construction rather than by the caller's discipline.
This module is that seam:

* :func:`allowed_regimes` / :func:`regime_allows` — the decision, on a label
  string, plus the named refusals;
* :func:`causal_regime_table` — one symbol's per-bar causal labels, with
  ``attrs`` that record **which mode produced them**;
* :func:`enforce_causal_table` — raise unless the table's labels are causal;
* :class:`RegimeContext` — an index→label cache an evaluation can look up a bar
  timestamp in, built once per (symbol, interval).

Causality (the whole argument)
------------------------------
``classify_regimes(df, with_hmm=False)`` labels every bar from

* ``volatility_terciles`` — the expanding 1/3 and 2/3 quantiles of the **past**
  volatility (``vol.shift(1).expanding().quantile(...)``), and
* ``trend_regimes`` — EMAs of the past close,

so the composite ``regime`` column is a function of bars ``≤ t`` only, for every
``t``.  The HMM columns are the only part that has a non-causal mode, and
:func:`causal_regime_table` therefore never asks for them
(``with_hmm=False``); :func:`enforce_causal_table` additionally *measures* the
table's ``attrs["causal_hmm"]`` flag, so a table handed in from anywhere else is
refused rather than trusted (``attrs`` is also what survives a column subset).

:data:`IN_SAMPLE_LABELS` names the two labels that only a whole-sample HMM
emits.  They are refused **by name** (:class:`InSampleRegimeLabelError`) before
any decision is made, so a caller that tries to condition on ``hmm_state``
instead of the composite ``regime`` column fails loudly instead of quietly
gating on look-ahead labels.

What is NOT here
----------------
Nothing in this module changes the live path.  ``REGIME_GATING_ENABLED`` stays
``False``, no production component constructs a :class:`RegimeGate`, and P7-S1
wires the attribute into the **backtest/GA** entry path only (S3 is the stage
that decides whether the live engine may consume it).  The GA-side master switch
``ga.regime_conditioning`` (``False`` by default) gates gene creation, so with
the shipped configuration no genome carries a regime gene and no evaluation
builds a :class:`RegimeContext`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from core.strategy.regime import (
    MIN_REGIME_ROWS,
    NonCausalRegimeError,
    classify_regimes,
    trend_regimes,
    volatility_terciles,
)

# ── The gate vocabulary ────────────────────────────────────────────────

#: Labels a strategy's regime filter may name — the composite ``regime`` column
#: of :func:`~core.strategy.regime.classify_regimes` (``trend_up`` /
#: ``trend_down`` when the tape trends, ``range_<vol>`` otherwise).  31 bars at
#: the very start of a frame are ``range_unknown`` (the volatility terciles need
#: ``min_periods`` observations) and are deliberately **not** gate-able: a
#: strategy may not filter on a label that means "not measured yet".
GATE_REGIME_LABELS: tuple[str, ...] = (
    "trend_up", "trend_down", "range_low", "range_mid", "range_high",
)

#: Labels only a **whole-sample** HMM emits.  Conditioning on one of these is the
#: exact look-ahead failure mode ``tests/test_p34_audit_fixes.py`` exists to
#: prevent, so :func:`regime_allows` refuses them by name.
IN_SAMPLE_LABELS: tuple[str, ...] = ("calm", "stressed")

#: The label a bar gets when the regime table has no row for it (a bar before the
#: table's first row, or a symbol/interval with no cached history).  It is not in
#: :data:`GATE_REGIME_LABELS`, so an unmeasured bar refuses entries — the
#: conservative direction.
UNKNOWN_REGIME = "unknown"


class RegimeFilterError(ValueError):
    """Base class for a strategy's regime filter being unusable."""


class UnknownRegimeLabelError(RegimeFilterError):
    """A filter names a label outside :data:`GATE_REGIME_LABELS`."""


class InSampleRegimeLabelError(RegimeFilterError):
    """A filter names an in-sample-only HMM label (see :data:`IN_SAMPLE_LABELS`).

    A subclass of :class:`RegimeFilterError`, and a refusal rather than a
    warning: a look-ahead label silently conditioning an evaluation is the one
    failure this whole module is built to make impossible.
    """


class UnknownRegimeSourceError(RegimeFilterError):
    """A regime *table* was built by a source the strategy filter cannot use."""


# ── The decision ───────────────────────────────────────────────────────

def parse_regime_filter(raw) -> list[str]:
    """``regime_filter`` value → validated, de-duplicated, ordered label list.

    ``None``/absent/blank ⇒ ``[]`` = **no filter** (the historical behaviour: the
    strategy trades every regime).  A plain string is accepted as one label (a
    YAML ``regime_filter: trend_up``), like the container form.  An unknown label
    raises :class:`UnknownRegimeLabelError` and an in-sample one raises
    :class:`InSampleRegimeLabelError`, both naming the offender and the accepted
    set — a bad declaration fails **at load**, not halfway through a backtest.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        items: Iterable = raw.split(",")
    elif isinstance(raw, (list, tuple, set, frozenset)):
        items = [piece for entry in raw for piece in str(entry).split(",")]
    else:
        raise RegimeFilterError(
            "regime_filter must be a list of regime labels (or a comma-separated "
            f"string), got {type(raw).__name__}")
    out: list[str] = []
    for item in items:
        label = str(item).strip()
        if not label:
            continue
        if label in IN_SAMPLE_LABELS:
            raise InSampleRegimeLabelError(
                f"regime_filter names '{label}', which only a whole-sample "
                f"(look-ahead) HMM produces -- refuse: condition on the causal "
                f"composite labels {', '.join(GATE_REGIME_LABELS)} instead")
        if label not in GATE_REGIME_LABELS:
            raise UnknownRegimeLabelError(
                f"unknown regime label '{label}' in regime_filter; accepted: "
                f"{', '.join(GATE_REGIME_LABELS)}")
        if label not in out:
            out.append(label)
    return out


def allowed_regimes(regime_filter) -> list[str]:
    """Validated filter labels (``[]`` = allow every regime)."""
    return parse_regime_filter(regime_filter)


def regime_allows(regime_filter, label) -> bool:
    """May a strategy with *regime_filter* enter on a bar labelled *label*?

    No filter (``None``/``[]``) ⇒ ``True`` for every label, which is exactly the
    pre-P7 behaviour.  With a filter, an in-sample label raises
    :class:`InSampleRegimeLabelError` (never ``False`` — a refusal must be
    visible, not a silent "you are not allowed to trade"), and membership in
    :data:`GATE_REGIME_LABELS` decides otherwise.  An unknown / ``None`` label is
    ``False``: a bar whose regime could not be measured is not traded.
    """
    allowed = parse_regime_filter(regime_filter)
    if not allowed:
        return True
    text = "" if label is None else str(label)
    if text in IN_SAMPLE_LABELS:
        raise InSampleRegimeLabelError(
            f"refusing to gate on the in-sample HMM label '{text}': the causal "
            f"composite labels are {', '.join(GATE_REGIME_LABELS)}")
    return text in set(allowed)


# ── The causal table ───────────────────────────────────────────────────

def causal_regime_table(df: pd.DataFrame, *,
                        vol_window: int | None = None,
                        trend_fast: int | None = None,
                        trend_slow: int | None = None) -> pd.DataFrame:
    """Per-bar **causal** regime labels for one OHLCV frame.

    A thin, single-purpose wrapper around
    :func:`~core.strategy.regime.classify_regimes` with ``with_hmm=False``: the
    volatility-tercile and trend columns are causal by construction (see the
    module docstring), and the HMM — the only part of that function with a
    non-causal mode — is not computed at all.  The returned frame carries the
    same index as *df* and the columns ``vol_regime`` / ``trend_regime`` /
    ``regime``; ``attrs["causal_hmm"]`` is ``True`` and ``attrs["hmm_present"]``
    is ``False``, so :func:`enforce_causal_table` can verify the mode from the
    object rather than from the call site.

    ``vol_window`` / ``trend_fast`` / ``trend_slow`` are forwarded only when
    given, so the defaults stay :mod:`core.strategy.regime`'s own module
    constants.
    """
    kwargs: dict = {"with_hmm": False}
    if vol_window is not None:
        kwargs["vol_window"] = int(vol_window)
    if trend_fast is not None:
        kwargs["trend_fast"] = int(trend_fast)
    if trend_slow is not None:
        kwargs["trend_slow"] = int(trend_slow)
    table = classify_regimes(df, **kwargs)
    # classify_regimes already reports the mode; re-assert it here so the flag is
    # a property of THIS function's output and not of the callee's future edits.
    table.attrs["hmm_present"] = False
    table.attrs["causal_hmm"] = True
    table.attrs["regime_source"] = "causal_composite"
    return table


def enforce_causal_table(table: pd.DataFrame) -> None:
    """Raise unless *table*'s labels are a causal source P7 may condition on.

    Two refusals, both named:

    * the table carries HMM labels that were fitted on the whole sample
      (``attrs["causal_hmm"] is False``) ⇒
      :class:`~core.strategy.regime.NonCausalRegimeError` — the existing error,
      reused rather than re-invented;
    * the table came from a source that is not the causal composite classifier
      (an explicit ``attrs["regime_source"]`` that is not
      ``"causal_composite"``, e.g. a hand-built frame of HMM states) ⇒
      :class:`UnknownRegimeSourceError`.

    A table with **no** ``attrs`` at all is refused too: "trust me" is not a
    source.  The name of the column is not checked — the caller looks up
    ``regime`` — but a frame whose index is empty is accepted and yields no
    labels, which is the honest answer for an empty frame.
    """
    attrs = getattr(table, "attrs", None)
    if attrs is None:
        raise UnknownRegimeSourceError(
            "refusing to condition on a regime table that carries no attrs: the "
            "table must come from core.strategy.regime_causal.causal_regime_table")
    if bool(attrs.get("hmm_present", False)) and not bool(
            attrs.get("causal_hmm", False)):
        raise NonCausalRegimeError(
            "refusing to condition on whole-sample (look-ahead) HMM labels: "
            "rebuild the table with "
            "core.strategy.regime_causal.causal_regime_table or "
            "classify_regimes(..., with_hmm=False)")
    source = attrs.get("regime_source")
    if source is not None and str(source) != "causal_composite":
        raise UnknownRegimeSourceError(
            f"refusing to condition on regime labels from source '{source}'; the "
            f"only accepted source is 'causal_composite'")
    if source is None and not bool(attrs.get("causal_hmm", False)):
        raise UnknownRegimeSourceError(
            "refusing to condition on a regime table whose causal flag is unset")
    return None


# ── The per-run lookup an evaluation uses ──────────────────────────────

class RegimeContext:
    """Index→label cache for one evaluation run, keyed by (symbol, interval).

    Built once per run (not per genome) from the **causal** table of each
    ``(symbol, interval)`` the run's genomes actually name, so a 30-genome
    generation labels each symbol once.  ``labels`` maps the tuple to a numpy
    array of strings aligned with ``index`` (the bar timestamps), and
    :meth:`lookup` answers with the label of the last bar at or before a
    timestamp — the same convention ``core.ga.fitness.VolumeContext`` uses for a
    trade stamp, and the only causal reading: a bar may consume the label of
    itself or of an earlier bar, never of a later one.

    ``summaries`` records what each label array looks like (bar count, first/last
    timestamp and the label histogram), so a run can report the sample a
    conditioned evaluation was restricted to instead of asserting it.
    """

    __slots__ = ("index", "labels", "summaries")

    def __init__(self, index: dict | None = None,
                 labels: dict | None = None,
                 summaries: dict | None = None):
        self.index = index or {}
        self.labels = labels or {}
        self.summaries = summaries or {}

    def __bool__(self) -> bool:
        return bool(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def keys(self) -> list[tuple]:
        return sorted(self.labels)

    def lookup(self, symbol: str, interval: str, ts):
        """The causal label at or before *ts*, or ``None`` when unmapped."""
        key = (str(symbol), str(interval or "1h"))
        index = self.index.get(key)
        labels = self.labels.get(key)
        if index is None or labels is None or len(index) == 0 or ts is None:
            return None
        try:
            stamp = np.datetime64(pd.Timestamp(ts).to_datetime64())
        except Exception:
            return None
        cut = int(np.searchsorted(index, stamp, side="right"))
        if cut <= 0:
            return None
        return str(labels[cut - 1])

    def allow(self, symbol: str, interval: str, ts, regime_filter) -> bool:
        """Checked ``regime_allows`` for one bar of this context."""
        return regime_allows(regime_filter, self.lookup(symbol, interval, ts))

    def counts(self, symbol: str, interval: str) -> dict:
        """``{label: bars}`` for one key (empty dict when unmapped)."""
        return dict(self.summaries.get((str(symbol), str(interval)), {})
                    .get("counts") or {})


def _label_summary(table: pd.DataFrame) -> dict:
    counts: dict[str, int] = {}
    if len(table):
        for label, n in table["regime"].value_counts().items():
            counts[str(label)] = int(n)
    return {
        "bars": int(len(table)),
        "first": (str(table.index[0]) if len(table) else None),
        "last": (str(table.index[-1]) if len(table) else None),
        "counts": counts,
        "causal_hmm": bool(getattr(table, "attrs", {}).get("causal_hmm", False)),
        "hmm_present": bool(getattr(table, "attrs", {}).get("hmm_present", False)),
        "regime_source": getattr(table, "attrs", {}).get("regime_source"),
    }


def build_regime_context(frames: dict, *, min_bars: int = MIN_REGIME_ROWS,
                         ) -> RegimeContext | None:
    """``{(symbol, interval): OHLCV frame}`` → :class:`RegimeContext`.

    *frames* may be keyed ``(symbol, interval)`` or ``symbol`` (an interval is
    then assumed to be the caller's single one and is filled in by the caller's
    key convention — a plain ``symbol`` key is stored under
    ``(symbol, "1h")``).  Returns ``None`` when no key yields a usable frame, so
    a caller can treat "no regime data" exactly like "no filter": the shipped
    default reads nothing here.

    A frame shorter than *min_bars* is skipped (its labels would all be
    ``range_unknown`` / ``range``) rather than admitted as a table full of
    gate-able-looking labels.
    """
    index: dict = {}
    labels: dict = {}
    summaries: dict = {}
    for key, df in (frames or {}).items():
        if df is None or len(df) == 0:
            continue
        if isinstance(key, tuple):
            symbol, interval = str(key[0]), str(key[1] or "1h")
        else:
            symbol, interval = str(key), "1h"
        if len(df) < int(min_bars):
            continue
        table = causal_regime_table(df)
        enforce_causal_table(table)
        canonical = (symbol, interval)
        index[canonical] = table.index.to_numpy()
        labels[canonical] = table["regime"].astype(str).to_numpy()
        summaries[canonical] = _label_summary(table)
    if not labels:
        return None
    return RegimeContext(index, labels, summaries)


__all__ = [
    "GATE_REGIME_LABELS", "IN_SAMPLE_LABELS", "UNKNOWN_REGIME",
    "RegimeFilterError", "UnknownRegimeLabelError", "InSampleRegimeLabelError",
    "UnknownRegimeSourceError",
    "parse_regime_filter", "allowed_regimes", "regime_allows",
    "causal_regime_table", "enforce_causal_table", "RegimeContext",
    "build_regime_context",
    # re-exported so a caller needs one import for the gate vocabulary + the
    # refusals (``NonCausalRegimeError`` is regime.py's own class, reused).
    "NonCausalRegimeError", "classify_regimes", "trend_regimes",
    "volatility_terciles",
]
