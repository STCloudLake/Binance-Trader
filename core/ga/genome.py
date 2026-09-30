"""Strategy genome encoding — maps between YAML StrategyConfig and GA chromosomes.

Each strategy is encoded as a mixed-type chromosome:

    [continuous_params | categorical | entry_conditions | exit_conditions]

- Continuous: indicator periods, thresholds → Gaussian mutation
- Categorical: mode, timeframes → random replacement
- Structural: entry/exit condition strings → crossover-exchange, add/remove mutation
"""

import copy
import random
import re
import itertools
from dataclasses import dataclass, field
from core.strategy.loader import (
    StrategyConfig, MLConfig, normalize_condition_logic,
)
from core.market_data.provider import (
    DEFAULT_INTERVALS, DEFAULT_TIMEFRAME, interval_minutes)


# ── Gene definitions ──────────────────────────────────────────────────

@dataclass
class ContinuousGene:
    """A numeric gene with range and mutation step."""
    name: str
    value: float
    min_val: float
    max_val: float
    step: float = 1.0  # for integer-ish params (periods)

    def mutate(self, strength: float = 1.0):
        delta = random.gauss(0, self.step * strength)
        self.value = max(self.min_val, min(self.max_val, self.value + delta))
        if self.step >= 1.0:
            self.value = round(self.value)


@dataclass
class CategoricalGene:
    """A gene with discrete choices."""
    name: str
    value: str
    options: list[str]

    def mutate(self):
        self.value = random.choice([o for o in self.options if o != self.value])


@dataclass
class BooleanGene:
    """On/off switch gene — controls whether an indicator/feature is active."""
    name: str
    value: bool = True

    def mutate(self):
        if random.random() < 0.15:
            self.value = not self.value


@dataclass
class StructuralGene:
    """Entry/exit conditions as a list of condition strings."""
    name: str  # e.g. "entry_long"
    conditions: list[str] = field(default_factory=list)
    template_pool: list[str] = field(default_factory=list)

    def mutate_add(self):
        if self.template_pool:
            new_cond = random.choice(self.template_pool)
            if new_cond not in self.conditions:
                self.conditions.append(new_cond)

    def mutate_remove(self):
        if len(self.conditions) > 1:
            self.conditions.pop(random.randrange(len(self.conditions)))

    def mutate(self):
        if random.random() < 0.5 and len(self.conditions) > 1:
            self.mutate_remove()
        else:
            self.mutate_add()


# ── Chromosome ↔ StrategyConfig ───────────────────────────────────────

# ── P6-D: volume / flow condition templates ───────────────────────────
#
# Every template below is a plain condition string evaluated by the ONE shared
# kernel (``core.strategy.indicators.evaluate_condition``) on a frame built by
# ``compute_all``.  The scalar path (``evaluate_entry_conditions`` /
# ``StrategyConfig.entry_sides``) and the vectorised path
# (``core/backtest/signal_matrix.py``) both call that exact function, so the new
# templates do not fork the evaluator — they are new *strings*, not a new
# predicate engine.
#
# They read only columns that exist on EVERY frame (the raw bar columns plus the
# auto-derived ``volume_sma`` / ``volume_ratio``) or a column an existing
# indicator gene owns (``obv``).  ``TEMPLATE_REQUIRED_COLUMNS`` declares that
# (and ``COLUMN_INDICATOR_OWNER`` is the indicator availability mapping);
# ``audit_template_ownership`` — run over every pool by
# ``tests/test_ga_volume_genes.py`` — refuses a template with no owner, a
# declaration that disagrees with the string, or an owner the sanitiser ignores.

#: Rolling 20-bar VWAP, ``Σ(close·volume)/Σ(volume)``.  ``sma(x, n)`` in the
#: condition language is a rolling mean, so the ratio of two ``sma`` calls is the
#: volume-weighted average price — no new indicator column is needed.
_VWAP20 = "sma(close * volume, 20) / sma(volume, 20)"

#: RVOL z-score of ``volume_ratio`` against its own 60-bar mean/variance.
#: The condition language has no sqrt or power operator, so ``|z| > k`` is
#: written as the equivalent ``z² > k²`` with the variance expanded as
#: ``E[x²] − E[x]²``; the trailing comparison carries the sign of ``z``.
#:
#: The two thresholds are deliberately different.  ``volume_ratio`` is
#: right-skewed (volume_ratio ≥ 0 and spikes are one-sided), so on 11 549 cached
#: BTCUSDT 1h bars P(z < −1) = 4.79 % but P(z < −1.5) = 0.026 % and
#: P(z < −2) = 0.000 %: a symmetric ±2 would make the "dry volume" template
#: unreachable — a template that never fires is exactly what the P6 plan says to
#: roll back, so the low side uses the measured reachable threshold.
def _rvol_z_squared(k: float) -> str:
    return ("(volume_ratio - sma(volume_ratio, 60))"
            " * (volume_ratio - sma(volume_ratio, 60))"
            f" > {k * k} * (sma(volume_ratio * volume_ratio, 60)"
            " - sma(volume_ratio, 60) * sma(volume_ratio, 60))")


RVOL_Z_THRESHOLD_HIGH = 2.0
RVOL_Z_THRESHOLD_LOW = 1.0
RVOL_ZSPIKE_HIGH = (f"{_rvol_z_squared(RVOL_Z_THRESHOLD_HIGH)}"
                    " and volume_ratio > sma(volume_ratio, 60)")
RVOL_ZDRY_LOW = (f"{_rvol_z_squared(RVOL_Z_THRESHOLD_LOW)}"
                 " and volume_ratio < sma(volume_ratio, 60)")

VWAP_RECLAIM = f"cross(close, {_VWAP20})"
VWAP_LOSS = f"cross({_VWAP20}, close)"
VWAP_BELOW = f"close < {_VWAP20}"
VWAP_ABOVE = f"close > {_VWAP20}"

#: OBV **slope**: a fast mean of the OBV line against a slow one.  The level
#: (``obv > obv_sma``) is the pre-P6 template; this one is the direction.
OBV_SLOPE_UP = "sma(obv, 5) > sma(obv, 20)"
OBV_SLOPE_DOWN = "sma(obv, 5) < sma(obv, 20)"

#: A/D **slope**: the accumulation/distribution line's increment per bar is
#: ``CLV · volume`` with ``CLV = ((close−low) − (high−close)) / (high−low)``.
#: Its 20-bar slope is proportional to the mean of those increments, and
#: ``close``/``high``/``low``/``volume`` are raw columns — no cumulative column.
_CLV_VOLUME = "((close - low) - (high - close)) / (high - low) * volume"
AD_SLOPE_UP = f"sma({_CLV_VOLUME}, 20) > 0"
AD_SLOPE_DOWN = f"sma({_CLV_VOLUME}, 20) < 0"

#: MFI(14) on the typical price ``tp = high+low+close`` (∝ ``(h+l+c)/3``) with the
#: one-bar change ``d = tp − sma(tp, 2)`` (``sma(tp, 2) = (tpₜ+tpₜ₋₁)/2``).
#: Signed money flow is ``mfd = tp · volume · d``; with ``pos = (mfd+|mfd|)/2``
#: and ``neg = (|mfd|−mfd)/2``, ``MFI = 100 − 100/(1+pos/neg)``, so
#: ``MFI > 80 ⇔ pos > 4·neg ⇔ sma(mfd) > 0.6·sma(|mfd|)`` and
#: ``MFI < 20 ⇔ sma(mfd) < −0.6·sma(|mfd|)`` (the 0.6 is exact, not a fit).
_MONEY_FLOW = ("(high + low + close) * volume"
               " * ((high + low + close) - sma(high + low + close, 2))")
MFI_OVERBOUGHT = f"sma({_MONEY_FLOW}, 14) > 0.6 * sma(abs({_MONEY_FLOW}), 14)"
MFI_OVERSOLD = f"sma({_MONEY_FLOW}, 14) < -0.6 * sma(abs({_MONEY_FLOW}), 14)"

#: Volume/price divergence: price and volume moved in OPPOSITE directions on the
#: bar (``x − sma(x, 2) = Δx/2``, so the comparison is a genuine one-bar change),
#: i.e. the move has no participation behind it.  Bullish = price fell while
#: volume dried up (seller exhaustion); bearish = price rose while volume dried
#: up (buyer exhaustion).
VP_DIVERGENCE_BULL = "close < sma(close, 2) and volume < sma(volume, 2)"
VP_DIVERGENCE_BEAR = "close > sma(close, 2) and volume < sma(volume, 2)"

# Template condition pool for mutation
CONDITION_POOL = {
    "long": [
        "rsi < 30",
        "rsi < 35",
        "macd_histogram > 0",
        "close > bollinger_lower",
        "close > ema_fast",
        "ema_fast > ema_slow",
        "volume_ratio > 1.5",
        "volume_ratio > 2.0",
        "adx > 20",
        "adx > 25",
        "stoch_k < 20",
        "stoch_k > stoch_d",
        "cci < -100",
        "cci < -200",
        "atr_ratio > 1.5",
        "obv > obv_sma",
        "close > sma",
        # Phase 4c: market structure
        "dist_to_low_pct < 0.02",
        "swing_range_pct > 0.03",
        "hurst > 0.55",
        # P6-D: volume / flow
        RVOL_ZSPIKE_HIGH,
        VWAP_RECLAIM,
        OBV_SLOPE_UP,
        AD_SLOPE_UP,
        MFI_OVERSOLD,
        VP_DIVERGENCE_BULL,
    ],
    "short": [
        "rsi > 70",
        "rsi > 65",
        "macd_histogram < 0",
        "close < bollinger_upper",
        "close < ema_fast",
        "ema_fast < ema_slow",
        "volume_ratio > 1.5",
        "volume_ratio > 2.0",
        "adx > 20",
        "adx > 25",
        "stoch_k > 80",
        "stoch_k < stoch_d",
        "cci > 100",
        "cci > 200",
        "atr_ratio < 0.7",
        "obv < obv_sma",
        "close < sma",
        # Phase 4c: market structure
        "dist_to_high_pct < 0.02",
        "swing_range_pct > 0.03",
        "hurst < 0.45",
        # P6-D: volume / flow
        RVOL_ZSPIKE_HIGH,
        VWAP_LOSS,
        OBV_SLOPE_DOWN,
        AD_SLOPE_DOWN,
        MFI_OVERBOUGHT,
        VP_DIVERGENCE_BEAR,
    ],
}

EXIT_CONDITION_POOL = {
    "long": [
        "rsi > 65",
        "rsi > 55",
        "close < bollinger_middle",
        "close < ema_slow",
        "close < bollinger_lower",
        "stoch_k > 75",
        "cci > 150",
        "close < sma",
        # P6-D: volume / flow
        VWAP_BELOW,
        RVOL_ZDRY_LOW,
        MFI_OVERBOUGHT,
    ],
    "short": [
        "rsi < 35",
        "rsi < 45",
        "close > bollinger_middle",
        "close > ema_slow",
        "close > bollinger_upper",
        "stoch_k < 25",
        "cci < -150",
        "close > sma",
        # P6-D: volume / flow
        VWAP_ABOVE,
        RVOL_ZDRY_LOW,
        MFI_OVERSOLD,
    ],
}

MODE_OPTIONS = ["trend", "range", "scalp", "momentum"]
#: Timeframes the GA evolves over — the intervals INTERVAL_SPEC declares as the
#: streamed/pre-fetched set, so a new interval added there joins the genes too.
TIMEFRAME_OPTIONS = list(DEFAULT_INTERVALS)

INDICATOR_NAMES = ["rsi", "macd", "bollinger", "adx", "ema", "atr", "stoch", "cci", "obv", "sma", "hurst", "swing_points", "frac_diff"]

# ── First-class volume / flow indicator columns (audit: clean columns) ──
#
# `rvol`/`rvol_z`/`vwap`/`mfi`/`ad_line`/`obv_slope` are the P6-B feature family's
# own series under readable names, produced by `compute_all` when — and only when
# — the indicator config carries :data:`VOLUME_FLOW_INDICATOR`
# (`core/strategy/indicators.py`; on demand because the family costs ~96 ms per
# 8 844 bars, measured, and most strategies read none of it).
#
# They are NOT a GA gene: adding one would change `random_chromosome`, i.e. the
# shipped search space.  The decoder instead enables the key automatically for any
# candidate condition that references one of the columns
# (:func:`_condition_reads_volume_flow`), which is what keeps
# "template references a column" and "the column is produced" one invariant — the
# alternative is the `'adx' is not defined` defect class the sanitiser exists for.
VOLUME_FLOW_INDICATOR = "volume_flow"
VOLUME_FLOW_COLUMNS = ("rvol", "rvol_z", "vwap", "mfi", "ad_line", "obv_slope")


def _condition_reads_volume_flow(condition) -> bool:
    """True when *condition* reads one of :data:`VOLUME_FLOW_COLUMNS`.

    Parsed with the condition grammar's own identifier extractor (so a column
    name inside a string or a call name cannot trigger it); an unparseable
    condition falls back to a substring test, which can only enable the key —
    never drop a column a surviving condition needs.
    """
    if not isinstance(condition, str) or not condition.strip():
        return False
    try:
        identifiers = template_identifier_columns(condition)
    except SyntaxError:
        return any(column in condition for column in VOLUME_FLOW_COLUMNS)
    return bool(identifiers & set(VOLUME_FLOW_COLUMNS))

# Indicator inclusion probability for random init (avoid all-on/all-off extremes)
INDICATOR_INIT_PROB = {
    "rsi": 0.6, "macd": 0.6, "bollinger": 0.5, "adx": 0.4, "ema": 0.4,
    "atr": 0.3, "stoch": 0.4, "cci": 0.3, "obv": 0.3, "sma": 0.3,
    "hurst": 0.3, "swing_points": 0.35, "frac_diff": 0.2,
}

# New indicator gene ranges (min, max, step, default)
NEW_GENE_RANGES = {
    "atr_period": (7, 28, 1, 14),
    "stoch_k_period": (5, 21, 1, 14),
    "stoch_d_period": (3, 9, 1, 3),
    "cci_period": (7, 28, 1, 14),
    "sma_period": (10, 100, 2, 50),
    # Phase 4c
    "hurst_lookback": (50, 200, 10, 100),
    "swing_lookback": (3, 10, 1, 5),
    "frac_diff_d": (10, 60, 5, 40),  # stored as int*100 → 0.10-0.60
    # P6-D volume genes (0.0 = off; both neutral by default)
    "volume_filter_rvol": (0.0, 3.0, 0.1, 0.0),
    "volume_scale_k": (0.0, 1.0, 0.05, 0.0),
}

#: Columns ``compute_all`` (core/strategy/indicators.py) ALWAYS adds to the frame,
#: whatever the indicator config is: the raw close, the volume ratio, EMA9/EMA21
#: (backfilled when the config does not ask for its own EMA pair) and the plain
#: ``sma`` alias.  A condition may reference these even when the matching
#: indicator gene is off, so sanitisation must keep them.
ALWAYS_AVAILABLE_COLUMNS = {"close", "volume_ratio", "ema_fast", "ema_slow", "sma"}

#: Condition → required indicator mapping (for sanitization)
CONDITION_INDICATOR_MAP = {
    "rsi": ["rsi"],
    "macd_histogram": ["macd"],
    "bollinger_lower": ["bollinger"], "bollinger_upper": ["bollinger"],
    "bollinger_middle": ["bollinger"],
    "ema_fast": ["ema"], "ema_slow": ["ema"],
    "adx": ["adx"],
    "stoch_k": ["stoch"], "stoch_d": ["stoch"],
    "cci": ["cci"],
    "atr_ratio": ["atr"],
    "obv": ["obv"], "obv_sma": ["obv"],
    "sma": ["sma"],
    # Phase 4c
    "dist_to_high_pct": ["swing_points"], "dist_to_low_pct": ["swing_points"],
    "swing_range_pct": ["swing_points"],
    "swing_high": ["swing_points"], "swing_low": ["swing_points"],
    "hurst": ["hurst"], "hurst_signal": ["hurst"],
    "frac_close": ["frac_diff"],
    "volume_ratio": [], "close": [],
    # P6-D: the raw bar columns the volume/flow templates read are present on
    # every frame, so they carry no indicator requirement.  Declared explicitly
    # so the mapping stays in step with ``RAW_ALWAYS_AVAILABLE_COLUMNS``.
    "volume": [], "high": [], "low": [], "volume_sma": [],
    # First-class volume/flow columns (audit: "clean indicator columns").  They
    # are produced on demand by ``compute_all(df, {"volume_flow": {}})``, so a
    # condition that reads one of them *requires* that config key — sanitisation
    # drops it otherwise, exactly like `adx` requires the adx gene.
    **{column: [VOLUME_FLOW_INDICATOR] for column in VOLUME_FLOW_COLUMNS},
}

# ── P6-D: indicator availability + template ownership ──────────────────
#
# Two tables and one auditor, so "no orphan template" is a property a test can
# fail, not a promise:
#
# * ``RAW_ALWAYS_AVAILABLE_COLUMNS`` — the columns ``compute_all`` adds to every
#   frame whatever the indicator config is (raw bars + auto-derived volume
#   statistics).  A template reading only these needs no indicator gene.
# * ``COLUMN_INDICATOR_OWNER`` — column → indicator gene ``compute_all`` needs to
#   add it (``None`` = raw/always available).
# * ``TEMPLATE_REQUIRED_COLUMNS`` — every template in every pool → the columns it
#   reads.  Declared, not inferred: ``audit_template_ownership`` compares the
#   declaration with the identifiers actually parsed out of the string and with
#   what the sanitiser does, so a stale or missing entry fails loudly.

#: Columns present on every OHLCV frame ``compute_all`` is handed.
RAW_ALWAYS_AVAILABLE_COLUMNS = frozenset({
    "open", "high", "low", "close", "volume", "volume_sma", "volume_ratio",
})

#: Columns the *sanitiser* keeps even though ``compute_all`` only adds them when
#: the matching indicator gene is on.  ``sma`` is the single pre-existing
#: exemption (``compute_all`` writes ``sma``/``sma_{period}`` only for the ``sma``
#: indicator, while ``ALWAYS_AVAILABLE_COLUMNS`` — P1's sanitisation contract —
#: lists it).  Recorded here so the guard can assert the set cannot grow
#: silently: that is a change to ``core/strategy/**``, outside P6-D's scope.
SANITISER_ONLY_COLUMNS = frozenset({"sma"})

#: Column → indicator gene required for ``compute_all`` to add it.
COLUMN_INDICATOR_OWNER: dict[str, str | None] = {
    "open": None, "high": None, "low": None, "close": None,
    "volume": None, "volume_sma": None, "volume_ratio": None,
    "ema_fast": None, "ema_slow": None,          # backfilled for every frame
    "sma": "sma",
    "rsi": "rsi", "macd_histogram": "macd",
    "bollinger_lower": "bollinger", "bollinger_middle": "bollinger",
    "bollinger_upper": "bollinger",
    "adx": "adx", "stoch_k": "stoch", "stoch_d": "stoch", "cci": "cci",
    "atr_ratio": "atr", "obv": "obv", "obv_sma": "obv",
    "hurst": "hurst", "hurst_signal": "hurst",
    "swing_high": "swing_points", "swing_low": "swing_points",
    "dist_to_high_pct": "swing_points", "dist_to_low_pct": "swing_points",
    "swing_range_pct": "swing_points", "frac_close": "frac_diff",
    # First-class volume/flow columns: produced only when the `volume_flow`
    # indicator key is on, so they are owned, not raw.
    **{column: VOLUME_FLOW_INDICATOR for column in VOLUME_FLOW_COLUMNS},
}

#: Indicator gene → the config ``compute_all`` needs to produce its columns.
#: Used by the guard to prove every declared column really is producible.
INDICATOR_CONFIG_FOR_COLUMNS: dict[str, dict] = {
    "rsi": {"period": 14, "source": "close"},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"period": 20, "stddev": 2.0},
    "adx": {"period": 14},
    "ema": {"fast_period": 9, "slow_period": 21, "source": "close"},
    "atr": {"period": 14},
    "stoch": {"period": 14, "slowk_period": 3, "slowd_period": 3},
    "cci": {"period": 14},
    "obv": {"period": 14},
    "sma": {"period": 20},
    "hurst": {"lookback": 100},
    "swing_points": {"lookback": 5},
    "frac_diff": {"d": 0.4},
    # The volume/flow column family takes no parameters.
    VOLUME_FLOW_INDICATOR: {},
}

#: Every pool template → the columns it reads.  Compact by construction: a
#: template's row groups templates that share a column set.
TEMPLATE_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {}
for _cols, _templates in (
    (("rsi",), ("rsi < 30", "rsi < 35", "rsi > 70", "rsi > 65",
                "rsi > 55", "rsi < 35", "rsi < 45", "rsi > 65", "rsi < 20",
                "rsi > 80")),
    (("macd_histogram",), ("macd_histogram > 0", "macd_histogram < 0")),
    (("close", "bollinger_lower"), ("close > bollinger_lower",
                                    "close < bollinger_lower")),
    (("close", "bollinger_upper"), ("close > bollinger_upper",
                                    "close < bollinger_upper")),
    (("close", "bollinger_middle"), ("close < bollinger_middle",
                                     "close > bollinger_middle")),
    (("close", "ema_fast"), ("close > ema_fast", "close < ema_fast")),
    (("close", "ema_slow"), ("close < ema_slow", "close > ema_slow")),
    (("ema_fast", "ema_slow"), ("ema_fast > ema_slow", "ema_fast < ema_slow")),
    (("volume_ratio",), ("volume_ratio > 1.5", "volume_ratio > 2.0")),
    (("adx",), ("adx > 20", "adx > 25")),
    (("stoch_k",), ("stoch_k < 20", "stoch_k > 80", "stoch_k > 75",
                    "stoch_k < 25")),
    (("stoch_k", "stoch_d"), ("stoch_k > stoch_d", "stoch_k < stoch_d")),
    (("cci",), ("cci < -100", "cci < -200", "cci > 100", "cci > 200",
                "cci > 150", "cci < -150")),
    (("atr_ratio",), ("atr_ratio > 1.5", "atr_ratio < 0.7")),
    (("obv", "obv_sma"), ("obv > obv_sma", "obv < obv_sma")),
    (("close", "sma"), ("close > sma", "close < sma")),
    (("dist_to_low_pct",), ("dist_to_low_pct < 0.02",)),
    (("dist_to_high_pct",), ("dist_to_high_pct < 0.02",)),
    (("swing_range_pct",), ("swing_range_pct > 0.03",)),
    (("hurst",), ("hurst > 0.55", "hurst < 0.45")),
    # ── P6-D: volume / flow ──
    (("volume_ratio",), (RVOL_ZSPIKE_HIGH, RVOL_ZDRY_LOW)),
    (("close", "volume"), (VWAP_RECLAIM, VWAP_LOSS, VWAP_BELOW, VWAP_ABOVE,
                           VP_DIVERGENCE_BULL, VP_DIVERGENCE_BEAR)),
    (("obv",), (OBV_SLOPE_UP, OBV_SLOPE_DOWN)),
    (("high", "low", "close", "volume"), (AD_SLOPE_UP, AD_SLOPE_DOWN,
                                          MFI_OVERBOUGHT, MFI_OVERSOLD)),
):
    for _template in _templates:
        TEMPLATE_REQUIRED_COLUMNS[_template] = tuple(_cols)
del _cols, _templates, _template


def template_identifier_columns(template: str) -> set[str]:
    """Column identifiers actually referenced by *template* (verified, not trusted).

    Parses the condition with the same grammar the evaluator accepts and returns
    the ``Name`` nodes that are **not** calls to the whitelisted condition
    functions (``sma``/``cross``/``abs``/``min``/``max``/``round``) — i.e. the
    columns the expression really reads.
    """
    import ast as _ast

    functions = {"sma", "cross", "abs", "min", "max", "round"}
    tree = _ast.parse(template, mode="eval")
    columns = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name):
            columns.add(node.id)
    return {c for c in columns if c not in functions}


def template_owners(template: str) -> set[str]:
    """Indicator genes *template* needs (from its declared columns)."""
    owners = set()
    for column in TEMPLATE_REQUIRED_COLUMNS.get(template, ()):
        owner = COLUMN_INDICATOR_OWNER.get(column)
        if owner:
            owners.add(owner)
    return owners


def audit_template_ownership(pools: dict[str, dict[str, list[str]]] | None = None,
                             extra: dict[str, tuple[str, ...]] | None = None
                             ) -> list[str]:
    """``[problem, ...]`` for the template pools — empty means "no orphans".

    Checks, for every template of every pool (entry/exit × long/short):

    1. it has a ``TEMPLATE_REQUIRED_COLUMNS`` declaration (missing ⇒ orphan);
    2. every declared column is in the availability mapping
       (``COLUMN_INDICATOR_OWNER`` / ``RAW_ALWAYS_AVAILABLE_COLUMNS``);
    3. the declaration matches the identifiers parsed from the string, and the
       string parses at all under the condition grammar;
    4. the sanitiser agrees with the ownership: the template survives
       ``_sanitize_conditions`` exactly when its owners are enabled.

    ``extra`` lets a test inject a deliberately broken template (for example one
    reading a column no indicator produces) together with the columns it claims;
    its problems are reported exactly like a pool template's.
    """
    pools = pools if pools is not None else {
        "entry": CONDITION_POOL, "exit": EXIT_CONDITION_POOL}
    declared = dict(TEMPLATE_REQUIRED_COLUMNS)
    injected = dict(extra or {})
    all_indicators = (set(INDICATOR_NAMES) | set(ALWAYS_AVAILABLE_COLUMNS)
                      | {VOLUME_FLOW_INDICATOR})
    problems: list[str] = []

    audit_targets: list[tuple[str, str]] = []
    for pool_name, sides in pools.items():
        for side, templates in sides.items():
            audit_targets.extend((f"{pool_name}.{side}", t) for t in templates)
    audit_targets.extend(("injected", t) for t in injected)

    for where, template in audit_targets:
        if template not in declared and template not in injected:
            problems.append(f"{where}: orphan template (no declared "
                            f"columns): {template[:60]}")
            continue
        columns = injected.get(template, declared.get(template, ()))
        try:
            parsed = template_identifier_columns(template)
        except SyntaxError as exc:
            problems.append(f"{where}: unparseable template "
                            f"({exc}): {template[:60]}")
            continue
        missing = parsed - set(columns)
        if missing:
            problems.append(f"{where}: undeclared column(s) "
                            f"{sorted(missing)} in {template[:60]}")
        unknown = [c for c in columns
                   if c not in COLUMN_INDICATOR_OWNER
                   and c not in RAW_ALWAYS_AVAILABLE_COLUMNS]
        if unknown:
            problems.append(f"{where}: column(s) {sorted(unknown)} are "
                            f"not produced by any indicator: "
                            f"{template[:60]}")
        # The sanitiser must keep the template iff its owners are on.  Owners come
        # from the columns *this* call resolved (so an injected template is
        # audited too), not from the pool-only declaration table.
        owners = {COLUMN_INDICATOR_OWNER[c] for c in columns
                  if COLUMN_INDICATOR_OWNER.get(c)}
        side = "long"
        for pool_name, sides in pools.items():
            for _side, templates in sides.items():
                if template in templates:
                    side = _side
        with_all = _sanitize_conditions([template], all_indicators, side)
        if template not in with_all:
            problems.append(f"{where}: sanitiser drops the template even "
                            f"with every indicator enabled: "
                            f"{template[:60]}")
        for owner in sorted(owners):
            without = _sanitize_conditions(
                [template], all_indicators - {owner}, side)
            if template in without:
                problems.append(
                    f"{where}: sanitiser keeps the template while its "
                    f"owner '{owner}' is off: {template[:60]}")
    return problems


# ── P6-D: the two volume genes ────────────────────────────────────────
#
# ``volume_filter_rvol`` — "only take entries while RVOL > x", 0.0 = off.
# ``volume_scale_k``    — "scale the traded size with recent volume", 0.0 = off.
#
# Both are neutral by default, so a config/chromosome that does not carry them
# decodes exactly as it did before P6-D.

#: Reserved pseudo-indicator key that parks the two volume genes inside a decoded
#: ``StrategyConfig``.  ``StrategyConfig`` is the runtime schema and carries no
#: filter/sizing field; P6-D's write scope deliberately excludes
#: ``core/strategy/loader.py``, so the genes ride in the free-form
#: ``indicators`` dict — which ``compute_all`` walks and ignores for any name its
#: elif chain does not know, i.e. the key is inert at evaluation time.  It is
#: written ONLY when a gene is non-neutral (a pre-P6 config decodes byte-for-byte
#: as before) and exists so the genome round-trips and a champion YAML stays
#: self-describing about the genes it was scored with.
VOLUME_GENE_INDICATOR_KEY = "_ga_volume_genes"

#: Documented gene bounds (docs/core-algorithms/16-ga-volume-genes.md).
VOLUME_FILTER_MAX_RVOL = 3.0
VOLUME_FILTER_STEP = 0.1
VOLUME_SCALE_MAX_K = 1.0
VOLUME_SCALE_STEP = 0.05

#: The conjunct the filter gene renders into every entry condition.
_VOLUME_FILTER_SUFFIX = re.compile(
    r"^(?P<base>\(.*\))\s+and\s+\(volume_ratio > (?P<value>[0-9.]+)\)$", re.S)


def volume_filter_condition(rvol: float) -> str:
    """The conjunct the filter gene adds: ``volume_ratio > rvol``."""
    return f"volume_ratio > {float(rvol):g}"


def with_volume_filter(condition: str, rvol: float) -> str:
    """AND the RVOL filter into *condition*.

    ANDing the filter into **each** entry condition is what makes the gene a
    real filter under both entry structures the decoder can emit:

        OR_i (c_i ∧ f)  =  (OR_i c_i) ∧ f
        AND_i (c_i ∧ f) =  (AND_i c_i) ∧ f

    so a volume filter narrows entries whether ``condition_logic`` is ``or`` or
    ``and``, through the ONE shared kernel (``evaluate_condition``) — no bespoke
    evaluation path, scalar and vectorised alike.
    """
    return f"({condition}) and ({volume_filter_condition(rvol)})"


def strip_volume_filter(condition: str) -> tuple[str, float | None]:
    """Inverse of :func:`with_volume_filter` → ``(base, rvol)``.

    Only the *wrapped* form matches, so the plain pool template
    ``volume_ratio > 1.5`` is never mistaken for the filter gene.
    """
    match = _VOLUME_FILTER_SUFFIX.match(condition or "")
    if not match:
        return condition, None
    # ``group("base")`` carries the wrapper's own parentheses; drop exactly those
    # so ``strip(with_volume_filter(c, v)) == (c, v)`` for every c.
    return match.group("base")[1:-1], float(match.group("value"))


def _volume_genes_from_conditions(config: StrategyConfig,
                                  parked: dict) -> tuple[float, float]:
    """``(filter_rvol, scale_k)`` recovered from a decoded config.

    The parked ``_ga_volume_genes`` entry is authoritative for the sizing gene
    (it has no condition footprint); the filter value is read back from the
    wrapped entry conditions so the emitted strategy — not a side-channel — is
    the source of truth for the filter that will actually be evaluated.
    """
    scale_k = 0.0
    try:
        scale_k = float(parked.get("scale_k", 0.0) or 0.0)
    except (TypeError, ValueError):
        scale_k = 0.0
    scale_k = max(0.0, min(VOLUME_SCALE_MAX_K, round(scale_k, 3)))

    filter_rvol = 0.0
    for side in ("long", "short"):
        for condition in (config.entry_conditions or {}).get(side, []) or []:
            _base, value = strip_volume_filter(condition)
            if value is not None:
                filter_rvol = max(filter_rvol, value)
    try:
        filter_rvol = max(filter_rvol, float(parked.get("filter_rvol", 0.0) or 0.0))
    except (TypeError, ValueError):
        pass
    filter_rvol = max(0.0, min(VOLUME_FILTER_MAX_RVOL, round(filter_rvol, 1)))
    return filter_rvol, scale_k


def _coerce_volume_gene(value, maximum: float, ndigits: int) -> float:
    """Clamp one volume gene to its documented bounds (never raises)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, min(maximum, round(number, ndigits)))


def _randomise_volume_genes(chrom: dict) -> None:
    """Random initial values for the two volume genes (off half of the time).

    The genes must be *reachable* for the GA (acceptance: a fixed-seed run
    selects each new gene), but never forced on: 50 % neutral keeps the search
    honest and keeps the "off" path the default.
    """
    values = {
        "volume_filter_rvol": (0.0 if random.random() < 0.5
                               else round(random.uniform(1.1, 2.5), 1)),
        "volume_scale_k": (0.0 if random.random() < 0.5
                           else round(random.uniform(0.2, VOLUME_SCALE_MAX_K), 2)),
    }
    for gene in chrom.get("continuous", []):
        if gene.name in values:
            gene.value = values[gene.name]


def strategy_to_chromosome(config: StrategyConfig) -> dict:
    """Encode a StrategyConfig into a chromosome dict.

    Returns a dict with keys: continuous, categorical, structural
    that can be mutated and decoded back.
    """
    ind = config.indicators

    # ── Continuous genes ──
    continuous = []
    # RSI
    if "rsi" in ind:
        rsi = ind["rsi"]
        continuous.append(ContinuousGene("rsi_period", rsi.get("period", 14), 5, 28, 1))

    # MACD
    if "macd" in ind:
        macd = ind["macd"]
        continuous.append(ContinuousGene("macd_fast", macd.get("fast", 12), 6, 20, 2))
        continuous.append(ContinuousGene("macd_slow", macd.get("slow", 26), 18, 40, 2))
        continuous.append(ContinuousGene("macd_signal", macd.get("signal", 9), 5, 15, 1))

    # Bollinger
    if "bollinger" in ind:
        bb = ind["bollinger"]
        continuous.append(ContinuousGene("bb_period", bb.get("period", 20), 10, 40, 2))
        continuous.append(ContinuousGene("bb_stddev", bb.get("stddev", 2), 1.0, 3.5, 0.25))

    # ADX
    if "adx" in ind:
        adx = ind["adx"]
        continuous.append(ContinuousGene("adx_period", adx.get("period", 14), 7, 28, 1))

    # EMA — a FAST/SLOW pair, not a single period.  The old single
    # ``ema_period`` gene emitted ``{"ema": {"period": p}}``, which
    # ``compute_all`` turns into an ``ema_{p}`` column while every condition
    # template uses ``ema_fast``/``ema_slow`` — and those are backfilled to a
    # hardcoded EMA9/EMA21, so the gene was inert (5 vs 50 → identical 672
    # trades).  ``fast_period``/``slow_period`` is the schema ``compute_all``
    # actually reads to build ``ema_fast``/``ema_slow``.
    if "ema" in ind:
        ema = ind["ema"]
        fast_default = int(ema.get("fast_period", ema.get("period", 9)) or 9)
        slow_default = int(ema.get("slow_period", 21) or 21)
        continuous.append(ContinuousGene("ema_fast_period", fast_default, 5, 30, 2))
        continuous.append(ContinuousGene("ema_slow_period", slow_default, 12, 60, 2))

    # ATR
    if "atr" in ind:
        a = ind["atr"]
        continuous.append(ContinuousGene("atr_period", a.get("period", 14), 7, 28, 1))

    # Stochastic
    if "stoch" in ind:
        s = ind["stoch"]
        # Accept BOTH spellings: ``_random_indicators`` and pre-P6 checkpoints
        # write ``k_period``/``d_period``, while ``chromosome_to_strategy`` emits
        # the schema ``compute_all`` reads (``period``/``slowk_period``).  Reading
        # only the first spelling made encode(decode(x)) reset a decoded stoch
        # genome to 14/3/3 — it could never round-trip (found while pinning the
        # P6-D round-trip property; the GA itself never re-encodes a champion, so
        # only the seed-strategy path is affected).
        continuous.append(ContinuousGene(
            "stoch_k_period",
            s.get("k_period", s.get("period", 14)), 5, 21, 1))
        continuous.append(ContinuousGene(
            "stoch_d_period",
            s.get("d_period", s.get("slowk_period", 3)), 3, 9, 1))

    # CCI
    if "cci" in ind:
        c = ind["cci"]
        continuous.append(ContinuousGene("cci_period", c.get("period", 14), 7, 28, 1))

    # OBV — no continuous genes (parameterless)

    # Hurst
    if "hurst" in ind:
        h = ind["hurst"]
        continuous.append(ContinuousGene("hurst_lookback",
            h.get("lookback", 100), 50, 200, 10))

    # Swing Points
    if "swing_points" in ind:
        sp = ind["swing_points"]
        continuous.append(ContinuousGene("swing_lookback",
            sp.get("lookback", 5), 3, 10, 1))

    # Fractional Differentiation
    if "frac_diff" in ind:
        fd = ind["frac_diff"]
        continuous.append(ContinuousGene("frac_diff_d",
            int(fd.get("d", 0.4) * 100), 10, 60, 5))

    # SMA
    if "sma" in ind:
        sm = ind["sma"]
        continuous.append(ContinuousGene("sma_period", sm.get("period", 50), 10, 100, 2))

    # ── ML genes deliberately carry NO effect ──
    # Scoring and live entry must agree.  ``evaluate_chromosome`` disables
    # ``ml_config`` (fitness.py) while the decoder used to emit
    # ``enabled = weight > 0`` — measured fusion difference: ML off 0.5000,
    # weight 0.1 → 0.6250, weight 0.5 → 0.4167, i.e. a GA champion was traded
    # with an entry set it was never scored on.  The score path is the one that
    # is measured, so the emitted weight is pinned to 0.0: the genes stay in the
    # chromosome (mutation/crossover shape, checkpoint compatibility) but can
    # never re-introduce a live/scored mismatch.
    ml_weight = 0.0
    ml_threshold = 0.6
    if config.ml_config:
        ml_threshold = config.ml_config.confidence_threshold
    continuous.append(ContinuousGene("ml_weight", ml_weight, 0.0, 0.0, 0.05))
    continuous.append(ContinuousGene("ml_threshold", ml_threshold, 0.5, 0.85, 0.05))

    # ── P6-D: volume genes (both NEUTRAL by default) ──
    # ``volume_filter_rvol`` 0.0 = no filter; ``volume_scale_k`` 0.0 = no sizing
    # model.  They are recovered from the decoded config when it carries them
    # (see ``VOLUME_GENE_INDICATOR_KEY``), so encode→decode round-trips; a
    # pre-P6 config, a checkpoint chromosome or a hand-written YAML without them
    # decodes to exactly HEAD's config.
    volume_genes = (config.indicators or {}).get(VOLUME_GENE_INDICATOR_KEY) or {}
    if not isinstance(volume_genes, dict):
        volume_genes = {}
    filter_rvol, scale_k = _volume_genes_from_conditions(
        config, volume_genes)
    continuous.append(ContinuousGene(
        "volume_filter_rvol", filter_rvol, 0.0, VOLUME_FILTER_MAX_RVOL,
        VOLUME_FILTER_STEP))
    continuous.append(ContinuousGene(
        "volume_scale_k", scale_k, 0.0, VOLUME_SCALE_MAX_K, VOLUME_SCALE_STEP))

    # ── Categorical genes ──
    categorical = [
        CategoricalGene("mode", config.mode, MODE_OPTIONS),
    ]
    # Timeframes: store as comma-separated for GA; decode splits back
    categorical.append(
        CategoricalGene("timeframes", ",".join(config.timeframes),
                        [",".join(c) for c in itertools.combinations(TIMEFRAME_OPTIONS, 2)]))

    # ── Evolvable entry logic (OR = looser, AND = stricter) ──
    # A first-class ``StrategyConfig`` field (schema-level), so the chromosome
    # gene survives into the published YAML and the reloaded strategy is
    # evaluated with the same entry structure the genome was scored under.
    # ``getattr`` keeps duck-typed/legacy configs (no field) working as "or".
    condition_logic = normalize_condition_logic(
        getattr(config, "condition_logic", None))

    # ── Structural genes ──
    structural = []
    for side in ["long", "short"]:
        # The P6-D filter gene is *rendered into* the entry conditions (it has to
        # be, to be honoured by the shared kernel); encoding therefore unwraps it
        # again, so ``encode(decode(x))`` is the identity and the gene is not
        # applied twice on the next decode.
        entry = []
        for condition in config.entry_conditions.get(side, []) or []:
            base, _value = strip_volume_filter(condition)
            entry.append(base)
        structural.append(StructuralGene(
            f"entry_{side}", entry,
            template_pool=CONDITION_POOL.get(side, [])))

    for side in ["long", "short"]:
        exit_conds = config.exit_conditions.get(side, [])
        structural.append(StructuralGene(
            f"exit_{side}", list(exit_conds),
            template_pool=EXIT_CONDITION_POOL.get(side, [])))

    # ── Indicator boolean genes ──
    indicator_genes = []
    for name in INDICATOR_NAMES:
        enabled = name in config.indicators
        indicator_genes.append(BooleanGene(name, enabled))

    return {
        "continuous": continuous,
        "categorical": categorical,
        "structural": structural,
        "indicator_genes": indicator_genes,
        "condition_logic": condition_logic,
        "name": config.name,
    }


def chromosome_to_strategy(chromosome: dict) -> StrategyConfig:
    """Decode a chromosome dict back into a StrategyConfig."""
    cont = {g.name: g.value for g in chromosome["continuous"]}
    cat = {g.name: g.value for g in chromosome["categorical"]}
    struct = {g.name: g.conditions for g in chromosome["structural"]}

    # Read indicator boolean genes (backward compat: all True if missing)
    ind_genes_list = chromosome.get("indicator_genes", [])
    ind_genes = {g.name: g.value for g in ind_genes_list} if ind_genes_list else {n: True for n in INDICATOR_NAMES}

    indicators = {}
    if ind_genes.get("rsi", True) and "rsi_period" in cont:
        indicators["rsi"] = {"period": int(cont.get("rsi_period", 14)), "source": "close"}
    if ind_genes.get("macd", True) and "macd_fast" in cont:
        indicators["macd"] = {
            "fast": int(cont.get("macd_fast", 12)),
            "slow": int(cont.get("macd_slow", 26)),
            "signal": int(cont.get("macd_signal", 9)),
        }
    if ind_genes.get("bollinger", True) and "bb_period" in cont:
        indicators["bollinger"] = {
            "period": int(cont.get("bb_period", 20)),
            "stddev": round(cont.get("bb_stddev", 2.0), 2),
        }
    if ind_genes.get("adx", True) and "adx_period" in cont:
        indicators["adx"] = {"period": int(cont.get("adx_period", 14))}
    if ind_genes.get("ema", True) and ("ema_fast_period" in cont or "ema_period" in cont):
        # Accept the legacy single ``ema_period`` gene too (old checkpoints).
        fast_default = int(cont.get("ema_fast_period", cont.get("ema_period", 9)) or 9)
        slow_default = int(cont.get("ema_slow_period", 21) or 21)
        if slow_default <= fast_default:
            slow_default = fast_default + 5
        indicators["ema"] = {
            "fast_period": max(2, fast_default),
            "slow_period": max(3, slow_default),
            "source": "close",
        }
    if ind_genes.get("atr", False) and "atr_period" in cont:
        indicators["atr"] = {"period": int(cont.get("atr_period", 14))}
    if ind_genes.get("stoch", False) and "stoch_k_period" in cont:
        # ``compute_all`` reads ``slowk_period``/``slowd_period`` + ``period`` as
        # the fast-K period (indicators.py).  Emitting ``k_period``/``d_period``
        # left the stochastic on its defaults, i.e. another inert gene.
        indicators["stoch"] = {
            "period": int(cont.get("stoch_k_period", 14)),
            "slowk_period": int(cont.get("stoch_d_period", 3)),
            "slowd_period": int(cont.get("stoch_d_period", 3)),
        }
    if ind_genes.get("cci", False) and "cci_period" in cont:
        indicators["cci"] = {"period": int(cont.get("cci_period", 14))}
    if ind_genes.get("obv", False):
        indicators["obv"] = {}
    if ind_genes.get("sma", False) and "sma_period" in cont:
        indicators["sma"] = {"period": int(cont.get("sma_period", 50))}
    if ind_genes.get("hurst", False) and "hurst_lookback" in cont:
        indicators["hurst"] = {"lookback": int(cont.get("hurst_lookback", 100))}
    if ind_genes.get("swing_points", False) and "swing_lookback" in cont:
        indicators["swing_points"] = {"lookback": int(cont.get("swing_lookback", 5))}
    if ind_genes.get("frac_diff", False) and "frac_diff_d" in cont:
        indicators["frac_diff"] = {"d": round(cont.get("frac_diff_d", 40) / 100.0, 2)}

    # ── Fallback: ensure at least one indicator is active when all genes disabled
    # but continuous genes exist (i.e., chromosome has the data but genes say "no")
    if not indicators:
        if "rsi_period" in cont:
            indicators["rsi"] = {"period": int(cont.get("rsi_period", 14)), "source": "close"}
        elif "macd_fast" in cont:
            indicators["macd"] = {
                "fast": int(cont.get("macd_fast", 12)),
                "slow": int(cont.get("macd_slow", 26)),
                "signal": int(cont.get("macd_signal", 9)),
            }
        elif "ema_fast_period" in cont or "ema_period" in cont:
            indicators["ema"] = {
                "fast_period": int(cont.get("ema_fast_period", cont.get("ema_period", 9)) or 9),
                "slow_period": int(cont.get("ema_slow_period", 21) or 21),
                "source": "close",
            }

    # ── ML config: must match what the fitness path scored ──
    # ``evaluate_chromosome`` forces ``ml_config.enabled = False``; the decoder
    # used to emit ``enabled = weight > 0``.  Measured fusion divergence:
    # off 0.5000 / w=0.1 → 0.6250 / w=0.5 → 0.4167, so the live entry set
    # differed from the scored one.  Both are pinned to "disabled, weight 0".
    ml_config = MLConfig(
        enabled=False,
        weight=0.0,
        confidence_threshold=round(cont.get("ml_threshold", 0.6), 2),
    )

    timeframes = cat.get("timeframes", DEFAULT_TIMEFRAME).split(",")

    # ── Condition sanitization ──
    # The old code built the enabled set from the BOOLEAN genes.  A chromosome
    # whose genes were switched off while the continuous params remained (or a
    # legacy chromosome missing the boolean genes) therefore kept conditions for
    # indicators the decoded config no longer carries — production logged
    # ``'adx > 20' — name 'adx' is not defined``.  The authoritative set is the
    # DECODED ``indicators`` dict, plus the columns ``compute_all`` always adds
    # (``close``/``volume_ratio``/``ema_fast``/``ema_slow``/``sma``).
    # First-class volume/flow columns: any candidate condition that reads one of
    # them makes the config ask for the family, so the column and the condition
    # are one invariant and sanitisation can treat them as owned.  Enabling the
    # key is additive (it adds columns, it never removes an indicator), so a
    # chromosome that reads none of them decodes byte-for-byte as before.
    if any(_condition_reads_volume_flow(c)
           for side in ("entry_long", "entry_short", "exit_long", "exit_short")
           for c in (struct.get(side, []) or [])):
        indicators.setdefault(VOLUME_FLOW_INDICATOR, {})

    enabled_set = set(indicators.keys()) | ALWAYS_AVAILABLE_COLUMNS
    if "sma" in indicators:
        enabled_set.add("sma")
    entry_long = _sanitize_conditions(struct.get("entry_long", []), enabled_set, "long")
    entry_short = _sanitize_conditions(struct.get("entry_short", []), enabled_set, "short")
    exit_long = _sanitize_conditions(struct.get("exit_long", []), enabled_set, "long")
    exit_short = _sanitize_conditions(struct.get("exit_short", []), enabled_set, "short")

    # ── P6-D volume filter gene ──
    # Applied AFTER sanitisation: the conjunct only reads ``volume_ratio``, which
    # every frame carries, so it can never be dropped for a missing indicator, and
    # the filter must hold even for a condition that survived on its own.
    filter_rvol = _coerce_volume_gene(cont.get("volume_filter_rvol", 0.0),
                                      VOLUME_FILTER_MAX_RVOL, 1)
    scale_k = _coerce_volume_gene(cont.get("volume_scale_k", 0.0),
                                  VOLUME_SCALE_MAX_K, 3)
    if filter_rvol > 0:
        entry_long = [with_volume_filter(c, filter_rvol) for c in entry_long]
        entry_short = [with_volume_filter(c, filter_rvol) for c in entry_short]
    if filter_rvol > 0 or scale_k > 0:
        # Round-trip carrier + self-describing champion YAML (see the constant).
        indicators[VOLUME_GENE_INDICATOR_KEY] = {
            "filter_rvol": filter_rvol, "scale_k": scale_k}

    config = StrategyConfig(
        name=chromosome.get("name", "ga_strategy"),
        enabled=True,
        mode=cat.get("mode", "trend"),
        timeframes=timeframes,
        indicators=indicators,
        entry_conditions={
            "long": entry_long,
            "short": entry_short,
        },
        # ── Evolvable entry logic (AND / OR) ──
        # A plain schema field: the gene is written into the champion YAML and
        # read back by the ONE shared entry-structure evaluator
        # (``StrategyConfig.entry_sides``), used by GA evaluation and by
        # post-load trading alike.  An unknown gene value warns and falls back
        # to "or" inside the model validator.
        condition_logic=chromosome.get("condition_logic", "or"),
        exit_conditions={
            "long": exit_long,
            "short": exit_short,
        },
        ml_config=ml_config,
    )
    return config


# ── Random initialization ─────────────────────────────────────────────

def random_chromosome(name: str = "ga_strategy") -> dict:
    """Create a random strategy chromosome with diverse indicator selection."""
    mode = random.choice(MODE_OPTIONS)
    tfs = random.sample(TIMEFRAME_OPTIONS, k=random.choice([1, 2]))
    # Shortest first, using the shared interval registry: an unknown timeframe
    # must not raise a KeyError here.
    tfs.sort(key=interval_minutes)

    indicators = _random_indicators()

    config = StrategyConfig(
        name=name,
        enabled=True,
        mode=mode,
        timeframes=tfs,
        indicators=indicators,
        entry_conditions=_random_conditions("entry"),
        exit_conditions=_random_conditions("exit"),
        ml_config=MLConfig(
            enabled=random.random() < 0.3,
            weight=random.choice([0.1, 0.2, 0.3]),
            confidence_threshold=random.uniform(0.55, 0.75),
        ),
    )
    chrom = strategy_to_chromosome(config)
    # indicator_genes are already set by strategy_to_chromosome based on config.indicators
    # Evolvable entry logic: random init explores both OR and AND.
    chrom["condition_logic"] = random.choice(["or", "or", "and"])
    # P6-D volume genes: random init explores them, half the time neutral.
    _randomise_volume_genes(chrom)
    return chrom


def _random_indicators() -> dict:
    """Generate random indicator config using INDICATOR_INIT_PROB."""
    ind = {}
    if random.random() < INDICATOR_INIT_PROB["rsi"]:
        ind["rsi"] = {"period": random.randint(7, 21), "source": "close"}
    if random.random() < INDICATOR_INIT_PROB["macd"]:
        ind["macd"] = {
            "fast": random.randint(8, 16),
            "slow": random.randint(22, 34),
            "signal": random.randint(6, 12),
        }
    if random.random() < INDICATOR_INIT_PROB["bollinger"]:
        ind["bollinger"] = {
            "period": random.randint(14, 30),
            "stddev": random.uniform(1.5, 3.0),
        }
    if random.random() < INDICATOR_INIT_PROB["adx"]:
        ind["adx"] = {"period": random.randint(10, 21)}
    if random.random() < INDICATOR_INIT_PROB["ema"]:
        ind["ema"] = {"period": random.randint(5, 30), "source": "close"}
    if random.random() < INDICATOR_INIT_PROB["atr"]:
        ind["atr"] = {"period": random.randint(10, 21)}
    if random.random() < INDICATOR_INIT_PROB["stoch"]:
        ind["stoch"] = {"k_period": random.randint(7, 18), "d_period": random.randint(3, 7)}
    if random.random() < INDICATOR_INIT_PROB["cci"]:
        ind["cci"] = {"period": random.randint(10, 21)}
    if random.random() < INDICATOR_INIT_PROB["obv"]:
        ind["obv"] = {}
    if random.random() < INDICATOR_INIT_PROB["sma"]:
        ind["sma"] = {"period": random.randint(20, 80)}
    if random.random() < INDICATOR_INIT_PROB["hurst"]:
        ind["hurst"] = {"lookback": random.randint(60, 150)}
    if random.random() < INDICATOR_INIT_PROB["swing_points"]:
        ind["swing_points"] = {"lookback": random.randint(3, 8)}
    if random.random() < INDICATOR_INIT_PROB["frac_diff"]:
        ind["frac_diff"] = {"d": random.uniform(0.25, 0.5)}
    # Guarantee at least one indicator (RSI fallback)
    if not ind:
        ind["rsi"] = {"period": 14, "source": "close"}
    return ind


def _sanitize_conditions(conditions: list[str], enabled_indicators: set[str],
                         direction: str = "long") -> list[str]:
    """Remove conditions referencing disabled indicators. Inject fallback if empty.

    Args:
        conditions: List of condition strings (e.g. "rsi < 30").
        enabled_indicators: Set of indicator names currently active.
        direction: "long" or "short" — used for fallback injection.

    Returns:
        Sanitized list with at least one condition (fallback if all filtered).
    """
    clean = []
    for cond in conditions or []:
        if not isinstance(cond, str) or not cond.strip():
            continue
        ok = True
        for col_pattern, required in CONDITION_INDICATOR_MAP.items():
            if col_pattern in cond and required:
                if not any(r in enabled_indicators for r in required):
                    ok = False
                    break
        if ok:
            clean.append(cond)

    if not clean:
        # Fallback: inject a safe condition — the result must NEVER be an empty
        # list (an empty entry/exit list makes ``evaluate_condition`` comparisons
        # meaningless and used to raise ValueError downstream).
        if "ema" in enabled_indicators:
            clean = ["close > ema_fast"] if direction == "long" else ["close < ema_fast"]
        elif "bollinger" in enabled_indicators:
            clean = ["close > bollinger_lower"] if direction == "long" else ["close < bollinger_upper"]
        elif "sma" in enabled_indicators:
            clean = ["close > sma"] if direction == "long" else ["close < sma"]
        else:
            clean = ["volume_ratio > 1.0"]  # always available

    return clean


def _random_conditions(cond_type: str) -> dict:
    """Generate random entry or exit conditions."""
    result = {}
    for side in ["long", "short"]:
        pool = CONDITION_POOL[side] if cond_type == "entry" else EXIT_CONDITION_POOL[side]
        n = random.randint(1, 3)
        result[side] = random.sample(pool, min(n, len(pool)))
    return result
