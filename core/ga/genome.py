"""Strategy genome encoding — maps between YAML StrategyConfig and GA chromosomes.

Each strategy is encoded as a mixed-type chromosome:

    [continuous_params | categorical | entry_conditions | exit_conditions]

- Continuous: indicator periods, thresholds → Gaussian mutation
- Categorical: mode, timeframes → random replacement
- Structural: entry/exit condition strings → crossover-exchange, add/remove mutation
"""

import copy
import random
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
    ],
}

MODE_OPTIONS = ["trend", "range", "scalp", "momentum"]
#: Timeframes the GA evolves over — the intervals INTERVAL_SPEC declares as the
#: streamed/pre-fetched set, so a new interval added there joins the genes too.
TIMEFRAME_OPTIONS = list(DEFAULT_INTERVALS)

INDICATOR_NAMES = ["rsi", "macd", "bollinger", "adx", "ema", "atr", "stoch", "cci", "obv", "sma", "hurst", "swing_points", "frac_diff"]

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
}


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
        continuous.append(ContinuousGene("stoch_k_period", s.get("k_period", 14), 5, 21, 1))
        continuous.append(ContinuousGene("stoch_d_period", s.get("d_period", 3), 3, 9, 1))

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
        entry = config.entry_conditions.get(side, [])
        structural.append(StructuralGene(
            f"entry_{side}", list(entry),
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
    enabled_set = set(indicators.keys()) | ALWAYS_AVAILABLE_COLUMNS
    if "sma" in indicators:
        enabled_set.add("sma")
    entry_long = _sanitize_conditions(struct.get("entry_long", []), enabled_set, "long")
    entry_short = _sanitize_conditions(struct.get("entry_short", []), enabled_set, "short")
    exit_long = _sanitize_conditions(struct.get("exit_long", []), enabled_set, "long")
    exit_short = _sanitize_conditions(struct.get("exit_short", []), enabled_set, "short")

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
