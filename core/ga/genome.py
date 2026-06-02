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
from core.strategy.loader import StrategyConfig, MLConfig


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
    ],
}

EXIT_CONDITION_POOL = {
    "long": [
        "rsi > 65",
        "rsi > 55",
        "close < bollinger_middle",
        "close < ema_slow",
        "close < bollinger_lower",
    ],
    "short": [
        "rsi < 35",
        "rsi < 45",
        "close > bollinger_middle",
        "close > ema_slow",
        "close > bollinger_upper",
    ],
}

MODE_OPTIONS = ["trend", "range", "scalp", "momentum"]
TIMEFRAME_OPTIONS = ["1m", "5m", "15m", "1h", "4h"]


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

    # EMA
    if "ema" in ind:
        ema = ind["ema"]
        continuous.append(ContinuousGene("ema_period", ema.get("period", 9), 5, 50, 2))

    # ML
    ml_weight = 0.0
    ml_threshold = 0.6
    if config.ml_config:
        ml_weight = config.ml_config.weight
        ml_threshold = config.ml_config.confidence_threshold
    continuous.append(ContinuousGene("ml_weight", ml_weight, 0.0, 0.5, 0.05))
    continuous.append(ContinuousGene("ml_threshold", ml_threshold, 0.5, 0.85, 0.05))

    # ── Categorical genes ──
    categorical = [
        CategoricalGene("mode", config.mode, MODE_OPTIONS),
    ]
    # Timeframes: store as comma-separated for GA; decode splits back
    categorical.append(
        CategoricalGene("timeframes", ",".join(config.timeframes),
                        [",".join(c) for c in itertools.combinations(TIMEFRAME_OPTIONS, 2)]))

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

    return {
        "continuous": continuous,
        "categorical": categorical,
        "structural": structural,
        "name": config.name,
    }


def chromosome_to_strategy(chromosome: dict) -> StrategyConfig:
    """Decode a chromosome dict back into a StrategyConfig."""
    cont = {g.name: g.value for g in chromosome["continuous"]}
    cat = {g.name: g.value for g in chromosome["categorical"]}
    struct = {g.name: g.conditions for g in chromosome["structural"]}

    indicators = {}
    if "rsi_period" in cont:
        indicators["rsi"] = {"period": int(cont.get("rsi_period", 14)), "source": "close"}
    if "macd_fast" in cont:
        indicators["macd"] = {
            "fast": int(cont.get("macd_fast", 12)),
            "slow": int(cont.get("macd_slow", 26)),
            "signal": int(cont.get("macd_signal", 9)),
        }
    if "bb_period" in cont:
        indicators["bollinger"] = {
            "period": int(cont.get("bb_period", 20)),
            "stddev": round(cont.get("bb_stddev", 2.0), 2),
        }
    if "adx_period" in cont:
        indicators["adx"] = {"period": int(cont.get("adx_period", 14))}
    if "ema_period" in cont:
        indicators["ema"] = {"period": int(cont.get("ema_period", 9)), "source": "close"}

    ml_config = MLConfig(
        enabled=cont.get("ml_weight", 0) > 0,
        weight=round(cont.get("ml_weight", 0), 2),
        confidence_threshold=round(cont.get("ml_threshold", 0.6), 2),
    )

    timeframes = cat.get("timeframes", "1h").split(",")

    return StrategyConfig(
        name=chromosome.get("name", "ga_strategy"),
        enabled=True,
        mode=cat.get("mode", "trend"),
        timeframes=timeframes,
        indicators=indicators,
        entry_conditions={
            "long": struct.get("entry_long", []),
            "short": struct.get("entry_short", []),
        },
        exit_conditions={
            "long": struct.get("exit_long", []),
            "short": struct.get("exit_short", []),
        },
        ml_config=ml_config,
    )


# ── Random initialization ─────────────────────────────────────────────

def random_chromosome(name: str = "ga_strategy") -> dict:
    """Create a random strategy chromosome."""
    # Build a minimal random config
    mode = random.choice(MODE_OPTIONS)
    tfs = random.sample(TIMEFRAME_OPTIONS, k=random.choice([1, 2]))
    tfs.sort(key=lambda t: {"1m":1,"5m":5,"15m":15,"1h":60,"4h":240}[t])

    config = StrategyConfig(
        name=name,
        enabled=True,
        mode=mode,
        timeframes=tfs,
        indicators=_random_indicators(),
        entry_conditions=_random_conditions("entry"),
        exit_conditions=_random_conditions("exit"),
        ml_config=MLConfig(
            enabled=random.random() < 0.3,
            weight=random.choice([0.1, 0.2, 0.3]),
            confidence_threshold=random.uniform(0.55, 0.75),
        ),
    )
    return strategy_to_chromosome(config)


def _random_indicators() -> dict:
    """Generate random indicator config."""
    ind = {}
    if random.random() < 0.9:
        ind["rsi"] = {"period": random.randint(7, 21), "source": "close"}
    if random.random() < 0.7:
        ind["macd"] = {
            "fast": random.randint(8, 16),
            "slow": random.randint(22, 34),
            "signal": random.randint(6, 12),
        }
    if random.random() < 0.7:
        ind["bollinger"] = {
            "period": random.randint(14, 30),
            "stddev": random.uniform(1.5, 3.0),
        }
    if random.random() < 0.5:
        ind["adx"] = {"period": random.randint(10, 21)}
    if random.random() < 0.4:
        ind["ema"] = {"period": random.randint(5, 30), "source": "close"}
    return ind


def _random_conditions(cond_type: str) -> dict:
    """Generate random entry or exit conditions."""
    result = {}
    for side in ["long", "short"]:
        pool = CONDITION_POOL[side] if cond_type == "entry" else EXIT_CONDITION_POOL[side]
        n = random.randint(1, 3)
        result[side] = random.sample(pool, min(n, len(pool)))
    return result
