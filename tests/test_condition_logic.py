"""``condition_logic`` (AND/OR entry structure): persistence + scoring/publishing parity.

Closes the P1 gap left open by ``tests/test_ga_credibility.py``: the GA chromosome
has an evolvable ``condition_logic`` gene and the engine honoured it, but
``StrategyConfig`` had no such field, so P1 attached it to the instance with
``object.__setattr__``.  GA evaluation therefore used AND while a champion YAML
published from that genome reloaded as OR — "scoring ≠ publishing", the exact
inconsistency P1 item 6 exists to remove.

These tests pin, in order:

1. the schema field + its fallback contract (missing/invalid YAML keeps loading)
2. the gene is a *field*, so it lands in the emitted YAML verbatim
3. the guard that would have caught the original bug: one genome, evaluated
   before serialisation and after YAML reload, must produce the identical
   entry-signal set (same bars, same side) — plus the same actual backtest
   entries through the GA/backtest engine
4. AND really changes the signal set versus OR on the same data (non-vacuous)
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── deterministic synthetic market (same generator as test_ga_credibility) ──

def _write_market(tmp_path: Path, symbols=("BTCUSDT",),
                  timeframes=("15m", "1h", "4h"), start="2026-01-01",
                  bars_15m=120 * 96):
    """Deterministic OHLCV parquet tree — no cached data, no network."""
    rng = np.random.default_rng(20261001)
    base = pd.date_range(start, periods=bars_15m, freq="15min")
    for symbol in symbols:
        close = 20000 + np.cumsum(rng.normal(0, 40, len(base)))
        m15 = pd.DataFrame({
            "open": close, "high": close + 30, "low": close - 30,
            "close": close, "volume": rng.random(len(base)) * 100 + 10,
        }, index=base)
        market_dir = tmp_path / "market" / symbol
        market_dir.mkdir(parents=True, exist_ok=True)
        for tf in timeframes:
            frame = m15 if tf == "15m" else m15.resample(tf).agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
            frame.to_parquet(market_dir / f"{tf}.parquet")
    return str(tmp_path)


@pytest.fixture()
def market_dir(tmp_path):
    return _write_market(tmp_path / "data")


def _engine(market_dir, tmp_path):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager
    from core.strategy.loader import StrategyLoader

    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = market_dir
    cfg.backtest_engine_mode = "legacy"
    cfg.backtest_ml_enabled = False
    cfg.backtest_live_spread_enabled = False
    bus = EventBus()
    loader = StrategyLoader(str(tmp_path / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus), OrderExecutor(cfg, bus))
    return cfg, engine, loader


# ── chromosomes / frames ──────────────────────────────────────────────

ENTRY_LONG = ["rsi < 70", "close > sma"]
ENTRY_SHORT = ["rsi > 30", "close < sma"]


def _logic_genome(name: str, logic: str):
    """Two long + two short conditions, so OR and AND are measurably different."""
    from core.ga.genome import (BooleanGene, CategoricalGene, ContinuousGene,
                                StructuralGene)

    return {
        "continuous": [ContinuousGene("rsi_period", 14, 5, 28, 1),
                       ContinuousGene("sma_period", 20, 10, 100, 2)],
        "categorical": [CategoricalGene("mode", "trend", ["trend"]),
                        CategoricalGene("timeframes", "1h", ["1h"])],
        "structural": [StructuralGene("entry_long", list(ENTRY_LONG), []),
                       StructuralGene("entry_short", list(ENTRY_SHORT), []),
                       StructuralGene("exit_long", ["rsi > 80"], []),
                       StructuralGene("exit_short", ["rsi < 20"], [])],
        "indicator_genes": [BooleanGene("rsi", True), BooleanGene("sma", True)],
        "condition_logic": logic,
        "name": name,
    }


def _indicator_frame(market_dir: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    from core.strategy.indicators import compute_all

    raw = pd.read_parquet(Path(market_dir) / "market" / symbol / "1h.parquet")
    return compute_all(raw, {"rsi": {"period": 14, "source": "close"},
                             "sma": {"period": 20}})


def _signal_sets(cfg, df: pd.DataFrame, start: int, bars: int) -> dict:
    """Per-bar entry activity, evaluated with a growing window (no look-ahead).

    Mirrors how the engines evaluate bar *i*: the frame ends at that bar.  A bar
    maps to ``"long"``/``"short"``/``"both"`` (the engines treat ``"both"`` as
    ambiguous → no trade); inactive bars are absent, so the dict *is* the
    entry-signal set.
    """
    out: dict = {}
    for i in range(bars):
        window = df.iloc[: start + i + 1]
        long_active, short_active = cfg.entry_sides(window)
        ts = window.index[-1]
        if long_active and short_active:
            out[ts] = "both"
        elif long_active:
            out[ts] = "long"
        elif short_active:
            out[ts] = "short"
    return out


# ══════════════════════════════════════════════════════════════════════
# 1 — the schema field and its fallback contract
# ══════════════════════════════════════════════════════════════════════

def test_missing_field_defaults_to_or_and_keeps_loading(tmp_path):
    """Pre-P1 / hand-written YAML has no field — it must load as OR."""
    from core.strategy.loader import StrategyLoader

    strategies = tmp_path / "strategies"
    strategies.mkdir()
    (strategies / "legacy.yaml").write_text(
        "name: legacy\n"
        "timeframes: [1h]\n"
        "indicators:\n  sma: {period: 20}\n"
        "entry_conditions:\n  long: ['close > sma']\n"
        "  short: ['close < sma']\n",
        encoding="utf-8")

    cfg = StrategyLoader(str(strategies)).load("legacy")
    assert cfg.condition_logic == "or"
    assert StrategyLoader(str(strategies)).load_all()[0].condition_logic == "or"


@pytest.mark.parametrize("raw,expected", [
    ("or", "or"),
    ("OR", "or"),
    (" and ", "and"),
    ("And", "and"),
    ("xor", "or"),          # typo / AI hallucination → fallback, not a crash
    ("", "or"),
    (None, "or"),
    (42, "or"),
])
def test_invalid_condition_logic_warns_and_falls_back(tmp_path, raw, expected):
    from loguru import logger
    from core.strategy.loader import StrategyConfig

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        cfg = StrategyConfig(name="probe", condition_logic=raw)
    finally:
        logger.remove(sink)

    assert cfg.condition_logic == expected
    if expected == "or" and str(raw).strip().lower() != "or":
        assert any("condition_logic" in m for m in messages), (
            f"invalid value {raw!r} fell back silently: {messages}")


def test_chromosome_gene_is_a_schema_field_not_an_instance_patch(tmp_path):
    """The gene must be a real field so ``model_dump``/YAML carry it."""
    from core.ga.genome import chromosome_to_strategy, strategy_to_chromosome
    from core.strategy.loader import StrategyConfig

    assert "condition_logic" in StrategyConfig.model_fields
    cfg = chromosome_to_strategy(_logic_genome("logic_and", "and"))
    assert cfg.condition_logic == "and"
    assert cfg.model_dump()["condition_logic"] == "and"
    # ... and it round-trips back into the chromosome.
    assert strategy_to_chromosome(cfg)["condition_logic"] == "and"

    # An unknown gene value falls back to OR (with a warning) instead of raising.
    bad = _logic_genome("logic_bad", "xor")
    assert chromosome_to_strategy(bad).condition_logic == "or"


def test_setattr_hack_is_gone_from_the_ga_package():
    """Regression pin: the P1 ``object.__setattr__`` workaround must not return."""
    import core.ga as ga_pkg

    ga_dir = Path(ga_pkg.__file__).parent
    offenders = [p.name for p in ga_dir.glob("*.py")
                 if "object.__setattr__" in p.read_text(encoding="utf-8")]
    assert offenders == [], f"instance-patched schema field in {offenders}"


# ══════════════════════════════════════════════════════════════════════
# 2 — the guard: evaluate → publish → reload → evaluate
# ══════════════════════════════════════════════════════════════════════

def test_entry_signal_sets_are_identical_after_yaml_round_trip(market_dir, tmp_path):
    """The bug P1 could not close: GA-scored AND genome reloaded as OR."""
    from core.ga.genome import chromosome_to_strategy
    from core.strategy.loader import StrategyLoader

    cfg_ga = chromosome_to_strategy(_logic_genome("logic_and", "and"))
    assert cfg_ga.condition_logic == "and"

    df = _indicator_frame(market_dir)
    start, bars = len(df) - 300, 300          # growing-window evaluation, no look-ahead
    ga_signals = _signal_sets(cfg_ga, df, start, bars)
    assert ga_signals, "test is vacuous — the AND genome never fires on this data"
    assert any(side == "long" for side in ga_signals.values())

    loader = StrategyLoader(str(tmp_path / "published"))
    loader.save(cfg_ga)                        # the champion YAML
    text = (loader.strategies_dir / "logic_and.yaml").read_text(encoding="utf-8")
    assert "condition_logic: and" in text, f"gene missing from the YAML:\n{text}"

    cfg_live = loader.load("logic_and")        # post-load trading
    assert cfg_live.condition_logic == "and"
    live_signals = _signal_sets(cfg_live, df, start, bars)

    assert live_signals == ga_signals, (
        f"entry-signal sets differ after reload: GA={len(ga_signals)} bars, "
        f"live={len(live_signals)} bars, "
        f"only-GA={sorted(set(ga_signals) - set(live_signals))[:3]}, "
        f"only-live={sorted(set(live_signals) - set(ga_signals))[:3]}")
    assert len(live_signals) == len(ga_signals)
    print(f"\n[evaluate vs reload] identical entry bars: {len(ga_signals)} "
          f"(window {start}..{start + bars - 1}, {bars} bars evaluated)")


def test_and_genome_trades_identically_before_and_after_publishing(market_dir, tmp_path):
    """End-to-end on the GA path itself: the backtest engine's entries agree.

    Before the fix the reloaded config silently fell back to OR, so the second
    run took a *different* set of entries than the genome was scored on.
    """
    from core.ga.genome import chromosome_to_strategy

    cfg_ga = chromosome_to_strategy(_logic_genome("logic_and", "and"))
    _, engine, loader = _engine(market_dir, tmp_path)
    res_ga = engine.run_with_exit_evaluation(
        strategies=[cfg_ga], symbols=["BTCUSDT"],
        date_start="2026-02-01", date_end="2026-03-15", mode="full",
        simulate_ai_weights=False, per_strategy_isolation=True,
        per_genome_ledger=True, use_live_spread=False)

    loader.save(cfg_ga)                        # publish
    assert "condition_logic: and" in (
        loader.strategies_dir / "logic_and.yaml").read_text(encoding="utf-8")
    cfg_live = loader.load("logic_and")        # reload = what live would trade

    _, engine2, _ = _engine(market_dir, tmp_path)
    res_live = engine2.run_with_exit_evaluation(
        strategies=[cfg_live], symbols=["BTCUSDT"],
        date_start="2026-02-01", date_end="2026-03-15", mode="full",
        simulate_ai_weights=False, per_strategy_isolation=True,
        per_genome_ledger=True, use_live_spread=False)

    def _entries(result):
        return [(e["time"], e["symbol"], e["side"]) for e in result["events"]
                if e.get("type") == "entry"]

    entries_ga, entries_live = _entries(res_ga), _entries(res_live)
    assert entries_ga, "test is vacuous — the AND genome took no entries"
    assert entries_live == entries_ga, (
        f"publishing changed the traded entries: GA={len(entries_ga)} "
        f"live={len(entries_live)}")
    print(f"\n[GA path vs reloaded] identical entries: {len(entries_ga)} "
          f"(sides: {sorted({s for _, _, s in entries_ga})})")


def test_helper_entry_rule_matches_the_engine_entry_bars(market_dir, tmp_path):
    """No drift between ``entry_sides`` and the GA/backtest entry path.

    Every entry the backtest engine actually took on this AND genome must be a
    bar where ``condition_sides`` reports that side active — i.e. the helper the
    live engine uses applies the *same* rule the GA scored the genome with.
    """
    from core.ga.genome import chromosome_to_strategy

    cfg = chromosome_to_strategy(_logic_genome("logic_and", "and"))
    df = _indicator_frame(market_dir)
    _, engine, _ = _engine(market_dir, tmp_path)
    res = engine.run_with_exit_evaluation(
        strategies=[cfg], symbols=["BTCUSDT"],
        date_start="2026-02-01", date_end="2026-03-15", mode="full",
        simulate_ai_weights=False, per_strategy_isolation=True,
        per_genome_ledger=True, use_live_spread=False)

    entries = [(e["time"], e["side"]) for e in res["events"]
               if e.get("type") == "entry"]
    assert entries, "test is vacuous — the AND genome took no entries"

    mismatches = []
    for time_str, side in entries:
        pos = df.index.get_loc(pd.Timestamp(time_str))
        long_active, short_active = cfg.entry_sides(df.iloc[:pos + 1])
        if not (long_active if side == "long" else short_active):
            mismatches.append((time_str, side, long_active, short_active))

    assert not mismatches, (
        f"{len(mismatches)}/{len(entries)} engine entries are not entry-signal "
        f"bars for the shared rule: {mismatches[:3]}")
    print(f"\n[helper vs engine] {len(entries)} entries, 0 rule mismatches")


# ══════════════════════════════════════════════════════════════════════
# 3 — AND is genuinely different from OR on the same data
# ══════════════════════════════════════════════════════════════════════

def test_and_is_stricter_than_or_on_the_same_data(market_dir, tmp_path):
    """AND must shrink the entry-signal set (otherwise the gene is inert)."""
    from core.ga.genome import chromosome_to_strategy

    df = _indicator_frame(market_dir)
    start, bars = len(df) - 300, 300

    cfg_or = chromosome_to_strategy(_logic_genome("logic_or", "or"))
    cfg_and = chromosome_to_strategy(copy.deepcopy(_logic_genome("logic_and", "and")))
    assert (cfg_or.condition_logic, cfg_and.condition_logic) == ("or", "and")

    or_signals = _signal_sets(cfg_or, df, start, bars)
    and_signals = _signal_sets(cfg_and, df, start, bars)

    assert and_signals and or_signals, "test is vacuous — no signals at all"
    assert set(and_signals) < set(or_signals), (
        f"AND did not strictly shrink the signal set: "
        f"AND={len(and_signals)} OR={len(or_signals)} bars")

    # Every AND bar is an OR bar and its side is *compatible*: OR may add the
    # other side ("both" = ambiguous → the engines skip the bar), never drop the
    # side AND found.
    compatible = {"long": ("long", "both"),
                  "short": ("short", "both"),
                  "both": ("both",)}
    mismatched = {ts: (and_signals[ts], or_signals[ts])
                  for ts in and_signals
                  if or_signals[ts] not in compatible[and_signals[ts]]}
    assert not mismatched, f"AND/OR sides disagree on {len(mismatched)} bars: " \
                           f"{list(mismatched.items())[:3]}"

    and_long = sum(1 for s in and_signals.values() if s in ("long", "both"))
    or_long = sum(1 for s in or_signals.values() if s in ("long", "both"))
    assert and_long < or_long, (
        f"AND did not shrink the long side either: AND={and_long} OR={or_long}")
    print(f"\n[AND vs OR] entry bars on identical data: "
          f"OR={len(or_signals)} bars ({or_long} long-side)  "
          f"AND={len(and_signals)} bars ({and_long} long-side)  "
          f"(dropped {len(or_signals) - len(and_signals)})")


def test_and_and_or_only_disagree_where_both_conditions_are_needed(tmp_path):
    """Unit-level proof of the two modes on a hand-built frame."""
    from core.strategy.loader import StrategyConfig

    df = pd.DataFrame({
        "close": [10.0, 10.0, 10.0, 10.0],
        "sma": [5.0, 5.0, 20.0, 20.0],
        #              bar:    0     1      2      3
        "rsi": [20.0, 80.0, 80.0, 20.0],
    })
    base = dict(name="unit", indicators={"rsi": {}, "sma": {}},
                entry_conditions={"long": ["rsi < 30", "close > sma"]},
                exit_conditions={"long": [], "short": []})

    cfg_or = StrategyConfig(**base, condition_logic="or")
    cfg_and = StrategyConfig(**base, condition_logic="and")

    # bar 3: rsi<30 True, close>sma False → OR long, AND not long
    assert cfg_or.entry_sides(df) == (True, False)
    assert cfg_and.entry_sides(df) == (False, False)
    # bar 0 only: both conditions true → both modes long
    assert cfg_or.entry_sides(df.iloc[:1]) == (True, False)
    assert cfg_and.entry_sides(df.iloc[:1]) == (True, False)
    # bar 2 only: rsi<30 False, close>sma False → neither
    assert cfg_and.entry_sides(df.iloc[:3]) == (False, False)
    # empty condition list is inactive in both modes (no unconditional entries)
    empty = StrategyConfig(name="empty", entry_conditions={"long": []},
                           condition_logic="and")
    assert empty.entry_sides(df) == (False, False)


# ══════════════════════════════════════════════════════════════════════
# 4 — post-load trading: the LIVE engine path honours the reloaded field
# ══════════════════════════════════════════════════════════════════════

class _FakeMarketData:
    """Minimal ``MarketDataProvider`` stand-in — deterministic, no network."""

    def __init__(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        self.df = pd.DataFrame({
            "open": close, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": rng.random(n) * 10 + 1,
        }, index=pd.date_range("2026-01-01", periods=n, freq="1h"))

    async def get_historical(self, symbol, interval, limit=None):
        return self.df.copy()

    def get_current_price(self, symbol):
        return float(self.df["close"].iloc[-1])


@pytest.mark.asyncio
async def test_live_engine_entry_path_honours_the_reloaded_field(tmp_path):
    """The live engine must not take the OR entry for an AND strategy.

    ``core/strategy/engine.py::_evaluate`` evaluates the loaded
    ``StrategyConfig``; before the field existed it always applied OR, so a
    champion scored with AND silently traded a looser entry set.
    """
    from app.config import Config
    from app.event_bus import EventBus
    from core.strategy.engine import StrategyEngine
    from core.strategy.loader import MLConfig, StrategyLoader, StrategyConfig

    def _cfg(logic: str):
        return StrategyConfig(
            name=f"live_{logic}", enabled=True, mode="trend", timeframes=["1h"],
            indicators={"rsi": {"period": 14, "source": "close"}},
            # ``close > 0`` holds on every bar, the impossible second condition
            # never does — so OR enters and AND does not, on identical data.
            entry_conditions={"long": ["close > 0", "close > 1000000"],
                              "short": ["close < 0", "close > 1000000"]},
            exit_conditions={"long": ["rsi > 100"], "short": ["rsi < 0"]},
            condition_logic=logic, ml_config=MLConfig(enabled=False))

    loader = StrategyLoader(str(tmp_path / "live_strategies"))
    loader.save(_cfg("and"))
    reloaded = loader.load("live_and")
    assert reloaded.condition_logic == "and"

    async def _indicator_signal(strategy):
        Config._instance = None       # Config is a singleton — start from defaults
        engine = StrategyEngine(Config.load("sim"), EventBus(), _FakeMarketData())
        await engine._evaluate("BTCUSDT", "1h", strategy)
        return engine._signal_cache[f"{strategy.name}|BTCUSDT"]["indicator_signal"]

    assert await _indicator_signal(reloaded) == 0.0, (
        "the live path entered on OR semantics for an AND strategy")
    assert await _indicator_signal(_cfg("or")) == 1.0, (
        "OR must keep entering on the same conditions (test would be vacuous)")
