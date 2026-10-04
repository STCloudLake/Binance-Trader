"""Engine-parity gate across the variants that used to diverge.

`test_hybrid_equivalence.py` covers the vanilla case (3 strategies, single 1h
timeframe). An independent audit then falsified that gate's coverage by finding
divergences in other configurations:

  * strategies sharing an indicator config but using DIFFERENT timeframes were
    silently dropped from the signal matrix (zero trades, zero fitness)
  * multi-timeframe strategies (higher-TF EMA alignment multiplier)
  * `risk_exit` overrides (stop_loss_pct / trailing_stop_pct / max_hold_hours /
    use_indicator_exits=False) were ignored by the hybrid engine
  * `per_strategy_isolation=True` (used by GA) disabled most exits in the legacy
    engine because it tested `symbol` instead of the composite position key
  * `reduce_conditions` are unsupported by the hybrid engine

Each test below runs both engines on identical inputs and requires identical
trade sets and headline metrics, so a future divergence fails loudly.
"""
import pytest

from app.config import Config, SignalWeights
from core.strategy.loader import MLConfig, RiskExitConfig, StrategyConfig

DATE_START = "2026-05-25"
DATE_END = "2026-05-31"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INITIAL_BALANCE = 10000.0


def _strategy(name, *, timeframes=("1h",), mode="trend", rsi_period=14,
              entry_long="rsi < 35", entry_short="rsi > 65",
              reduce_conditions=None, risk_exit=None, enabled=True):
    return StrategyConfig(
        name=name, enabled=enabled, mode=mode, timeframes=list(timeframes),
        indicators={
            "rsi": {"period": rsi_period, "source": "close"},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
        },
        entry_conditions={"long": [entry_long], "short": [entry_short]},
        exit_conditions={"long": ["rsi > 60"], "short": ["rsi < 40"]},
        reduce_conditions=reduce_conditions or {},
        ml_config=MLConfig(enabled=False),
        risk_exit=risk_exit,
    )


def _config(mode="legacy"):
    Config._instance = None
    config = Config.load("sim")
    config.backtest_engine_mode = mode
    config.backtest_ml_enabled = False
    config.signal_weights = SignalWeights(indicator=1.0, ml=0.0, news=0.0)
    # P9 re-pin: the shipped default is now `next_open`, but the hybrid engine has
    # no fill seam and prices every fill at the decision bar's close.  The parity
    # contract between the two engines is therefore defined on the historical
    # `close` convention, which is requested explicitly here (values unchanged).
    config.backtest_fill_convention = "close"
    return config


def _legacy_engine(config, loader):
    from core.backtest.engine import BacktestEngine
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config
    engine.strategy_engine = type("obj", (object,), {"loader": loader})()
    return engine


def _run_both(strategies, *, per_strategy_isolation=False, date_start=DATE_START,
              date_end=DATE_END, symbols=SYMBOLS, data_dir=None):
    """Run legacy and hybrid on the same input; return (legacy_result, hybrid_result)."""
    import tempfile
    from pathlib import Path

    from core.backtest.engine_hybrid import run_hybrid
    from core.strategy.loader import StrategyLoader

    tmp = Path(tempfile.mkdtemp(prefix="bt_parity_")) / "strategies"
    tmp.mkdir(parents=True, exist_ok=True)
    loader = StrategyLoader(str(tmp))
    for s in strategies:
        loader.save(s)

    legacy_cfg = _config("legacy")
    if data_dir:
        legacy_cfg.data_dir = data_dir
    legacy = _legacy_engine(legacy_cfg, loader).run_with_exit_evaluation(
        strategies=[s.name for s in strategies], symbols=symbols,
        date_start=date_start, date_end=date_end,
        initial_balance=INITIAL_BALANCE, mode="full", simulate_ai_weights=False,
        per_strategy_isolation=per_strategy_isolation)

    hybrid_cfg = _config("hybrid")
    if data_dir:
        hybrid_cfg.data_dir = data_dir
    hybrid = run_hybrid(
        strategies=strategies, symbols=symbols, date_start=date_start,
        date_end=date_end, config=hybrid_cfg, loader=loader,
        initial_balance=INITIAL_BALANCE,
        per_strategy_isolation=per_strategy_isolation)
    return legacy, hybrid


def _assert_parity(legacy, hybrid, label):
    lt, ht = legacy.get("trades", []), hybrid.get("trades", [])

    def key(t):
        return (t.get("strategy", ""), t.get("symbol", ""), t.get("side", ""),
                str(t.get("opened_at", "")))

    lmap = {key(t): t for t in lt}
    hmap = {key(t): t for t in ht}
    assert set(lmap) == set(hmap), (
        f"[{label}] trade sets differ: legacy-only={sorted(set(lmap) - set(hmap))[:3]} "
        f"hybrid-only={sorted(set(hmap) - set(lmap))[:3]} "
        f"(legacy={len(lt)} hybrid={len(ht)} trades)")
    for k, ltrade in lmap.items():
        htrade = hmap[k]
        assert ltrade["quantity"] == pytest.approx(htrade["quantity"], abs=1e-6), \
            f"[{label}] size mismatch on {k}"
        assert ltrade["pnl"] == pytest.approx(htrade["pnl"], abs=0.05), \
            f"[{label}] pnl mismatch on {k}: {ltrade['pnl']} vs {htrade['pnl']}"
        assert ltrade["exit_price"] == pytest.approx(htrade["exit_price"], abs=0.05), \
            f"[{label}] exit price mismatch on {k}"
    for metric in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                   "win_rate_pct", "profit_factor"):
        lv = legacy["metrics"].get(metric)
        hv = hybrid["metrics"].get(metric)
        assert lv == pytest.approx(hv, abs=1e-6, rel=1e-6), \
            f"[{label}] metric '{metric}' differs: legacy={lv} hybrid={hv}"


@pytest.mark.slow
def test_parity_multi_timeframe_strategies():
    """Higher-timeframe alignment AND multi-timeframe exits must match.

    Uses the shape an independent audit used to falsify an earlier version of this
    suite: 15m + 1h + 4h. The legacy engine evaluates indicator exits on EVERY
    configured timeframe (and closes at the triggering timeframe's close), while the
    matrix initially only carried the primary timeframe — 178 vs 77 trades on a
    single strategy and 389 vs 207 on the 3-strategy GA shape.
    """
    strategies = [
        _strategy("mtf_a", timeframes=("15m", "1h", "4h")),
        _strategy("mtf_b", timeframes=("15m", "1h", "4h"), rsi_period=7),
        _strategy("mtf_c", timeframes=("15m", "1h", "4h"), rsi_period=21),
    ]
    _assert_parity(*_run_both(strategies), label="multi-timeframe 15m/1h/4h")


@pytest.mark.slow
def test_parity_multi_timeframe_single_strategy():
    strategies = [_strategy("mtf_solo", timeframes=("15m", "1h", "4h"))]
    _assert_parity(*_run_both(strategies), label="single multi-timeframe strategy")


@pytest.mark.slow
def test_parity_multi_timeframe_with_isolation():
    """GA evaluates with per_strategy_isolation=True — cover it for multi-TF too."""
    strategies = [
        _strategy("mtf_iso_a", timeframes=("15m", "1h", "4h")),
        _strategy("mtf_iso_b", timeframes=("15m", "1h", "4h"), rsi_period=7),
        _strategy("mtf_iso_c", timeframes=("15m", "1h", "4h"), rsi_period=21),
    ]
    _assert_parity(*_run_both(strategies, per_strategy_isolation=True),
                   label="multi-timeframe + isolation")


@pytest.mark.slow
def test_parity_shared_indicator_config_different_timeframes():
    """Regression: a strategy must not be dropped when a sibling in the same
    indicator group uses a different timeframe."""
    a = _strategy("twin_a", timeframes=("1h",))
    b = _strategy("twin_b", timeframes=("4h",))  # identical indicators, other tf
    legacy, hybrid = _run_both([a, b])

    assert legacy.get("trades") is not None and hybrid.get("trades") is not None
    matrix_strategies = {t["strategy"] for t in hybrid["trades"]}
    # Both strategies must be represented in the matrix (the matrix carries rows
    # for every strategy; trades depend on signals, so check the matrix instead).
    from core.backtest.data_feeder import DataFeeder
    from core.backtest.signal_matrix import SignalMatrixBuilder
    feeder = DataFeeder(str(Config.load("sim").data_dir) + "/market", SYMBOLS,
                        ["1h", "4h"], DATE_START, DATE_END)
    feeder.load()
    matrix = SignalMatrixBuilder(feeder).build([a, b], SYMBOLS)
    rows = {idx[0] for idx in matrix.signals.index}
    assert rows == {"twin_a", "twin_b"}, (
        f"strategies dropped from the signal matrix: missing {({'twin_a','twin_b'} - rows)}")
    _assert_parity(legacy, hybrid, label="shared-config/different-tf")


@pytest.mark.slow
def test_parity_risk_exit_overrides():
    """risk_exit stop/trailing/max-hold must be honoured by both engines."""
    risk_exit = RiskExitConfig(stop_loss_pct=1.0, trailing_stop_pct=0.8,
                               max_hold_hours=12.0, use_indicator_exits=True)
    strategies = [
        _strategy("re_a", risk_exit=risk_exit),
        _strategy("re_b", risk_exit=risk_exit, rsi_period=7),
        _strategy("re_c", risk_exit=risk_exit, rsi_period=21),
    ]
    _assert_parity(*_run_both(strategies), label="risk_exit overrides")


@pytest.mark.slow
def test_parity_risk_only_exits():
    """use_indicator_exits=False must disable indicator exits in both engines."""
    risk_exit = RiskExitConfig(stop_loss_pct=2.0, trailing_stop_pct=1.5,
                               max_hold_hours=0.0, use_indicator_exits=False)
    strategies = [
        _strategy("ro_a", risk_exit=risk_exit),
        _strategy("ro_b", risk_exit=risk_exit, rsi_period=7),
        _strategy("ro_c", risk_exit=risk_exit, rsi_period=21),
    ]
    _assert_parity(*_run_both(strategies), label="risk-only exits")


@pytest.mark.slow
def test_parity_per_strategy_isolation():
    """GA runs with per_strategy_isolation=True — both engines must agree there."""
    strategies = [
        _strategy("iso_a"), _strategy("iso_b", rsi_period=7),
        _strategy("iso_c", rsi_period=21),
    ]
    _assert_parity(*_run_both(strategies, per_strategy_isolation=True),
                   label="per-strategy isolation")


def _synthetic_market(tmp_path, symbols=("BTCUSDT", "ETHUSDT"),
                      timeframes=("15m", "1h", "4h"), days=45):
    """Write synthetic OHLCV parquet so the parity gate needs no cached data.

    `data/market/**` is gitignored, so the real-data gate silently skips on a fresh
    clone — which is exactly how a green suite can hide a broken engine. This
    helper builds a deterministic market in a temp dir instead.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20260929)
    base = pd.date_range("2026-01-01", periods=days * 96, freq="15min")  # 15m grid
    frames = {}
    for symbol in symbols:
        close = 20000 + np.cumsum(rng.normal(0, 40, len(base)))
        m15 = pd.DataFrame({
            "open": close, "high": close + 30, "low": close - 30,
            "close": close, "volume": rng.random(len(base)) * 100 + 10,
        }, index=base)
        frames[symbol] = {}
        for tf in timeframes:
            rule = {"15m": None, "1h": "1h", "4h": "4h"}[tf]
            if rule is None:
                frames[symbol][tf] = m15
            else:
                agg = m15.resample(rule).agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna()
                frames[symbol][tf] = agg

        market_dir = tmp_path / "market" / symbol
        market_dir.mkdir(parents=True, exist_ok=True)
        for tf, frame in frames[symbol].items():
            frame.to_parquet(market_dir / f"{tf}.parquet")
    return str(tmp_path)


def test_parity_uses_synthetic_market_without_cached_data(tmp_path):
    """Always-run gate: multi-timeframe parity on synthetic data (no data/ needed).

    Uses the same 15m/1h/4h shape that exposed the multi-timeframe exit divergence,
    plus a risk_exit variant, so a fresh clone exercises the real equivalence logic.
    """
    data_dir = _synthetic_market(tmp_path)
    start, end = "2026-02-01", "2026-02-20"

    risk_exit = RiskExitConfig(stop_loss_pct=1.0, trailing_stop_pct=1.5,
                               max_hold_hours=24.0, use_indicator_exits=False)
    strategies = [
        _strategy("syn_mtf_a", timeframes=("15m", "1h", "4h")),
        _strategy("syn_mtf_b", timeframes=("15m", "1h", "4h"), rsi_period=7),
        _strategy("syn_risk_c", timeframes=("15m", "1h", "4h"), rsi_period=21,
                  risk_exit=risk_exit),
    ]

    legacy, hybrid = _run_both(strategies, date_start=start, date_end=end,
                               data_dir=data_dir)
    assert legacy.get("trades"), "synthetic market produced no trades — test is vacuous"
    _assert_parity(legacy, hybrid, label="synthetic multi-tf")


def test_reduce_conditions_route_to_legacy():
    """The hybrid engine has no reduce path, so auto-routing must pick legacy."""
    from core.backtest.engine import BacktestEngine
    config = _config("auto")
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config

    with_reduce = [
        _strategy("r1", reduce_conditions={"long": [{"condition": "rsi > 55", "reduce_pct": 50}]}),
        _strategy("r2"), _strategy("r3"),
    ]
    assert engine._select_engine(with_reduce, "auto") == "legacy"

    without_reduce = [_strategy("n1"), _strategy("n2"), _strategy("n3")]
    assert engine._select_engine(without_reduce, "auto") == "hybrid"
