"""P9 — the fill convention: config validation, the shift, and the bit-identity.

The convention decides **when a fill is priced**:

``close``      (shipped default) the signal bar's own close — zero execution
               latency, and bit-identical to the pre-P9 engine.
``next_open``  the signal is unchanged (still the bar at ``ts``) and the fill —
               entry AND exit — is the OPEN of the bar one row later on the same
               series.

These tests pin, in order:

1. the config contract (default is ``close``; anything else raises the NAMED
   ``UnknownFillConventionError`` at config load, never a silent fallback);
2. the primitive (``next_bar_open``) on a hand-computable frame;
3. the end-to-end shift on a synthetic series whose prices are known by hand —
   entry and exit each move by exactly one bar;
4. the two window-end cases (an entry that cannot be filled is refused; an exit
   that cannot be filled falls back to the close) — both COUNTED, never silent;
5. the provenance field on the result and in the metrics;
6. the strongest claim: the default is **bit-identical** to a ``git worktree`` at
   the pre-P9 revision, compared as bytes over trades, equity points and a fixed
   metrics digest (``tools/p9_fill_convention_identity.py``).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The revision this work was developed on (HEAD when P9 started).  The worktree
#: is created from this commit, not from "HEAD", so the proof keeps meaning the
#: same thing after later commits land on top.
BASELINE_REVISION = "cafdc1d"


# ── synthetic market whose prices are known by hand ─────────────────────

#: close[i] = 100 + 0.5 i   →  close[0] = 100.0, close[1] = 100.5, close[2] = 101.0
#: open[i]  = close[i-1] + 0.25 (open[0] = close[0])  →  open[1] = 100.25,
#:            open[2] = 100.75, open[3] = 101.25
#: So for a decision on bar k:  close arm fills at close[k]; next_open arm fills
#: at open[k+1] = close[k] + 0.25 — a difference of exactly +25 bps at 100.
BAR_COUNT = 400
PRICE_BASE = 100.0
PRICE_STEP = 0.5
OPEN_STEP = 0.25


def _frame(seed: int = 20261003):
    import numpy as np
    import pandas as pd

    close = PRICE_BASE + PRICE_STEP * np.arange(BAR_COUNT, dtype=float)
    open_ = close.copy()
    open_[1:] = close[:-1] + OPEN_STEP
    return pd.DataFrame(
        {"open": open_,
         "high": np.maximum(open_, close) + 0.05,
         "low": np.minimum(open_, close) - 0.05,
         "close": close,
         "volume": 100.0},
        index=pd.date_range("2026-01-01", periods=BAR_COUNT, freq="1h"))


def _write_market(root: Path, timeframe: str = "1h") -> str:
    frame = _frame()
    for symbol in ("BTCUSDT", "ETHUSDT"):
        target = root / "market" / symbol
        target.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target / f"{timeframe}.parquet")
    return str(root)


@pytest.fixture(scope="module")
def market_dir(tmp_path_factory):
    return _write_market(tmp_path_factory.mktemp("p9") / "data")


def _config(market_dir, convention=None):
    from app.config import Config, SignalWeights

    Config._instance = None
    config = Config.load("sim")
    config.data_dir = market_dir
    config.backtest_engine_mode = "legacy"
    config.backtest_ml_enabled = False
    config.backtest_live_spread_enabled = False
    config.signal_weights = SignalWeights(indicator=1.0, ml=0.0, news=0.0)
    if convention is not None:
        config.backtest_fill_convention = convention
    return config


def _strategy(name="p9_synthetic", *, entry_long="close >= 100",
              timeframes=("1h",), max_hold_hours=1.0):
    from core.strategy.loader import MLConfig, RiskExitConfig, StrategyConfig

    return StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=list(timeframes),
        indicators={"sma": {"period": 5}},
        entry_conditions={"long": [entry_long], "short": ["close < 0"]},
        exit_conditions={"long": ["close < 0"], "short": ["close > 1e9"]},
        ml_config=MLConfig(enabled=False),
        risk_exit=RiskExitConfig(stop_loss_pct=2.0, trailing_stop_pct=0.0,
                                 max_hold_hours=max_hold_hours,
                                 use_indicator_exits=False))


def _engine(config):
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager

    bus = EventBus()
    return BacktestEngine(config, None, RiskManager(config, bus),
                          OrderExecutor(config, bus))


def _run(market_dir, *, convention, date_start="2026-01-05",
         date_end="2026-02-01", symbols=("BTCUSDT",), strategy=None,
         per_genome_ledger=False, with_fill_kwarg=True):
    engine = _engine(_config(market_dir))
    kwargs = {}
    if with_fill_kwarg:
        kwargs["fill_convention"] = convention
    return engine.run_with_exit_evaluation(
        strategies=[strategy or _strategy()], symbols=list(symbols),
        date_start=date_start, date_end=date_end, initial_balance=10_000.0,
        mode="full", simulate_ai_weights=False, use_live_spread=False,
        per_strategy_isolation=True, per_genome_ledger=per_genome_ledger,
        benchmark_mode="none", **kwargs)


def _frame_pos(ts) -> int:
    return int(_frame().index.get_loc(ts))


# ══════════════════════════════════════════════════════════════════════════
# 1 — the config contract
# ══════════════════════════════════════════════════════════════════════════

def test_the_two_conventions_and_the_code_default():
    from core.backtest.fill_convention import (
        DEFAULT_FILL_CONVENTION, FILL_CONVENTIONS, parse_fill_convention)

    assert FILL_CONVENTIONS == ("close", "next_open")
    assert DEFAULT_FILL_CONVENTION == "close"
    assert parse_fill_convention(None) == "close"          # key absent
    assert parse_fill_convention("close") == "close"
    assert parse_fill_convention(" NEXT_OPEN ") == "next_open"


def test_an_unknown_convention_is_refused_by_name():
    from core.backtest.fill_convention import (
        UnknownFillConventionError, parse_fill_convention)

    for bad in ("nextopen", "open", "next", "", "next-bar-open"):
        with pytest.raises(UnknownFillConventionError) as excinfo:
            parse_fill_convention(bad)
        assert "valid values" in str(excinfo.value)
    with pytest.raises(UnknownFillConventionError):
        parse_fill_convention(3)
    assert issubclass(UnknownFillConventionError, ValueError)


def test_the_shipped_key_is_close(monkeypatch):
    """The shipped YAML says `close` and the loaded config agrees."""
    import yaml
    from app.config import Config

    shipped = yaml.safe_load(
        (ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert (shipped.get("backtest") or {}).get("fill_convention") == "close"
    Config._instance = None
    try:
        assert Config.load("sim").backtest_fill_convention == "close"
    finally:
        Config._instance = None


def test_an_unknown_convention_is_rejected_at_config_load(tmp_path, monkeypatch):
    """`backtest.fill_convention: nonsense` must fail at LOAD, not mid-run."""
    import yaml
    import app.config as config_mod
    from app.config import Config
    from core.backtest.fill_convention import UnknownFillConventionError

    data = yaml.safe_load(
        (ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    data.setdefault("backtest", {})["fill_convention"] = "next-bar-open"
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)

    Config._instance = None
    try:
        with pytest.raises(UnknownFillConventionError) as excinfo:
            Config.load("sim")
        assert "next-bar-open" in str(excinfo.value)
    finally:
        Config._instance = None


# ══════════════════════════════════════════════════════════════════════════
# 2 — the primitive, on a hand-computable frame
# ══════════════════════════════════════════════════════════════════════════

def test_next_bar_open_is_the_following_row_and_none_at_the_window_end():
    """Hand-computed on this frame: close[10]=105.0, open[11]=105.25 → +23.81 bps."""
    from core.backtest.fill_convention import decision_bar_position, next_bar_open

    frame = _frame()
    ts = frame.index[10]
    assert decision_bar_position(frame, ts) == 10
    assert float(frame["close"].iloc[10]) == pytest.approx(PRICE_BASE + PRICE_STEP * 10)
    assert next_bar_open(frame, ts) == pytest.approx(
        PRICE_BASE + PRICE_STEP * 10 + OPEN_STEP)
    bps = (next_bar_open(frame, ts) - float(frame["close"].iloc[10])) \
        / float(frame["close"].iloc[10]) * 10_000.0
    assert bps == pytest.approx(OPEN_STEP / (PRICE_BASE + PRICE_STEP * 10) * 10_000.0)
    assert bps == pytest.approx(0.25 / 105.0 * 10_000.0)
    assert bps == pytest.approx(23.81, abs=0.01)
    # The last bar has no following bar: the helper must say so, not guess.
    assert next_bar_open(frame, frame.index[-1]) is None
    # A timestamp between two bars resolves to the bar at/before it (the engine's
    # own `df[df.index <= ts].iloc[-1]` rule), and the fill is the NEXT row.
    between = frame.index[10] + pytest.importorskip("pandas").Timedelta(minutes=20)
    assert decision_bar_position(frame, between) == 10


# ══════════════════════════════════════════════════════════════════════════
# 3 — the end-to-end shift on the synthetic series
# ══════════════════════════════════════════════════════════════════════════

def test_next_open_shifts_entry_and_exit_by_exactly_one_bar(market_dir):
    """Entry at close[k] → open[k+1]; exit at close[k+1] → open[k+2]."""
    frame = _frame()
    close_run = _run(market_dir, convention="close")
    next_run = _run(market_dir, convention="next_open")

    close_trades = close_run["trades"]
    next_trades = next_run["trades"]
    assert close_trades and next_trades, "the synthetic market produced no trades"
    # Signals are unchanged: both arms decide on the SAME bar.
    assert close_trades[0]["opened_at"] == next_trades[0]["opened_at"]

    k = _frame_pos(close_trades[0]["opened_at"])
    assert close_trades[0]["entry_price"] == pytest.approx(float(frame["close"].iloc[k]))
    assert close_trades[0]["exit_reason"] == "max_hold"
    assert close_trades[0]["exit_price"] == pytest.approx(float(frame["close"].iloc[k + 1]))

    assert next_trades[0]["entry_price"] == pytest.approx(float(frame["open"].iloc[k + 1]))
    assert next_trades[0]["entry_price"] == pytest.approx(
        float(frame["close"].iloc[k]) + OPEN_STEP)
    assert next_trades[0]["exit_reason"] == "max_hold"
    assert next_trades[0]["exit_price"] == pytest.approx(float(frame["open"].iloc[k + 2]))
    assert next_trades[0]["exit_price"] == pytest.approx(
        float(frame["close"].iloc[k + 1]) + OPEN_STEP)

    # …and the sizes move with the fill price: sizing uses the price you fill at.
    assert next_trades[0]["quantity"] != close_trades[0]["quantity"]


def test_the_close_arm_reports_zero_accounting(market_dir):
    """`close` never refuses an entry nor falls back — the counters stay empty."""
    run = _run(market_dir, convention="close")
    accounting = run["metrics"]["fill_convention_accounting"]
    assert accounting["convention"] == "close"
    assert accounting["unfilled_entries"] == 0
    assert accounting["window_end_fallback_fills"] == 0


# ══════════════════════════════════════════════════════════════════════════
# 4 — the window end, handled explicitly
# ══════════════════════════════════════════════════════════════════════════

def test_an_entry_on_the_last_window_bar_is_refused_and_counted(market_dir):
    """No next bar → no fill → no trade, and the refusal is counted."""
    decision = _run(market_dir, convention="close")["trades"][0]["opened_at"]

    close_run = _run(market_dir, convention="close", date_end=decision)
    assert len(close_run["trades"]) == 1, "the close arm fills on the decision bar"

    next_run = _run(market_dir, convention="next_open", date_end=decision)
    assert next_run["trades"] == [], (
        "an entry whose fill falls outside the window must not be booked")
    accounting = next_run["metrics"]["fill_convention_accounting"]
    assert accounting["unfilled_entries"] == 1
    assert accounting["window_end_fallback_fills"] == 0


def test_an_exit_on_the_last_window_bar_falls_back_and_is_counted(market_dir):
    """The forced exit at the window end is priced at the last close — counted."""
    first = _run(market_dir, convention="close")["trades"][0]
    k = _frame_pos(first["opened_at"])
    last_bar = _frame().index[k + 1]          # the max_hold exit bar

    frame = _frame()
    close_run = _run(market_dir, convention="close", date_end=last_bar)
    next_run = _run(market_dir, convention="next_open", date_end=last_bar)

    close_exit = next(t for t in close_run["trades"] if t["exit_reason"] == "max_hold")
    next_exit = next(t for t in next_run["trades"] if t["exit_reason"] == "max_hold")
    assert close_exit["exit_price"] == pytest.approx(float(frame["close"].iloc[k + 1]))
    # No next bar exists for this exit either, so the fallback IS the last close…
    assert next_exit["exit_price"] == pytest.approx(float(frame["close"].iloc[k + 1]))
    # …and it is reported, per reason, instead of silently reverting.
    accounting = next_run["metrics"]["fill_convention_accounting"]
    assert accounting["window_end_fallback_by_reason"].get("max_hold") == 1
    assert accounting["window_end_fallback_fills"] >= 1
    # The re-entry the close arm makes on that same last bar has no fill either.
    assert accounting["unfilled_entries"] >= 1


# ══════════════════════════════════════════════════════════════════════════
# 5 — provenance
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("convention", ["close", "next_open"])
def test_the_result_and_the_metrics_name_the_convention(market_dir, convention):
    run = _run(market_dir, convention=convention)
    assert run["fill_convention"] == convention
    assert run["metrics"]["fill_convention"] == convention
    assert run["metrics"]["fill_convention_accounting"]["convention"] == convention


def test_the_config_key_is_honoured_without_the_explicit_kwarg(market_dir):
    """The GA path passes no kwarg: the convention must come from the config."""
    default_run = _run(market_dir, convention=None, with_fill_kwarg=False)
    assert default_run["fill_convention"] == "close"

    engine = _engine(_config(market_dir, convention="next_open"))
    run = engine.run_with_exit_evaluation(
        strategies=[_strategy()], symbols=["BTCUSDT"], date_start="2026-01-05",
        date_end="2026-02-01", initial_balance=10_000.0, mode="full",
        simulate_ai_weights=False, use_live_spread=False,
        benchmark_mode="none")
    assert run["fill_convention"] == "next_open"
    # …and it really shifted: the first entry is the next bar's open, not a close.
    frame = _frame()
    k = _frame_pos(run["trades"][0]["opened_at"])
    assert run["trades"][0]["entry_price"] == pytest.approx(
        float(frame["open"].iloc[k + 1]))


def test_the_hybrid_engine_never_prices_a_next_open_fill(market_dir):
    """`hybrid` refuses by name; `auto` falls back to legacy (and says so).

    The hybrid executor has no fill seam, so the only two honest behaviours are
    "refuse" (an explicit `engine_mode: hybrid`) and "run it on the engine that
    can" (`auto`, with the fallback recorded in the run's own accounting).
    """
    from core.backtest.fill_convention import FillConventionUnsupportedError

    strategies = [_strategy(f"hybrid_{i}") for i in range(3)]

    explicit = _config(market_dir, convention="next_open")
    explicit.backtest_engine_mode = "hybrid"
    with pytest.raises(FillConventionUnsupportedError) as excinfo:
        _engine(explicit).run_with_exit_evaluation(
            strategies=strategies, symbols=["BTCUSDT"], date_start="2026-01-05",
            date_end="2026-02-01", initial_balance=10_000.0, mode="full",
            simulate_ai_weights=False, use_live_spread=False)
    assert "legacy" in str(excinfo.value)

    auto = _config(market_dir, convention="next_open")
    auto.backtest_engine_mode = "auto"
    assert _engine(auto)._select_engine(strategies, "auto") == "hybrid"
    run = _engine(auto).run_with_exit_evaluation(
        strategies=strategies, symbols=["BTCUSDT"], date_start="2026-01-05",
        date_end="2026-02-01", initial_balance=10_000.0, mode="full",
        simulate_ai_weights=False, use_live_spread=False)
    assert run["fill_convention"] == "next_open"
    accounting = run["metrics"]["fill_convention_accounting"]
    assert "hybrid" in accounting["engine_fallback"]
    frame = _frame()
    for trade in run["trades"]:
        k = _frame_pos(trade["opened_at"])
        assert trade["entry_price"] == pytest.approx(float(frame["open"].iloc[k + 1]))


# ══════════════════════════════════════════════════════════════════════════
# 6 — the default is BIT-IDENTICAL to the pre-P9 revision
# ══════════════════════════════════════════════════════════════════════════

IDENTITY_HARNESS = ROOT / "tools" / "p9_fill_convention_identity.py"


def _run_identity(tree: Path, tmp_path: Path, tag: str, data_dir: str):
    harness = tmp_path / f"p9_identity_{tag}.py"
    harness.write_text(IDENTITY_HARNESS.read_text(encoding="utf-8"),
                       encoding="utf-8", newline="\n")
    out = tmp_path / f"p9_identity_{tag}.json"
    proc = subprocess.run(
        [sys.executable, str(harness), "--tree", str(tree),
         "--data-dir", data_dir, "--out", str(out)],
        cwd=str(tree), capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"identity harness failed in {tree}:\n{proc.stderr}"
    return out.read_bytes(), proc.stdout.strip()


def test_close_is_bit_identical_to_the_pre_p9_worktree(market_dir, tmp_path):
    """The same harness, the same synthetic market, two trees, one digest.

    Compares every trade (timestamps, prices, PnL, cost), every per-genome equity
    point and a **fixed** metrics digest as bytes.  The pre-P9 tree has no
    ``backtest.fill_convention`` key at all — that is the "key absent" case — and
    the payload therefore excludes P9's own two metrics (whose presence in the
    working tree is asserted separately, so the exclusion cannot hide a change).
    """
    worktree = tmp_path / "pre_p9_tree"
    add = subprocess.run(["git", "worktree", "add", "--detach", str(worktree),
                          BASELINE_REVISION],
                         cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    if add.returncode != 0:
        pytest.skip(f"cannot create a worktree at {BASELINE_REVISION}: "
                    f"{add.stderr.strip()}")
    try:
        head_bytes, head_digest = _run_identity(worktree, tmp_path, "baseline",
                                                market_dir)
        tree_bytes, tree_digest = _run_identity(ROOT, tmp_path, "working",
                                                market_dir)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=600)

    payload = json.loads(tree_bytes.decode("utf-8"))
    assert payload["trade_count"] > 0, "the identity harness traded nothing"
    assert len(next(iter(payload["equity"].values()))) > 0
    assert payload["metrics"]["total_return_pct"] is not None
    print(f"IDENTITY baseline={head_digest} working={tree_digest} "
          f"trades={payload['trade_count']} "
          f"equity_points={len(next(iter(payload['equity'].values())))} "
          f"payload_sha16={hashlib.sha256(tree_bytes).hexdigest()[:16]}")
    assert head_digest == tree_digest, (
        "the default (`close`) path is no longer bit-identical to "
        f"{BASELINE_REVISION}: baseline={head_digest} working={tree_digest}")
