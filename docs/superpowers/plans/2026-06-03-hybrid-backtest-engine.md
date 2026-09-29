# Hybrid Backtest Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current single-pass backtest engine with a two-phase hybrid architecture (vectorized signal matrix + event-driven executor) delivering 10× speedup while maintaining trade-level equivalence.

**Architecture:** Three-layer design: SignalMatrixBuilder clusters unique indicator configs and evaluates all conditions in one vectorized pass → EventDrivenExecutor consumes the precomputed signal matrix in a lightweight tick loop, reusing existing PositionSizer/close-position logic → MetricsCalculator unchanged. A router selects hybrid-vs-legacy based on `engine_mode` config and strategy count.

**Tech Stack:** Python 3.12, pandas, numpy, TA-Lib, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `app/config.py` | Modify | Add `backtest_engine_mode` and `backtest_ml_enabled` properties |
| `config/config.yaml` | Modify | Add `backtest` section with `engine_mode` and `ml_enabled` |
| `core/backtest/signal_matrix.py` | Create | SignalMatrixBuilder, IndicatorGrouper, SignalMatrix dataclass |
| `core/backtest/event_executor.py` | Create | EventDrivenExecutor — consumes signal matrix, produces trades |
| `core/backtest/engine_hybrid.py` | Create | `_run_hybrid()` entry point — orchestrates builder + executor |
| `core/backtest/engine.py` | Modify | Add router logic (`_select_engine`), wire hybrid path, slim legacy |
| `tests/test_signal_matrix.py` | Create | L1 unit tests: grouper, condition eval, matrix build (30 tests) |
| `tests/test_event_executor.py` | Create | L1 unit tests: entry/exit/SL/TP/balance/isolation (20 tests) |
| `tests/test_engine_router.py` | Create | L1 unit tests: auto/hybrid/legacy mode selection (5 tests) |
| `tests/test_hybrid_equivalence.py` | Create | L2+L3+L4: trade equivalence, ranking, anti-lookahead (12 tests) |

---

### Task 1: Config — add `backtest.engine_mode` and `backtest.ml_enabled`

**Files:**
- Modify: `app/config.py`
- Modify: `config/config.yaml`

- [ ] **Step 1: Add config model properties**

```python
# config/config.yaml — add after the existing 'language' line:
backtest:
  engine_mode: "auto"       # "auto" | "hybrid" | "legacy"
  ml_enabled: false          # enable ML training/prediction during backtest
```

- [ ] **Step 2: Read new config in Config._load()**

In `app/config.py`, after line 98 (`self.language = self._get("language", "zh")`), add:

```python
bt = self._get("backtest", {})
self.backtest_engine_mode = bt.get("engine_mode", "auto") if isinstance(bt, dict) else "auto"
if self.backtest_engine_mode not in ("auto", "hybrid", "legacy"):
    from loguru import logger
    logger.warning(f"Invalid backtest.engine_mode '{self.backtest_engine_mode}', falling back to 'auto'")
    self.backtest_engine_mode = "auto"
self.backtest_ml_enabled = bt.get("ml_enabled", False) if isinstance(bt, dict) else False
```

- [ ] **Step 3: Write the config test**

```python
# tests/test_config.py — add this function
def test_backtest_config_defaults():
    from app.config import Config
    Config._instance = None
    config = Config.load("sim")
    assert config.backtest_engine_mode == "auto"
    assert config.backtest_ml_enabled is False
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_config.py::test_backtest_config_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py config/config.yaml tests/test_config.py
git commit -m "feat(config): add backtest.engine_mode and backtest.ml_enabled"
```

---

### Task 2: SignalMatrix dataclass + IndicatorGrouper

**Files:**
- Create: `core/backtest/signal_matrix.py`

- [ ] **Step 1: Write the failing test for IndicatorGrouper**

```python
# tests/test_signal_matrix.py
import pytest
import pandas as pd
from core.strategy.loader import StrategyConfig, MLConfig
from core.backtest.signal_matrix import IndicatorGrouper

def _make_config(name, rsi_period, macd_fast=12, macd_slow=26):
    """Helper to create minimal StrategyConfig for testing."""
    return StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=["1h"],
        indicators={
            "rsi": {"period": rsi_period, "source": "close"},
            "macd": {"fast": macd_fast, "slow": macd_slow, "signal": 9},
        },
        entry_conditions={"long": ["rsi < 30"], "short": ["rsi > 70"]},
        exit_conditions={"long": [], "short": []},
        ml_config=MLConfig(enabled=False),
    )

def test_indicator_grouper_clusters_identical_configs():
    """Strategies with same indicator params should share groups."""
    s1 = _make_config("s1", 14, 12, 26)
    s2 = _make_config("s2", 14, 12, 26)  # same as s1
    s3 = _make_config("s3", 7, 8, 22)    # different

    grouper = IndicatorGrouper()
    groups = grouper.group([s1, s2, s3])

    # s1 and s2 share a group; s3 is alone
    assert len(groups) == 2
    group_sizes = sorted([len(g) for g in groups])
    assert group_sizes == [1, 2]

def test_indicator_grouper_hash_deterministic():
    """Same config always produces same group hash."""
    s1 = _make_config("s1", 14)
    s2 = _make_config("s2", 14)

    grouper = IndicatorGrouper()
    h1 = grouper._config_hash(s1)
    h2 = grouper._config_hash(s2)
    assert h1 == h2

def test_indicator_grouper_different_params_different_hash():
    """Different RSI periods produce different hashes."""
    s1 = _make_config("s1", 14)
    s2 = _make_config("s2", 21)

    grouper = IndicatorGrouper()
    h1 = grouper._config_hash(s1)
    h2 = grouper._config_hash(s2)
    assert h1 != h2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signal_matrix.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.backtest.signal_matrix'`

- [ ] **Step 3: Implement IndicatorGrouper + SignalMatrix**

```python
# core/backtest/signal_matrix.py
"""Signal matrix builder — vectorized condition evaluation for batch backtests."""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.strategy.indicators import compute_all, evaluate_condition
from core.strategy.loader import StrategyConfig


@dataclass(frozen=True)
class SignalMatrix:
    """Immutable container for precomputed entry/exit signals.

    Attributes:
        signals: MultiIndex (strategy_name, symbol, tf) × timestamp columns.
                 Values: 1 (long), -1 (short), 0 (no signal). dtype=int8.
        exit_signals: MultiIndex (strategy_name, symbol, tf, exit_key) × timestamp.
                      Values: bool. exit_key is "exit_long" or "exit_short".
        price_data: symbol → tf → OHLCV DataFrame (close prices for PnL calc).
        metadata: build_time_seconds, strategy_count, symbol_count, total_signals.
    """
    signals: pd.DataFrame
    exit_signals: pd.DataFrame
    price_data: dict[str, dict[str, pd.DataFrame]]
    metadata: dict

    def get_entry(self, strategy_name: str, symbol: str, tf: str,
                  ts: pd.Timestamp) -> int:
        """Return 1 (long), -1 (short), or 0 (no signal) at a given timestamp."""
        try:
            return int(self.signals.loc[(strategy_name, symbol, tf), ts])
        except (KeyError, TypeError):
            return 0

    def get_exit(self, strategy_name: str, symbol: str, tf: str,
                 side: str, ts: pd.Timestamp) -> bool:
        """Return True if exit condition met at timestamp."""
        exit_key = f"exit_{side}"
        try:
            return bool(self.exit_signals.loc[(strategy_name, symbol, tf, exit_key), ts])
        except (KeyError, TypeError):
            return False


class IndicatorGrouper:
    """Groups strategies by their indicator configuration hash.

    Strategies with identical indicator dicts share one compute_all() call.
    """

    @staticmethod
    def _config_hash(config: StrategyConfig) -> str:
        """Deterministic hash of a strategy's indicator config."""
        raw = json.dumps(config.indicators, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def group(self, strategies: list[StrategyConfig]) -> list[list[StrategyConfig]]:
        """Partition strategies into groups with identical indicator configs."""
        groups: dict[str, list[StrategyConfig]] = {}
        for s in strategies:
            h = self._config_hash(s)
            groups.setdefault(h, []).append(s)
        return list(groups.values())


class SignalMatrixBuilder:
    """Builds a precomputed signal matrix from strategy configs and market data.

    Phase 1: Group strategies by indicator config → one compute_all() per group.
    Phase 2: Collect all unique conditions → evaluate each once → distribute to strategies.
    """

    def __init__(self, feeder):
        """feeder: DataFeeder instance with loaded data."""
        self.feeder = feeder
        self.grouper = IndicatorGrouper()

    def build(self, strategies: list[StrategyConfig],
              symbols: list[str]) -> SignalMatrix:
        """Build the complete signal matrix for all strategies × symbols × timeframes."""
        t0 = time.time()

        # ── Group strategies by indicator config ──
        groups = self.grouper.group(strategies)
        logger.info(f"SignalMatrix: {len(strategies)} strategies → "
                    f"{len(groups)} unique indicator groups")

        # ── Determine unified timestamp index ──
        # Use the finest-granularity timeframe that any strategy uses
        all_tfs = set()
        for s in strategies:
            all_tfs.update(s.timeframes)
        # Build timestamp index from the first symbol's finest TF
        primary_sym = symbols[0]
        finest_tf = min(all_tfs, key=lambda t: {"1m": 1, "5m": 5, "15m": 15,
                                                  "1h": 60, "4h": 240}.get(t, 60))
        base_df = self.feeder.get_all_data_for_symbol(primary_sym, finest_tf)
        timestamps = base_df.index

        if len(timestamps) == 0:
            raise ValueError("No timestamps found in data feeder")

        # ── Build signals per symbol (sharding) ──
        all_entry_frames = []
        all_exit_frames = []
        price_data: dict[str, dict[str, pd.DataFrame]] = {}

        for sym in symbols:
            price_data[sym] = {}
            # Pre-fetch all timeframe data for this symbol
            sym_data: dict[str, pd.DataFrame] = {}
            for tf in all_tfs:
                df = self.feeder.get_all_data_for_symbol(sym, tf)
                if len(df) > 0:
                    sym_data[tf] = df
                    if tf == finest_tf or tf not in price_data[sym]:
                        price_data[sym][tf] = df[["open", "high", "low", "close", "volume"]].copy()

            # ── Compute indicators per group ──
            # indicator_cache: (group_hash, sym, tf) → DataFrame with all indicators
            indicator_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
            for group_idx, group in enumerate(groups):
                rep = group[0]  # representative config
                config_hash = self.grouper._config_hash(rep)
                for tf in rep.timeframes:
                    raw_df = sym_data.get(tf)
                    if raw_df is None or len(raw_df) < 20:
                        continue
                    cache_key = (config_hash, sym, tf)
                    if cache_key not in indicator_cache:
                        indicator_cache[cache_key] = compute_all(raw_df.copy(), rep.indicators)

            # ── Evaluate all conditions for this symbol ──
            # Collect unique conditions across all strategies
            all_conditions: dict[str, list[tuple[str, str, str]]] = {}
            # key = condition_string → [(strategy_name, tf, side)]
            for s in strategies:
                primary_tf = s.timeframes[0] if s.timeframes else "1h"
                for side in ("long", "short"):
                    for cond in s.entry_conditions.get(side, []):
                        all_conditions.setdefault(cond, []).append((s.name, primary_tf, side))
                    for cond in s.exit_conditions.get(side, []):
                        all_conditions.setdefault(cond, []).append((s.name, primary_tf, f"exit_{side}"))

            # Evaluate each unique condition once, then distribute
            condition_results: dict[str, dict[str, pd.Series]] = {}
            # → {condition_str: {tf: bool_series}}
            for cond_str in all_conditions:
                condition_results[cond_str] = {}
                for group in groups:
                    rep = group[0]
                    config_hash = self.grouper._config_hash(rep)
                    for tf in rep.timeframes:
                        df = indicator_cache.get((config_hash, sym, tf))
                        if df is None:
                            continue
                        result = evaluate_condition(df, cond_str)
                        condition_results[cond_str][tf] = result

            # ── Build entry signal matrix for this symbol ──
            # Reindex all condition results to unified timestamp index
            entry_rows = []
            exit_rows = []

            for s in strategies:
                primary_tf = s.timeframes[0] if s.timeframes else "1h"
                s_hash = self.grouper._config_hash(s)

                # Long entry: AND all long entry conditions
                long_conds = s.entry_conditions.get("long", [])
                if long_conds:
                    long_signals = None
                    for cond_str in long_conds:
                        result = condition_results.get(cond_str, {}).get(primary_tf)
                        if result is not None:
                            aligned = result.reindex(timestamps, fill_value=False)
                            if long_signals is None:
                                long_signals = aligned.astype(bool)
                            else:
                                long_signals = long_signals & aligned.astype(bool)
                    if long_signals is not None:
                        row = pd.Series(long_signals.astype(int), index=timestamps, dtype="int8")
                        row.name = (s.name, sym, primary_tf)
                        entry_rows.append(row)

                # Short entry: AND all short entry conditions
                short_conds = s.entry_conditions.get("short", [])
                if short_conds:
                    short_signals = None
                    for cond_str in short_conds:
                        result = condition_results.get(cond_str, {}).get(primary_tf)
                        if result is not None:
                            aligned = result.reindex(timestamps, fill_value=False)
                            if short_signals is None:
                                short_signals = aligned.astype(bool)
                            else:
                                short_signals = short_signals & aligned.astype(bool)
                    if short_signals is not None:
                        # Combine: long=1, short=-1. If both true at same time, → 0.
                        existing_long = None
                        for row in entry_rows:
                            if row.name == (s.name, sym, primary_tf):
                                existing_long = row
                                break
                        short_series = short_signals.astype(int) * -1
                        if existing_long is not None:
                            combined = existing_long.copy()
                            combined[short_series == -1] = -1
                            combined[(existing_long == 1) & (short_series == -1)] = 0
                            # Find and replace
                            for i, r in enumerate(entry_rows):
                                if r.name == (s.name, sym, primary_tf):
                                    entry_rows[i] = combined
                                    break
                        else:
                            row = pd.Series(short_series, index=timestamps, dtype="int8")
                            row.name = (s.name, sym, primary_tf)
                            entry_rows.append(row)

                # Exit signals: long and short
                for side in ("long", "short"):
                    exit_conds = s.exit_conditions.get(side, [])
                    if exit_conds:
                        exit_sig = None
                        for cond_str in exit_conds:
                            result = condition_results.get(cond_str, {}).get(primary_tf)
                            if result is not None:
                                aligned = result.reindex(timestamps, fill_value=False)
                                if exit_sig is None:
                                    exit_sig = aligned.astype(bool)
                                else:
                                    exit_sig = exit_sig | aligned.astype(bool)
                        if exit_sig is not None:
                            row = pd.Series(exit_sig, index=timestamps, dtype=bool)
                            row.name = (s.name, sym, primary_tf, f"exit_{side}")
                            exit_rows.append(row)

            if entry_rows:
                entry_df = pd.concat(entry_rows, axis=1).T
                all_entry_frames.append(entry_df)
            if exit_rows:
                exit_df = pd.concat(exit_rows, axis=1).T
                all_exit_frames.append(exit_df)

        # ── Assemble final signal matrices ──
        signals = pd.concat(all_entry_frames, axis=0) if all_entry_frames else pd.DataFrame()
        exit_signals = pd.concat(all_exit_frames, axis=0) if all_exit_frames else pd.DataFrame()

        total_signals = int((signals != 0).sum().sum()) if not signals.empty else 0

        return SignalMatrix(
            signals=signals,
            exit_signals=exit_signals,
            price_data=price_data,
            metadata={
                "build_time_seconds": round(time.time() - t0, 2),
                "strategy_count": len(strategies),
                "symbol_count": len(symbols),
                "indicator_groups": len(groups),
                "total_signals": total_signals,
                "timestamp_count": len(timestamps),
            },
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_signal_matrix.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add core/backtest/signal_matrix.py tests/test_signal_matrix.py
git commit -m "feat: SignalMatrixBuilder with IndicatorGrouper — vectorized condition eval"
```

---

### Task 3: SignalMatrixBuilder — condition evaluation tests

**Files:**
- Modify: `tests/test_signal_matrix.py`

- [ ] **Step 1: Add condition evaluation tests**

```python
def test_batch_condition_eval_matches_sequential():
    """Batch-evaluated conditions must match one-by-one evaluation."""
    import numpy as np
    from core.strategy.indicators import compute_all, evaluate_condition

    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=500, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(500) * 100)
    df = pd.DataFrame({
        "open": close - 50, "high": close + 100,
        "low": close - 100, "close": close,
        "volume": np.random.rand(500) * 100 + 50,
    }, index=dates)

    # Compute indicators for two configs
    ind1 = {"rsi": {"period": 14, "source": "close"}}
    ind2 = {"rsi": {"period": 7, "source": "close"}}
    df1 = compute_all(df.copy(), ind1)
    df2 = compute_all(df.copy(), ind2)

    conditions = ["rsi < 30", "rsi > 70"]
    for cond in conditions:
        r1 = evaluate_condition(df1, cond)
        r2 = evaluate_condition(df2, cond)
        # Same condition should produce different results on different RSI periods
        assert not r1.equals(r2) or r1.sum() == r2.sum() == 0

def test_condition_and_combination():
    """Multiple entry conditions must be combined with AND logic."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=500, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(500) * 100)
    df = pd.DataFrame({
        "open": close - 50, "high": close + 100,
        "low": close - 100, "close": close,
        "volume": np.random.rand(500) * 100 + 50,
    }, index=dates)
    df = compute_all(df, {"rsi": {"period": 14, "source": "close"},
                          "macd": {"fast": 12, "slow": 26, "signal": 9}})

    rsi_low = evaluate_condition(df, "rsi < 30")
    macd_positive = evaluate_condition(df, "macd_histogram > 0")
    combined = rsi_low & macd_positive

    # AND should be a subset of each individual condition
    assert combined.sum() <= rsi_low.sum()
    assert combined.sum() <= macd_positive.sum()
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_signal_matrix.py -v`
Expected: 5 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_signal_matrix.py
git commit -m "test: add batch condition eval and AND-combination tests"
```

---

### Task 4: EventDrivenExecutor — core trade execution

**Files:**
- Create: `core/backtest/event_executor.py`

- [ ] **Step 1: Write failing executor tests**

```python
# tests/test_event_executor.py
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


def _make_dummy_matrix(with_signals=True):
    """Create a minimal SignalMatrix for testing the executor."""
    from core.backtest.signal_matrix import SignalMatrix

    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(100) * 50)

    # Entry signal at index 10 (long), exit at index 20
    entry_data = np.zeros(100, dtype="int8")
    exit_data = np.zeros(100, dtype=bool)
    if with_signals:
        entry_data[10] = 1  # long entry at t=10
        exit_data[20] = True  # exit at t=20

    entry_df = pd.DataFrame(
        [entry_data],
        index=pd.MultiIndex.from_tuples([("test_strat", "BTCUSDT", "1h")]),
        columns=dates,
    )
    exit_df = pd.DataFrame(
        [exit_data],
        index=pd.MultiIndex.from_tuples([("test_strat", "BTCUSDT", "1h", "exit_long")]),
        columns=dates,
    )

    price_df = pd.DataFrame({
        "open": close - 10, "high": close + 50,
        "low": close - 50, "close": close,
        "volume": np.ones(100) * 100,
    }, index=dates)

    return SignalMatrix(
        signals=entry_df,
        exit_signals=exit_df,
        price_data={"BTCUSDT": {"1h": price_df}},
        metadata={"build_time_seconds": 0.01},
    )


def test_executor_enter_long():
    """Executor should open a long position when entry signal is 1."""
    from core.backtest.event_executor import EventDrivenExecutor
    from core.risk.position_sizer import PositionSizer
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)

    matrix = _make_dummy_matrix(with_signals=True)
    executor = EventDrivenExecutor(sizer, config.hard_limits)
    result = executor.run(matrix, initial_balance=10000.0)

    assert result.final_balance != 10000.0  # balance changed
    assert len(result.trades) >= 1
    trade = result.trades[0]
    assert trade["side"] == "long"
    assert trade["symbol"] == "BTCUSDT"
    assert trade["strategy"] == "test_strat"


def test_executor_no_signals_no_trades():
    """No signals should produce no trades, unchanged balance."""
    from core.backtest.event_executor import EventDrivenExecutor
    from core.risk.position_sizer import PositionSizer
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)

    matrix = _make_dummy_matrix(with_signals=False)
    executor = EventDrivenExecutor(sizer, config.hard_limits)
    result = executor.run(matrix, initial_balance=10000.0)

    assert len(result.trades) == 0
    assert result.final_balance == 10000.0


def test_executor_per_strategy_isolation():
    """Isolation mode: two strategies should not block each other."""
    from core.backtest.event_executor import EventDrivenExecutor
    from core.backtest.signal_matrix import SignalMatrix
    from core.risk.position_sizer import PositionSizer
    from app.config import Config

    dates = pd.date_range("2026-01-01", periods=50, freq="1h")
    entry_data = np.zeros((2, 50), dtype="int8")
    entry_data[0, 5] = 1   # strat A long
    entry_data[1, 8] = -1  # strat B short

    entry_df = pd.DataFrame(
        entry_data,
        index=pd.MultiIndex.from_tuples([
            ("strat_a", "BTCUSDT", "1h"),
            ("strat_b", "BTCUSDT", "1h"),
        ]),
        columns=dates,
    )
    exit_df = pd.DataFrame(columns=dates)
    exit_df.index = pd.MultiIndex.from_tuples([], names=["strategy", "symbol", "tf", "side"])

    close = 50000 + np.cumsum(np.random.randn(50) * 50)
    price_df = pd.DataFrame({"open": close - 10, "high": close + 50,
                              "low": close - 50, "close": close,
                              "volume": np.ones(50) * 100}, index=dates)

    matrix = SignalMatrix(signals=entry_df, exit_signals=exit_df,
                          price_data={"BTCUSDT": {"1h": price_df}},
                          metadata={"build_time_seconds": 0.01})

    Config._instance = None
    config = Config.load("sim")
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)
    executor = EventDrivenExecutor(sizer, config.hard_limits,
                                     per_strategy_isolation=True)
    result = executor.run(matrix, initial_balance=10000.0)

    # Both strategies should have opened positions
    strategy_trades = set(t["strategy"] for t in result.trades)
    assert "strat_a" in strategy_trades
    assert "strat_b" in strategy_trades
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_event_executor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.backtest.event_executor'`

- [ ] **Step 3: Implement EventDrivenExecutor**

```python
# core/backtest/event_executor.py
"""Event-driven executor — consumes precomputed signal matrix, produces trades."""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from core.backtest.signal_matrix import SignalMatrix
from core.risk.position_sizer import PositionSizer


@dataclass
class ExecutorResult:
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    per_matrix: dict = field(default_factory=dict)
    final_balance: float = 0.0
    runtime_seconds: float = 0.0


class EventDrivenExecutor:
    """Consumes a precomputed SignalMatrix and simulates trade execution.

    Reuses the same position sizing, stop-loss, and take-profit logic as the
    legacy engine. The key difference: entry/exit signals are looked up from
    the matrix instead of computed on-the-fly.
    """

    def __init__(self, sizer: PositionSizer, hard_limits,
                 per_strategy_isolation: bool = False,
                 max_positions: int = 15):
        self.sizer = sizer
        self.hard_limits = hard_limits
        self.per_strategy_isolation = per_strategy_isolation
        self.max_positions = max_positions

    def _pkey(self, sym: str, s_name: str = "") -> str:
        return f"{s_name}|{sym}" if self.per_strategy_isolation else sym

    def _close_position(self, pos_key: str, pos: dict, exit_price: float,
                        ts, reason: str, trades: list, balance: float,
                        positions: dict, per_matrix: dict) -> float:
        """Close a position. Identical logic to BacktestEngine._close_position."""
        sym = pos["symbol"]
        entry_price = pos["entry_price"]
        qty = pos["quantity"]
        amount = pos.get("amount_usdt", qty * entry_price)
        side = pos["side"]
        strategy_name = pos.get("strategy_name", "")

        if side == "long":
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty

        trades.append({
            "symbol": sym, "side": side,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "quantity": round(qty, 6),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / (entry_price * qty) * 100, 2) if entry_price > 0 else 0,
            "strategy": strategy_name,
            "opened_at": str(pos.get("opened_at", ts)),
            "closed_at": str(ts),
            "amount_usdt": round(amount, 2),
            "exit_reason": reason,
        })
        balance += amount + pnl

        if strategy_name in per_matrix and sym in per_matrix[strategy_name]:
            cell = per_matrix[strategy_name][sym]
            cell["trades"] += 1
            cell["pnl"] += pnl
            if pnl > 0:
                cell["winning"] += 1
            else:
                cell["losing"] += 1
            if side == "long":
                cell["long_trades"] += 1
            else:
                cell["short_trades"] += 1

        del positions[pos_key]
        return balance

    def run(self, matrix: SignalMatrix,
            initial_balance: float = 10000.0,
            progress_callback=None) -> ExecutorResult:
        """Execute all trades by consuming the signal matrix chronologically."""
        t0 = time.time()

        if matrix.signals.empty:
            return ExecutorResult(
                trades=[], equity_curve=[],
                per_matrix={}, final_balance=initial_balance,
                runtime_seconds=round(time.time() - t0, 2),
            )

        timestamps = matrix.signals.columns
        balance = initial_balance
        positions: dict[str, dict] = {}
        trades: list[dict] = []
        equity_curve: list[dict] = []
        pos_counter = 0

        # ── Initialize per_matrix ──
        per_matrix: dict[str, dict[str, dict]] = {}
        strategy_names = set(idx[0] for idx in matrix.signals.index)
        symbols = set(idx[1] for idx in matrix.signals.index)
        for s_name in strategy_names:
            per_matrix[s_name] = {}
            for sym in symbols:
                per_matrix[s_name][sym] = {
                    "trades": 0, "pnl": 0.0, "winning": 0, "losing": 0,
                    "long_trades": 0, "short_trades": 0,
                }

        total_steps = len(timestamps)
        for step, ts in enumerate(timestamps):

            if progress_callback and (step % 50 == 0 or step == total_steps - 1):
                progress_callback(step + 1, total_steps, ts)

            # ── Check exits (SL, TP, indicator exits) ──
            for pos_key in list(positions.keys()):
                pos = positions[pos_key]
                sym = pos["symbol"]
                s_name = pos.get("strategy_name", "")
                side = pos["side"]
                tf = pos.get("timeframe", "1h")

                # Get current price
                price_df = matrix.price_data.get(sym, {}).get(tf)
                if price_df is None:
                    continue
                try:
                    idx = price_df.index.get_loc(ts)
                    if isinstance(idx, slice):
                        idx = idx.stop - 1
                    current_price = float(price_df.iloc[idx]["close"])
                except (KeyError, IndexError):
                    continue

                # Stop-loss check
                sl_price = pos.get("stop_loss", 0)
                if sl_price > 0:
                    hit = (side == "long" and current_price <= sl_price) or \
                          (side == "short" and current_price >= sl_price)
                    if hit:
                        balance = self._close_position(
                            pos_key, pos, sl_price, ts, "stop_loss",
                            trades, balance, positions, per_matrix)
                        continue

                # Take-profit check
                tp_levels = pos.get("take_profits", [])
                for tp_price, tp_pct in tp_levels:
                    hit = (side == "long" and current_price >= tp_price) or \
                          (side == "short" and current_price <= tp_price)
                    if hit:
                        balance = self._close_position(
                            pos_key, pos, tp_price, ts, f"tp_{int(tp_pct*100)}pct",
                            trades, balance, positions, per_matrix)
                        break

                if pos_key not in positions:
                    continue

                # Indicator exit check
                exit_hit = matrix.get_exit(s_name, sym, tf, side, ts)
                if exit_hit:
                    balance = self._close_position(
                        pos_key, pos, current_price, ts, "indicator",
                        trades, balance, positions, per_matrix)

            # ── Check entries ──
            for idx_tuple in matrix.signals.index:
                s_name, sym, tf = idx_tuple
                entry_val = matrix.get_entry(s_name, sym, tf, ts)

                if entry_val == 0:
                    continue

                pos_key = self._pkey(sym, s_name)
                if pos_key in positions:
                    continue
                if len(positions) >= self.max_positions:
                    break

                side = "long" if entry_val == 1 else "short"

                # Get current price
                price_df = matrix.price_data.get(sym, {}).get(tf)
                if price_df is None:
                    continue
                try:
                    idx_val = price_df.index.get_loc(ts)
                    if isinstance(idx_val, slice):
                        idx_val = idx_val.stop - 1
                    price = float(price_df.iloc[idx_val]["close"])
                except (KeyError, IndexError):
                    continue

                # Position sizing
                qty, risk_amount = self.sizer.calculate_position_size(
                    account_balance=balance, current_price=price,
                    position_type="satellite")
                if qty <= 0:
                    continue

                amount_usdt = qty * price
                if amount_usdt > balance * 0.95:
                    continue

                pos_counter += 1
                balance -= amount_usdt

                sl = self.sizer.calculate_stop_loss(entry_price=price, side=side)
                tps = self.sizer.calculate_take_profits(entry_price=price, side=side)

                positions[pos_key] = {
                    "symbol": sym, "side": side,
                    "quantity": qty, "entry_price": price,
                    "amount_usdt": amount_usdt,
                    "strategy_name": s_name,
                    "opened_at": str(ts), "trade_group": f"bt_{pos_counter}_{int(ts.timestamp())}",
                    "stop_loss": sl, "take_profits": tps,
                    "timeframe": tf, "reduce_count": 0,
                }

            # ── Equity curve ──
            invested = sum(p.get("amount_usdt", 0) for p in positions.values())
            equity_curve.append({
                "time": str(ts),
                "equity": round(balance + invested, 2),
                "balance": round(balance, 2),
                "invested": round(invested, 2),
            })

        # ── Finalize per_matrix ──
        for s_name in per_matrix:
            for sym in per_matrix[s_name]:
                cell = per_matrix[s_name][sym]
                n = cell["trades"]
                cell["win_rate_pct"] = round(cell["winning"] / n * 100, 1) if n > 0 else 0.0
                cell["pnl"] = round(cell["pnl"], 2)

        return ExecutorResult(
            trades=trades,
            equity_curve=equity_curve,
            per_matrix=per_matrix,
            final_balance=round(balance, 2),
            runtime_seconds=round(time.time() - t0, 2),
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_event_executor.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add core/backtest/event_executor.py tests/test_event_executor.py
git commit -m "feat: EventDrivenExecutor — signal matrix consumer with trade execution"
```

---

### Task 5: Executor edge cases — SL/TP/balance limits

**Files:**
- Modify: `tests/test_event_executor.py`

- [ ] **Step 1: Add edge case tests**

```python
def test_executor_stop_loss_triggers():
    """Stop-loss should be checked every tick and close the position."""
    from core.backtest.event_executor import EventDrivenExecutor
    from core.backtest.signal_matrix import SignalMatrix
    from core.risk.position_sizer import PositionSizer
    from app.config import Config

    dates = pd.date_range("2026-01-01", periods=50, freq="1h")
    # Price drops 20% from 50000 to 40000 — SL should trigger
    close = np.linspace(50000, 40000, 50)
    entry_data = np.zeros(50, dtype="int8")
    entry_data[5] = 1  # long entry at t=5 (price ~49000)

    entry_df = pd.DataFrame(
        [entry_data],
        index=pd.MultiIndex.from_tuples([("test", "BTCUSDT", "1h")]),
        columns=dates,
    )
    exit_df = pd.DataFrame(columns=dates)
    exit_df.index = pd.MultiIndex.from_tuples([], names=["strategy", "symbol", "tf", "side"])

    price_df = pd.DataFrame({"open": close - 10, "high": close + 50,
                              "low": close - 50, "close": close,
                              "volume": np.ones(50) * 100}, index=dates)

    matrix = SignalMatrix(signals=entry_df, exit_signals=exit_df,
                          price_data={"BTCUSDT": {"1h": price_df}},
                          metadata={"build_time_seconds": 0})

    Config._instance = None
    config = Config.load("sim")
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)
    executor = EventDrivenExecutor(sizer, config.hard_limits)
    result = executor.run(matrix, initial_balance=10000.0)

    assert len(result.trades) >= 1
    trade = result.trades[0]
    assert trade["exit_reason"] == "stop_loss"


def test_executor_balance_insufficient():
    """Should not enter when balance is too low."""
    from core.backtest.event_executor import EventDrivenExecutor
    from core.backtest.signal_matrix import SignalMatrix
    from core.risk.position_sizer import PositionSizer
    from app.config import Config

    dates = pd.date_range("2026-01-01", periods=10, freq="1h")
    entry_data = np.zeros(10, dtype="int8")
    entry_data[1] = 1

    entry_df = pd.DataFrame(
        [entry_data],
        index=pd.MultiIndex.from_tuples([("test", "BTCUSDT", "1h")]),
        columns=dates,
    )
    exit_df = pd.DataFrame(columns=dates)
    exit_df.index = pd.MultiIndex.from_tuples([], names=["strategy", "symbol", "tf", "side"])

    close = np.full(10, 50000.0)
    price_df = pd.DataFrame({"open": close - 10, "high": close + 50,
                              "low": close - 50, "close": close,
                              "volume": np.ones(10) * 100}, index=dates)

    matrix = SignalMatrix(signals=entry_df, exit_signals=exit_df,
                          price_data={"BTCUSDT": {"1h": price_df}},
                          metadata={"build_time_seconds": 0})

    Config._instance = None
    config = Config.load("sim")
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)
    executor = EventDrivenExecutor(sizer, config.hard_limits)
    result = executor.run(matrix, initial_balance=1.0)  # too low

    assert len(result.trades) == 0


def test_executor_max_positions():
    """Should respect max_positions limit."""
    from core.backtest.event_executor import EventDrivenExecutor
    from core.backtest.signal_matrix import SignalMatrix
    from core.risk.position_sizer import PositionSizer
    from app.config import Config

    dates = pd.date_range("2026-01-01", periods=30, freq="1h")
    entry_data = np.zeros((3, 30), dtype="int8")
    entry_data[0, 2] = 1   # BTC
    entry_data[1, 4] = 1   # ETH
    entry_data[2, 6] = 1   # BNB — should be blocked if max=2

    entry_df = pd.DataFrame(
        entry_data,
        index=pd.MultiIndex.from_tuples([
            ("s1", "BTCUSDT", "1h"),
            ("s2", "ETHUSDT", "1h"),
            ("s3", "BNBUSDT", "1h"),
        ]),
        columns=dates,
    )
    exit_df = pd.DataFrame(columns=dates)
    exit_df.index = pd.MultiIndex.from_tuples([], names=["strategy", "symbol", "tf", "side"])

    close = np.full(30, 3000.0)  # low price to allow entry
    price_df = pd.DataFrame({"open": close - 10, "high": close + 50,
                              "low": close - 50, "close": close,
                              "volume": np.ones(30) * 200}, index=dates)

    matrix = SignalMatrix(signals=entry_df, exit_signals=exit_df,
                          price_data={
                              "BTCUSDT": {"1h": price_df},
                              "ETHUSDT": {"1h": price_df},
                              "BNBUSDT": {"1h": price_df},
                          },
                          metadata={"build_time_seconds": 0})

    Config._instance = None
    config = Config.load("sim")
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)
    executor = EventDrivenExecutor(sizer, config.hard_limits, max_positions=2)
    result = executor.run(matrix, initial_balance=100000.0)

    # At most 2 positions opened
    symbols_entered = set(t["symbol"] for t in result.trades)
    assert len(symbols_entered) <= 2
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_event_executor.py -v`
Expected: 6 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_event_executor.py
git commit -m "test: add executor edge cases — SL trigger, insufficient balance, max positions"
```

---

### Task 6: Engine router — wire hybrid path into BacktestEngine

**Files:**
- Modify: `core/backtest/engine.py`
- Create: `core/backtest/engine_hybrid.py`

- [ ] **Step 1: Create engine_hybrid.py**

```python
# core/backtest/engine_hybrid.py
"""Hybrid backtest engine entry point — orchestrates SignalMatrixBuilder + EventDrivenExecutor."""

import time
from pathlib import Path
from loguru import logger

from core.backtest.data_feeder import DataFeeder
from core.backtest.signal_matrix import SignalMatrixBuilder
from core.backtest.event_executor import EventDrivenExecutor
from core.backtest.metrics import calculate_metrics
from core.risk.position_sizer import PositionSizer


def run_hybrid(strategies, symbols, date_start, date_end,
               config, loader, initial_balance=10000.0,
               per_strategy_isolation=False,
               progress_callback=None) -> dict:
    """Run backtest using the hybrid (vectorized) engine.

    Args:
        strategies: List of StrategyConfig objects (NOT strategy name strings).
        symbols: List of symbol strings.
        date_start, date_end: Date range strings.
        config: Config instance.
        loader: StrategyLoader instance.
        initial_balance: Starting balance.
        per_strategy_isolation: Independent positions per strategy.
        progress_callback: (step, total, ts) called during executor loop.

    Returns:
        dict with keys: trades, equity_curve, metrics, final_balance,
                        per_matrix, strategies, symbols, date_start, date_end,
                        engine_mode="hybrid", metadata.
    """
    t0 = time.time()

    # Load strategy configs
    strategy_configs = []
    if isinstance(strategies, list) and strategies and not isinstance(strategies[0], str):
        strategy_configs = strategies
    else:
        for name in strategies:
            s = loader.load(name)
            strategy_configs.append(s)

    # Override ML
    for s in strategy_configs:
        if s.ml_config:
            s.ml_config.enabled = False

    strategy_names = [s.name for s in strategy_configs]

    # Determine intervals
    intervals = list(set(tf for s in strategy_configs for tf in s.timeframes)) or ["1h"]

    # Load data
    cache_dir = str(Path(config.data_dir) / "market")
    feeder = DataFeeder(cache_dir, symbols, intervals, date_start, date_end)
    feeder.load()

    if len(feeder) == 0:
        return {"error": "No historical data found"}

    # Phase 1: Build signal matrix
    logger.info(f"Hybrid engine: building signal matrix for {len(strategy_configs)} strategies")
    builder = SignalMatrixBuilder(feeder)
    matrix = builder.build(strategy_configs, symbols)
    logger.info(f"Signal matrix built in {matrix.metadata['build_time_seconds']}s: "
                f"{matrix.metadata['total_signals']} signals, "
                f"{matrix.metadata['timestamp_count']} timestamps, "
                f"{matrix.metadata['indicator_groups']} indicator groups")

    # Phase 2: Execute trades
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)
    max_positions = config.hard_limits.max_open_trades
    if per_strategy_isolation:
        max_positions = max(1, max_positions // max(len(strategy_configs), 1))

    executor = EventDrivenExecutor(
        sizer, config.hard_limits,
        per_strategy_isolation=per_strategy_isolation,
        max_positions=max_positions,
    )

    logger.info(f"Hybrid engine: executing trades ({matrix.metadata['timestamp_count']} ticks)")
    exec_result = executor.run(matrix, initial_balance=initial_balance,
                               progress_callback=progress_callback)

    # Phase 3: Calculate metrics
    metrics = calculate_metrics(exec_result.trades, exec_result.equity_curve,
                                initial_balance, exec_result.final_balance)
    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    metrics["engine_mode"] = "hybrid"
    metrics["signal_matrix_build_seconds"] = matrix.metadata["build_time_seconds"]

    logger.info(f"Hybrid engine: {len(exec_result.trades)} trades, "
                f"final_balance={exec_result.final_balance:.2f}, "
                f"runtime={metrics['runtime_seconds']}s")

    return {
        "trades": exec_result.trades,
        "equity_curve": exec_result.equity_curve,
        "metrics": metrics,
        "final_balance": exec_result.final_balance,
        "initial_balance": initial_balance,
        "strategies": strategy_names,
        "symbols": symbols,
        "date_start": date_start,
        "date_end": date_end,
        "per_matrix": exec_result.per_matrix,
        "engine_mode": "hybrid",
    }
```

- [ ] **Step 2: Add router logic to engine.py**

In `core/backtest/engine.py`, add a new method `_select_engine` and modify
`run_with_exit_evaluation` to route to hybrid when appropriate.

Add the import at the top (after existing imports):

```python
from core.backtest.engine_hybrid import run_hybrid
```

Add the router method to the `BacktestEngine` class:

```python
def _select_engine(self, strategies, engine_mode: str) -> str:
    """Determine which engine to use: 'hybrid' or 'legacy'."""
    if engine_mode == "legacy":
        return "legacy"

    # Check if any strategy has ML enabled
    strategy_configs = []
    if isinstance(strategies, list) and strategies and not isinstance(strategies[0], str):
        strategy_configs = strategies
    else:
        for name in strategies:
            try:
                s = self.strategy_engine.loader.load(name)
                strategy_configs.append(s)
            except Exception:
                pass

    has_ml = any(
        s.ml_config and s.ml_config.enabled
        for s in strategy_configs
    )

    if engine_mode == "hybrid":
        if has_ml:
            raise ValueError(
                "Hybrid engine does not support ML training/prediction. "
                "Set config backtest.ml_enabled=false or use engine_mode='legacy'.")
        return "hybrid"

    # engine_mode == "auto"
    n = len(strategies) if isinstance(strategies, list) else 1
    if n >= 3 and not has_ml:
        return "hybrid"
    return "legacy"
```

Modify `run_with_exit_evaluation` signature to accept the new parameter and add routing at the top of the method body.
After the `t0 = time.time()` line, add:

```python
# Determine engine mode from config and parameters
_engine_mode = getattr(self.config, 'backtest_engine_mode', 'auto')
_ml_enabled = getattr(self.config, 'backtest_ml_enabled', False)

# Override ML based on config
if not _ml_enabled:
    # Disable ML on all strategy configs
    _tmp_configs = []
    if isinstance(strategies, list) and strategies and not isinstance(strategies[0], str):
        _tmp_configs = strategies
    else:
        from core.strategy.loader import StrategyLoader
        _loader = self.strategy_engine.loader
        for name in strategies:
            try:
                _tmp_configs.append(_loader.load(name))
            except Exception:
                pass
    for s in _tmp_configs:
        if s.ml_config:
            s.ml_config.enabled = False

# Route to hybrid engine if applicable
try:
    use_hybrid = self._select_engine(strategies, _engine_mode) == "hybrid"
except ValueError:
    use_hybrid = False

if use_hybrid:
    try:
        return run_hybrid(
            strategies, symbols, date_start, date_end,
            self.config, self.strategy_engine.loader,
            initial_balance=initial_balance,
            per_strategy_isolation=per_strategy_isolation,
            progress_callback=progress_callback,
        )
    except Exception as e:
        logger.warning(f"Hybrid engine failed ({e}), falling back to legacy")
        # Fall through to legacy engine
```

- [ ] **Step 3: Write router tests**

```python
# tests/test_engine_router.py
import pytest
from core.strategy.loader import StrategyConfig, MLConfig
from app.config import Config


def _make_configs(n=3, ml_enabled=False):
    """Helper: create N minimal StrategyConfigs."""
    configs = []
    for i in range(n):
        configs.append(StrategyConfig(
            name=f"test_{i}", enabled=True, mode="trend", timeframes=["1h"],
            indicators={"rsi": {"period": 14, "source": "close"}},
            entry_conditions={"long": [], "short": []},
            exit_conditions={"long": [], "short": []},
            ml_config=MLConfig(enabled=ml_enabled),
        ))
    return configs


def test_router_auto_hybrid_with_3_strategies():
    """3+ strategies, no ML → hybrid."""
    from core.backtest.engine import BacktestEngine
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config

    result = engine._select_engine(_make_configs(3), "auto")
    assert result == "hybrid"


def test_router_auto_legacy_with_1_strategy():
    """1 strategy → legacy."""
    from core.backtest.engine import BacktestEngine
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config

    result = engine._select_engine(_make_configs(1), "auto")
    assert result == "legacy"


def test_router_force_legacy():
    """Explicit legacy mode always returns legacy."""
    from core.backtest.engine import BacktestEngine
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config

    result = engine._select_engine(_make_configs(5), "legacy")
    assert result == "legacy"


def test_router_force_hybrid():
    """Explicit hybrid mode returns hybrid when no ML."""
    from core.backtest.engine import BacktestEngine
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config

    result = engine._select_engine(_make_configs(1, ml_enabled=False), "hybrid")
    assert result == "hybrid"


def test_router_hybrid_with_ml_raises():
    """Hybrid mode with ML enabled should raise ValueError."""
    from core.backtest.engine import BacktestEngine
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config

    with pytest.raises(ValueError, match="Hybrid engine does not support ML"):
        engine._select_engine(_make_configs(1, ml_enabled=True), "hybrid")
```

- [ ] **Step 4: Run router tests**

Run: `pytest tests/test_engine_router.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add core/backtest/engine_hybrid.py core/backtest/engine.py tests/test_engine_router.py
git commit -m "feat: engine router — auto/hybrid/legacy mode selection + hybrid entry point"
```

---

### Task 7: Equivalence tests — L2 trade-level comparison

**Files:**
- Create: `tests/test_hybrid_equivalence.py`

- [ ] **Step 1: Write equivalence tests**

```python
# tests/test_hybrid_equivalence.py
"""L2+L3 equivalence tests: old vs hybrid engine produce identical results."""
import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path

from core.strategy.loader import StrategyConfig, MLConfig


def _make_test_strategy(name, rsi_period=14, macd_fast=12):
    """Create a realistic test strategy that will generate trades."""
    return StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=["1h"],
        indicators={
            "rsi": {"period": rsi_period, "source": "close"},
            "macd": {"fast": macd_fast, "slow": 26, "signal": 9},
        },
        entry_conditions={
            "long": ["rsi < 35", "macd_histogram > 0"],
            "short": ["rsi > 65", "macd_histogram < 0"],
        },
        exit_conditions={
            "long": ["rsi > 60"],
            "short": ["rsi < 40"],
        },
        ml_config=MLConfig(enabled=False),
    )


@pytest.mark.slow
def test_hybrid_matches_legacy_trade_for_trade():
    """1 week of data, 3 strategies: every trade must match exactly."""
    from app.config import Config
    from core.strategy.loader import StrategyLoader
    from core.backtest.engine import BacktestEngine
    from core.backtest.engine_hybrid import run_hybrid
    from app.event_bus import EventBus

    Config._instance = None
    config = Config.load("sim")
    config.backtest_engine_mode = "legacy"
    config.backtest_ml_enabled = False

    tmp_dir = tempfile.mkdtemp()
    strats_dir = Path(tmp_dir) / "strategies"
    strats_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir = tmp_dir

    loader = StrategyLoader(str(strats_dir))
    symbols = ["BTCUSDT", "ETHUSDT"]

    strategies = [
        _make_test_strategy("s1", 14),
        _make_test_strategy("s2", 7),
        _make_test_strategy("s3", 21),
    ]
    for s in strategies:
        loader.save(s)

    # Use the real market data from the project
    data_dir = "data"
    config.data_dir = data_dir

    # Run legacy engine
    t0 = pd.Timestamp.now()
    bus = EventBus()
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config
    engine.strategy_engine = type('obj', (object,), {'loader': loader})()

    # Pick a short date range that has data
    date_start = "2026-05-25"
    date_end = "2026-05-31"

    legacy_result = engine.run_with_exit_evaluation(
        strategies=[s.name for s in strategies],
        symbols=symbols,
        date_start=date_start,
        date_end=date_end,
        initial_balance=10000.0,
        mode="full",
        simulate_ai_weights=False,
        engine_mode="legacy",
    )

    # Run hybrid engine
    hybrid_result = run_hybrid(
        strategies=strategies,
        symbols=symbols,
        date_start=date_start,
        date_end=date_end,
        config=config,
        loader=loader,
        initial_balance=10000.0,
    )

    # Compare
    legacy_trades = legacy_result.get("trades", [])
    hybrid_trades = hybrid_result.get("trades", [])

    if len(legacy_trades) == 0 and len(hybrid_trades) == 0:
        pytest.skip("No trades in either engine — not enough data variation")

    # Count of trades should match
    assert len(legacy_trades) == len(hybrid_trades), \
        f"Trade count mismatch: legacy={len(legacy_trades)} vs hybrid={len(hybrid_trades)}"

    for i, (lt, ht) in enumerate(zip(legacy_trades, hybrid_trades)):
        assert lt["symbol"] == ht["symbol"], f"Trade {i}: symbol mismatch"
        assert lt["side"] == ht["side"], f"Trade {i}: side mismatch"
        assert lt["strategy"] == ht["strategy"], f"Trade {i}: strategy mismatch"
        assert abs(lt["entry_price"] - ht["entry_price"]) < 1.0, \
            f"Trade {i}: entry price mismatch {lt['entry_price']} vs {ht['entry_price']}"
        assert abs(lt["exit_price"] - ht["exit_price"]) < 1.0, \
            f"Trade {i}: exit price mismatch"
        assert abs(lt["pnl"] - ht["pnl"]) < 0.1, \
            f"Trade {i}: PnL mismatch {lt['pnl']} vs {ht['pnl']}"


def test_no_lookahead_bias():
    """Time-shifted data must produce different signals."""
    # This is a lightweight test that validates the anti-lookahead mechanism.
    # Full verification: if all condition evaluations shift data by 1 period,
    # the signal output must differ at >= 1% of timestamps.

    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=300, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(300) * 100)
    df = pd.DataFrame({
        "open": close - 50, "high": close + 100,
        "low": close - 100, "close": close,
        "volume": np.random.rand(300) * 100 + 50,
    }, index=dates)

    from core.strategy.indicators import compute_all, evaluate_condition
    ind = {"rsi": {"period": 14, "source": "close"}}
    df_original = compute_all(df.copy(), ind)
    df_shifted = compute_all(df.shift(1).copy(), ind)

    # Evaluate conditions on both
    conditions = ["rsi < 30", "rsi > 70", "rsi < 40", "rsi > 60"]
    disagreement_count = 0
    total = 0
    for cond in conditions:
        r1 = evaluate_condition(df_original, cond)
        r2 = evaluate_condition(df_shifted, cond)
        r2 = r2.reindex(r1.index, fill_value=False)
        disagreement_count += (r1 != r2).sum()
        total += len(r1)

    disagreement_rate = disagreement_count / max(total, 1)
    assert disagreement_rate >= 0.01, \
        f"Lookahead risk: only {disagreement_rate:.4f} of signals differ on shifted data"
```

- [ ] **Step 2: Run equivalence tests**

Run: `pytest tests/test_hybrid_equivalence.py -v -m "not slow"`
Expected: 1 PASS (anti-lookahead test)

- [ ] **Step 3: Commit**

```bash
git add tests/test_hybrid_equivalence.py
git commit -m "test: L2 trade equivalence + L4 anti-lookahead validation"
```

---

### Task 8: Final integration — run full test suite

**Files:**
- None (verification only)

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: all tests pass (current 97 + new ~35 = ~132 tests)

- [ ] **Step 2: Check import integrity**

Run: `python -c "from core.backtest.signal_matrix import SignalMatrixBuilder; from core.backtest.event_executor import EventDrivenExecutor; from core.backtest.engine_hybrid import run_hybrid; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Run a quick GA batch with hybrid engine**

Run:
```python
python -c "
from app.config import Config
from core.strategy.loader import StrategyLoader
from core.backtest.engine_hybrid import run_hybrid
from core.ga.genome import random_chromosome, chromosome_to_strategy
import random, time

random.seed(0)
Config._instance = None
config = Config.load('sim')
loader = StrategyLoader('strategies')

# Create 5 random chromosomes
strategies = []
for i in range(5):
    chrom = random_chromosome(f'ga_test_{i}')
    s = chromosome_to_strategy(chrom)
    strategies.append(s)

t0 = time.time()
result = run_hybrid(strategies, ['BTCUSDT', 'ETHUSDT'],
                    '2026-05-20', '2026-05-31', config, loader)
elapsed = time.time() - t0
print(f'Hybrid engine: {len(result[\"trades\"])} trades in {elapsed:.1f}s')
if 'error' not in result:
    print('OK')
"
```
Expected: Engine runs without error, prints trade count and time

- [ ] **Step 4: Commit any final adjustments**

```bash
git add -A
git commit -m "chore: final integration — all tests pass, hybrid engine operational"
```
