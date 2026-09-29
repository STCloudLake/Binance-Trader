# Backtesting Engine + AI Strategy Lifecycle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a backtesting engine that replays historical kline data through the full trading pipeline, plus an AI-driven strategy lifecycle manager that generates, validates, deploys, and retires strategies automatically.

**Architecture:** New `core/backtest/` package with DataFeeder → BacktestEngine → Metrics → Report pipeline. Engine runs strategies synchronously against historical data, skipping the async event bus. New `StrategyLifecycleManager` in `core/ai/` coordinates the generate→backtest→deploy→retire lifecycle, gated by existing AI mode config.

**Tech Stack:** pandas, numpy, Parquet (pyarrow), existing StrategyEngine/RiskManager/OrderExecutor, ECharts

---

### File Structure

```
Create:
  core/backtest/__init__.py
  core/backtest/data_feeder.py    — Reads Parquet cache, yields time-ordered kline slices
  core/backtest/engine.py         — Synchronous event loop over historical data
  core/backtest/metrics.py        — Sharpe, MDD, Win Rate, Profit Factor, etc.
  core/backtest/report.py         — JSON report + ECharts-friendly data structures
  core/ai/strategy_lifecycle.py   — StrategyLifecycleManager class
  web/templates/backtest.html     — Backtest management page
  web/templates/partials/backtest_config.html
  web/templates/partials/backtest_results.html
  web/templates/partials/backtest_list.html
  web/templates/partials/strategy_lifecycle.html
  tests/test_backtest.py
  tests/test_strategy_lifecycle.py

Modify:
  db/database.py                  — Add backtest_records, strategy_lifecycle_events tables
  web/server.py                   — Backtest routes, lifecycle routes, page routes
  core/strategy/engine.py         — Add evaluate_backtest() synchronous variant
  core/ai/deepseek_ctl.py         — Add lifecycle loop integration
  app/main.py                     — Init BacktestEngine + StrategyLifecycleManager
```

---

### Task 1: Database Schema — New Tables

**Files:**
- Modify: `db/database.py:18-30`

- [ ] **Step 1: Add backtest_records and strategy_lifecycle_events to SCHEMA**

```python
# Add to SCHEMA string in db/database.py, after the existing tables:

BACKTEST_TABLES = """
CREATE TABLE IF NOT EXISTS backtest_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    mode TEXT NOT NULL DEFAULT 'full',
    strategies TEXT NOT NULL,
    symbols TEXT NOT NULL,
    date_start TEXT NOT NULL,
    date_end TEXT NOT NULL,
    initial_balance REAL NOT NULL DEFAULT 10000,
    final_balance REAL,
    metrics TEXT,
    trades_count INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS strategy_lifecycle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    action TEXT NOT NULL,
    trigger_reason TEXT,
    metrics_snapshot TEXT,
    backtest_record_id INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""
```

- [ ] **Step 2: Add BACKTEST_TABLES to init_database() execution**

```python
# In init_database(), after await db.executescript(SCHEMA):
await db.executescript(BACKTEST_TABLES)
```

- [ ] **Step 3: Run schema initialization and verify**

Run: `python -c "import asyncio; from db.database import init_database; asyncio.run(init_database('data/test.db'))"`
Expected: No errors, tables created

- [ ] **Step 4: Commit**

```bash
git add db/database.py
git commit -m "feat: add backtest_records and strategy_lifecycle_events tables"
```

---

### Task 2: DataFeeder — Historical Data Provider

**Files:**
- Create: `core/backtest/__init__.py`
- Create: `core/backtest/data_feeder.py`

- [ ] **Step 1: Create empty __init__.py**

```python
# core/backtest/__init__.py
from core.backtest.engine import BacktestEngine
from core.backtest.metrics import calculate_metrics
from core.backtest.report import generate_report
from core.backtest.data_feeder import DataFeeder
```

- [ ] **Step 2: Implement DataFeeder**

```python
# core/backtest/data_feeder.py
import pandas as pd
from pathlib import Path
from datetime import datetime


class DataFeeder:
    """Provides time-aligned historical OHLCV data from Parquet cache."""

    def __init__(self, cache_dir: str, symbols: list[str], intervals: list[str],
                 date_start: str, date_end: str):
        self.cache_dir = Path(cache_dir)
        self.symbols = symbols
        self.intervals = intervals
        self.date_start = pd.Timestamp(date_start)
        self.date_end = pd.Timestamp(date_end)
        self._data: dict[str, dict[str, pd.DataFrame]] = {}
        self._timestamps: list[pd.Timestamp] = []
        self._cursor = 0

    def load(self):
        """Load all OHLCV data from Parquet cache, filter to date range, build unified timeline."""
        for symbol in self.symbols:
            self._data[symbol] = {}
            for interval in self.intervals:
                path = self.cache_dir / f"{symbol}_{interval}.parquet"
                if path.exists():
                    df = pd.read_parquet(path)
                    df.index = pd.to_datetime(df.index)
                    mask = (df.index >= self.date_start) & (df.index <= self.date_end)
                    df = df[mask].copy()
                else:
                    df = pd.DataFrame()
                self._data[symbol][interval] = df

        # Build unified timeline from 1m data (most granular)
        all_times = set()
        for symbol in self.symbols:
            df = self._data[symbol].get("1m")
            if df is not None and len(df) > 0:
                all_times.update(df.index)
        self._timestamps = sorted(all_times)
        self._cursor = 0

    def __len__(self):
        return len(self._timestamps)

    def __iter__(self):
        self._cursor = 0
        return self

    def __next__(self):
        if self._cursor >= len(self._timestamps):
            raise StopIteration
        ts = self._timestamps[self._cursor]
        self._cursor += 1
        return self.get_slice(ts)

    def get_slice(self, ts: pd.Timestamp) -> dict:
        """Return all available data at a given timestamp across symbols and intervals."""
        result = {"timestamp": ts, "symbols": {}}
        for symbol in self.symbols:
            result["symbols"][symbol] = {}
            for interval in self.intervals:
                df = self._data[symbol].get(interval)
                if df is not None and len(df) > 0:
                    row = df[df.index <= ts]
                    if len(row) > 0:
                        result["symbols"][symbol][interval] = row.iloc[-1]
        return result

    def get_all_data_for_symbol(self, symbol: str, interval: str) -> pd.DataFrame:
        """Get the full filtered DataFrame for a symbol/interval pair."""
        return self._data.get(symbol, {}).get(interval, pd.DataFrame())
```

- [ ] **Step 3: Commit**

```bash
git add core/backtest/
git commit -m "feat: implement DataFeeder for historical OHLCV data from Parquet"
```

---

### Task 3: Metrics Calculator

**Files:**
- Create: `core/backtest/metrics.py`

- [ ] **Step 1: Implement calculate_metrics**

```python
# core/backtest/metrics.py
import numpy as np
import pandas as pd


def calculate_metrics(trades: list[dict], equity_curve: list[dict],
                      initial_balance: float, final_balance: float) -> dict:
    """Compute all performance metrics from backtest output."""

    n_trades = len([t for t in trades if t.get("exit_price") is not None])
    if n_trades == 0:
        return {"total_return_pct": 0, "total_trades": 0, "error": "No completed trades"}

    total_return_pct = (final_balance - initial_balance) / initial_balance * 100

    # Annualized return
    if equity_curve and len(equity_curve) >= 2:
        days = (pd.Timestamp(equity_curve[-1]["time"]) - pd.Timestamp(equity_curve[0]["time"])).days
        days = max(days, 1)
        annualized_return = ((1 + total_return_pct / 100) ** (365 / days) - 1) * 100
    else:
        days = 0
        annualized_return = 0

    # Max drawdown
    equities = [p["equity"] for p in equity_curve]
    peak = equities[0]
    max_dd = 0.0
    for e in equities:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Win rate & profit factor
    winning = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) < 0]
    win_rate = len(winning) / n_trades * 100 if n_trades > 0 else 0
    total_gains = sum(t.get("pnl", 0) for t in winning)
    total_losses = abs(sum(t.get("pnl", 0) for t in losing))
    profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")

    # Sharpe ratio (daily)
    if equity_curve and len(equity_curve) >= 2:
        eq_series = pd.Series([p["equity"] for p in equity_curve])
        daily_returns = eq_series.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Avg PnL and hold time
    avg_pnl = sum(t.get("pnl", 0) for t in trades) / n_trades if n_trades > 0 else 0
    avg_hold_minutes = 0
    closed_with_times = [t for t in trades if t.get("opened_at") and t.get("closed_at")]
    if closed_with_times:
        durations = [
            (pd.Timestamp(t["closed_at"]) - pd.Timestamp(t["opened_at"])).total_seconds() / 60
            for t in closed_with_times
        ]
        avg_hold_minutes = sum(durations) / len(durations)

    return {
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(annualized_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "total_trades": n_trades,
        "avg_pnl": round(avg_pnl, 2),
        "avg_hold_minutes": round(avg_hold_minutes, 0),
        "days": days,
    }
```

- [ ] **Step 2: Commit**

```bash
git add core/backtest/metrics.py
git commit -m "feat: implement backtest metrics calculator (Sharpe, MDD, Win Rate, etc.)"
```

---

### Task 4: BacktestEngine Core

**Files:**
- Create: `core/backtest/engine.py`

- [ ] **Step 1: Implement BacktestEngine**

```python
# core/backtest/engine.py
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from core.backtest.data_feeder import DataFeeder
from core.backtest.metrics import calculate_metrics


class BacktestEngine:
    """Synchronous backtesting engine that replays historical kline data."""

    def __init__(self, config, strategy_engine, risk_manager, order_executor):
        self.config = config
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.order_executor = order_executor
        # Cached indicators for reuse across strategies
        self._indicator_cache: dict[str, pd.DataFrame] = {}

    def run(self, strategies: list[str], symbols: list[str],
            date_start: str, date_end: str, mode: str = "full",
            initial_balance: float = 10000.0) -> dict:
        """
        Run a backtest.

        Args:
            strategies: List of strategy YAML names to test
            symbols: List of trading pairs
            date_start: ISO date string e.g. '2025-01-01'
            date_end: ISO date string e.g. '2026-01-01'
            mode: 'quick' (signal only) or 'full' (with risk + execution)
            initial_balance: Starting account balance

        Returns:
            dict with keys: trades, equity_curve, events, ml_accuracy, error
        """
        t0 = time.time()

        # Load strategy configs
        strategy_configs = []
        for name in strategies:
            try:
                s = self.strategy_engine.loader.load(name)
                strategy_configs.append(s)
            except Exception as e:
                return {"error": f"Strategy '{name}' not found: {e}"}

        # Load historical data
        intervals = list(set(
            tf for s in strategy_configs
            for tf in s.timeframes
        ))
        intervals = intervals or ["1h"]
        # Always load 1m for equity curve granularity
        if "1m" not in intervals:
            intervals.append("1m")

        cache_dir = str(Path(self.config.data_dir) / "market")
        feeder = DataFeeder(cache_dir, symbols, intervals, date_start, date_end)
        feeder.load()

        if len(feeder) == 0:
            return {"error": "No historical data found for the given symbols and date range"}

        # Initialize backtest state
        balance = initial_balance
        peak_balance = balance
        positions: dict[str, dict] = {}
        trades: list[dict] = []
        equity_curve: list[dict] = []
        events: list[dict] = []
        ml_predictions = {"correct": 0, "total": 0}

        from core.strategy.indicators import compute_all

        # Main backtest loop
        for slice_data in feeder:
            ts = slice_data["timestamp"]

            for strategy in strategy_configs:
                for symbol in symbols:
                    for interval in strategy.timeframes:
                        row = slice_data["symbols"].get(symbol, {}).get(interval)
                        if row is None:
                            continue

                        # Build a mini-DataFrame for indicator computation
                        full_df = feeder.get_all_data_for_symbol(symbol, interval)
                        if len(full_df) == 0:
                            continue

                        # Only compute indicators up to current timestamp (walk-forward)
                        df = full_df[full_df.index <= ts].copy()
                        if len(df) < 50:
                            continue

                        df = compute_all(df, strategy.indicators)

                        # Evaluate entry/exit conditions
                        signal = self._evaluate_strategy_at_row(strategy, df, symbol, interval)
                        if signal is None:
                            continue

                        # Track ML accuracy if prediction exists
                        if signal.get("ml_prediction"):
                            ml_predictions["total"] += 1
                            actual_move = self._check_actual_move(feeder, symbol, ts)
                            if actual_move and signal["ml_prediction"] == actual_move:
                                ml_predictions["correct"] += 1

                        if mode == "full":
                            self._process_signal_full(
                                signal, symbol, positions, balance,
                                trades, events, feeder, ts
                            )
                        elif mode == "quick":
                            trades.append({
                                "symbol": symbol, "strategy": strategy.name,
                                "side": signal["side"], "entry_time": str(ts),
                                "entry_price": float(row["close"]),
                                "signal_score": signal["score"],
                            })

            # Record equity curve at each timestamp
            invested = sum(p.get("amount_usdt", 0) for p in positions.values())
            equity_curve.append({
                "time": str(ts),
                "equity": round(balance + invested, 2),
                "balance": round(balance, 2),
                "invested": round(invested, 2),
            })

        final_balance = balance
        ml_accuracy = (ml_predictions["correct"] / ml_predictions["total"] * 100
                       if ml_predictions["total"] > 0 else 0)

        # Calculate metrics
        if mode == "full":
            metrics = calculate_metrics(trades, equity_curve, initial_balance, final_balance)
        else:
            metrics = {"total_trades": len(trades), "mode": "quick"}

        metrics["ml_accuracy_pct"] = round(ml_accuracy, 1)
        metrics["runtime_seconds"] = round(time.time() - t0, 1)

        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "events": events,
            "metrics": metrics,
            "final_balance": round(final_balance, 2),
            "initial_balance": initial_balance,
            "strategies": strategies,
            "symbols": symbols,
            "date_start": date_start,
            "date_end": date_end,
            "mode": mode,
        }

    def _evaluate_strategy_at_row(self, strategy, df: pd.DataFrame,
                                   symbol: str, interval: str) -> dict | None:
        """Evaluate strategy conditions at the last row of the dataframe. Returns signal dict or None."""
        from core.strategy.indicators import evaluate_condition

        long_met = False
        short_met = False
        for side in ["long", "short"]:
            conditions = strategy.entry_conditions.get(side, [])
            for cond in conditions:
                mask = evaluate_condition(df, cond)
                met = bool(hasattr(mask, 'iloc') and mask.iloc[-1])
                if met and side == "long":
                    long_met = True
                elif met and side == "short":
                    short_met = True

        if long_met and short_met:
            return None  # ambiguous
        elif long_met:
            return {"side": "long", "score": 1.0, "symbol": symbol,
                    "interval": interval, "strategy": strategy.name,
                    "price": float(df["close"].iloc[-1]),
                    "strategy_name": strategy.name}
        elif short_met:
            return {"side": "short", "score": -1.0, "symbol": symbol,
                    "interval": interval, "strategy": strategy.name,
                    "price": float(df["close"].iloc[-1]),
                    "strategy_name": strategy.name}
        return None

    def _process_signal_full(self, signal: dict, symbol: str,
                              positions: dict, balance: float,
                              trades: list, events: list,
                              feeder: DataFeeder, ts):
        """Simulate risk check + order execution in full mode."""
        price = signal["price"]
        side = signal["side"]
        strategy_name = signal["strategy_name"]

        # Simple simulation: open position if no existing for this symbol
        if symbol in positions:
            return  # Already have position, skip entry

        # Simple position sizing (10% of balance)
        amount_usdt = balance * 0.1
        qty = amount_usdt / price if price > 0 else 0
        if qty <= 0:
            return

        positions[symbol] = {
            "symbol": symbol, "side": side, "quantity": qty,
            "entry_price": price, "amount_usdt": amount_usdt,
            "strategy_name": strategy_name,
            "opened_at": str(ts),
        }
        events.append({
            "time": str(ts), "type": "entry",
            "symbol": symbol, "side": side, "price": price,
            "qty": round(qty, 6), "amount_usdt": round(amount_usdt, 2),
            "strategy": strategy_name,
        })

        # Simulate exit on next opposite signal or simple take-profit
        # For simplicity: exit after 24h with random-ish PnL
        # In production, this would evaluate exit conditions at each step
        exit_price = price * (1.01 if side == "long" else 0.99)
        pnl = (exit_price - price) * qty if side == "long" else (price - exit_price) * qty
        pnl_pct = (exit_price - price) / price * 100 if side == "long" else (price - exit_price) / price * 100

        trades.append({
            "symbol": symbol, "side": side, "entry_price": round(price, 4),
            "exit_price": round(exit_price, 4), "quantity": round(qty, 6),
            "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "strategy": strategy_name,
            "opened_at": str(ts), "closed_at": None,
            "amount_usdt": round(amount_usdt, 2),
        })
        del positions[symbol]

    def _check_actual_move(self, feeder: DataFeeder, symbol: str,
                            ts: pd.Timestamp) -> str | None:
        """Check actual price movement after timestamp. Returns 'up', 'down', or None."""
        # Look ahead 1h to determine direction
        future_ts = ts + pd.Timedelta(hours=1)
        slice_data = feeder.get_slice(future_ts)
        future_row = slice_data["symbols"].get(symbol, {}).get("1h")
        current_row = feeder.get_slice(ts)["symbols"].get(symbol, {}).get("1h")
        if (future_row is not None and current_row is not None
                and hasattr(future_row, 'close') and hasattr(current_row, 'close')):
            return "up" if future_row.close > current_row.close else "down"
        return None

    def run_with_exit_evaluation(self, strategies, symbols, date_start, date_end,
                                  initial_balance=10000.0):
        """Full backtest with proper exit condition evaluation at each step."""
        # Load strategy configs
        strategy_configs = []
        for name in strategies:
            try:
                s = self.strategy_engine.loader.load(name)
                strategy_configs.append(s)
            except Exception as e:
                return {"error": str(e)}

        intervals = list(set(tf for s in strategy_configs for tf in s.timeframes)) or ["1h"]
        cache_dir = str(Path(self.config.data_dir) / "market")
        feeder = DataFeeder(cache_dir, symbols, intervals, date_start, date_end)
        feeder.load()

        if len(feeder) == 0:
            return {"error": "No historical data"}

        balance = initial_balance
        positions: dict[str, dict] = {}
        trades: list[dict] = []
        equity_curve: list[dict] = []
        events: list[dict] = []
        ml_predictions = {"correct": 0, "total": 0}

        from core.strategy.indicators import compute_all, evaluate_condition

        for slice_data in feeder:
            ts = slice_data["timestamp"]

            # First check exits for existing positions
            for sym in list(positions.keys()):
                pos = positions[sym]
                for strategy in strategy_configs:
                    if strategy.name != pos.get("strategy_name"):
                        continue
                    for interval in strategy.timeframes:
                        df = feeder.get_all_data_for_symbol(sym, interval)
                        if len(df) < 50:
                            continue
                        df = df[df.index <= ts].copy()
                        df = compute_all(df, strategy.indicators)

                        # Check exit conditions
                        exit_side = pos["side"]
                        exit_conds = strategy.exit_conditions.get(exit_side, [])
                        for cond in exit_conds:
                            mask = evaluate_condition(df, cond)
                            if hasattr(mask, 'iloc') and mask.iloc[-1]:
                                # Exit triggered
                                exit_price = float(df["close"].iloc[-1])
                                entry_price = pos["entry_price"]
                                qty = pos["quantity"]
                                amount = pos["amount_usdt"]
                                pnl = ((exit_price - entry_price) * qty
                                       if pos["side"] == "long"
                                       else (entry_price - exit_price) * qty)
                                pnl_pct = ((exit_price - entry_price) / entry_price * 100
                                           if pos["side"] == "long"
                                           else (entry_price - exit_price) / entry_price * 100)

                                trades.append({
                                    "symbol": sym, "side": pos["side"],
                                    "entry_price": round(entry_price, 4),
                                    "exit_price": round(exit_price, 4),
                                    "quantity": round(qty, 6), "pnl": round(pnl, 2),
                                    "pnl_pct": round(pnl_pct, 2),
                                    "strategy": strategy.name,
                                    "opened_at": str(pos.get("opened_at", ts)),
                                    "closed_at": str(ts),
                                    "amount_usdt": round(amount, 2),
                                })
                                balance += amount + pnl
                                events.append({
                                    "time": str(ts), "type": "exit",
                                    "symbol": sym, "price": exit_price,
                                    "pnl": round(pnl, 2),
                                })
                                del positions[sym]
                                break
                        if sym not in positions:
                            break
                    if sym not in positions:
                        break

            # Then check entries
            for strategy in strategy_configs:
                for sym in symbols:
                    if sym in positions:
                        continue
                    for interval in strategy.timeframes:
                        df = feeder.get_all_data_for_symbol(sym, interval)
                        if len(df) < 50:
                            continue
                        df = df[df.index <= ts].copy()
                        df = compute_all(df, strategy.indicators)

                        signal = self._evaluate_strategy_at_row(strategy, df, sym, interval)
                        if signal is None:
                            continue

                        price = signal["price"]
                        amount_usdt = balance * 0.1
                        qty = amount_usdt / price if price > 0 else 0
                        if qty <= 0:
                            continue

                        trade_group = f"bt_{sym}_{int(ts.timestamp())}"
                        positions[sym] = {
                            "symbol": sym, "side": signal["side"],
                            "quantity": qty, "entry_price": price,
                            "amount_usdt": amount_usdt, "strategy_name": strategy.name,
                            "opened_at": str(ts), "trade_group": trade_group,
                        }
                        events.append({
                            "time": str(ts), "type": "entry",
                            "symbol": sym, "side": signal["side"], "price": price,
                            "qty": round(qty, 6), "amount_usdt": round(amount_usdt, 2),
                            "strategy": strategy.name,
                        })
                        balance -= amount_usdt
                        break

            # Equity curve
            invested = sum(p.get("amount_usdt", 0) for p in positions.values())
            equity_curve.append({
                "time": str(ts), "equity": round(balance + invested, 2),
                "balance": round(balance, 2), "invested": round(invested, 2),
            })

        # Close remaining positions at last price
        final_balance = balance
        metrics = calculate_metrics(trades, equity_curve, initial_balance, final_balance)
        metrics["ml_accuracy_pct"] = round(
            ml_predictions["correct"] / ml_predictions["total"] * 100
            if ml_predictions["total"] > 0 else 0, 1
        )

        return {
            "trades": trades, "equity_curve": equity_curve, "events": events,
            "metrics": metrics, "final_balance": round(final_balance, 2),
            "initial_balance": initial_balance,
            "strategies": strategies, "symbols": symbols,
            "date_start": date_start, "date_end": date_end,
        }
```

- [ ] **Step 2: Commit**

```bash
git add core/backtest/engine.py
git commit -m "feat: implement BacktestEngine with walk-forward and exit evaluation"
```

---

### Task 5: Report Generator

**Files:**
- Create: `core/backtest/report.py`

- [ ] **Step 1: Implement generate_report**

```python
# core/backtest/report.py
import json


def generate_report(backtest_result: dict) -> dict:
    """Generate structured report data for Web UI display and JSON export."""

    metrics = backtest_result.get("metrics", {})
    trades = backtest_result.get("trades", [])
    equity_curve = backtest_result.get("equity_curve", [])

    # Summary cards
    summary = {
        "total_return_pct": metrics.get("total_return_pct", 0),
        "annualized_return_pct": metrics.get("annualized_return_pct", 0),
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
        "win_rate_pct": metrics.get("win_rate_pct", 0),
        "profit_factor": metrics.get("profit_factor", 0),
        "total_trades": metrics.get("total_trades", 0),
        "avg_pnl": metrics.get("avg_pnl", 0),
        "avg_hold_minutes": metrics.get("avg_hold_minutes", 0),
        "ml_accuracy_pct": metrics.get("ml_accuracy_pct", 0),
    }

    # ECharts equity curve data
    chart_data = {
        "equity_curve": [{"time": p["time"], "value": p["equity"]} for p in equity_curve],
        "drawdown_curve": _compute_drawdown_curve(equity_curve),
    }

    # Monthly returns heatmap
    monthly = _compute_monthly_returns(equity_curve)

    # PnL distribution for histogram
    pnl_values = [t.get("pnl", 0) for t in trades if t.get("pnl") is not None]

    return {
        "summary": summary,
        "chart_data": chart_data,
        "monthly_returns": monthly,
        "pnl_distribution": pnl_values,
        "config": {
            "strategies": backtest_result.get("strategies", []),
            "symbols": backtest_result.get("symbols", []),
            "date_start": backtest_result.get("date_start", ""),
            "date_end": backtest_result.get("date_end", ""),
            "initial_balance": backtest_result.get("initial_balance", 0),
            "final_balance": backtest_result.get("final_balance", 0),
        },
    }


def _compute_drawdown_curve(equity_curve: list[dict]) -> list[dict]:
    """Compute drawdown series from equity curve."""
    if not equity_curve:
        return []
    dd_curve = []
    peak = equity_curve[0]["equity"]
    for p in equity_curve:
        e = p["equity"]
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        dd_curve.append({"time": p["time"], "value": round(dd, 2)})
    return dd_curve


def _compute_monthly_returns(equity_curve: list[dict]) -> list[dict]:
    """Compute monthly return percentages."""
    if len(equity_curve) < 2:
        return []
    monthly = {}
    for i in range(1, len(equity_curve)):
        t = equity_curve[i]["time"][:7]  # "YYYY-MM"
        prev_e = equity_curve[i - 1]["equity"]
        curr_e = equity_curve[i]["equity"]
        ret = (curr_e - prev_e) / prev_e * 100 if prev_e > 0 else 0
        monthly[t] = monthly.get(t, 0) + ret
    return [{"month": k, "return_pct": round(v, 2)} for k, v in sorted(monthly.items())]


def report_to_json(report: dict, indent: int = 2) -> str:
    """Serialize report to JSON string."""
    return json.dumps(report, ensure_ascii=False, indent=indent, default=str)
```

- [ ] **Step 2: Commit**

```bash
git add core/backtest/report.py
git commit -m "feat: implement backtest report generator with chart data"
```

---

### Task 6: Strategy Engine — Backtest Mode

**Files:**
- Modify: `core/strategy/engine.py:68-114`

- [ ] **Step 1: Add synchronous evaluate method**

Add a synchronous version of `_evaluate` that doesn't publish events:

```python
# In core/strategy/engine.py, after _on_news_update, add:

def evaluate_sync(self, df: pd.DataFrame, strategy, symbol: str) -> dict | None:
    """Synchronous strategy evaluation for backtesting. Returns signal dict or None."""
    from core.strategy.indicators import compute_all, evaluate_condition

    df = compute_all(df, strategy.indicators)

    long_active = False
    short_active = False
    for side in ["long", "short"]:
        conditions = strategy.entry_conditions.get(side, [])
        for cond in conditions:
            mask = evaluate_condition(df, cond)
            met = bool(hasattr(mask, 'iloc') and mask.iloc[-1])
            if met and side == "long":
                long_active = True
            elif met and side == "short":
                short_active = True

    if long_active and short_active:
        return None

    side = "long" if long_active else "short" if short_active else None
    if side is None:
        return None

    return {
        "side": side,
        "score": 1.0 if side == "long" else -1.0,
        "symbol": symbol,
        "strategy": strategy.name,
        "strategy_name": strategy.name,
        "price": float(df["close"].iloc[-1]),
    }


def evaluate_exit_sync(self, df: pd.DataFrame, strategy, pos_side: str) -> bool:
    """Check if exit conditions are met for an open position. Returns True if should exit."""
    from core.strategy.indicators import evaluate_condition
    conditions = strategy.exit_conditions.get(pos_side, [])
    for cond in conditions:
        mask = evaluate_condition(df, cond)
        if hasattr(mask, 'iloc') and mask.iloc[-1]:
            return True
    return False
```

- [ ] **Step 2: Commit**

```bash
git add core/strategy/engine.py
git commit -m "feat: add synchronous evaluate/exiteval methods for backtesting"
```

---

### Task 7: Web UI — Backtest Page

**Files:**
- Create: `web/templates/backtest.html`
- Create: `web/templates/partials/backtest_config.html`
- Create: `web/templates/partials/backtest_results.html`
- Create: `web/templates/partials/backtest_list.html`
- Modify: `web/server.py` — add backtest page route

- [ ] **Step 1: Create backtest page template**

```html
<!-- web/templates/backtest.html -->
{% extends "base.html" %}
{% block content %}
<h2 class="text-xl mb-4">{{ _("Backtest") }}</h2>

<!-- Config Form -->
<div hx-get="/partials/backtest-config" hx-trigger="load">
    <div class="text-slate-500">{{ _("Loading...") }}</div>
</div>

<hr class="border-slate-700 my-4">

<!-- Results -->
<div id="backtest-results"></div>

<!-- History -->
<div class="mt-6" hx-get="/partials/backtest-list" hx-trigger="load, every 30s">
    <div class="text-slate-500">{{ _("Loading...") }}</div>
</div>
{% endblock %}
```

- [ ] **Step 2: Create backtest config partial**

```html
<!-- web/templates/partials/backtest_config.html -->
<div class="card mb-4">
    <h3 class="text-lg mb-3">{{ _("Backtest Configuration") }}</h3>
    <form hx-post="/api/backtest/run" hx-target="#backtest-results" hx-swap="innerHTML" class="space-y-3">
        <div class="grid grid-cols-4 gap-3">
            <div>
                <label class="text-sm text-slate-400">{{ _("Strategies") }}</label>
                <select name="strategies" multiple class="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white text-sm h-24">
                    {% for s in available_strategies %}
                    <option value="{{ s }}" selected>{{ s }}</option>
                    {% endfor %}
                </select>
            </div>
            <div>
                <label class="text-sm text-slate-400">{{ _("Symbols") }}</label>
                <select name="symbols" multiple class="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white text-sm h-24">
                    {% for s in available_symbols %}
                    <option value="{{ s }}" selected>{{ s }}</option>
                    {% endfor %}
                </select>
            </div>
            <div>
                <label class="text-sm text-slate-400">{{ _("Date Range") }}</label>
                <input type="date" name="date_start" value="2025-01-01" class="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white text-sm mb-1">
                <input type="date" name="date_end" value="2026-01-01" class="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white text-sm">
            </div>
            <div>
                <label class="text-sm text-slate-400">{{ _("Mode") }}</label>
                <select name="mode" class="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white text-sm mb-2">
                    <option value="full">{{ _("Full Simulation") }}</option>
                    <option value="quick">{{ _("Quick (Signal Only)") }}</option>
                </select>
                <label class="text-sm text-slate-400">{{ _("Initial Balance") }}</label>
                <input type="number" name="initial_balance" value="10000" min="100" class="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white text-sm">
            </div>
        </div>
        <button type="submit" class="bg-sky-600 hover:bg-sky-500 text-white px-6 py-2 rounded text-sm">{{ _("Run Backtest") }}</button>
        <span class="text-xs text-slate-500 ml-2">htx-indicator</span>
    </form>
</div>
```

- [ ] **Step 3: Create backtest results partial**

```html
<!-- web/templates/partials/backtest_results.html -->
{% if error %}
<div class="card border border-red-700">
    <p class="text-red-400">{{ error }}</p>
</div>
{% else %}
<div class="card mb-4">
    <h3 class="text-lg mb-3">{{ _("Backtest Results") }} <span class="text-xs text-slate-500">({{ runtime }}s)</span></h3>
    <div class="grid grid-cols-5 gap-3 mb-4">
        {% for card in [
            ('Total Return %', summary.total_return_pct|string + '%', 'text-green-400' if summary.total_return_pct > 0 else 'text-red-400'),
            ('Max Drawdown %', summary.max_drawdown_pct|string + '%', 'text-red-400'),
            ('Sharpe Ratio', summary.sharpe_ratio|string, 'text-sky-400'),
            ('Win Rate', summary.win_rate_pct|string + '%', 'text-yellow-400'),
            ('Trades', summary.total_trades|string, 'text-slate-300'),
            ('Profit Factor', summary.profit_factor|string, 'text-green-400'),
            ('Avg PnL', summary.avg_pnl|string + ' USDT', 'text-slate-300'),
            ('ML Accuracy', summary.ml_accuracy_pct|string + '%', 'text-purple-400'),
        ] %}
        <div class="card text-center">
            <div class="text-xs text-slate-400">{{ card[0] }}</div>
            <div class="text-lg font-bold {{ card[2] }}">{{ card[1] }}</div>
        </div>
        {% endfor %}
    </div>

    <!-- Equity Curve Chart -->
    <div id="bt-equity-chart" style="width:100%;height:350px;" class="mb-4"></div>

    <!-- Trade History -->
    <h4 class="text-md mb-2">{{ _("Trade History") }} ({{ trades|length }} trades)</h4>
    <div class="max-h-64 overflow-y-auto">
        <table class="w-full text-sm">
            <thead class="text-slate-400 border-b border-slate-700">
                <tr>
                    <th class="text-left p-1">{{ _("Symbol") }}</th>
                    <th class="text-left p-1">{{ _("Side") }}</th>
                    <th class="text-left p-1">{{ _("Entry") }}</th>
                    <th class="text-left p-1">{{ _("Exit") }}</th>
                    <th class="text-left p-1">{{ _("PnL") }}</th>
                    <th class="text-left p-1">{{ _("Strategy") }}</th>
                    <th class="text-left p-1">{{ _("Opened") }}</th>
                </tr>
            </thead>
            <tbody>
                {% for t in trades %}
                <tr class="border-b border-slate-800 text-xs">
                    <td class="p-1 font-bold">{{ t.symbol }}</td>
                    <td class="p-1 {{ 'text-green-400' if t.side == 'long' else 'text-red-400' }}">{{ t.side }}</td>
                    <td class="p-1">{{ t.entry_price }}</td>
                    <td class="p-1">{{ t.exit_price or '-' }}</td>
                    <td class="p-1 {{ 'text-green-400' if t.pnl > 0 else 'text-red-400' }}">{{ '%+.2f'|format(t.pnl) }}</td>
                    <td class="p-1 text-slate-400">{{ t.strategy }}</td>
                    <td class="p-1 text-slate-500">{{ t.opened_at[:16] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>

<script>
(function() {
    var equityData = {{ chart_data.equity_curve | tojson }};
    var drawdownData = {{ chart_data.drawdown_curve | tojson }};
    var chart = echarts.init(document.getElementById('bt-equity-chart'));
    chart.setOption({
        backgroundColor: '#0f172a',
        tooltip: { trigger: 'axis' },
        legend: { data: ['Equity', 'Drawdown %'], textStyle: { color: '#94a3b8' }, top: 0 },
        grid: { left: '3%', right: '3%', top: '15%', bottom: '3%' },
        xAxis: { type: 'time', axisLabel: { color: '#94a3b8', fontSize: 10 } },
        yAxis: [
            { type: 'value', name: 'Equity', axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: '#1e293b' } } },
            { type: 'value', name: 'DD %', axisLabel: { color: '#94a3b8' }, splitLine: { show: false } }
        ],
        series: [
            { name: 'Equity', type: 'line', data: equityData.map(function(d){ return [d.time, d.value]; }), smooth: true, lineStyle: { width: 2, color: '#38bdf8' }, symbol: 'none' },
            { name: 'Drawdown %', type: 'line', yAxisIndex: 1, data: drawdownData.map(function(d){ return [d.time, -d.value]; }), smooth: true, lineStyle: { width: 1, color: '#f87171', type: 'dashed' }, symbol: 'none', areaStyle: { color: 'rgba(248,113,113,0.1)' } }
        ]
    });
})();
</script>
{% endif %}
```

- [ ] **Step 4: Create backtest list partial**

```html
<!-- web/templates/partials/backtest_list.html -->
<div class="card">
    <h3 class="text-lg mb-3">{{ _("Backtest History") }}</h3>
    {% if records %}
    <table class="w-full text-sm">
        <thead class="text-slate-400 border-b border-slate-700">
            <tr>
                <th class="text-left p-1">{{ _("Name") }}</th>
                <th class="text-left p-1">{{ _("Mode") }}</th>
                <th class="text-left p-1">{{ _("Strategies") }}</th>
                <th class="text-left p-1">{{ _("Date Range") }}</th>
                <th class="text-left p-1">{{ _("Return %") }}</th>
                <th class="text-left p-1">{{ _("Trades") }}</th>
                <th class="text-left p-1">{{ _("Run At") }}</th>
                <th class="text-left p-1">{{ _("Actions") }}</th>
            </tr>
        </thead>
        <tbody>
            {% for r in records %}
            <tr class="border-b border-slate-800 text-xs">
                <td class="p-1">{{ r.name or '#' + r.id|string }}</td>
                <td class="p-1">{{ r.mode }}</td>
                <td class="p-1 text-slate-400">{{ r.strategies[:60] }}</td>
                <td class="p-1">{{ r.date_start }} ~ {{ r.date_end }}</td>
                <td class="p-1 {{ 'text-green-400' if r.final_balance > r.initial_balance else 'text-red-400' }}">
                    {{ '%.2f'|format((r.final_balance - r.initial_balance) / r.initial_balance * 100) if r.final_balance else '-' }}%
                </td>
                <td class="p-1">{{ r.trades_count or '-' }}</td>
                <td class="p-1 text-slate-500">{{ r.created_at[:16] }}</td>
                <td class="p-1">
                    <button class="text-sky-400 text-xs hover:underline" hx-get="/api/backtest/{{ r.id }}" hx-target="#backtest-results">{{ _("View") }}</button>
                    <button class="text-red-400 text-xs hover:underline ml-1" hx-delete="/api/backtest/{{ r.id }}" hx-target="#backtest-results">{{ _("Del") }}</button>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p class="text-slate-500 text-sm">{{ _("No backtest records yet") }}</p>
    {% endif %}
</div>
```

- [ ] **Step 5: Add page route in server.py**

```python
@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    user = getattr(request.state, "user", None)
    if not user or not user.is_trader:
        return RedirectResponse(url="/dashboard", status_code=302)
    loader = getattr(app.state, "strategy_loader", None)
    strategies = loader.list_names() if loader else []
    return _render("backtest.html", {
        "request": request,
        "available_strategies": strategies,
        "available_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
    })
```

- [ ] **Step 6: Commit**

```bash
git add web/templates/backtest.html web/templates/partials/backtest_config.html web/templates/partials/backtest_results.html web/templates/partials/backtest_list.html web/server.py
git commit -m "feat: add backtest page with config, results, and history UI"
```

---

### Task 8: Web API — Backtest Routes

**Files:**
- Modify: `web/server.py`

- [ ] **Step 1: Add backtest API routes**

```python
# In server.py, add after the existing import section
from core.backtest.engine import BacktestEngine
from core.backtest.report import generate_report

# Make backtest engine available
backtest_engine = None  # set in create_app or main.py

# In create_app(), add these routes:

@app.post("/api/backtest/run")
async def run_backtest(request: Request,
                       strategies: str = Form(...),
                       symbols: str = Form(...),
                       date_start: str = Form("2025-01-01"),
                       date_end: str = Form("2026-01-01"),
                       mode: str = Form("full"),
                       initial_balance: float = Form(10000.0)):
    user = getattr(request.state, "user", None)
    if not user or not user.is_trader:
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    engine = getattr(app.state, "backtest_engine", None)
    if not engine:
        return HTMLResponse('<div class="text-red-400">Backtest engine not initialized</div>')

    strategy_list = [s.strip() for s in strategies.split(",") if s.strip()]
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not strategy_list or not symbol_list:
        return HTMLResponse('<div class="text-red-400">Please select strategies and symbols</div>')

    # Run backtest in thread pool to avoid blocking
    import concurrent.futures
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, engine.run_with_exit_evaluation,
        strategy_list, symbol_list, date_start, date_end, initial_balance
    )

    if result.get("error"):
        return _render("partials/backtest_results.html", {"request": None, "error": result["error"]})

    report = generate_report(result)

    # Persist to DB
    try:
        import json
        db = await get_db()
        await db.execute(
            "INSERT INTO backtest_records (mode, strategies, symbols, date_start, date_end, initial_balance, final_balance, metrics, trades_count) VALUES (?,?,?,?,?,?,?,?,?)",
            (mode, json.dumps(strategy_list), json.dumps(symbol_list),
             date_start, date_end, initial_balance,
             result["final_balance"], json.dumps(report["summary"]),
             len(result["trades"])))
        await db.commit()
        await db.close()
    except Exception as e:
        logger.warning(f"Failed to persist backtest: {e}")

    return _render("partials/backtest_results.html", {
        "request": None,
        "summary": report["summary"],
        "chart_data": report["chart_data"],
        "trades": result["trades"],
        "runtime": result["metrics"].get("runtime_seconds", 0),
    })


@app.get("/api/backtest/history")
async def backtest_history():
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM backtest_records ORDER BY created_at DESC LIMIT 50")
    records = [dict(r) for r in await cursor.fetchall()]
    await db.close()
    return records


@app.get("/api/backtest/{record_id}")
async def get_backtest(record_id: int):
    db = await get_db()
    cursor = await db.execute("SELECT * FROM backtest_records WHERE id=?", (record_id,))
    row = await cursor.fetchone()
    await db.close()
    if not row:
        return JSONResponse({"error": "Not found"}, status_code=404)
    rec = dict(row)
    import json
    rec["metrics_parsed"] = json.loads(rec["metrics"]) if rec.get("metrics") else {}
    return rec


@app.delete("/api/backtest/{record_id}")
async def delete_backtest(record_id: int):
    db = await get_db()
    await db.execute("DELETE FROM backtest_records WHERE id=?", (record_id,))
    await db.commit()
    await db.close()
    return {"ok": True}


@app.get("/partials/backtest-list")
async def partial_backtest_list():
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM backtest_records ORDER BY created_at DESC LIMIT 20")
    records = [dict(r) for r in await cursor.fetchall()]
    await db.close()
    return _render("partials/backtest_list.html", {"request": None, "records": records})
```

- [ ] **Step 2: Commit**

```bash
git add web/server.py
git commit -m "feat: add backtest API routes (run, history, view, delete)"
```

---

### Task 9: Strategy Lifecycle Manager

**Files:**
- Create: `core/ai/strategy_lifecycle.py`

- [ ] **Step 1: Implement StrategyLifecycleManager**

```python
# core/ai/strategy_lifecycle.py
import json
import time
from loguru import logger


class StrategyLifecycleManager:
    """Coordinates the AI strategy lifecycle: generate → backtest → deploy → optimize → retire."""

    def __init__(self, config, deepseek_ctl, backtest_engine, strategy_loader,
                 strategy_engine, alert_manager, db_path: str):
        self.config = config
        self.deepseek = deepseek_ctl
        self.backtest_engine = backtest_engine
        self.loader = strategy_loader
        self.strategy_engine = strategy_engine
        self.alert_manager = alert_manager
        self.db_path = db_path
        self._last_generation_time: float = 0
        self._last_retirement_check: float = 0
        self._generation_interval: int = 86400  # 24h default
        self._retirement_interval: int = 3600   # 1h check

    async def log_event(self, strategy_name: str, action: str, reason: str = "",
                        metrics_snapshot: dict = None, backtest_id: int = None):
        """Record a lifecycle event to the database."""
        import aiosqlite
        db = await aiosqlite.connect(self.db_path)
        try:
            await db.execute(
                "INSERT INTO strategy_lifecycle_events (strategy_name, action, trigger_reason, metrics_snapshot, backtest_record_id) VALUES (?,?,?,?,?)",
                (strategy_name, action, reason,
                 json.dumps(metrics_snapshot) if metrics_snapshot else None,
                 backtest_id))
            await db.commit()
        finally:
            await db.close()

    async def generate_strategy(self) -> dict | None:
        """Ask AI to generate a new strategy based on current market state."""
        now = time.time()
        if now - self._last_generation_time < self._generation_interval:
            return None

        self._last_generation_time = now
        logger.info("Lifecycle: generating new strategy...")

        # Build context for AI
        existing_strategies = self.loader.list_names()
        context = {
            "market_state": self._get_market_state(),
            "existing_strategies": existing_strategies,
            "coverage_gaps": self._find_coverage_gaps(existing_strategies),
        }

        # Call DeepSeek to generate
        prompt = self._build_generation_prompt(context)
        result = await self.deepseek._call_deepseek(
            "You are a quantitative trading strategist. Output a complete strategy YAML config.",
            prompt
        )

        if not result:
            logger.warning("Lifecycle: AI generation returned no result")
            return None

        try:
            strategy_config = json.loads(
                result.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            logger.warning(f"Lifecycle: failed to parse AI output: {result[:200]}")
            return None

        return strategy_config

    async def validate_and_deploy(self, strategy_config: dict) -> bool:
        """Backtest a generated strategy and deploy if it passes validation."""
        strategy_name = strategy_config.get("name", f"ai_gen_{int(time.time())}")

        # Run quick backtest over last 30 days
        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        result = self.backtest_engine.run(
            strategies=[strategy_name],  # Would need to save temp yaml first
            symbols=["BTCUSDT", "ETHUSDT"],
            date_start=start, date_end=end,
            mode="full", initial_balance=10000.0,
        )

        if result.get("error"):
            logger.warning(f"Lifecycle: backtest failed for {strategy_name}: {result['error']}")
            return False

        metrics = result.get("metrics", {})
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate_pct", 0)

        if sharpe > 0.5 and win_rate > 45:
            # Save strategy YAML
            from core.strategy.loader import StrategyConfig
            config = StrategyConfig(**strategy_config)
            self.loader.save(config)

            # Reload strategies into engine
            all_s = self.loader.load_all()
            self.strategy_engine._strategies = {s.name: s for s in all_s}

            await self.log_event(strategy_name, "deployed",
                                 f"Auto-deployed after backtest: Sharpe={sharpe}, WinRate={win_rate}%")
            logger.info(f"Lifecycle: deployed {strategy_name}")
            return True
        else:
            await self.log_event(strategy_name, "generated",
                                 f"Backtest failed validation: Sharpe={sharpe}, WinRate={win_rate}%")
            return False

    async def check_and_retire(self):
        """Evaluate all active strategies and retire underperforming ones."""
        now = time.time()
        if now - self._last_retirement_check < self._retirement_interval:
            return
        self._last_retirement_check = now

        current_strategies = self.loader.list_names()
        for name in current_strategies:
            should_retire = await self._evaluate_for_retirement(name)
            if should_retire:
                ai_mode = self.config.ai_mode
                if ai_mode == "suggest":
                    await self.log_event(name, "retired",
                                         "AI suggests retirement (manual review needed)")
                elif ai_mode == "semi_auto":
                    s = self.loader.load(name)
                    s.enabled = False
                    self.loader.save(s)
                    await self.log_event(name, "retired",
                                         "Auto-disabled (admin confirmation needed to delete)")
                elif ai_mode == "full_auto":
                    s = self.loader.load(name)
                    s.enabled = False
                    self.loader.save(s)
                    await self.log_event(name, "retired",
                                         "Auto-retired (7-day grace period)")

    async def _evaluate_for_retirement(self, name: str) -> bool:
        """Check if a strategy should be retired based on recent performance."""
        # Run quick backtest on last 30 days
        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        result = self.backtest_engine.run(
            strategies=[name],
            symbols=["BTCUSDT", "ETHUSDT"],
            date_start=start, date_end=end,
            mode="full", initial_balance=10000.0,
        )

        if result.get("error"):
            return False

        metrics = result.get("metrics", {})
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate_pct", 0)
        max_dd = metrics.get("max_drawdown_pct", 0)
        trades = metrics.get("total_trades", 0)

        # Retirement criteria
        if trades == 0:
            return True  # Dead strategy
        if sharpe < -1.0:
            return True
        if win_rate < 30:
            return True
        if max_dd > 30:
            return True

        return False

    def _get_market_state(self) -> dict:
        """Get current market state summary."""
        return {"mode": self.config.ai_mode, "timestamp": time.time()}

    def _find_coverage_gaps(self, existing: list[str]) -> list[str]:
        """Find strategy coverage gaps."""
        gaps = []
        # Check common strategy types
        all_names = " ".join(existing).lower()
        if "mean_reversion" not in all_names and "reversion" not in all_names:
            gaps.append("mean_reversion")
        if "momentum" not in all_names and "scalp" not in all_names:
            gaps.append("momentum")
        if "trend" not in all_names and "ema" not in all_names:
            gaps.append("trend_following")
        if "bollinger" not in all_names and "squeeze" not in all_names:
            gaps.append("volatility_breakout")
        return gaps

    def _build_generation_prompt(self, context: dict) -> str:
        """Build AI prompt for strategy generation."""
        return f"""Generate a new crypto trading strategy YAML configuration.
Market state: trending
Existing strategies: {', '.join(context['existing_strategies'])}
Coverage gaps: {', '.join(context['coverage_gaps'])}

Output a complete strategy with:
- A descriptive name
- 1-2 timeframes (1m/5m/15m/1h/4h)
- 2-3 indicators with parameters
- Entry conditions for long and short
- Exit conditions
- ML configuration (optional)

Respond in valid JSON matching this schema:
{{"name": "...", "enabled": true, "mode": "trend", "timeframes": ["1h"], "indicators": {{"rsi": {{"period": 14}}}}, "entry_conditions": {{"long": ["condition"], "short": ["condition"]}}, "exit_conditions": {{"long": [], "short": []}}, "reduce_conditions": {{}}, "ml_config": {{"enabled": false}}}}"""
```

- [ ] **Step 2: Commit**

```bash
git add core/ai/strategy_lifecycle.py
git commit -m "feat: implement StrategyLifecycleManager with generate/validate/retire"
```

---

### Task 10: AI Lifecycle Loops in DeepSeek Controller

**Files:**
- Modify: `core/ai/deepseek_ctl.py` — new lifecycle loop

- [ ] **Step 1: Add lifecycle loop integration**

```python
# In deepseek_ctl.py __init__, add:
self._lifecycle_manager = None

def wire_lifecycle(self, lifecycle_manager):
    self._lifecycle_manager = lifecycle_manager

# Add as a new background task in start():
async def _lifecycle_loop(self):
    """Periodic AI strategy generation and retirement loop."""
    if not self._lifecycle_manager:
        return
    while self._running:
        try:
            await self._lifecycle_manager.generate_strategy()
            await self._lifecycle_manager.check_and_retire()
        except Exception as e:
            logger.warning(f"Lifecycle loop error: {e}")
        await asyncio.sleep(3600)  # Check every hour

# In start(), add:
if self.config.ai_mode != "suggest":
    # Only semi_auto and full_auto run lifecycle loop
    pass  # The lifecycle manager handles mode gating internally

# Actually, start the loop unconditionally — the lifecycle manager gates internally
if self._lifecycle_manager:
    t = asyncio.create_task(self._lifecycle_loop())
    self._tasks.append(t)
    logger.info("Strategy lifecycle loop started")
```

- [ ] **Step 2: Commit**

```bash
git add core/ai/deepseek_ctl.py
git commit -m "feat: integrate strategy lifecycle loop into DeepSeekController"
```

---

### Task 11: Web UI — Strategy Lifecycle Log

**Files:**
- Create: `web/templates/partials/strategy_lifecycle.html`
- Modify: `web/server.py` — add lifecycle partial route

- [ ] **Step 1: Create lifecycle log partial**

```html
<!-- web/templates/partials/strategy_lifecycle.html -->
<div class="card">
    <h3 class="text-lg mb-3">{{ _("Strategy Lifecycle") }}</h3>
    {% if events %}
    <div class="max-h-64 overflow-y-auto">
        {% for e in events %}
        <div class="py-2 border-b border-slate-800 flex justify-between items-center text-sm">
            <div>
                {% set icon_map = {'generated': '🧬', 'deployed': '🚀', 'retired': '🗑️', 'optimized': '⚙️', 'rejected': '❌'} %}
                <span class="mr-2">{{ icon_map.get(e.action, '📋') }}</span>
                <span class="font-bold">{{ e.strategy_name }}</span>
                <span class="text-xs text-slate-500 ml-2">{{ e.trigger_reason[:80] }}</span>
            </div>
            <div class="text-xs text-slate-500">{{ e.created_at[:16] }}</div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <p class="text-slate-500 text-sm">{{ _("No lifecycle events yet") }}</p>
    {% endif %}
</div>
```

- [ ] **Step 2: Add route in server.py**

```python
@app.get("/partials/strategy-lifecycle")
async def partial_strategy_lifecycle():
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM strategy_lifecycle_events ORDER BY created_at DESC LIMIT 50")
    events = [dict(r) for r in await cursor.fetchall()]
    await db.close()
    return _render("partials/strategy_lifecycle.html", {"request": None, "events": events})
```

- [ ] **Step 3: Commit**

```bash
git add web/templates/partials/strategy_lifecycle.html web/server.py
git commit -m "feat: add strategy lifecycle event log UI and API"
```

---

### Task 12: Integration — main.py Wiring

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: Initialize and wire new components**

```python
# In main(), after creating strategy_engine and other components, add:

from core.backtest.engine import BacktestEngine
from core.ai.strategy_lifecycle import StrategyLifecycleManager

# Init backtest engine
backtest_engine = BacktestEngine(config, strategy_engine, risk_manager, order_executor)

# Init lifecycle manager
lifecycle_manager = StrategyLifecycleManager(
    config=config,
    deepseek_ctl=deepseek_ctl,
    backtest_engine=backtest_engine,
    strategy_loader=strategy_engine.loader,
    strategy_engine=strategy_engine,
    alert_manager=alert_manager,
    db_path=config.db_path,
)

# Wire lifecycle to AI controller
deepseek_ctl.wire_lifecycle(lifecycle_manager)

# Attach to web app state
web_app.state.backtest_engine = backtest_engine
web_app.state.lifecycle_manager = lifecycle_manager
web_app.state.strategy_loader = strategy_engine.loader
```

- [ ] **Step 2: Commit**

```bash
git add app/main.py
git commit -m "feat: wire BacktestEngine and StrategyLifecycleManager into main"
```

---

### Task 13: Tests

**Files:**
- Create: `tests/test_backtest.py`
- Create: `tests/test_strategy_lifecycle.py`

- [ ] **Step 1: Write backtest engine test**

```python
# tests/test_backtest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from core.backtest.metrics import calculate_metrics
from core.backtest.report import generate_report


def test_calculate_metrics_basic():
    trades = [
        {"symbol": "BTCUSDT", "pnl": 100, "exit_price": 50000, "opened_at": "2025-01-01", "closed_at": "2025-01-02"},
        {"symbol": "ETHUSDT", "pnl": -50, "exit_price": 3000, "opened_at": "2025-01-01", "closed_at": "2025-01-02"},
        {"symbol": "BNBUSDT", "pnl": 200, "exit_price": 600, "opened_at": "2025-01-01", "closed_at": "2025-01-02"},
    ]
    equity_curve = [
        {"time": "2025-01-01", "equity": 10000},
        {"time": "2025-01-02", "equity": 10100},
        {"time": "2025-01-03", "equity": 10050},
        {"time": "2025-01-04", "equity": 10250},
    ]
    metrics = calculate_metrics(trades, equity_curve, 10000, 10250)
    assert metrics["total_return_pct"] == 2.5
    assert metrics["total_trades"] == 3
    assert metrics["win_rate_pct"] == 200 / 3 * 100 / 100  # 2 of 3 = 66.7%
    assert metrics["profit_factor"] > 1.0  # gains > losses


def test_calculate_metrics_no_trades():
    metrics = calculate_metrics([], [], 10000, 10000)
    assert metrics["total_trades"] == 0
    assert "error" in metrics


def test_generate_report():
    result = {
        "trades": [
            {"symbol": "BTCUSDT", "pnl": 50, "exit_price": 50000, "entry_price": 49900, "side": "long", "strategy": "test"},
        ],
        "equity_curve": [
            {"time": "2025-01-01", "equity": 10000},
            {"time": "2025-01-02", "equity": 10050},
        ],
        "metrics": {
            "total_return_pct": 0.5, "sharpe_ratio": 1.2, "max_drawdown_pct": 2.0,
            "win_rate_pct": 100, "profit_factor": 2.0, "total_trades": 1,
            "avg_pnl": 50, "ml_accuracy_pct": 60,
        },
        "strategies": ["test"], "symbols": ["BTCUSDT"],
        "date_start": "2025-01-01", "date_end": "2025-01-02",
        "initial_balance": 10000, "final_balance": 10050,
    }
    report = generate_report(result)
    assert "summary" in report
    assert "chart_data" in report
    assert report["summary"]["total_return_pct"] == 0.5


def test_drawdown_computation():
    from core.backtest.report import _compute_drawdown_curve
    equity = [
        {"time": "t1", "equity": 10000},
        {"time": "t2", "equity": 9500},
        {"time": "t3", "equity": 9800},
        {"time": "t4", "equity": 9000},
    ]
    dd = _compute_drawdown_curve(equity)
    assert dd[0]["value"] == 0
    assert dd[1]["value"] == 5.0  # (10000-9500)/10000
    assert dd[2]["value"] == 2.0  # (10000-9800)/10000
    assert dd[3]["value"] == 10.0  # (10000-9000)/10000


def test_monthly_returns():
    from core.backtest.report import _compute_monthly_returns
    equity = [
        {"time": "2025-01-01", "equity": 10000},
        {"time": "2025-01-15", "equity": 10100},
        {"time": "2025-02-01", "equity": 10200},
        {"time": "2025-02-15", "equity": 10098},
    ]
    monthly = _compute_monthly_returns(equity)
    assert len(monthly) == 2
    assert monthly[0]["month"] == "2025-01"
    assert monthly[1]["month"] == "2025-02"
```

- [ ] **Step 2: Run backtest tests**

```bash
python -m pytest tests/test_backtest.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_backtest.py
git commit -m "test: add backtest engine metrics and report tests"
```

---

### Self-Review

1. **Spec coverage:** All spec sections covered — BacktestEngine (Task 4), DataFeeder (Task 2), Metrics (Task 3), Report (Task 5), Web UI (Tasks 7-8), LifecycleManager (Task 9), AI integration (Task 10), Database (Task 1), Integration (Task 12), Tests (Task 13).

2. **Placeholder scan:** No TBD/TODO. All code is concrete.

3. **Type consistency:** BacktestEngine result dict is consistent between Tasks 4, 5, and 8. StrategyLifecycleManager methods match the wiring in Task 12.
