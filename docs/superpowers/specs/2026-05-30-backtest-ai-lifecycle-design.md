# Backtesting Engine + AI Strategy Lifecycle Management — Design Spec

**Date:** 2026-05-30
**Status:** Approved

## 1. Overview

Add a backtesting engine that replays historical kline data through the full strategy→risk→execution pipeline, and an AI-driven strategy lifecycle manager that can generate, validate, deploy, optimize, and retire trading strategies.

The two systems are tightly coupled — backtesting is the validation mechanism for AI-generated strategies.

## 2. Architecture

### 2.1 New Modules

```
core/backtest/
├── __init__.py
├── engine.py          # BacktestEngine: event replay loop
├── metrics.py         # Performance metrics (Sharpe, MDD, Win Rate, etc.)
├── data_feeder.py     # Historical data feeder (reads from Parquet cache)
└── report.py          # Report generator (JSON + ECharts data)

core/ai/
└── strategy_lifecycle.py  # StrategyLifecycleManager: generate→test→deploy→retire loop

web/templates/
├── backtest.html              # Backtest management page
└── partials/
    ├── backtest_config.html   # HTMX partial: config form
    ├── backtest_results.html  # HTMX partial: metric cards + charts
    ├── backtest_list.html     # HTMX partial: historical backtest records
    └── strategy_lifecycle.html # HTMX partial: AI lifecycle event log

db/database.py                  # New tables: backtest_records, strategy_lifecycle_events
```

### 2.2 Modified Modules

| Module | Changes |
|--------|---------|
| `core/strategy/engine.py` | Add backtest mode — skip async event publish, return results directly |
| `core/risk/manager.py` | Backtest-compatible check_signal (no live balance dependency) |
| `core/ai/deepseek_ctl.py` | New loops: strategy generation loop, strategy retirement loop |
| `web/server.py` | New routes: /backtest page, CRUD API, lifecycle events API |
| `app/main.py` | Init BacktestEngine + StrategyLifecycleManager |

### 2.3 Data Flow

```
[Parquet OHLCV Cache] → DataFeeder → BacktestEngine
                                         ↓
                              StrategyEngine (backtest mode)
                                         ↓
                              RiskManager (check_signal)
                                         ↓
                              OrderExecutor (sim, isolated balance)
                                         ↓
                              MetricsCalculator
                                         ↓
                              ReportGenerator → [Web UI] / [AI Decision]
```

## 3. BacktestEngine Core

### 3.1 Input

- Strategy list (1-N)
- Symbol list (1-N)
- Date range (start/end)
- Mode: `"quick"` (signal-only) | `"full"` (with risk + execution)
- Initial balance (default: 10000 USDT)

### 3.2 Internal Loop

```
for each timestamp (sorted kline time across all symbols/intervals):
    1. DataFeeder pushes kline + ML prediction + news sentiment for this time
    2. StrategyEngine evaluates → produces signals
    3. (full mode) RiskManager audits → OrderExecutor sim-executes
    4. Record: signal snapshots, orders, balance changes, position snapshots
    5. (if breaker trips) Execute breaker action, log event
```

### 3.3 Key Differences from Live Trading

| Aspect | Live/Sim | Backtest |
|--------|----------|----------|
| Time control | asyncio (real-time) | Synchronous loop (fast-forward) |
| Data source | WebSocket + REST | Parquet cache via DataFeeder |
| Event handling | Async pub/sub via EventBus | Synchronous direct call |
| Balance | Shared DB with atomic lock | Isolated in-memory balance |
| ML prediction | Real-time from MLPredictor | Historical: pre-computed or on-the-fly |

### 3.4 Output

- Complete trade records per strategy/symbol (entry/exit time, price, PnL)
- Equity curve time series (account equity at each kline close)
- Event log (signals, rejections, breaker trips)
- ML model prediction accuracy during the backtest period

## 4. Performance Metrics

All metrics computed per-strategy and per-combined-portfolio:

### 4.1 Standard Metrics

| Metric | Formula / Description |
|--------|-----------------------|
| Total Return % | `(final_equity - initial) / initial × 100` |
| Annualized Return | `(1 + total_return)^(365/days) - 1` |
| Max Drawdown (MDD) | `max(peak - trough) / peak × 100` over equity curve |
| Sharpe Ratio | `(mean(daily_return) - risk_free) / std(daily_return)` |
| Win Rate | `winning_trades / total_closed_trades` |
| Profit Factor | `sum(gains) / abs(sum(losses))` |
| Total Trades | Count of all closed trades |
| Avg Hold Time | Mean duration between entry and exit |
| Avg PnL per Trade | Mean PnL across all closed trades |

### 4.2 Visualization Data (for ECharts)

- **Equity Curve**: `[{time, equity}]` — line chart with drawdown overlay
- **Monthly Returns Heatmap**: `[{year, month, return_pct}]`
- **PnL Distribution**: histogram of trade PnL values
- **Strategy Comparison**: side-by-side equity curves

### 4.3 ML Model Accuracy

- Per-strategy: `ml_correct_predictions / ml_total_predictions` during the backtest period
- Direction accuracy: how often the ML `up`/`down` prediction matched actual price movement
- Displayed alongside strategy metrics for AI model improvement feedback

## 5. DataFeeder

### 5.1 Responsibility

Provide historical OHLCV data to the backtesting engine, simulating the same data pipeline that `MarketDataProvider` provides in live trading, but reading from the Parquet cache instead of WebSocket.

### 5.2 Data Loading

```
1. Accept: symbol list + interval list + date range
2. For each symbol/interval pair:
   a. Read from Parquet cache (data/market/{symbol}_{interval}.parquet)
   b. If missing, fetch from Binance REST API and cache
   c. Filter to date range
   d. Align all intervals to a unified timeline
3. Yield: per-timestamp data frames with OHLCV + indicators pre-computed
```

### 5.3 ML Prediction Simulation

During backtest, ML predictions are either:
- Replayed from pre-computed values (if model existed at that time)
- Computed on-the-fly using a "walk-forward" approach (train up to time T, predict T+1)

The walk-forward approach is preferred for accuracy evaluation.

## 6. Web UI

### 6.1 Backtest Page (`/backtest`)

**Config Panel** (left / top):
- Strategy selector (multi-select with "select all")
- Symbol selector (multi-select)
- Date range picker (start / end)
- Mode toggle: Quick / Full
- Initial balance input
- "Run Backtest" button

**Results Panel** (center / bottom):
- Metric cards grid (Total Return, Sharpe, MDD, Win Rate, Trades)
- ECharts equity curve with drawdown overlay
- Trade history table (sortable, filterable)
- Strategy comparison table (when multiple strategies selected)
- ML accuracy display per strategy

**History Panel** (tab or sidebar):
- List of previous backtest runs
- Click to reload previous results
- Delete old reports

### 6.2 Strategy Lifecycle Log

Displayed in the AI Panel or a dedicated section:
- Chronological event log: generated → backtested → deployed → retired
- Each entry shows: strategy name, action icon, trigger reason, metrics summary, timestamp
- Filter by action type or strategy name

## 7. StrategyLifecycleManager

### 7.1 AI Strategy Generation

**Trigger**: Periodic (default: every 24h) or on-demand via Web UI button.

**AI Prompt Input**:
- Current market state assessment (trending/ranging/volatile)
- Existing strategy portfolio with recent performance
- Strategy coverage gaps: missing types, timeframes, symbols
- Recent ML model feedback: which features are most predictive

**AI Output**: A complete `StrategyConfig` (indicators, conditions, timeframes, ML config) + rationale.

**Automated Backtest**:
1. Generated strategy → run backtest over last 30 days (quick mode)
2. If Sharpe > 0.5 and Win Rate > 45% → passes validation
3. (semi_auto/full_auto) Deploy to live strategy list
4. (suggest) Save as AI suggestion with backtest report attached

**Initial Risk Constraints** (full_auto only):
- New strategy starts on 1h timeframe only
- Satellite position type only
- Max position size = 1/3 of normal
- 24h observation period before full deployment

### 7.2 Strategy Optimization

**Trigger**: Periodic (default: every 24h) or on underperformance.

**Process**:
1. Identify underperforming strategies (Sharpe < 0 or Win Rate < 40%)
2. AI proposes parameter adjustments (indicator periods, thresholds, timeframes)
3. Backtest the adjusted version vs original over the same period
4. If adjusted version outperforms → apply changes

### 7.3 Strategy Retirement

**Retirement Criteria** (configurable):
- Consecutive 7 days with Sharpe < 0
- Win Rate < 30% over last 30 days
- Max Drawdown exceeds 2× the strategy's initial backtest MDD
- Zero signals generated in 48h (dead strategy)
- A new strategy of the same type significantly outperforms it (backtest comparison)

**Retirement Process by Mode**:
| Mode | Action |
|------|--------|
| `suggest` | AI marks as "retirement suggested" → appears in AI suggestions → manual review |
| `semi_auto` | AI disables strategy (can be re-enabled) → logs reason → manual confirmation needed to delete |
| `full_auto` | AI disables immediately → 7-day grace period (kept in DB, restorable) → auto-delete after grace |

## 8. Database Schema

### 8.1 New Tables

```sql
-- Backtest records
CREATE TABLE backtest_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,                           -- user-given name or auto-generated
    mode TEXT NOT NULL DEFAULT 'full',   -- 'quick' or 'full'
    strategies TEXT NOT NULL,            -- JSON array of strategy names
    symbols TEXT NOT NULL,               -- JSON array of symbols
    date_start TEXT NOT NULL,
    date_end TEXT NOT NULL,
    initial_balance REAL NOT NULL DEFAULT 10000,
    final_balance REAL,
    metrics TEXT,                        -- JSON: all computed metrics
    trades_count INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Strategy lifecycle events
CREATE TABLE strategy_lifecycle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    action TEXT NOT NULL,                -- 'generated' | 'deployed' | 'optimized' | 'retired' | 'rejected'
    trigger_reason TEXT,                 -- AI rationale or user action
    metrics_snapshot TEXT,               -- JSON: key metrics at time of action
    backtest_record_id INTEGER,          -- FK to backtest_records (for generated/deployed actions)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

## 9. API Endpoints

### 9.1 Backtest Routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/backtest` | Backtest page (HTML) |
| POST | `/api/backtest/run` | Start a backtest run (Form: strategies, symbols, date_start, date_end, mode, initial_balance) |
| GET | `/api/backtest/{id}` | Get backtest result (JSON with metrics + trade list) |
| GET | `/api/backtest/history` | List historical backtest runs |
| DELETE | `/api/backtest/{id}` | Delete a backtest record |
| POST | `/api/backtest/compare` | Run comparison backtest across multiple strategies |
| GET | `/partials/backtest-results/{id}` | HTMX partial: result display |

### 9.2 Lifecycle Routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/strategy-lifecycle/events` | List lifecycle events (filterable) |
| POST | `/api/strategy-lifecycle/generate` | Trigger AI strategy generation (trader+) |
| POST | `/api/strategy-lifecycle/retire/{name}` | Manually retire a strategy (trader+) |
| POST | `/api/strategy-lifecycle/restore/{name}` | Restore a retired strategy within grace period |
| GET | `/partials/strategy-lifecycle` | HTMX partial: event log |

## 10. Integration with Existing AI Modes

The lifecycle manager behavior is gated by the existing `ai.mode` config:

```
ai.mode = "suggest":
  - strategy_lifecycle_loop runs, but only generates suggestions
  - No auto-deploy, no auto-retire
  - All actions go to ai_suggestions table for manual review

ai.mode = "semi_auto":
  - Strategy generation + backtest → auto-deploy if passes validation
  - Strategy retirement → auto-disable, require manual delete confirmation
  - Strategy optimization → auto-apply parameter changes

ai.mode = "full_auto":
  - Full autonomy: generate → validate → deploy → observe → optimize → retire
  - New strategies start with restricted risk, graduate after observation period
  - 7-day grace period before permanent deletion
```

## 11. Self-Review Checklist

- [x] No TBD/TODO placeholders
- [x] Architecture matches feature scope
- [x] API endpoints are concrete with HTTP methods
- [x] Database schema has column types
- [x] AI mode gating is consistent with existing system
- [x] Backtest engine design accounts for both quick and full modes
- [x] ML accuracy tracking integrated
- [x] Strategy lifecycle covers full generate→deploy→optimize→retire loop
