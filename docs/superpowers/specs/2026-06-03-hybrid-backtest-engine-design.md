# Hybrid Backtest Engine — Design Spec

> **Status:** Draft → User Review
> **Date:** 2026-06-03
> **Branch:** `feature/hybrid-backtest`

## Goal

Replace the current single-pass tick-by-tick backtest engine with a two-phase hybrid
architecture that delivers **10–20× speedup** for GA batch evaluation while
producing **provably equivalent** results (trade-for-trade, signal-for-signal)
to the current engine.

## Motivation

- GA evolution (80 individuals × 30 generations = 2400 backtests) currently takes
  ~60 hours on 1 year of data. Target: ~3–6 hours.
- The current engine's bottleneck is per-tick re-computation of indicators and
  condition evaluation for every strategy. With 30 strategies, this repeats
  30× the same work per timestamp.
- GA needs ranking consistency, not perfect PnL fidelity. But we set a harder
  bar: **provable trade-level equivalence** on the no-ML path.

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Layer 1: SignalMatrixBuilder (vectorized, once) │
│  ┌────────────────────────────────────────────┐  │
│  │ Input: N × StrategyConfig                   │  │
│  │ 1. Cluster unique indicator configs         │  │
│  │ 2. Batch compute_all() per unique group     │  │
│  │ 3. Vectorized evaluate_condition() → matrix │  │
│  │ Output: SignalMatrix (immutable)            │  │
│  └────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────┤
│  Layer 2: EventDrivenExecutor (tick loop)        │
│  ┌────────────────────────────────────────────┐  │
│  │ Input: SignalMatrix + price data            │  │
│  │ Per timestamp:                              │  │
│  │   → Check SL/TP/reduce/exit (matrix lookup) │  │
│  │   → Check entry signals (matrix lookup)     │  │
│  │   → Position sizing (reuse PositionSizer)   │  │
│  │   → Balance/position update                 │  │
│  │ Output: trades[], equity_curve[], per_matrix│  │
│  └────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────┤
│  Layer 3: MetricsCalculator (unchanged)           │
│  ┌────────────────────────────────────────────┐  │
│  │ Reuses existing calculate_metrics()         │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

## File Structure

```
core/backtest/
├── engine.py              # Retained, slimmed to router + legacy engine
├── engine_hybrid.py       # NEW: _run_hybrid entry point
├── signal_matrix.py       # NEW: SignalMatrixBuilder + IndicatorGrouper
├── event_executor.py      # NEW: EventDrivenExecutor
├── data_feeder.py         # Unchanged
├── metrics.py             # Unchanged
└── report.py              # Unchanged
```

Each new file ~200–300 lines. `engine.py` shrinks from ~850 to ~350 lines
(router + legacy engine + shared `_close_position`).

## Configuration

Two explicit config items added to `config/config.yaml` and the `Config` model:

```yaml
# config/config.yaml
backtest:
  engine_mode: "auto"       # "auto" | "hybrid" | "legacy"
  ml_enabled: false          # whether to train/use ML models during backtest
```

### `engine_mode`

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | N ≥ 3 strategies → hybrid; else legacy. ML-enabled strategies always route to legacy regardless of count. |
| `"hybrid"` | Force hybrid engine. If any strategy has `ml_config.enabled = True`, raise an explicit error telling the user to either disable ML or switch to legacy. |
| `"legacy"` | Force legacy engine. Works with all features including ML. |

### `ml_enabled`

| Value | Behavior |
|-------|----------|
| `false` (default) | All strategies' `ml_config.enabled` is overridden to `False` during backtest. ML models are not trained or loaded. No GPU memory consumed. 3–5× speedup on legacy engine alone. |
| `true` | Strategies respect their individual `ml_config.enabled` setting. ML training/prediction runs normally. |

**Rationale for defaults:** ML direction prediction accuracy is ~50% (theoretical ceiling
on OHLCV data). The cost (GPU time, training overhead) outweighs the benefit for
most backtest scenarios. Users explicitly opt in when they want to test ML integration.

**GA override:** GA fitness evaluation always sets `ml_enabled = False` internally
regardless of config — this is already implemented and unchanged.

## SignalMatrixBuilder Design

### Indicator Clustering

30 GA strategies typically collapse to 5–8 unique indicator configs:

```
Input: 30 StrategyConfig objects
    ↓
Hash each config's indicators dict → group by hash
    ↓
Group A: rsi(14) + macd(12,26,9)           → 18 strategies
Group B: rsi(7)  + macd(8,22,6)            →  7 strategies
Group C: rsi(21) + bb(20,2)                →  5 strategies
    ↓
compute_all() called 3 times instead of 30
```

### Condition Evaluation

Conditions are evaluated once per unique indicator group:

```
For each Group:
  1. Compute full indicator DataFrame once
  2. Collect all unique conditions across all strategies in the group
  3. pandas.eval() each condition → boolean mask
  4. Map masks back to individual strategies via condition→strategy lookup
  5. Combine entry conditions per strategy with logical AND
  6. Store final entry/exit signals in SignalMatrix
```

### Signal Matrix Structure

```
SignalMatrix.signals:
  MultiIndex: (strategy_name, symbol, timeframe)
  Columns:    timestamp index (datetime64)
  Values:     int8: 1 (long_entry), -1 (short_entry), 0 (no signal)

SignalMatrix.exit_signals:
  MultiIndex: (strategy_name, symbol, timeframe, "exit_long"|"exit_short")
  Columns:    timestamp index
  Values:     bool

Memory (1 year × 1m data):
  30 strategies × 5 symbols × 2 TFs = 300 rows
  526,000 columns
  dtype int8
  → ~158 MB per batch

  Mitigation: signal matrix is built per-symbol (sharding).
  Each shard: ~30 MB. Event loop processes one shard, then merges results.
```

### Anti-Lookahead Guarantee

All condition evaluation uses `df.iloc[:cutoff_idx]` for each timestamp column.
A dedicated test validates that shifting input data by 1 period changes signal
output at ≥1% of timestamps — this proves the engine is time-sensitive and
not leaking future information.

## EventDrivenExecutor Design

Reuses existing logic from the legacy engine with zero behavioral changes:

| Capability | Source | How Reused |
|------------|--------|------------|
| Position sizing | `PositionSizer` | Same instance, same calls |
| Stop-loss calculation | `sizer.calculate_stop_loss()` | Same |
| Take-profit calculation | `sizer.calculate_take_profits()` | Same |
| Position close | `_close_position()` | Copied (returns balance) |
| per_matrix tracking | existing structure | Same dict shape |
| per_strategy_isolation | `_pkey()` helper | Same key format |

### Key change: condition evaluation becomes matrix lookup

```python
# OLD: every tick, every strategy, every condition
for strategy in strategies:
    for cond in strategy.entry_conditions["long"]:
        mask = evaluate_condition(df, cond)  # pandas eval, ~50µs each
        if mask.iloc[-1]:
            signal = True

# NEW: single matrix lookup
entry_signal = matrix.get_entry(strategy.name, symbol, tf, timestamp)
# returns 1, -1, or 0 — precomputed
```

### Exit signal isolation

When `per_strategy_isolation = True`, exit signals in the matrix are keyed by
`(strategy_name, symbol, tf, "exit_long"|"exit_short")`. The executor looks up
exit signals using the position's owning strategy name, preventing strategy A's
exit condition from closing strategy B's position.

## Compatibility & Migration

### No breaking changes to public API

```python
# All existing call sites work unchanged:
engine.run_with_exit_evaluation(strategies, symbols, date_start, date_end)
# ↑ engine_mode="auto" by default

# Explicit control when needed:
engine.run_with_exit_evaluation(..., engine_mode="hybrid")
engine.run_with_exit_evaluation(..., engine_mode="legacy")
```

### Routing logic

```python
def _select_engine(self, strategies, engine_mode, ml_enabled):
    if engine_mode == "legacy":
        return "legacy"
    
    if engine_mode == "hybrid":
        # Block ML-dependent strategies from hybrid path
        has_ml = any(s.ml_config and s.ml_config.enabled for s in strategies)
        if has_ml:
            raise ValueError(
                "Hybrid engine does not support ML. "
                "Set ml_enabled=false or use engine_mode='legacy'.")
        return "hybrid"
    
    # engine_mode == "auto"
    n = len(strategies) if isinstance(strategies, list) else 1
    has_ml = any(s.ml_config and s.ml_config.enabled for s in strategies)
    if n >= 3 and not has_ml:
        return "hybrid"
    return "legacy"
```

## Verification Plan

### L1: Unit Tests (CI, every commit)

| Test Target | Count |
|-------------|-------|
| `IndicatorGrouper` — clustering correctness | 4 |
| `ConditionEvaluator` — batch vs per-condition equivalence | 6 |
| `SignalMatrixBuilder` — dimensions, dtypes, index structure | 5 |
| `EventDrivenExecutor` — entry, exit, SL, TP, balance, position cap | 10 |
| `EngineRouter` — auto/hybrid/legacy mode selection | 3 |
| `Config` — ml_enabled and engine_mode parsing | 2 |

### L2: Equivalence Tests (CI, every commit)

```python
class TestSignalEquivalence:
    def test_same_trades(self):
        """1 week × 5 symbols × 3 strategies: every trade matches exactly."""
    
    def test_same_equity_curve(self):
        """Equity at each timestamp deviates < 0.01 USD."""
    
    def test_same_per_matrix(self):
        """Per-strategy×symbol metrics: trades, pnl, win_rate all match."""
```

### L3: GA Ranking Tests (CI, every commit)

```python
class TestGARanking:
    def test_top5_overlap(self):
        """≥4 of top 5 champions overlap between engines."""
    
    def test_spearman_correlation(self):
        """Rank correlation across 30 strategies > 0.95."""
```

### L4: Anti-Lookahead Test (CI, every commit)

```python
def test_no_lookahead_bias():
    """Time-shifted data produces ≥1% different signals."""
```

### L5: End-to-End GA Comparison (manual, per release)

```bash
python -m tests.e2e_ga_compare --generations 5 --population 20
# Output: speedup, fitness delta, rank correlation, PASS/FAIL
```

### Fallback Safety Net

```python
try:
    result = engine.run_with_exit_evaluation(..., engine_mode="hybrid")
except Exception:
    logger.warning("Hybrid engine failed, falling back to legacy", exc_info=True)
    result = engine.run_with_exit_evaluation(..., engine_mode="legacy")
```

The fallback is automatic and transparent — GA evolution continues uninterrupted.

## Performance Targets

| Scenario | Legacy | Hybrid | Speedup |
|----------|--------|--------|---------|
| Single strategy (1 month, 5 symbols) | ~45s | ~8s | ~6× |
| GA batch (15 strategies, 1 month) | ~90s | ~10s | ~9× |
| GA full cycle (80 pop × 30 gen, 1 year) | ~60h | ~6h | ~10× |
| Single strategy (1 year, 5 symbols) | ~6min | ~30s | ~12× |

## Out of Scope (for this phase)

- ML training/prediction in hybrid engine (routes to legacy)
- GPU-accelerated condition evaluation
- Streaming/real-time hybrid evaluation (backtest only)
- Multi-process parallelization of signal matrix build

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Signal mismatch between engines | Medium | High | L2 equivalence tests run on every commit |
| GA ranking divergence | Low | Medium | L3 ranking tests; auto-fallback to legacy |
| Memory pressure (1m data) | Low | Low | Per-symbol sharding; 30 MB per shard |
| Condition AND/OR logic bug in batch eval | Medium | High | L2 tests cover every condition combinatorics |
| Future ML integration blocked | Low | Low | Legacy engine preserved; hybrid gets ML in phase 2 |
