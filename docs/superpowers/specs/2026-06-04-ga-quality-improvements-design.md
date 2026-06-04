# GA Quality Improvements — Design Spec

> **Status:** Approved | **Date:** 2026-06-04 | **Branch:** feature/hybrid-backtest

**Goal:** Improve GA-evolved strategy quality via expanded gene expression, rolling walk-forward validation, and data-driven fitness weight calibration.

**Three independent subsystems** — implemented sequentially (B → C → D) due to dependency: D requires Walk-Forward as its evaluation metric.

---

## B. Full-Coverage Gene Expression

### B1. New Gene Type: `BooleanGene`

```python
@dataclass
class BooleanGene:
    """On/off switch gene — controls whether a feature is active."""
    name: str
    value: bool = True
    mutation_rate: float = 0.15

    def mutate(self) -> None:
        if random.random() < self.mutation_rate:
            self.value = not self.value
```

Crossover: random inheritance from either parent. Initialization: `random.random() < p` where `p` is indicator-specific (0.3–0.7 range) to avoid all-on/all-off extremes.

### B2. Expanded Chromosome Structure

```python
chromosome = {
    "continuous": [...],     # unchanged: 10+ continuous genes
    "categorical": [...],    # unchanged: mode, timeframes
    "structural": [...],     # unchanged: entry/exit conditions
    "indicator_genes": [...], # NEW: 10 BooleanGene, one per indicator
    "name": "...",
}
```

### B3. New Continuous Genes

| Gene | Min | Max | Step | Default |
|------|-----|-----|------|---------|
| `atr_period` | 7 | 28 | 1 | 14 |
| `stoch_k_period` | 5 | 21 | 1 | 14 |
| `stoch_d_period` | 3 | 9 | 1 | 3 |
| `cci_period` | 7 | 28 | 1 | 14 |
| `sma_period` | 10 | 100 | 2 | 50 |

OBV has no tunable parameters — controlled solely by its BooleanGene.

### B4. Expanded Condition Template Pools

**Entry conditions — new templates:**

| Direction | New Templates |
|-----------|--------------|
| Long | `stoch_k < 20`, `stoch_k > stoch_d`, `cci < -100`, `cci < -200`, `atr_ratio > 1.5`, `obv > obv_sma`, `close > sma` |
| Short | `stoch_k > 80`, `stoch_k < stoch_d`, `cci > 100`, `cci > 200`, `atr_ratio < 0.7`, `obv < obv_sma`, `close < sma` |

**Exit conditions — new templates:**

| Direction | New Templates |
|-----------|--------------|
| Long | `stoch_k > 75`, `cci > 150`, `close < sma` |
| Short | `stoch_k < 25`, `cci < -150`, `close > sma` |

Total pools: entry 17/direction, exit 14/direction (original 10+5 + new 7+3 + removed duplicates).

### B5. Condition–Indicator Mapping & Sanitization

```python
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
    "volume_ratio": [],  # always available (auto-computed)
    "close": [],          # always available (raw price)
}
```

After `chromosome_to_strategy()` decodes conditions, run `_sanitize_conditions(chrom)`:
1. For each condition string, extract column references (e.g. `"stoch_k"` from `"stoch_k < 20"`)
2. Map each column to its required indicator via `CONDITION_INDICATOR_MAP`
3. If any required indicator is disabled in `indicator_genes`, remove the condition
4. If after sanitization a direction has zero conditions, inject one safe fallback from enabled indicators (prefer RSI → MACD → Bollinger → close-based)

### B6. Derived Columns in `compute_all()`

Add auto-computation when corresponding indicators are present:
- `atr_ratio` = `atr / close` (normalized volatility)
- `obv_sma` = 20-period SMA of `obv` (OBV trend)
- `sma` = SMA of close, period from config (already implemented via `sma_{period}` naming)

### B7. `complexity_penalty()` Update

Accept `indicator_genes` list; count enabled indicators directly instead of parsing continuous gene names.

---

## C. Rolling Walk-Forward Validation

### C1. Window Configuration

```python
@dataclass
class WalkForwardConfig:
    enabled: bool = False
    train_months: int = 6       # training window length
    validation_months: int = 1  # out-of-sample window length
    step_months: int = 1        # advance between windows
    ga_population: int = 80     # GA params per window (can be lower than standalone)
    ga_generations: int = 20
```

### C2. Window Calculation

```python
def compute_windows(date_start: str, date_end: str, cfg: WalkForwardConfig) -> list[tuple[str, str, str]]:
    """Returns list of (train_start, train_end, val_end) tuples.
    
    Rolling: train window slides forward by step_months each iteration.
    Validation period immediately follows training period.
    """
    start = pd.Timestamp(date_start)
    end = pd.Timestamp(date_end)
    train_delta = pd.DateOffset(months=cfg.train_months)
    val_delta = pd.DateOffset(months=cfg.validation_months)
    step_delta = pd.DateOffset(months=cfg.step_months)
    
    windows = []
    cursor = start
    while cursor + train_delta + val_delta <= end:
        train_start = cursor
        train_end = cursor + train_delta
        val_start = train_end
        val_end = train_end + val_delta
        windows.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            val_start.strftime("%Y-%m-%d"),
            val_end.strftime("%Y-%m-%d"),
        ))
        cursor += step_delta
    
    return windows
```

### C3. Walk-Forward Runner

New file: `core/ga/walkforward.py`

```python
class WalkForwardRunner:
    def run(self, symbols, date_start, date_end, wf_config, ga_config, engine, loader):
        """Execute rolling walk-forward.
        
        Returns WFReport with per-window results + aggregate metrics.
        """
        windows = compute_windows(date_start, date_end, wf_config)
        results = []
        
        for i, (tr_start, tr_end, val_start, val_end) in enumerate(windows):
            # Run GA on training window
            evolver = GAStrategyEvolver(engine, loader, ga_config)
            champion = evolver.evolve(symbols, tr_start, tr_end,
                                      validation_start=val_start)
            
            # Collect window result
            results.append(WindowResult(
                window=i + 1, total=len(windows),
                train_start=tr_start, train_end=tr_end,
                val_start=val_start, val_end=val_end,
                train_sharpe=champion.get("sharpe", 0),
                val_sharpe=champion["validation"]["sharpe"] if champion.get("validation") else 0,
                val_win_rate=champion["validation"]["win_rate"] if champion.get("validation") else 0,
                champion_name=champion.get("champion_name", ""),
            ))
            
            # Checkpoint WF state for resume
            self._save_state(i, results)
        
        return WFReport.from_results(results)
```

### C4. WFReport Metrics

```python
@dataclass
class WFReport:
    windows: list[WindowResult]
    mean_val_sharpe: float
    std_val_sharpe: float
    min_val_sharpe: float
    max_val_sharpe: float
    wf_efficiency: float         # mean / std (higher = more stable)
    positive_window_pct: float   # % of windows with positive val_sharpe
    train_val_correlation: float # Pearson r between train & val sharpe
    best_champion_name: str
    best_val_sharpe: float
```

### C5. State Persistence for Resume

File: `data/ga_wf_state.json`
```json
{
  "started_at": "2026-06-04T12:00:00",
  "date_start": "2025-06-01",
  "date_end": "2026-06-01",
  "train_months": 6,
  "val_months": 1,
  "step_months": 1,
  "current_window": 3,
  "total_windows": 6,
  "completed": [
    {"window": 1, "train_sharpe": 1.2, "val_sharpe": 0.8, "champion": "ga_champion_xxx"},
    {"window": 2, "train_sharpe": 1.5, "val_sharpe": 0.3, "champion": "ga_champion_yyy"}
  ],
  "stopped": false
}
```

### C6. API & UI

**`POST /api/ga/walkforward`** — Start WF run (async, similar to `/api/ga/evolve`). Accepts `wf_enabled`, `train_months`, `val_months`, `step_months`.

**`GET /api/ga/wf_status`** — Poll WF progress (current window, per-window results, aggregate).

**UI additions** to `ga_panel.html`:
- Mode toggle: Single Validation / Walk-Forward
- WF config fields: train months, val months, step months (shown when WF mode selected)
- WF progress: window-level progress bar + per-window result cards
- WF summary: efficiency, mean/std/min sharpe, positive window %

---

## D. Fitness Weight Calibration

### D1. Two-Stage Method

**Stage 1 — Spearman Rank Correlation (fast, ~10 min):**
1. Generate 200 random chromosomes uniformly across gene space (ensuring indicator diversity via stratified sampling: each of 10 indicators appears in 60–140 strategies)
2. Run one batch backtest for all 200 strategies (train period only)
3. For each strategy, compute: (win_rate, profit_factor, roc, imbalance) → 4-tuple of fitness components
4. Also compute each strategy's validation Sharpe (hold-out period)
5. For each weight combination in the search grid, compute a Spearman rank correlation between fitness scores and validation Sharpe. The correlation measures: does this weighting select strategies whose validation performance ranks match their fitness rank?
6. Select top 5 weight combinations by Spearman ρ

```python
# Grid: ~5,000 combinations (pruned from 15,000 by correlation clustering)
w_wr_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
w_pf_values = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0]
w_roc_values = [10, 20, 30, 40, 50, 60, 80, 100]
w_bal_values = [2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
# Total: 6 × 8 × 8 × 7 = 2,688 combinations (fast — just correlation math)
```

**Stage 2 — Walk-Forward Validation (slower, ~75 min):**
1. For each of the top 5 weight combos, run a reduced Walk-Forward: 3 windows, population=50, generations=15
2. Rank by WF efficiency (mean val_sharpe / std val_sharpe)
3. Winner becomes the calibrated weight set

### D2. Persistence

File: `data/ga_fitness_weights.json`
```json
{
  "calibrated_at": "2026-06-04T14:30:00",
  "method": "spearman_wf",
  "stage1_spearman": 0.72,
  "stage2_wf_efficiency": 1.85,
  "weights": {
    "wr": 0.20,
    "pf": 7.0,
    "roc": 40,
    "bal": 10.0
  },
  "search_space": {
    "n_random_strategies": 200,
    "n_weight_combos_stage1": 2688,
    "n_top_stage2": 5,
    "wf_windows_stage2": 3,
    "ga_population_stage2": 50,
    "ga_generations_stage2": 15
  }
}
```

### D3. Integration with GA Runner

`evolver.py` reads weights at evolution start:
```python
def _load_fitness_weights(self) -> dict:
    try:
        path = Path(self.loader.strategies_dir).parent / "data" / "ga_fitness_weights.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data["weights"]
    except Exception:
        pass
    return {"wr": 0.15, "pf": 5.0, "roc": 50, "bal": 10.0}  # defaults
```

`fitness.py` accepts weights as parameter:
```python
def evaluate_population_batch(..., weights: dict | None = None):
    w = weights or DEFAULT_WEIGHTS
    fitness = (
        win_rate * w["wr"]
        + max(profit_factor, 0.1) * w["pf"]
        + roc * w["roc"]
        - imbalance * w["bal"]
    )
```

### D4. API & UI

**`POST /api/ga/calibrate`** — Launch calibration (async). Returns immediately; poll `/api/ga/calibrate_status`.

**UI additions:**
- "Calibrate Weights" button in GA panel
- Current weights display with source label (Default / Calibrated 2026-06-04)
- Calibration progress: stage (1/2), progress within stage

---

## Implementation Order & Dependencies

```
Phase 1: B (Gene Expression)        ← can start immediately
Phase 2: C (Walk-Forward)           ← requires B for meaningful champion quality
Phase 3: D (Weight Calibration)     ← requires B + C for WF-based evaluation
```

## Files Changed / Created

| File | Action | Phase |
|------|--------|-------|
| `core/ga/genome.py` | Modify — BooleanGene, new genes, expanded pools, sanitization | B |
| `core/strategy/indicators.py` | Modify — atr_ratio, obv_sma derived columns | B |
| `core/ga/fitness.py` | Modify — complexity_penalty accepts indicator_genes, weight params | B, D |
| `core/ga/walkforward.py` | **Create** — WalkForwardRunner + WFReport | C |
| `core/ga/fitness_calibrate.py` | **Create** — two-stage weight calibration | D |
| `core/ga/evolver.py` | Modify — weight loading, WF integration | C, D |
| `web/server.py` | Modify — WF + calibrate endpoints | C, D |
| `web/templates/partials/ga_panel.html` | Modify — WF + calibrate UI | C, D |

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Old checkpoint incompatible (new chromosome fields) | Detect old format → warn → clear checkpoint → restart |
| All conditions sanitized away (indicator disabled, no matching template) | Fallback: inject `close > ema_fast` or `close < ema_slow` from enabled indicators |
| WF compute time too long (6 windows × full GA) | Per-window GA uses reduced params by default; configurable |
| Calibration overfits to specific market period | Weights tagged with calibration date; periodic recalibration recommended |
| `per_matrix` backward compatibility (old JSON lacks `gross_win_pnl`) | `.get("gross_win_pnl", 0.0)` fallback in fitness.py (already implemented) |
| Concurrent GA + manual backtest race condition on `bt_config` cost settings | Acceptable risk — cost overrides are ephemeral per-request; documented limitation |

---

## Success Criteria

1. **B:** GA can produce strategies using any combination of 10 indicators (verified by inspecting champion YAML)
2. **B:** No more all-short/all-long degeneracy (imbalance penalty + diverse condition pools)
3. **C:** Walk-Forward produces 6+ window results with per-window train/val Sharpe and aggregate WF efficiency
4. **C:** WF state is resumable after stop/crash
5. **D:** Calibrated weights produce measurably higher WF efficiency than default weights (Δ > 0.2)
6. **D:** Calibration completes in < 90 minutes on current hardware
7. All 124 existing tests still pass; new tests for BooleanGene, sanitization, WF window calculation, calibration math
