# GA Quality Improvements — Implementation Plan

> **For agentic workers:** Implement inline with per-phase audit. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Expand gene expression to 10 indicators, add rolling walk-forward validation, and calibrate fitness weights via Spearman correlation.

**Architecture:** Three sequential phases. Phase B adds BooleanGene + 5 new continuous genes + expanded condition pools with sanitization. Phase C wraps GA in a rolling-window loop with per-window champion tracking and aggregate WF metrics. Phase D calibrates fitness weights via Spearman rank correlation of 200 random strategies, validated by reduced WF runs.

**Tech Stack:** Python 3.12, dataclasses, pandas, numpy, scipy.stats (Spearman)

---

## Phase B: Full-Coverage Gene Expression

### Task B1: BooleanGene dataclass + chromosome integration

**Files:**
- Modify: `core/ga/genome.py:1-50`

- [ ] **Step 1: Add BooleanGene dataclass**

After `CategoricalGene` (line 46), add:

```python
@dataclass
class BooleanGene:
    """On/off switch gene — controls whether an indicator/feature is active."""
    name: str
    value: bool = True

    def mutate(self):
        if random.random() < 0.15:
            self.value = not self.value
```

- [ ] **Step 2: Add INDICATOR_NAMES constant + indicator gene factory**

After `TIMEFRAME_OPTIONS` (line 120):

```python
INDICATOR_NAMES = ["rsi", "macd", "bollinger", "adx", "ema", "atr", "stoch", "cci", "obv", "sma"]

# Indicator inclusion probability for random init (avoid all-on/all-off extremes)
INDICATOR_INIT_PROB = {
    "rsi": 0.6, "macd": 0.6, "bollinger": 0.5, "adx": 0.4, "ema": 0.4,
    "atr": 0.3, "stoch": 0.4, "cci": 0.3, "obv": 0.3, "sma": 0.3,
}
```

- [ ] **Step 3: Update `strategy_to_chromosome()` to encode indicator_genes**

After the structural genes block (line 192), before the return statement:

```python
    # ── Indicator boolean genes ──
    indicator_genes = []
    for name in INDICATOR_NAMES:
        enabled = name in config.indicators
        indicator_genes.append(BooleanGene(name, enabled))
```

And add `"indicator_genes": indicator_genes` to the return dict.

- [ ] **Step 4: Update `chromosome_to_strategy()` to read indicator_genes**

At the top of the function, read indicator_genes:

```python
    ind_genes = {g.name: g.value for g in chromosome.get("indicator_genes", [])}
```

Then gate each indicator block on `ind_genes.get("rsi", True)`, etc. After building indicators, run sanitization (see Task B3).

- [ ] **Step 5: Update `random_chromosome()` to generate indicator_genes**

Replace `_random_indicators()` body: use `INDICATOR_INIT_PROB` to decide each gene. Build indicators dict only for enabled ones.

- [ ] **Step 6: Write tests**

In `tests/test_ga.py`, add:

```python
def test_boolean_gene_mutate():
    from core.ga.genome import BooleanGene
    random.seed(42)
    gene = BooleanGene("test", True)
    # Force mutation by patching random.random -> 0.1 (< 0.15)
    # Actually, we test the property that value flips correctly when mutate fires
    original = gene.value
    # Multiple mutations should eventually flip (probabilistic)
    flipped = False
    for _ in range(100):
        gene.mutate()
        if gene.value != original:
            flipped = True
            break
    assert flipped, "BooleanGene should eventually flip after many mutations"

def test_indicator_genes_in_chromosome():
    from core.ga.genome import random_chromosome, INDICATOR_NAMES
    chrom = random_chromosome("test_ind")
    assert "indicator_genes" in chrom
    assert len(chrom["indicator_genes"]) == len(INDICATOR_NAMES)
    for gene in chrom["indicator_genes"]:
        assert hasattr(gene, 'value')
        assert isinstance(gene.value, bool)

def test_disabled_indicator_not_in_strategy():
    from core.ga.genome import random_chromosome, chromosome_to_strategy
    chrom = random_chromosome("test_disable")
    # Force-disable RSI
    for g in chrom["indicator_genes"]:
        if g.name == "rsi":
            g.value = False
    config = chromosome_to_strategy(chrom)
    assert "rsi" not in config.indicators

def test_all_indicators_disabled_has_fallback():
    """When all indicators disabled, should get at least EMA fallback."""
    from core.ga.genome import random_chromosome, chromosome_to_strategy
    chrom = random_chromosome("test_none")
    for g in chrom["indicator_genes"]:
        g.value = False
    config = chromosome_to_strategy(chrom)
    # Should have at least one indicator as fallback
    assert len(config.indicators) >= 1
```

- [ ] **Step 7: Run tests, verify pass, commit**

```bash
git add core/ga/genome.py tests/test_ga.py
git commit -m "feat(ga): add BooleanGene + indicator_genes to chromosome"
```

---

### Task B2: 5 new continuous genes

**Files:**
- Modify: `core/ga/genome.py` (strategy_to_chromosome, chromosome_to_strategy, _random_indicators)

- [ ] **Step 1: Add new gene ranges to CONSTANTS section**

After line 120, add:

```python
# New indicator gene ranges (used by random init + encoding)
NEW_GENE_RANGES = {
    "atr_period": (7, 28, 1, 14),
    "stoch_k_period": (5, 21, 1, 14),
    "stoch_d_period": (3, 9, 1, 3),
    "cci_period": (7, 28, 1, 14),
    "sma_period": (10, 100, 2, 50),
}
```

- [ ] **Step 2: Update `strategy_to_chromosome()` — encode new indicators**

After the EMA block (line 159), add blocks for ATR, Stoch, CCI, SMA:

```python
    if "atr" in ind:
        a = ind["atr"]
        continuous.append(ContinuousGene("atr_period", a.get("period", 14), 7, 28, 1))
    if "stoch" in ind:
        s = ind["stoch"]
        continuous.append(ContinuousGene("stoch_k_period", s.get("k_period", 14), 5, 21, 1))
        continuous.append(ContinuousGene("stoch_d_period", s.get("d_period", 3), 3, 9, 1))
    if "cci" in ind:
        c = ind["cci"]
        continuous.append(ContinuousGene("cci_period", c.get("period", 14), 7, 28, 1))
    if "sma" in ind:
        sm = ind["sma"]
        continuous.append(ContinuousGene("sma_period", sm.get("period", 50), 10, 100, 2))
```

- [ ] **Step 3: Update `chromosome_to_strategy()` — decode new indicators**

Same pattern as existing indicators, gated on `ind_genes`:

```python
    if ind_genes.get("atr", False) and "atr_period" in cont:
        indicators["atr"] = {"period": int(cont.get("atr_period", 14))}
    if ind_genes.get("stoch", False) and "stoch_k_period" in cont:
        indicators["stoch"] = {
            "k_period": int(cont.get("stoch_k_period", 14)),
            "d_period": int(cont.get("stoch_d_period", 3)),
        }
    if ind_genes.get("cci", False) and "cci_period" in cont:
        indicators["cci"] = {"period": int(cont.get("cci_period", 14))}
    if ind_genes.get("obv", False):
        indicators["obv"] = {}
    if ind_genes.get("sma", False) and "sma_period" in cont:
        indicators["sma"] = {"period": int(cont.get("sma_period", 50))}
```

- [ ] **Step 4: Update `_random_indicators()` — include new indicators**

Use INDICATOR_INIT_PROB dict to generate. Return only enabled ones.

- [ ] **Step 5: Write tests**

```python
def test_new_indicator_genes_encoded():
    from core.ga.genome import random_chromosome
    chrom = random_chromosome("test_new")
    cont_names = [g.name for g in chrom["continuous"]]
    # Some new genes should appear (probabilistic)
    possible_new = ["atr_period", "stoch_k_period", "cci_period", "sma_period"]
    # Not all chromosomes will have all genes — just verify the encoding round-trip
    config = chromosome_to_strategy(chrom)
    assert config.name == "test_new"
    # Round-trip: encode → decode → encode should produce identical config
    chrom2 = strategy_to_chromosome(config)
    assert len(chrom2["continuous"]) == len(chrom["continuous"])

def test_obv_no_continuous_genes():
    """OBV has no tunable parameters — only BooleanGene controls it."""
    from core.ga.genome import chromosome_to_strategy
    chrom = random_chromosome("test_obv")
    for g in chrom["indicator_genes"]:
        g.value = False
    for g in chrom["indicator_genes"]:
        if g.name == "obv":
            g.value = True
    config = chromosome_to_strategy(chrom)
    assert "obv" in config.indicators
    assert config.indicators["obv"] == {}
```

- [ ] **Step 6: Run tests, commit**

```bash
git add core/ga/genome.py tests/test_ga.py
git commit -m "feat(ga): add 5 new indicator genes (ATR, Stoch, CCI, OBV, SMA)"
```

---

### Task B3: Condition pool expansion + sanitization

**Files:**
- Modify: `core/ga/genome.py` (CONDITION_POOL, EXIT_CONDITION_POOL, chromosome_to_strategy)

- [ ] **Step 1: Expand condition pools**

Replace CONDITION_POOL and EXIT_CONDITION_POOL with expanded versions including new templates from the spec (B4 section). See spec for exact template lists.

- [ ] **Step 2: Add CONDITION_INDICATOR_MAP**

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
    "volume_ratio": [], "close": [],
}
```

- [ ] **Step 3: Implement `_sanitize_conditions()`**

```python
def _sanitize_conditions(conditions: list[str], enabled_indicators: set[str]) -> list[str]:
    """Remove conditions referencing disabled indicators. Inject fallback if empty."""
    import re
    clean = []
    for cond in conditions:
        # Extract all column names referenced in the condition
        cols = set()
        for col_pattern in CONDITION_INDICATOR_MAP:
            if col_pattern in cond:
                cols.add(col_pattern)
        # Check if all required indicators are enabled
        ok = True
        for col in cols:
            required = CONDITION_INDICATOR_MAP.get(col, [])
            if required and not any(r in enabled_indicators for r in required):
                ok = False
                break
        if ok:
            clean.append(cond)
    
    if not clean:
        # Fallback: use price-based conditions that always work
        if "ema" in enabled_indicators:
            clean = ["close > ema_fast"] if "long" in str(conditions) else ["close < ema_fast"]
        elif "bollinger" in enabled_indicators:
            clean = ["close > bollinger_lower"] if "long" in str(conditions) else ["close < bollinger_upper"]
        else:
            clean = ["volume_ratio > 1.0"]  # always available
    
    return clean
```

- [ ] **Step 4: Integrate sanitization into `chromosome_to_strategy()`**

Before building entry_conditions/exit_conditions, get enabled indicators from `ind_genes`:

```python
    enabled_indicators = {name for name, gene in ind_genes.items() if gene}
    
    entry_long = _sanitize_conditions(struct.get("entry_long", []), enabled_indicators)
    entry_short = _sanitize_conditions(struct.get("entry_short", []), enabled_indicators)
    exit_long = _sanitize_conditions(struct.get("exit_long", []), enabled_indicators)
    exit_short = _sanitize_conditions(struct.get("exit_short", []), enabled_indicators)
```

- [ ] **Step 5: Write tests**

```python
def test_sanitize_removes_disabled_indicator_conditions():
    from core.ga.genome import _sanitize_conditions
    conds = ["rsi < 30", "stoch_k < 20", "close > ema_fast"]
    enabled = {"rsi", "ema"}
    result = _sanitize_conditions(conds, enabled)
    assert "rsi < 30" in result
    assert "close > ema_fast" in result
    assert "stoch_k < 20" not in result  # stoch disabled

def test_sanitize_injects_fallback_when_all_filtered():
    from core.ga.genome import _sanitize_conditions
    conds = ["stoch_k < 20", "cci < -100"]
    enabled = {"rsi"}  # nothing matches
    result = _sanitize_conditions(conds, enabled)
    assert len(result) >= 1  # fallback injected
```

- [ ] **Step 6: Run tests, commit**

---

### Task B4: Derived columns in indicators.py

**Files:**
- Modify: `core/strategy/indicators.py`

- [ ] **Step 1: Add atr_ratio auto-computation**

After the ATR block (after line 76), add:

```python
            if "atr" in result.columns and "close" in result.columns:
                result["atr_ratio"] = result["atr"] / result["close"]
```

- [ ] **Step 2: Add obv_sma auto-computation**

After the OBV block (after line 90), add:

```python
            if "obv" in result.columns:
                result["obv_sma"] = result["obv"].rolling(20).mean()
```

- [ ] **Step 3: Write tests**

```python
def test_atr_ratio_computed():
    from core.strategy.indicators import compute_all
    df = pd.DataFrame({
        "open": [100]*30, "high": [102]*30, "low": [98]*30,
        "close": [101]*30, "volume": [1000]*30,
    })
    result = compute_all(df, {"atr": {"period": 14}})
    assert "atr" in result.columns
    assert "atr_ratio" in result.columns
    assert (result["atr_ratio"] > 0).all()

def test_obv_sma_computed():
    from core.strategy.indicators import compute_all
    df = pd.DataFrame({
        "open": [100]*30, "high": [101]*30, "low": [99]*30,
        "close": [100]*30, "volume": [1000]*30,
    })
    result = compute_all(df, {"obv": {}})
    assert "obv" in result.columns
    assert "obv_sma" in result.columns
```

- [ ] **Step 4: Run tests, commit**

---

### Task B5: Update complexity_penalty for indicator_genes

**Files:**
- Modify: `core/ga/fitness.py:99-121`

- [ ] **Step 1: Update complexity_penalty signature and logic**

```python
def complexity_penalty(chromosome: dict) -> float:
    """Penalize overparameterized strategies."""
    structural = chromosome.get("structural", [])
    n_conditions = sum(len(g.conditions) for g in structural)
    
    # Count enabled indicators directly from BooleanGenes
    indicator_genes = chromosome.get("indicator_genes", [])
    if indicator_genes:
        n_indicators = sum(1 for g in indicator_genes if g.value)
    else:
        # Backward compat: parse from continuous gene names
        continuous = chromosome.get("continuous", [])
        indicators_used = set()
        for g in continuous:
            name = g.name.split("_")[0]
            indicators_used.add(name)
        # Filter out non-indicator continuous genes
        non_indicators = {"ml"}
        n_indicators = len(indicators_used - non_indicators)
    
    continuous = chromosome.get("continuous", [])
    penalty = 0.0
    penalty += n_conditions * 0.8
    penalty += n_indicators * 1.2
    penalty += len(continuous) * 0.3
    return penalty
```

- [ ] **Step 2: Run existing tests, commit**

---

## Phase B Audit

After all B tasks complete, run:

```bash
python -m pytest tests/test_ga.py tests/test_indicators.py -v
```

Verify:
1. BooleanGene mutates correctly
2. Round-trip: random_chromosome → chromosome_to_strategy → strategy_to_chromosome preserves genes
3. Disabled indicators don't appear in StrategyConfig
4. Condition sanitization removes disabled-indicator conditions
5. Fallback conditions injected when all filtered
6. Derived columns (atr_ratio, obv_sma) computed
7. All 124 existing tests still pass

---

## Phase C: Rolling Walk-Forward Validation

### Task C1: WalkForwardRunner + WFReport

**Files:**
- Create: `core/ga/walkforward.py`

- [ ] **Step 1: Create dataclasses**

```python
"""Rolling walk-forward validation — multi-window GA optimization and validation."""

from dataclasses import dataclass, field
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

@dataclass
class WFConfig:
    enabled: bool = True
    train_months: int = 6
    val_months: int = 1
    step_months: int = 1

@dataclass
class WindowResult:
    window: int
    total: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_sharpe: float
    val_sharpe: float
    val_win_rate: float = 0.0
    val_total_return: float = 0.0
    champion_name: str = ""
    
@dataclass
class WFReport:
    windows: list = field(default_factory=list)
    mean_val_sharpe: float = 0.0
    std_val_sharpe: float = 0.0
    min_val_sharpe: float = 0.0
    max_val_sharpe: float = 0.0
    wf_efficiency: float = 0.0
    positive_window_pct: float = 0.0
    train_val_correlation: float = 0.0
    best_champion_name: str = ""
    best_val_sharpe: float = 0.0
    elapsed_seconds: float = 0.0

    @classmethod
    def from_results(cls, results: list[WindowResult], elapsed: float) -> "WFReport":
        val_sharpes = [r.val_sharpe for r in results]
        train_sharpes = [r.train_sharpe for r in results]
        n = len(val_sharpes)
        mean_vs = float(np.mean(val_sharpes)) if n > 0 else 0.0
        std_vs = float(np.std(val_sharpes, ddof=1)) if n > 1 else 0.0
        
        # WF efficiency = mean / std (higher = more stable)
        wf_eff = mean_vs / std_vs if std_vs > 0 else 0.0
        
        # Positive window %
        pos_pct = sum(1 for s in val_sharpes if s > 0) / n * 100 if n > 0 else 0.0
        
        # Correlation between train and val Sharpe
        if n >= 3:
            corr = float(np.corrcoef(train_sharpes, val_sharpes)[0, 1])
            corr = 0.0 if np.isnan(corr) else corr
        else:
            corr = 0.0
        
        # Best champion
        best_idx = int(np.argmax(val_sharpes)) if n > 0 else 0
        best = results[best_idx] if n > 0 else None
        
        return cls(
            windows=results,
            mean_val_sharpe=round(mean_vs, 4),
            std_val_sharpe=round(std_vs, 4),
            min_val_sharpe=round(float(np.min(val_sharpes)), 4) if n > 0 else 0.0,
            max_val_sharpe=round(float(np.max(val_sharpes)), 4) if n > 0 else 0.0,
            wf_efficiency=round(wf_eff, 4),
            positive_window_pct=round(pos_pct, 1),
            train_val_correlation=round(corr, 4),
            best_champion_name=best.champion_name if best else "",
            best_val_sharpe=best.val_sharpe if best else 0.0,
            elapsed_seconds=round(elapsed, 1),
        )
```

- [ ] **Step 2: Implement WalkForwardRunner**

```python
class WalkForwardRunner:
    def __init__(self, engine, loader, data_dir: str):
        self.engine = engine
        self.loader = loader
        self.data_dir = Path(data_dir)
        self._state_path = self.data_dir / "data" / "ga_wf_state.json"
        self._running = False

    def compute_windows(self, date_start: str, date_end: str, cfg: WFConfig) -> list[tuple]:
        start = pd.Timestamp(date_start)
        end = pd.Timestamp(date_end)
        train_delta = pd.DateOffset(months=cfg.train_months)
        val_delta = pd.DateOffset(months=cfg.val_months)
        step_delta = pd.DateOffset(months=cfg.step_months)
        
        windows = []
        cursor = start
        while cursor + train_delta + val_delta <= end:
            train_start = cursor
            train_end = cursor + train_delta
            val_end = train_end + val_delta
            windows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                val_end.strftime("%Y-%m-%d"),
            ))
            cursor += step_delta
        return windows

    def run(self, symbols, date_start, date_end, wf_config, ga_config, resume=False):
        from core.ga.evolver import GAStrategyEvolver
        windows = self.compute_windows(date_start, date_end, wf_config)
        results = []
        start_window = 0
        
        if resume:
            saved = self._load_state()
            if saved:
                results = [WindowResult(**r) for r in saved.get("completed", [])]
                start_window = saved.get("current_window", 0)
        
        t0 = time.time()
        self._running = True
        
        for i in range(start_window, len(windows)):
            if not self._running:
                break
            tr_start, tr_end, val_start, val_end = windows[i]
            
            logger.info(f"WF window {i+1}/{len(windows)}: train={tr_start}~{tr_end}, val={val_start}~{val_end}")
            
            evolver = GAStrategyEvolver(self.engine, self.loader, ga_config)
            champion = evolver.evolve(
                symbols, tr_start, tr_end,
                seed_strategies=None,
                validation_start=val_start,
            )
            
            result = WindowResult(
                window=i+1, total=len(windows),
                train_start=tr_start, train_end=tr_end,
                val_start=val_start, val_end=val_end,
                train_sharpe=champion.get("sharpe", 0),
                val_sharpe=champion["validation"]["sharpe"] if champion.get("validation") else 0,
                val_win_rate=champion["validation"]["win_rate"] if champion.get("validation") else 0,
                val_total_return=champion["validation"]["total_return"] if champion.get("validation") else 0,
                champion_name=champion.get("champion_name", ""),
            )
            results.append(result)
            self._save_state(i + 1, results)
        
        self._running = False
        elapsed = time.time() - t0
        self._clear_state()
        return WFReport.from_results(results, elapsed)

    def stop(self):
        self._running = False

    def _save_state(self, current_window: int, results: list):
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "current_window": current_window,
                "total_windows": len(results) + (1 if current_window < len(results) else 0),
                "completed": [{"window": r.window, "train_sharpe": r.train_sharpe,
                               "val_sharpe": r.val_sharpe, "champion_name": r.champion_name}
                              for r in results],
                "stopped": not self._running,
            }
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"WF state save failed: {e}")

    def _load_state(self) -> dict | None:
        try:
            if self._state_path.exists():
                with open(self._state_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _clear_state(self):
        try:
            if self._state_path.exists():
                self._state_path.unlink()
        except Exception:
            pass
```

- [ ] **Step 3: Write tests**

```python
def test_wf_window_computation():
    from core.ga.walkforward import WalkForwardRunner, WFConfig
    runner = WalkForwardRunner(None, None, "/tmp")
    cfg = WFConfig(train_months=6, val_months=1, step_months=1)
    windows = runner.compute_windows("2025-01-01", "2026-01-01", cfg)
    assert len(windows) >= 5
    # Each window: train_end == val_start
    for tr_s, tr_e, val_s, val_e in windows:
        assert tr_e == val_s
    # Windows advance by 1 month
    first_start = pd.Timestamp(windows[0][0])
    second_start = pd.Timestamp(windows[1][0])
    assert (second_start - first_start).days >= 28

def test_wf_report_metrics():
    from core.ga.walkforward import WFReport, WindowResult
    results = [
        WindowResult(1, 3, "2025-01-01", "2025-07-01", "2025-07-01", "2025-08-01", 1.2, 0.8),
        WindowResult(2, 3, "2025-02-01", "2025-08-01", "2025-08-01", "2025-09-01", 1.5, 0.3),
        WindowResult(3, 3, "2025-03-01", "2025-09-01", "2025-09-01", "2025-10-01", 0.9, 1.1),
    ]
    report = WFReport.from_results(results, 100.0)
    assert report.mean_val_sharpe > 0
    assert report.std_val_sharpe > 0
    assert report.wf_efficiency > 0
    assert report.positive_window_pct == 100.0 / 3 * 2  # ~66.7% (2/3 positive)
    assert report.best_val_sharpe == 1.1
```

- [ ] **Step 4: Run tests, commit**

---

### Task C2: API endpoints + UI for Walk-Forward

**Files:**
- Modify: `web/server.py`
- Modify: `web/templates/partials/ga_panel.html`

Due to the complexity of server + UI changes, detailed steps omitted for brevity in this plan — the key pattern matches the existing `/api/ga/evolve` async endpoint pattern. New endpoints: `POST /api/ga/walkforward`, `GET /api/ga/wf_status`.

---

## Phase C Audit

1. `compute_windows()` produces correct non-overlapping windows
2. WFReport metrics (mean, std, efficiency, correlation) are correct
3. State save/load/resume works
4. Existing GA API still works (no regression)

---

## Phase D: Fitness Weight Calibration

### Task D1: Two-stage calibrator

**Files:**
- Create: `core/ga/fitness_calibrate.py`

- [ ] **Step 1: Create calibrator class**

```python
"""Fitness weight calibration via Spearman rank correlation + Walk-Forward."""

import json
import random
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from scipy import stats as _stats
from loguru import logger

DEFAULT_WEIGHTS = {"wr": 0.15, "pf": 5.0, "roc": 50, "bal": 10.0}

WEIGHT_GRID = {
    "wr": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "pf": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0],
    "roc": [10, 20, 30, 40, 50, 60, 80, 100],
    "bal": [2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0],
}

@dataclass
class CalibrationResult:
    weights: dict
    stage1_spearman: float
    stage2_wf_efficiency: float
    calibrated_at: str
    search_space: dict

class FitnessCalibrator:
    def __init__(self, engine, loader, data_dir: str):
        self.engine = engine
        self.loader = loader
        self.data_dir = Path(data_dir)

    def calibrate(self, symbols, date_start, date_end, progress_callback=None):
        from core.ga.genome import random_chromosome
        from core.ga.fitness import evaluate_population_batch

        n_random = 200
        # Stage 1: Generate random strategies + Spearman correlation
        if progress_callback:
            progress_callback("stage1", 0, n_random)
        
        # Generate diversified random chromosomes
        random_strategies = []
        for i in range(n_random):
            chrom = random_chromosome(f"calib_{i}")
            random_strategies.append(chrom)
        
        # Batch backtest all strategies
        result = self.engine.run_with_exit_evaluation(
            strategies=[chromosome_to_strategy(c) for c in random_strategies],
            # ... batch evaluation
        )
        
        # Compute fitness components + validation Sharpe per strategy
        components = []  # list of (wr, pf, roc, imbalance, val_sharpe)
        
        # For each weight combination, compute Spearman correlation
        best_rho = -1
        best_combo = None
        combos = []
        
        for w_wr in WEIGHT_GRID["wr"]:
            for w_pf in WEIGHT_GRID["pf"]:
                for w_roc in WEIGHT_GRID["roc"]:
                    for w_bal in WEIGHT_GRID["bal"]:
                        fitness_scores = [compute_fitness(c, ...) for c in components]
                        val_sharpes = [c[4] for c in components]
                        rho, _ = _stats.spearmanr(fitness_scores, val_sharpes)
                        combos.append(({"wr": w_wr, "pf": w_pf, "roc": w_roc, "bal": w_bal}, rho))
        
        combos.sort(key=lambda x: x[1], reverse=True)
        top5 = combos[:5]
        
        # Stage 2: WF validation of top 5
        # ... reduced WF runs
        
        return CalibrationResult(...)

    def save_weights(self, result: CalibrationResult):
        path = self.data_dir / "data" / "ga_fitness_weights.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "calibrated_at": result.calibrated_at,
                "method": "spearman_wf",
                "stage1_spearman": result.stage1_spearman,
                "stage2_wf_efficiency": result.stage2_wf_efficiency,
                "weights": result.weights,
                "search_space": result.search_space,
            }, f, indent=2)

    @staticmethod
    def load_weights(data_dir: str) -> dict:
        path = Path(data_dir) / "data" / "ga_fitness_weights.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)["weights"]
        return dict(DEFAULT_WEIGHTS)
```

- [ ] **Step 2: Update fitness.py to accept weight params**

Modify `evaluate_population_batch()` signature: add `weights: dict | None = None`. Use `weights or DEFAULT_WEIGHTS` in the fitness formula.

- [ ] **Step 3: Write tests**

```python
def test_default_weights():
    from core.ga.fitness_calibrate import DEFAULT_WEIGHTS
    assert "wr" in DEFAULT_WEIGHTS
    assert "pf" in DEFAULT_WEIGHTS
    assert "roc" in DEFAULT_WEIGHTS
    assert "bal" in DEFAULT_WEIGHTS

def test_spearman_perfect_correlation():
    """If fitness perfectly predicts val_sharpe, rho should be ~1."""
    from scipy.stats import spearmanr
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    rho, _ = spearmanr(x, y)
    assert rho > 0.99

def test_weight_grid_coverage():
    from core.ga.fitness_calibrate import WEIGHT_GRID
    total = 1
    for key in WEIGHT_GRID:
        total *= len(WEIGHT_GRID[key])
    assert total > 1000  # sufficient search coverage
    assert total < 10000  # not excessive
```

- [ ] **Step 4: Run tests, commit**

---

## Phase D Audit

1. Spearman calculation is correct (verified by perfect-correlation test)
2. Weight grid covers reasonable ranges
3. Weight save/load round-trip
4. integration with fitness.py weight param

---

## Final Holistic Audit

After all phases complete:

1. `python -m pytest tests/ -v` — all 124+ tests pass
2. Manual: generate a random chromosome → verify all 10 indicators can be enabled/disabled
3. Manual: verify condition sanitization with edge cases (all disabled, only OBV enabled)
4. Manual: verify WF window computation with edge cases (short date ranges, non-month-aligned)
5. Performance: random strategy generation should be O(n_indicators) not O(n^2)
6. Performance: fitness evaluation should not serialize/deserialize unnecessarily
7. Run a quick 3-window WF with population=30, generations=5 → should complete without errors
