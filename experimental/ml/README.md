# `experimental/ml` — never-wired ML extras

## What this is

ML code that was written for the ML roadmap (Phase 4d, R15) but **never called
by anything**. It was confirmed dead by AST/grep analysis, verified to have zero
callers in `app/`, `core/`, `web/`, `db/`, `scripts/` and `tests/`, and moved
out of `core/ml/` so the production ML surface only exposes what the trading
path actually uses.

## What it is NOT

**This is NOT part of the trading path.** Nothing here runs during signal
generation, risk management, execution, backtesting, or serving the Web API.
No production module imports this package. Deleting the whole directory would
not change trading behaviour — it is kept only so the designs are not lost.

What *is* live in `core/ml/` (do not confuse these with the orphans below):
`MLPredictor.predict` / `_predict_lgb` / `_predict_tft` / `_predict_volatility`,
`MLPredictor.train_model` / `train_tft_model` / `train_volatility_model`,
`MLTrainer.train_binary` / `load_model` / `save_training_data`, the
`TFTTrainer` / `PatchTSTTrainer` pair (used by `core/backtest/engine.py` when
`ml_engine` is `tft` / `patchtst`), and the label helpers
`create_binary_label`, `create_regression_label`,
`create_triple_barrier_label`, `create_volatility_label`.

## Contents

| Module | Moved from | Why it was dead |
| --- | --- | --- |
| `feature_store.py` | `core/ml/feature_store.py` (whole file) | `FeatureStore` was never imported anywhere; `has_data()` had no caller either |
| `features_extras.py` | `core/ml/features.py` | `create_label`, `triple_barrier_probabilities`, `create_regime_label`, `create_volatility_magnitude_label` — label helpers for prediction heads no trainer ever implemented |
| `predictor_methods.py` | `core/ml/predictor.py` | `load_tft_model`, `ensemble_predict` (+ its helper `_predict_patchtst`), `incremental_retrain`, `load_patchtst_model` — no caller; `predict()` routes to a single architecture, so the ensemble vote was unreachable |
| `trainer_methods.py` | `core/ml/trainer.py` | `train_regression` (+ private helpers `_train_lgb_reg` / `_train_xgb_reg`), `load_training_data` — only the binary classifier is wired up |

The moved functions are **verbatim** apart from one mechanical change: where a
method used `self`, the standalone function now takes the owning object as an
explicit first argument named `trainer` / `predictor`. The bodies, defaults and
return shapes are unchanged, so the behaviour is preserved exactly.

## How to wire a symbol back

1. Move the function back into the class/module it came from (see the table
   above) and restore `self`.
2. Remove the corresponding entry from this README and the "MOVED OUT OF
   PRODUCTION" banner in the file.
3. Add the call site that makes it live, and a test that exercises it through
   the real entry point (not just a direct import) — every symbol here was dead
   precisely because no test drove it through production.
4. Run `python -m pytest tests/ -q` and `python -m compileall -q app core web db scripts`.

## Importing it

The package is importable from the repo root, so the code stays honest and
reviewable:

```python
from experimental.ml.features_extras import create_regime_label
from experimental.ml.feature_store import FeatureStore
```
