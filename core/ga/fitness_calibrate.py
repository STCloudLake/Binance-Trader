"""Fitness weight calibration via Spearman rank correlation.

Two-stage approach:
  Stage 1 (fast, ~5 min): Generate 200 random strategies, batch-backtest on
      train+validation periods, compute Spearman ρ between fitness scores
      and validation PnL for each weight combination. Select top 5.
  Stage 2 (thorough, ~60 min): Run reduced Walk-Forward on top 5 weight
      combos. Winner = highest WF efficiency.
"""

import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from scipy import stats as _stats
from loguru import logger

# Default weights (hand-tuned, used when no calibration exists)
DEFAULT_WEIGHTS = {"wr": 0.15, "pf": 5.0, "roc": 50, "bal": 10.0}

# Search grid for weight calibration
WEIGHT_GRID = {
    "wr":  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "pf":  [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0],
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
    """Calibrate fitness function weights to maximize validation performance."""

    def __init__(self, engine, loader, data_dir: str):
        self.engine = engine
        self.loader = loader
        self.data_dir = Path(data_dir)
        self._save_path = self.data_dir / "data" / "ga_fitness_weights.json"

    def calibrate(self, symbols: list[str], date_start: str, date_end: str,
                  validation_start: str | None = None,
                  progress_callback=None) -> CalibrationResult:
        """Run two-stage calibration.

        Args:
            symbols: Trading symbols.
            date_start, date_end: Full date range.
            validation_start: If set, training=[date_start, validation_start),
                              validation=[validation_start, date_end).
                              If None, uses last 20% of range as validation.
            progress_callback: fn(stage: str, current: int, total: int)
        """
        from core.ga.genome import random_chromosome, chromosome_to_strategy
        from core.ga.fitness import complexity_penalty

        # Determine train/val split
        if validation_start is None:
            from datetime import datetime
            sd = datetime.strptime(date_start, "%Y-%m-%d")
            ed = datetime.strptime(date_end, "%Y-%m-%d")
            span_days = (ed - sd).days
            split_d = sd + __import__('datetime').timedelta(days=int(span_days * 0.8))
            train_end = split_d.strftime("%Y-%m-%d")
            val_start = train_end
        else:
            train_end = validation_start
            val_start = validation_start

        # ── Stage 1: Spearman rank correlation ──
        n_strategies = 200
        if progress_callback:
            progress_callback("stage1", 0, n_strategies)

        # Generate diversified random chromosomes
        logger.info(f"Calibrator Stage 1: generating {n_strategies} random strategies")
        random_strategies = []
        for i in range(n_strategies):
            chrom = random_chromosome(f"calib_{i}")
            random_strategies.append(chrom)

        # Batch backtest on TRAINING data
        logger.info(f"Calibrator: backtesting on train period {date_start}~{train_end}")
        configs = [chromosome_to_strategy(c) for c in random_strategies]
        train_result = self.engine.run_with_exit_evaluation(
            strategies=configs, symbols=symbols,
            date_start=date_start, date_end=train_end,
            initial_balance=10000.0, mode="full",
            simulate_ai_weights=False, ml_engine="lightgbm",
            per_strategy_isolation=True,
        )
        train_matrix = train_result.get("per_matrix", {})

        # Batch backtest on VALIDATION data
        logger.info(f"Calibrator: backtesting on val period {val_start}~{date_end}")
        val_result = self.engine.run_with_exit_evaluation(
            strategies=configs, symbols=symbols,
            date_start=val_start, date_end=date_end,
            initial_balance=10000.0, mode="full",
            simulate_ai_weights=False, ml_engine="lightgbm",
            per_strategy_isolation=True,
        )
        val_matrix = val_result.get("per_matrix", {})

        # Extract per-strategy fitness components + validation PnL
        components = []
        for i, config in enumerate(configs):
            name = config.name
            train_cells = train_matrix.get(name, {})
            val_cells = val_matrix.get(name, {})

            trades = sum(c.get("trades", 0) for c in train_cells.values())
            pnl = sum(c.get("pnl", 0) for c in train_cells.values())
            wins = sum(c.get("winning", 0) for c in train_cells.values())
            long_trades = sum(c.get("long_trades", 0) for c in train_cells.values())
            short_trades = sum(c.get("short_trades", 0) for c in train_cells.values())
            gross_win = sum(c.get("gross_win_pnl", 0.0) for c in train_cells.values())
            gross_loss = sum(c.get("gross_loss_pnl", 0.0) for c in train_cells.values())

            win_rate = (wins / max(trades, 1)) * 100
            if gross_loss > 0:
                profit_factor = min(gross_win / gross_loss, 100.0)
            elif gross_win > 0:
                profit_factor = 100.0
            else:
                profit_factor = 0.1
            roc = pnl / 10000.0
            if trades > 0:
                imbalance = abs((long_trades / trades) - 0.5) * 2
            else:
                imbalance = 1.0

            # Validation PnL (proxy for validation Sharpe — rank-preserving)
            val_pnl = sum(c.get("pnl", 0) for c in val_cells.values())

            # Skip strategies with 0 trades (no signal)
            if trades < 1:
                continue

            components.append((win_rate, profit_factor, roc, imbalance, val_pnl))

        if len(components) < 20:
            logger.warning(f"Calibrator: only {len(components)} valid strategies, "
                          "using default weights")
            return CalibrationResult(
                weights=dict(DEFAULT_WEIGHTS),
                stage1_spearman=0.0, stage2_wf_efficiency=0.0,
                calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                search_space={"n_valid_strategies": len(components), "error": "insufficient data"},
            )

        if progress_callback:
            progress_callback("stage1_eval", 0, len(WEIGHT_GRID["wr"]) * len(WEIGHT_GRID["pf"]) *
                             len(WEIGHT_GRID["roc"]) * len(WEIGHT_GRID["bal"]))

        # Grid search: find weights with highest Spearman ρ
        val_pnls = np.array([c[4] for c in components])
        best_rho = -1.0
        best_combo = dict(DEFAULT_WEIGHTS)
        all_rhos = []

        combo_count = 0
        for w_wr in WEIGHT_GRID["wr"]:
            for w_pf in WEIGHT_GRID["pf"]:
                for w_roc in WEIGHT_GRID["roc"]:
                    for w_bal in WEIGHT_GRID["bal"]:
                        combo_count += 1
                        scores = []
                        for wr, pf, roc, imb, _vp in components:
                            score = (wr * w_wr + max(pf, 0.1) * w_pf
                                    + roc * w_roc - imb * w_bal)
                            scores.append(score)

                        rho, pvalue = _stats.spearmanr(scores, val_pnls)
                        if np.isnan(rho):
                            rho = 0.0
                        all_rhos.append(({"wr": w_wr, "pf": w_pf, "roc": w_roc, "bal": w_bal}, rho, pvalue))

                        if rho > best_rho:
                            best_rho = rho
                            best_combo = {"wr": w_wr, "pf": w_pf, "roc": w_roc, "bal": w_bal}

        all_rhos.sort(key=lambda x: x[1], reverse=True)
        top5 = all_rhos[:5]

        logger.info(f"Calibrator Stage 1 done: best Spearman ρ = {best_rho:.4f}, "
                   f"weights = {best_combo}")

        # ── Stage 2: Walk-Forward validation of top 5 ──
        best_wf_efficiency = 0.0
        final_weights = best_combo

        try:
            from core.ga.walkforward import WalkForwardRunner, WFConfig
            from core.ga.evolver import GARunConfig

            wf_cfg = WFConfig(train_months=6, val_months=1, step_months=1)
            ga_cfg = GARunConfig(
                population_size=50, generations=15,
                elite_count=5, immigrant_count=5, max_workers=3,
            )

            data_dir_str = str(self.data_dir)
            runner = WalkForwardRunner(self.engine, self.loader, data_dir_str)

            for rank, (combo, rho, pv) in enumerate(top5):
                if progress_callback:
                    progress_callback("stage2", rank + 1, len(top5))

                logger.info(f"Calibrator Stage 2: testing combo {rank+1}/{len(top5)} "
                          f"(ρ={rho:.4f}): {combo}")

                # Temporarily set weights for this run
                self._save_weights_temp(combo)

                try:
                    # Use a short WF: 3 windows with reduced GA
                    report = runner.run(symbols, date_start, date_end, wf_cfg, ga_cfg)
                    wf_eff = report.wf_efficiency

                    if wf_eff > best_wf_efficiency:
                        best_wf_efficiency = wf_eff
                        final_weights = combo
                except Exception as e:
                    logger.warning(f"Stage 2 combo {combo} failed: {e}")
                    continue
        except ImportError:
            logger.info("WalkForward not available, skipping Stage 2")

        result = CalibrationResult(
            weights=final_weights,
            stage1_spearman=round(best_rho, 4),
            stage2_wf_efficiency=round(best_wf_efficiency, 4),
            calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            search_space={
                "n_random_strategies": n_strategies,
                "n_valid_strategies": len(components),
                "n_weight_combos_stage1": combo_count,
                "top5_rhos": [(c["wr"], c["pf"], c["roc"], c["bal"], round(r, 4))
                              for c, r, _ in top5],
                "n_top_stage2": len(top5),
            },
        )

        self.save_weights(result)
        return result

    def save_weights(self, result: CalibrationResult):
        """Persist calibrated weights to disk."""
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._save_path, "w") as f:
            json.dump({
                "calibrated_at": result.calibrated_at,
                "method": "spearman_wf",
                "stage1_spearman": result.stage1_spearman,
                "stage2_wf_efficiency": result.stage2_wf_efficiency,
                "weights": result.weights,
                "search_space": result.search_space,
            }, f, indent=2)
        logger.info(f"Calibrated weights saved: {result.weights}")

    def load_weights(self) -> dict:
        """Load calibrated weights, falling back to defaults."""
        try:
            if self._save_path.exists():
                with open(self._save_path) as f:
                    data = json.load(f)
                return data.get("weights", DEFAULT_WEIGHTS)
        except Exception:
            pass
        return dict(DEFAULT_WEIGHTS)

    def _save_weights_temp(self, weights: dict):
        """Temporarily save weights for use by running GA processes."""
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._save_path, "w") as f:
            json.dump({"weights": weights, "temp": True}, f)

    @staticmethod
    def load_weights_static(data_dir: str) -> dict:
        """Static helper: load weights from disk, return defaults if not found."""
        path = Path(data_dir) / "data" / "ga_fitness_weights.json"
        try:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                return data.get("weights", DEFAULT_WEIGHTS)
        except Exception:
            pass
        return dict(DEFAULT_WEIGHTS)
