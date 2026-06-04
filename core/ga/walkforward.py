"""Rolling walk-forward validation — multi-window GA optimization and validation.

Each window: train GA on [T_start, T_end], validate champion on [T_end, T_end+N].
Windows roll forward by step_months, producing N independent champion evaluations.
The aggregate WF report measures strategy stability across different market regimes.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class WFConfig:
    """Walk-forward configuration."""
    enabled: bool = True
    train_months: int = 6
    val_months: int = 1
    step_months: int = 1


@dataclass
class WindowResult:
    """Result of a single walk-forward window."""
    window: int
    total: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_sharpe: float = 0.0
    val_sharpe: float = 0.0
    val_win_rate: float = 0.0
    val_total_return: float = 0.0
    champion_name: str = ""


@dataclass
class WFReport:
    """Aggregate walk-forward validation report."""
    windows: list = field(default_factory=list)
    mean_val_sharpe: float = 0.0
    std_val_sharpe: float = 0.0
    min_val_sharpe: float = 0.0
    max_val_sharpe: float = 0.0
    wf_efficiency: float = 0.0       # mean / std (higher = more stable)
    positive_window_pct: float = 0.0  # % of windows with positive val Sharpe
    train_val_correlation: float = 0.0  # Pearson r between train & val Sharpe
    best_champion_name: str = ""
    best_val_sharpe: float = 0.0
    best_window: int = 0
    elapsed_seconds: float = 0.0

    @classmethod
    def from_results(cls, results: list[WindowResult], elapsed: float) -> "WFReport":
        """Build aggregate report from per-window results."""
        if not results:
            return cls(elapsed_seconds=round(elapsed, 1))

        val_sharpes = [r.val_sharpe for r in results]
        train_sharpes = [r.train_sharpe for r in results]
        n = len(val_sharpes)

        mean_vs = float(np.mean(val_sharpes))
        std_vs = float(np.std(val_sharpes, ddof=1)) if n > 1 else 0.0
        wf_eff = mean_vs / std_vs if std_vs > 0 else 0.0
        pos_pct = sum(1 for s in val_sharpes if s > 0) / n * 100

        # Train-val correlation (need >= 3 windows for meaningful correlation)
        if n >= 3:
            corr = float(np.corrcoef(train_sharpes, val_sharpes)[0, 1])
            corr = 0.0 if np.isnan(corr) else corr
        else:
            corr = 0.0

        best_idx = int(np.argmax(val_sharpes))
        best = results[best_idx]

        return cls(
            windows=results,
            mean_val_sharpe=round(mean_vs, 4),
            std_val_sharpe=round(std_vs, 4),
            min_val_sharpe=round(float(np.min(val_sharpes)), 4),
            max_val_sharpe=round(float(np.max(val_sharpes)), 4),
            wf_efficiency=round(wf_eff, 4),
            positive_window_pct=round(pos_pct, 1),
            train_val_correlation=round(corr, 4),
            best_champion_name=best.champion_name,
            best_val_sharpe=best.val_sharpe,
            best_window=best.window,
            elapsed_seconds=round(elapsed, 1),
        )

    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return {
            "windows": [
                {
                    "window": w.window, "total": w.total,
                    "train_start": w.train_start, "train_end": w.train_end,
                    "val_start": w.val_start, "val_end": w.val_end,
                    "train_sharpe": w.train_sharpe, "val_sharpe": w.val_sharpe,
                    "val_win_rate": w.val_win_rate,
                    "val_total_return": w.val_total_return,
                    "champion_name": w.champion_name,
                }
                for w in self.windows
            ],
            "mean_val_sharpe": self.mean_val_sharpe,
            "std_val_sharpe": self.std_val_sharpe,
            "min_val_sharpe": self.min_val_sharpe,
            "max_val_sharpe": self.max_val_sharpe,
            "wf_efficiency": self.wf_efficiency,
            "positive_window_pct": self.positive_window_pct,
            "train_val_correlation": self.train_val_correlation,
            "best_champion_name": self.best_champion_name,
            "best_val_sharpe": self.best_val_sharpe,
            "best_window": self.best_window,
            "elapsed_seconds": self.elapsed_seconds,
        }


class WalkForwardRunner:
    """Execute rolling walk-forward GA validation.

    Usage:
        runner = WalkForwardRunner(engine, loader, data_dir)
        wf_cfg = WFConfig(train_months=6, val_months=1, step_months=1)
        report = runner.run(symbols, "2025-01-01", "2026-01-01", wf_cfg, ga_cfg)
    """

    def __init__(self, engine, loader, data_dir: str):
        self.engine = engine
        self.loader = loader
        self.data_dir = Path(data_dir)
        self._state_path = self.data_dir / "data" / "ga_wf_state.json"
        self._running = False

    def compute_windows(self, date_start: str, date_end: str,
                        cfg: WFConfig) -> list[tuple[str, str, str, str]]:
        """Compute rolling window boundaries.

        Returns list of (train_start, train_end, val_start, val_end) tuples.
        """
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

    def run(self, symbols: list[str], date_start: str, date_end: str,
            wf_config: WFConfig, ga_config,
            seed_strategies: list[str] | None = None,
            resume: bool = False) -> WFReport:
        """Execute full walk-forward validation.

        Args:
            symbols: Trading symbols.
            date_start, date_end: Full date range to walk-forward across.
            wf_config: Walk-forward window configuration.
            ga_config: GARunConfig for per-window GA runs.
            seed_strategies: Optional seed strategies for each window's GA.
            resume: If True, try to resume from saved WF state.

        Returns:
            WFReport with per-window results and aggregate metrics.
        """
        from core.ga.evolver import GAStrategyEvolver

        windows = self.compute_windows(date_start, date_end, wf_config)
        if not windows:
            logger.warning("WF: no valid windows for the given date range")
            return WFReport()

        results: list[WindowResult] = []
        start_window = 0

        if resume:
            saved = self._load_state()
            if saved:
                for r in saved.get("completed", []):
                    results.append(WindowResult(**r))
                start_window = saved.get("current_window", 0)
                logger.info(f"WF resume: starting at window {start_window + 1}/{len(windows)}")

        t0 = time.time()
        self._running = True

        for i in range(start_window, len(windows)):
            if not self._running:
                break

            tr_start, tr_end, val_start, val_end = windows[i]
            logger.info(
                f"WF window {i + 1}/{len(windows)}: "
                f"train={tr_start}~{tr_end}, val={val_start}~{val_end}"
            )

            evolver = GAStrategyEvolver(self.engine, self.loader, ga_config)
            champion = evolver.evolve(
                symbols, tr_start, tr_end,
                seed_strategies=seed_strategies,
                validation_start=val_start,
            )

            val_data = champion.get("validation", {}) or {}
            result = WindowResult(
                window=i + 1,
                total=len(windows),
                train_start=tr_start,
                train_end=tr_end,
                val_start=val_start,
                val_end=val_end,
                train_sharpe=champion.get("sharpe", 0),
                val_sharpe=val_data.get("sharpe", 0),
                val_win_rate=val_data.get("win_rate", 0),
                val_total_return=val_data.get("total_return", 0),
                champion_name=champion.get("champion_name", ""),
            )
            results.append(result)
            self._save_state(i + 1, results)

            logger.info(
                f"WF window {i + 1} done: "
                f"train_sharpe={result.train_sharpe:.2f}, "
                f"val_sharpe={result.val_sharpe:.2f}"
            )

        self._running = False
        elapsed = time.time() - t0
        self._clear_state()

        report = WFReport.from_results(results, elapsed)
        logger.info(
            f"WF complete: {len(results)} windows, "
            f"mean_val_sharpe={report.mean_val_sharpe:.2f}, "
            f"wf_efficiency={report.wf_efficiency:.2f}"
        )
        return report

    def stop(self):
        """Signal the runner to stop after the current window completes."""
        self._running = False

    def get_state(self) -> dict | None:
        """Return current WF state for status polling."""
        return self._load_state()

    # ── State persistence ──────────────────────────────────────────

    def _save_state(self, current_window: int, results: list[WindowResult]):
        """Persist WF progress for resume after stop/crash."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "current_window": current_window,
                "total_windows": len(results) + max(0, current_window - len(results)),
                "completed": [
                    {
                        "window": r.window, "train_sharpe": r.train_sharpe,
                        "val_sharpe": r.val_sharpe, "champion_name": r.champion_name,
                        "train_start": r.train_start, "train_end": r.train_end,
                        "val_start": r.val_start, "val_end": r.val_end,
                        "val_win_rate": r.val_win_rate,
                        "val_total_return": r.val_total_return,
                        "total": r.total,
                    }
                    for r in results
                ],
                "stopped": not self._running,
            }
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"WF state save failed: {e}")

    def _load_state(self) -> dict | None:
        """Load saved WF state. Returns None if no state exists."""
        try:
            if self._state_path.exists():
                with open(self._state_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _clear_state(self):
        """Remove state file after successful completion."""
        try:
            if self._state_path.exists():
                self._state_path.unlink()
        except Exception:
            pass
