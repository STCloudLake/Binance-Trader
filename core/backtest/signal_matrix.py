"""Signal matrix builder — vectorized condition evaluation for batch backtests."""

import hashlib
import json
import time
from dataclasses import dataclass

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
    price_data: dict
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
        all_tfs = set()
        for s in strategies:
            all_tfs.update(s.timeframes)
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
            sym_data: dict[str, pd.DataFrame] = {}
            for tf in all_tfs:
                df = self.feeder.get_all_data_for_symbol(sym, tf)
                if len(df) > 0:
                    sym_data[tf] = df
                    if tf == finest_tf or tf not in price_data[sym]:
                        price_data[sym][tf] = df[["open", "high", "low", "close", "volume"]].copy()

            # ── Compute indicators per group ──
            indicator_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
            for group_idx, group in enumerate(groups):
                rep = group[0]
                config_hash = self.grouper._config_hash(rep)
                for tf in rep.timeframes:
                    raw_df = sym_data.get(tf)
                    if raw_df is None or len(raw_df) < 20:
                        continue
                    cache_key = (config_hash, sym, tf)
                    if cache_key not in indicator_cache:
                        indicator_cache[cache_key] = compute_all(raw_df.copy(), rep.indicators)

            # ── Evaluate all conditions for this symbol ──
            all_conditions: dict[str, list[tuple[str, str, str]]] = {}
            for s in strategies:
                primary_tf = s.timeframes[0] if s.timeframes else "1h"
                for side in ("long", "short"):
                    for cond in s.entry_conditions.get(side, []):
                        all_conditions.setdefault(cond, []).append((s.name, primary_tf, side))
                    for cond in s.exit_conditions.get(side, []):
                        all_conditions.setdefault(cond, []).append((s.name, primary_tf, f"exit_{side}"))

            condition_results: dict[str, dict[tuple[str, str], pd.Series]] = {}
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
                        condition_results[cond_str][(config_hash, tf)] = result

            # ── Build signal rows for this symbol ──
            entry_rows = []
            exit_rows = []

            for s in strategies:
                primary_tf = s.timeframes[0] if s.timeframes else "1h"
                config_hash = self.grouper._config_hash(s)

                # Long entry: AND all long entry conditions
                long_conds = s.entry_conditions.get("long", [])
                if long_conds:
                    long_signals = None
                    for cond_str in long_conds:
                        result = condition_results.get(cond_str, {}).get((config_hash, primary_tf))
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
                        result = condition_results.get(cond_str, {}).get((config_hash, primary_tf))
                        if result is not None:
                            aligned = result.reindex(timestamps, fill_value=False)
                            if short_signals is None:
                                short_signals = aligned.astype(bool)
                            else:
                                short_signals = short_signals & aligned.astype(bool)
                    if short_signals is not None and short_signals.any():
                        short_series = short_signals.astype(int) * -1
                        # Check if there's already a long row for this (strategy, sym, tf)
                        combined = short_series.copy()
                        for i, r in enumerate(entry_rows):
                            if r.name == (s.name, sym, primary_tf):
                                existing = r.astype(int)
                                combined[existing == 1] = 0  # both true → 0
                                combined[existing == 0] = short_series[existing == 0]
                                entry_rows.pop(i)
                                break
                        if combined.sum() != 0:
                            row = pd.Series(combined, index=timestamps, dtype="int8")
                            row.name = (s.name, sym, primary_tf)
                            entry_rows.append(row)

                # Exit signals
                for side in ("long", "short"):
                    exit_conds = s.exit_conditions.get(side, [])
                    if exit_conds:
                        exit_sig = None
                        for cond_str in exit_conds:
                            result = condition_results.get(cond_str, {}).get((config_hash, primary_tf))
                            if result is not None:
                                aligned = result.reindex(timestamps, fill_value=False)
                                if exit_sig is None:
                                    exit_sig = aligned.astype(bool)
                                else:
                                    exit_sig = exit_sig | aligned.astype(bool)
                        if exit_sig is not None and exit_sig.sum() > 0:
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
