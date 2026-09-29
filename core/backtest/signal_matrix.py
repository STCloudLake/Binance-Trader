"""Signal matrix builder — vectorized condition evaluation for batch backtests."""

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from core.strategy.indicators import compute_all, evaluate_condition
from core.strategy.evaluation_kernel import build_entry_signals
from core.strategy.loader import StrategyConfig
from core.market_data.provider import DEFAULT_TIMEFRAME, interval_minutes


#: Canonical "nothing to backtest" message — the SINGLE text every engine path
#: returns (legacy, hybrid and the signal-matrix builder) and the one the web
#: result card shows.  It keeps the legacy "No historical data ..." wording (the
#: phrase callers test for) and names the fix, because an empty data directory is
#: an operator problem, not a crash: the user must download candles first.
NO_MARKET_DATA_MESSAGE = (
    "No historical data found for the selected symbols and date range. "
    "Download candles first with `python scripts/download_history.py`, "
    "or use POST /api/backtest/fetch-data (Data page), then run the backtest again."
)


def _tf_minutes(tf: str) -> int:
    """Bar length in minutes, from the shared interval registry.

    Timeframe ordering (shortest = primary) used to live in a private
    ``_TF_MINUTES`` copy here; ``INTERVAL_SPEC`` in ``core.market_data.provider``
    is now the only place an interval is declared.
    """
    return interval_minutes(tf, 60)


def _primary_timeframe(strategy: StrategyConfig) -> str:
    """Shortest configured timeframe (the one that produces entry signals)."""
    tfs = list(strategy.timeframes or [])
    if not tfs:
        return DEFAULT_TIMEFRAME
    return min(tfs, key=_tf_minutes)



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
              symbols: list[str],
              signal_weights=None,
              regimes: dict | None = None) -> SignalMatrix:
        """Build the complete signal matrix for all strategies × symbols × timeframes.

        Args:
            strategies: Strategy configs to evaluate.
            symbols: Symbols to evaluate.
            signal_weights: ``SignalWeights`` used for indicator/ML/news fusion.
                Defaults to 1.0/0.0/0.0 when omitted (pure indicator signals),
                which keeps the matrix self-consistent for unit tests.
            regimes: Optional per-symbol market regime ("bull"/"bear"/"range")
                used by the regime-aware entry threshold.
        """
        t0 = time.time()

        w_indicator = float(getattr(signal_weights, "indicator", 1.0)) if signal_weights else 1.0
        w_ml = float(getattr(signal_weights, "ml", 0.0)) if signal_weights else 0.0
        w_news = float(getattr(signal_weights, "news", 0.0)) if signal_weights else 0.0
        regimes = regimes or {}

        # ── Group strategies by indicator config ──
        groups = self.grouper.group(strategies)
        logger.info(f"SignalMatrix: {len(strategies)} strategies → "
                    f"{len(groups)} unique indicator groups")

        # ── Determine unified timestamp index ──
        all_tfs = set()
        for s in strategies:
            all_tfs.update(s.timeframes)
        finest_tf = min(all_tfs, key=_tf_minutes)
        # The timeline is the UNION of every symbol's finest-timeframe index, not
        # `symbols[0]`'s index.  A symbol listed later than the first one used to
        # truncate the shared index (all of the other symbols' earlier, tradable
        # timestamps were silently dropped) and, when the first symbol had no data
        # at all, broke the run outright.  Symbols that have no bar at a given
        # timestamp simply contribute no signal there (they are reindexed with
        # fill_value=False below).
        union_index: pd.Index | None = None
        for sym in symbols:
            df = self.feeder.get_all_data_for_symbol(sym, finest_tf)
            if df is None or len(df) == 0:
                continue
            # The frame carries warm-up rows before date_start (needed for correct
            # indicators); only post-start timestamps are tradable.
            sym_times = df.index[df.index >= self.feeder.date_start]
            if len(sym_times) == 0:
                sym_times = df.index
            union_index = (sym_times if union_index is None
                           else union_index.union(sym_times))

        if union_index is None:
            # No symbol has data for the primary timeframe.  Fall back to the
            # first symbol's frame when there is one, so the caller still gets
            # the unified "no market data" result rather than a KeyError.
            base_df = (self.feeder.get_all_data_for_symbol(symbols[0], finest_tf)
                       if symbols else None)
            if base_df is None:
                base_df = pd.DataFrame()
            index = base_df.index
            # A symbol/interval with no parquet behind it carries a plain
            # RangeIndex, and comparing that with a Timestamp raises TypeError;
            # only a datetime index can be filtered against date_start.
            if isinstance(index, pd.DatetimeIndex):
                timestamps = index[index >= self.feeder.date_start]
                if len(timestamps) == 0:
                    timestamps = index
            else:
                timestamps = index
        else:
            timestamps = union_index

        if len(timestamps) == 0:
            # A run with no candles is a NORMAL outcome, not an exception: it used
            # to raise ``ValueError("No timestamps found in data feeder")``, which
            # the web layer could only surface as a raw message.  Returning an
            # empty matrix tagged with the unified, actionable message lets every
            # caller (hybrid engine → result card) report the same fix.
            logger.warning(NO_MARKET_DATA_MESSAGE)
            return SignalMatrix(
                signals=pd.DataFrame(),
                exit_signals=pd.DataFrame(),
                price_data={},
                metadata={
                    "error": NO_MARKET_DATA_MESSAGE,
                    "build_time_seconds": round(time.time() - t0, 2),
                    "strategy_count": len(strategies),
                    "symbol_count": len(symbols),
                    "indicator_groups": len(groups),
                    "total_signals": 0,
                    "timestamp_count": 0,
                },
            )

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
            # NOTE: use the UNION of the group's timeframes, not just the group
            # representative's. Strategies sharing an indicator config but using a
            # different timeframe used to get no condition results at all and were
            # silently dropped from the matrix (no entries, no exits, and a
            # zero-trade per_matrix row that GA scored as fitness -20).
            indicator_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
            group_timeframes: dict[str, list[str]] = {}
            for group in groups:
                rep = group[0]
                config_hash = self.grouper._config_hash(rep)
                tfs = sorted({tf for s in group for tf in (s.timeframes or [])},
                             key=_tf_minutes)
                group_timeframes[config_hash] = tfs
                for tf in tfs:
                    raw_df = sym_data.get(tf)
                    if raw_df is None or len(raw_df) < 20:
                        continue
                    cache_key = (config_hash, sym, tf)
                    if cache_key not in indicator_cache:
                        indicator_cache[cache_key] = compute_all(raw_df.copy(), rep.indicators)

            # ── Evaluate all conditions for this symbol ──
            all_conditions: dict[str, list[tuple[str, str, str]]] = {}
            for s in strategies:
                primary_tf = _primary_timeframe(s)
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
                    for tf in group_timeframes.get(config_hash, []):
                        df = indicator_cache.get((config_hash, sym, tf))
                        if df is None:
                            continue
                        result = evaluate_condition(df, cond_str)
                        condition_results[cond_str][(config_hash, tf)] = result

            # ── Build signal rows for this symbol ──
            entry_rows = []
            exit_rows = []

            for s in strategies:
                primary_tf = _primary_timeframe(s)
                config_hash = self.grouper._config_hash(s)

                # ── Entry signals — MUST use the shared kernel semantics ──
                # (OR conditions + weighted fusion + regime-aware threshold + HTF
                # alignment). The previous AND-logic implementation made the
                # vectorized engine behave differently from live trading, which is
                # exactly the divergence the equivalence gate test guards against.
                def _or_conditions(side: str):
                    series = None
                    for cond_str in s.entry_conditions.get(side, []):
                        result = condition_results.get(cond_str, {}).get((config_hash, primary_tf))
                        if result is None:
                            continue
                        aligned = result.reindex(timestamps, fill_value=False).astype(bool)
                        series = aligned if series is None else (series | aligned)
                    return series

                long_active = _or_conditions("long")
                short_active = _or_conditions("short")

                if long_active is not None or short_active is not None:
                    htf_frames = []
                    primary_min = _tf_minutes(primary_tf)
                    for tf in sorted(s.timeframes, key=_tf_minutes):
                        if _tf_minutes(tf) > primary_min and tf in sym_data:
                            htf_frames.append(sym_data[tf])

                    ml_enabled = bool(s.ml_config and s.ml_config.enabled)
                    signal_series = build_entry_signals(
                        long_active, short_active,
                        index=timestamps,
                        w_indicator=w_indicator,
                        w_ml=w_ml,
                        w_news=w_news,
                        ml_enabled=ml_enabled,
                        strategy_ml_weight=(s.ml_config.weight if ml_enabled else None),
                        htf_frames=htf_frames,
                        regime=regimes.get(sym, "range"),
                    )
                    if int(signal_series.abs().sum()) != 0:
                        row = pd.Series(signal_series.values, index=timestamps, dtype="int8")
                        row.name = (s.name, sym, primary_tf)
                        entry_rows.append(row)

                # Exit signals — evaluated on EVERY timeframe of the strategy, not
                # just the primary one. The legacy engine checks each configured
                # timeframe and exits if any of them fires (engine.py "INDICATOR
                # EXITS" loop), so restricting the matrix to the primary timeframe
                # made multi-timeframe strategies exit at different times in the
                # two engines (389 vs 207 trades on the standard GA shape).
                for tf in sorted(s.timeframes or [primary_tf],
                                 key=_tf_minutes):
                    for side in ("long", "short"):
                        exit_conds = s.exit_conditions.get(side, [])
                        if not exit_conds:
                            continue
                        exit_sig = None
                        for cond_str in exit_conds:
                            result = condition_results.get(cond_str, {}).get((config_hash, tf))
                            if result is None:
                                continue
                            # Legacy reads the last CLOSED bar of that timeframe at
                            # each tick (`df[df.index <= ts]`), i.e. a forward-fill of
                            # the timeframe's own series onto the trading timeline.
                            aligned = result.reindex(timestamps, method="ffill")
                            aligned = aligned.fillna(False).astype(bool)
                            exit_sig = aligned if exit_sig is None else (exit_sig | aligned)
                        if exit_sig is not None and exit_sig.sum() > 0:
                            row = pd.Series(exit_sig, index=timestamps, dtype=bool)
                            row.name = (s.name, sym, tf, f"exit_{side}")
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

        # Order rows strategy-major, then symbol — the same order the legacy engine
        # iterates entries in. Position sizing depends on the running balance, so a
        # different within-tick order silently produced different position sizes
        # (and therefore different PnL) between the two engines.
        if not signals.empty:
            s_order = {s.name: i for i, s in enumerate(strategies)}
            sym_order = {s: i for i, s in enumerate(symbols)}
            signals = signals.reindex(sorted(
                signals.index,
                key=lambda k: (s_order.get(k[0], 10**6), sym_order.get(k[1], 10**6),
                               _tf_minutes(k[2])),
            ))
        if not exit_signals.empty:
            s_order = {s.name: i for i, s in enumerate(strategies)}
            sym_order = {s: i for i, s in enumerate(symbols)}
            exit_signals = exit_signals.reindex(sorted(
                exit_signals.index,
                key=lambda k: (s_order.get(k[0], 10**6), sym_order.get(k[1], 10**6),
                               _tf_minutes(k[2]), k[3]),
            ))

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
