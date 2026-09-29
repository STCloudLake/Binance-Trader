"""Hybrid backtest engine entry point — orchestrates SignalMatrixBuilder + EventDrivenExecutor."""

import time
from pathlib import Path
from loguru import logger

from core.backtest.data_feeder import DataFeeder
from core.backtest.signal_matrix import NO_MARKET_DATA_MESSAGE, SignalMatrixBuilder
from core.backtest.event_executor import EventDrivenExecutor
from core.backtest.metrics import calculate_metrics
from core.risk.position_sizer import PositionSizer
from core.strategy.evaluation_kernel import detect_market_regime


def _detect_regimes(feeder, symbols: list[str], date_start: str) -> dict[str, str]:
    """Per-symbol market regime using data up to the first evaluated timestamp.

    Mirrors `BacktestEngine`'s one-shot detection so the regime-aware entry
    threshold behaves identically in both engines.
    """
    regimes: dict[str, str] = {}
    for sym in symbols:
        try:
            df_1h = feeder.get_all_data_for_symbol(sym, "1h")
        except Exception:
            regimes[sym] = "range"
            continue
        if df_1h is None or len(df_1h) == 0:
            regimes[sym] = "range"
            continue
        try:
            first_ts = feeder.first_timestamp
            window = df_1h[df_1h.index <= first_ts] if first_ts is not None else df_1h
        except Exception:
            window = df_1h
        regimes[sym] = detect_market_regime(window if len(window) else df_1h)
    return regimes


def run_hybrid(strategies, symbols, date_start, date_end,
               config, loader, initial_balance=10000.0,
               per_strategy_isolation=False,
               progress_callback=None) -> dict:
    """Run backtest using the hybrid (vectorized) engine.

    Args:
        strategies: List of StrategyConfig objects (NOT strategy name strings).
        symbols: List of symbol strings.
        date_start, date_end: Date range strings.
        config: Config instance.
        loader: StrategyLoader instance.
        initial_balance: Starting balance.
        per_strategy_isolation: Independent positions per strategy.
        progress_callback: (step, total, ts) called during executor loop.

    Returns:
        dict with keys: trades, equity_curve, metrics, final_balance,
                        per_matrix, strategies, symbols, date_start, date_end,
                        engine_mode="hybrid", metadata.
    """
    t0 = time.time()

    # Load strategy configs
    strategy_configs = []
    if isinstance(strategies, list) and strategies and not isinstance(strategies[0], str):
        strategy_configs = strategies
    else:
        for name in strategies:
            s = loader.load(name)
            strategy_configs.append(s)

    # Override ML
    for s in strategy_configs:
        if s.ml_config:
            s.ml_config.enabled = False

    strategy_names = [s.name for s in strategy_configs]

    # Determine intervals
    intervals = list(set(tf for s in strategy_configs for tf in s.timeframes)) or ["1h"]

    # Load data
    cache_dir = str(Path(config.data_dir) / "market")
    feeder = DataFeeder(cache_dir, symbols, intervals, date_start, date_end)
    feeder.load()

    if len(feeder) == 0:
        return {"error": NO_MARKET_DATA_MESSAGE}

    # Phase 1: Build signal matrix
    logger.info(f"Hybrid engine: building signal matrix for {len(strategy_configs)} strategies")

    # Market regime per symbol — detected exactly like the legacy engine (once,
    # from the data available before the first evaluated timestamp) so both
    # engines apply the same counter-trend penalty.
    regimes = _detect_regimes(feeder, symbols, date_start)

    builder = SignalMatrixBuilder(feeder)
    matrix = builder.build(strategy_configs, symbols,
                           signal_weights=getattr(config, "signal_weights", None),
                           regimes=regimes)
    if matrix.metadata.get("error"):
        # No candles on the primary timeframe: return the unified, actionable
        # message instead of executing an empty matrix as a "successful" run.
        return {"error": matrix.metadata["error"]}
    logger.info(f"Signal matrix built in {matrix.metadata['build_time_seconds']}s: "
                f"{matrix.metadata['total_signals']} signals, "
                f"{matrix.metadata['timestamp_count']} timestamps, "
                f"{matrix.metadata['indicator_groups']} indicator groups")

    # Phase 2: Execute trades
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)
    max_positions = config.hard_limits.max_open_trades
    if per_strategy_isolation:
        max_positions = max(1, max_positions // max(len(strategy_configs), 1))

    # Cost model config tuple: (enabled, fee_pct, spread_dict)
    cost_cfg = (getattr(config, 'backtest_cost_enabled', True),
                getattr(config, 'backtest_taker_fee_pct', 0.04),
                getattr(config, 'backtest_spread_pct', {}))

    executor = EventDrivenExecutor(
        sizer, config.hard_limits,
        per_strategy_isolation=per_strategy_isolation,
        max_positions=max_positions,
        cost_config=cost_cfg,
        strategy_risk={
            s.name: {
                "max_hold_hours": s.risk_exit.max_hold_hours if s.risk_exit else 0,
                "use_indicator_exits": s.risk_exit.use_indicator_exits if s.risk_exit else True,
                # Also propagate the per-strategy stop/trailing distances: the
                # executor needs them to mirror the legacy engine exactly, and the
                # Kelly-lite sizing branch is dead without stop_loss_pct.
                "stop_loss_pct": (s.risk_exit.stop_loss_pct if s.risk_exit else None),
                "trailing_stop_pct": (s.risk_exit.trailing_stop_pct if s.risk_exit else None),
                # Indicator exits are evaluated on every configured timeframe.
                "timeframes": list(s.timeframes or []),
            }
            for s in strategy_configs
        },
    )

    logger.info(f"Hybrid engine: executing trades ({matrix.metadata['timestamp_count']} ticks)")
    exec_result = executor.run(matrix, initial_balance=initial_balance,
                               progress_callback=progress_callback)

    # Phase 3: Calculate metrics
    metrics = calculate_metrics(exec_result.trades, exec_result.equity_curve,
                                initial_balance, exec_result.final_balance)
    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    metrics["engine_mode"] = "hybrid"
    metrics["signal_matrix_build_seconds"] = matrix.metadata["build_time_seconds"]

    logger.info(f"Hybrid engine: {len(exec_result.trades)} trades, "
                f"final_balance={exec_result.final_balance:.2f}, "
                f"runtime={metrics['runtime_seconds']}s")

    return {
        "trades": exec_result.trades,
        "equity_curve": exec_result.equity_curve,
        "metrics": metrics,
        "final_balance": exec_result.final_balance,
        "initial_balance": initial_balance,
        "strategies": strategy_names,
        "symbols": symbols,
        "date_start": date_start,
        "date_end": date_end,
        "per_matrix": exec_result.per_matrix,
        "engine_mode": "hybrid",
    }
