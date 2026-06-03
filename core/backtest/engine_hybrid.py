"""Hybrid backtest engine entry point — orchestrates SignalMatrixBuilder + EventDrivenExecutor."""

import time
from pathlib import Path
from loguru import logger

from core.backtest.data_feeder import DataFeeder
from core.backtest.signal_matrix import SignalMatrixBuilder
from core.backtest.event_executor import EventDrivenExecutor
from core.backtest.metrics import calculate_metrics
from core.risk.position_sizer import PositionSizer


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
        return {"error": "No historical data found"}

    # Phase 1: Build signal matrix
    logger.info(f"Hybrid engine: building signal matrix for {len(strategy_configs)} strategies")
    builder = SignalMatrixBuilder(feeder)
    matrix = builder.build(strategy_configs, symbols)
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
