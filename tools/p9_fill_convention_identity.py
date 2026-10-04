"""P9 bit-identity harness — the same run, two trees, one digest.

This file is BOTH a runnable tool and the harness the test
``tests/test_fill_convention.py::test_close_is_bit_identical_to_the_pre_p9_worktree``
copies into a temp dir and executes inside a ``git worktree`` checked out at the
pre-P9 revision.  One source of truth, so the proof in the evidence document and
the standing test can never drift apart.

What it does
------------
Runs ONE deterministic backtest with an **explicitly selected fill convention**
(``--fill-convention``, default ``close``: the historical convention whose
bit-identity with the pre-P9 engine is the claim being checked — the shipped
default is now ``next_open``, which by construction produces different bytes) and
writes every *traded fact* to a JSON payload, then prints its ``sha256``:

* every trade's signal/exit timestamps, prices, PnL and cost;
* every per-genome equity point (the GA ledger shape);
* a metrics digest over every metric key **that existed before P9**
  (``fill_convention`` / ``fill_convention_accounting`` are excluded, because the
  baseline cannot have them — their presence is asserted separately).

The convention is requested through the ``fill_convention=`` kwarg where the
target tree's engine has that parameter, and through
``config.backtest_fill_convention`` where it does not (the pre-P9 tree has
neither; there ``close`` is the only behaviour there is, so the requested value is
asserted to be ``close`` and nothing is passed).  Two trees producing the same
digest means the selected path is bit-identical: not "equivalent", the same bytes.

Usage::

    python tools/p9_fill_convention_identity.py --tree . --data-dir data --out %TEMP%\\id.json
    python tools/p9_fill_convention_identity.py --tree <worktree> --data-dir data --out %TEMP%\\id_head.json
    python tools/p9_fill_convention_identity.py --tree . --fill-convention next_open --out %TEMP%\\id_next.json
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

#: Metrics added by P9 itself — excluded from the digest so a baseline tree (which
#: cannot have them) still reaches the comparison instead of failing on a key set.
P9_METRIC_KEYS = ("fill_convention", "fill_convention_accounting")

#: Fixed metric key list: a *named* subset, so "the metrics digest" can never
#: silently shrink (an empty intersection would compare equal and prove nothing).
#: ``runtime_seconds`` is deliberately absent — it is wall-clock, not a result.
METRIC_KEYS = (
    "total_return_pct", "annualized_return_pct", "sharpe_ratio", "sortino_ratio",
    "calmar_ratio", "max_drawdown_pct", "profit_factor", "win_rate_pct",
    "total_trades", "avg_pnl", "avg_hold_minutes", "max_consecutive_losses",
    "recovery_factor", "omega_ratio", "tail_ratio", "var_95_daily_pct",
    "cvar_95_daily_pct", "var_99_daily_pct", "cvar_99_daily_pct", "days",
    "buy_hold_pct", "spread_sources", "spread_pct",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True,
                        help="repository tree whose code must be imported")
    parser.add_argument("--data-dir", default=None,
                        help="data dir (the parquet cache lives in <data-dir>/market)")
    parser.add_argument("--symbols", default="BTCUSDT")
    parser.add_argument("--start", default="2026-01-05")
    parser.add_argument("--end", default="2026-02-01")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--fill-convention", default="close",
                        help="explicit convention for this run (default: close, "
                             "the historical convention this harness proves "
                             "reproducible; the shipped default is next_open)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tree = str(Path(args.tree).resolve())
    sys.path.insert(0, tree)

    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager
    from core.strategy.loader import MLConfig, RiskExitConfig, StrategyConfig

    Config._instance = None
    cfg = Config.load("sim")
    if args.data_dir:
        cfg.data_dir = args.data_dir
    cfg.backtest_engine_mode = "legacy"
    cfg.backtest_ml_enabled = False
    cfg.backtest_live_spread_enabled = False
    # EXPLICIT convention.  The shipped default is `next_open`; the claim checked
    # here is the `close` path's bit-identity with the pre-P9 engine, so the
    # historical convention is requested by name rather than inherited.  A tree
    # with the P9 seam honours this; a pre-P9 tree ignores the attribute (there
    # `close` is the only behaviour that exists).
    cfg.backtest_fill_convention = args.fill_convention
    bus = EventBus()
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus),
                            OrderExecutor(cfg, bus))

    # Passed as the run kwarg where the tree's signature has the parameter (the
    # strongest form of "explicit"); a pre-P9 tree cannot accept it, and there the
    # request must be `close` — the convention it already prices.
    run_kwargs = {}
    if "fill_convention" in inspect.signature(
            engine.run_with_exit_evaluation).parameters:
        run_kwargs["fill_convention"] = args.fill_convention
    elif args.fill_convention != "close":
        raise SystemExit(
            f"this tree has no fill-convention seam: {args.fill_convention!r} "
            "cannot be run here (only the historical 'close')")

    strategy = StrategyConfig(
        name="p9_identity", enabled=True, mode="trend",
        timeframes=[args.timeframe],
        indicators={"rsi": {"period": 14, "source": "close"},
                    "sma": {"period": 5}},
        entry_conditions={"long": ["close > sma"], "short": ["close < sma"]},
        exit_conditions={"long": ["rsi > 99"], "short": ["rsi < 1"]},
        risk_exit=RiskExitConfig(stop_loss_pct=1.5, trailing_stop_pct=1.0,
                                 max_hold_hours=24.0, use_indicator_exits=True),
        ml_config=MLConfig(enabled=False))

    result = engine.run_with_exit_evaluation(
        strategies=[strategy], symbols=[args.symbols],
        date_start=args.start, date_end=args.end, initial_balance=10_000.0,
        mode="full", simulate_ai_weights=False, per_strategy_isolation=True,
        per_genome_ledger=True, use_live_spread=False, benchmark_mode="none",
        **run_kwargs)

    metrics = result.get("metrics") or {}

    # The convention the run actually resolved is part of the PROOF, not of the
    # compared payload (the baseline tree cannot report it) — printed on stderr so
    # a reader of this tool's output can see the historical path was selected.
    resolved = (result.get("fill_convention")
                or (metrics.get("fill_convention_accounting") or {}).get("convention")
                or "unreported (pre-P9 tree)")
    sys.stderr.write(
        f"resolved fill_convention = {resolved} "
        f"(requested {args.fill_convention!r})\n")

    payload = {
        "trades": [{"opened_at": str(t.get("opened_at")),
                    "closed_at": str(t.get("closed_at")),
                    "entry_price": t.get("entry_price"),
                    "exit_price": t.get("exit_price"),
                    "quantity": t.get("quantity"),
                    "pnl": t.get("pnl"), "cost": t.get("cost"),
                    "exit_reason": t.get("exit_reason")}
                   for t in (result.get("trades") or [])],
        "equity": {name: [point["equity"] for point in
                          (entry.get("equity_curve") or [])]
                   for name, entry in
                   (result.get("per_strategy_equity") or {}).items()},
        "metrics": {key: metrics.get(key) for key in METRIC_KEYS},
        "metrics_missing_from_tree": [key for key in METRIC_KEYS
                                      if key not in metrics],
        "trade_count": len(result.get("trades") or []),
    }
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sys.stdout.write(digest + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
