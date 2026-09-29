"""Tests for `core/ai/strategy_lifecycle.py:StrategyLifecycleManager`.

The manager is instantiated with stub collaborators (AI, backtest engine,
strategy engine) and a real `StrategyLoader` pointed at a temp directory, so no
network call, no real strategy file and no production DB is ever touched.

What is pinned here:
  * `log_event()` round-trips through `strategy_lifecycle_events`
  * the retirement decision matrix, driven by the REAL thresholds in
    `_evaluate_for_retirement` (trades == 0; sharpe < -1.0; win_rate < 30;
    max_drawdown_pct > 30) including the exact boundary values
  * `generate_strategy()` on valid / invalid / empty / rate-limited AI output
  * `validate_and_deploy()` saves on pass, cleans up on failure, and never
    corrupts pre-existing strategy files
"""
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
import pytest

from core.ai.strategy_lifecycle import StrategyLifecycleManager
from core.strategy.loader import MLConfig, StrategyConfig, StrategyLoader
from db.database import init_database

PROD_DB = Path(__file__).resolve().parents[1] / "data" / "binance_trader.db"

VALID_AI_CONFIG = {
    "name": "ai_valid_strategy",
    "enabled": True,
    "mode": "trend",
    "timeframes": ["1h"],
    "symbols": ["BTCUSDT"],
    "indicators": {"rsi": {"period": 14}},
    "entry_conditions": {"long": ["rsi < 30"], "short": ["rsi > 70"]},
    "exit_conditions": {"long": ["rsi > 70"], "short": ["rsi < 30"]},
    "reduce_conditions": {},
    "ml_config": {"enabled": False},
}
VALID_AI_JSON = json.dumps(VALID_AI_CONFIG)


# ── collaborators ────────────────────────────────────────────────────────

class FakeConfig:
    """Only the surface StrategyLifecycleManager actually reads."""

    def __init__(self, ai_mode="semi_auto"):
        self.ai_mode = ai_mode


class StubDeepSeek:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def _call_deepseek(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        return self.response


class StubBacktestEngine:
    """Sync `run()` — the manager calls it through run_in_executor."""

    def __init__(self, result=None, by_strategy=None):
        self.result = result if result is not None else {}
        self.by_strategy = by_strategy or {}
        self.calls: list[dict] = []

    def run(self, strategies, symbols, date_start, date_end, mode, initial_balance):
        self.calls.append({
            "strategies": list(strategies), "symbols": list(symbols),
            "date_start": date_start, "date_end": date_end,
            "mode": mode, "initial_balance": initial_balance,
        })
        if self.by_strategy:
            for name in strategies:
                if name in self.by_strategy:
                    return self.by_strategy[name]
            return {"error": f"no stub metrics for {strategies}"}
        return self.result


class StubStrategyEngine:
    def __init__(self):
        self._strategies: dict = {}


@dataclass
class Harness:
    manager: StrategyLifecycleManager
    loader: StrategyLoader
    db_path: str
    deepseek: StubDeepSeek
    backtest: StubBacktestEngine
    engine: StubStrategyEngine
    strategies_dir: Path


async def _make_harness(tmp_path, ai_response=None, ai_error=None,
                        backtest_result=None, backtest_by_strategy=None,
                        ai_mode="semi_auto") -> Harness:
    db_path = str(tmp_path / "lifecycle_test.db")
    await init_database(db_path)
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    loader = StrategyLoader(str(strategies_dir))
    deepseek = StubDeepSeek(response=ai_response, error=ai_error)
    backtest = StubBacktestEngine(result=backtest_result,
                                  by_strategy=backtest_by_strategy)
    engine = StubStrategyEngine()
    manager = StrategyLifecycleManager(
        config=FakeConfig(ai_mode),
        deepseek_ctl=deepseek,
        backtest_engine=backtest,
        strategy_loader=loader,
        strategy_engine=engine,
        alert_manager=None,
        db_path=db_path,
    )
    return Harness(manager, loader, db_path, deepseek, backtest, engine, strategies_dir)


def _bt_metrics(sharpe=1.0, win_rate=55.0, max_dd=10.0, trades=30):
    return {"metrics": {
        "sharpe_ratio": sharpe,
        "win_rate_pct": win_rate,
        "max_drawdown_pct": max_dd,
        "total_trades": trades,
    }}


def _strategy(name, **overrides) -> StrategyConfig:
    cfg = {
        "name": name,
        "enabled": True,
        "mode": "trend",
        "timeframes": ["1h"],
        "symbols": ["BTCUSDT"],
        "indicators": {"rsi": {"period": 14}},
        "entry_conditions": {"long": ["rsi < 30"], "short": ["rsi > 70"]},
        "exit_conditions": {"long": ["rsi > 70"], "short": ["rsi < 30"]},
        "ml_config": MLConfig(enabled=False),
    }
    cfg.update(overrides)
    return StrategyConfig(**cfg)


def _snapshot(directory: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(directory.glob("*.yaml"))}


def _fingerprint(path: Path):
    if not path.exists():
        return None
    st = path.stat()
    return (st.st_size, st.st_mtime_ns)


async def _rows(db_path: str, where: str = "1=1", params: tuple = ()) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            f"SELECT * FROM strategy_lifecycle_events WHERE {where} ORDER BY id", params)
        return [dict(r) for r in await cursor.fetchall()]


# ── 1. log_event ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_log_event_writes_readable_row(tmp_path):
    h = await _make_harness(tmp_path)

    await h.manager.log_event("strat_a", "deployed", "Sharpe=1.5",
                              {"sharpe_ratio": 1.5, "win_rate_pct": 58.0},
                              backtest_id=7)

    rows = await _rows(h.db_path, "strategy_name=?", ("strat_a",))
    assert len(rows) == 1, "log_event must insert exactly one row"
    row = rows[0]
    assert row["action"] == "deployed"
    assert row["trigger_reason"] == "Sharpe=1.5"
    assert json.loads(row["metrics_snapshot"]) == {"sharpe_ratio": 1.5, "win_rate_pct": 58.0}
    assert row["backtest_record_id"] == 7
    assert row["created_at"], "created_at must be defaulted by the schema"


@pytest.mark.asyncio
async def test_log_event_without_metrics_stores_null_snapshot(tmp_path):
    h = await _make_harness(tmp_path)

    await h.manager.log_event("strat_b", "retired")

    rows = await _rows(h.db_path, "strategy_name=?", ("strat_b",))
    assert len(rows) == 1
    assert rows[0]["metrics_snapshot"] is None
    assert rows[0]["trigger_reason"] == ""
    assert rows[0]["backtest_record_id"] is None


# ── 2. retirement decision ──────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("metrics,expected", [
    # dead strategy: no trades at all
    ({"sharpe_ratio": 0.5, "win_rate_pct": 60.0, "max_drawdown_pct": 5.0,
      "total_trades": 0}, True),
    # sharpe < -1.0
    ({"sharpe_ratio": -1.5, "win_rate_pct": 60.0, "max_drawdown_pct": 5.0,
      "total_trades": 20}, True),
    # win_rate < 30
    ({"sharpe_ratio": 0.9, "win_rate_pct": 20.0, "max_drawdown_pct": 5.0,
      "total_trades": 20}, True),
    # max_drawdown_pct > 30
    ({"sharpe_ratio": 0.9, "win_rate_pct": 60.0, "max_drawdown_pct": 45.0,
      "total_trades": 20}, True),
    # healthy: none of the four thresholds tripped
    ({"sharpe_ratio": 1.2, "win_rate_pct": 55.0, "max_drawdown_pct": 10.0,
      "total_trades": 40}, False),
])
async def test_evaluate_for_retirement_decision_matrix(tmp_path, metrics, expected):
    h = await _make_harness(tmp_path, backtest_result={"metrics": metrics})
    h.loader.save(_strategy("probe"))

    decision = await h.manager._evaluate_for_retirement("probe")

    assert decision is expected, (
        f"metrics {metrics} -> retire={decision}, expected {expected} "
        f"(thresholds live in strategy_lifecycle.py:232-239)"
    )
    assert h.backtest.calls, "retirement decision must come from a backtest"


@pytest.mark.asyncio
async def test_evaluate_for_retirement_boundary_values_are_kept(tmp_path):
    """Exactly at the thresholds the strategy is KEPT (strict </> comparisons)."""
    metrics = {"sharpe_ratio": -1.0, "win_rate_pct": 30.0,
               "max_drawdown_pct": 30.0, "total_trades": 5}
    h = await _make_harness(tmp_path, backtest_result={"metrics": metrics})
    h.loader.save(_strategy("boundary"))

    assert await h.manager._evaluate_for_retirement("boundary") is False


@pytest.mark.asyncio
async def test_evaluate_for_retirement_ignores_backtest_failure(tmp_path):
    h = await _make_harness(tmp_path, backtest_result={"error": "No historical data found"})
    h.loader.save(_strategy("no_data"))

    assert await h.manager._evaluate_for_retirement("no_data") is False, \
        "a backtest failure must not be treated as a retirement signal"


@pytest.mark.asyncio
async def test_check_and_retire_disables_only_the_poor_strategy(tmp_path):
    h = await _make_harness(tmp_path, backtest_by_strategy={
        "poor_strat": _bt_metrics(sharpe=-2.0, win_rate=10.0, max_dd=55.0, trades=12),
        "good_strat": _bt_metrics(sharpe=1.5, win_rate=60.0, max_dd=5.0, trades=50),
    }, ai_mode="semi_auto")
    h.loader.save(_strategy("poor_strat"))
    h.loader.save(_strategy("good_strat"))

    await h.manager.check_and_retire()

    assert h.loader.load("poor_strat").enabled is False, \
        "semi_auto retirement must persist enabled=False to the YAML"
    assert h.loader.load("good_strat").enabled is True
    retired = await _rows(h.db_path, "action=?", ("retired",))
    assert [r["strategy_name"] for r in retired] == ["poor_strat"]
    assert {c["strategies"][0] for c in h.backtest.calls} == {"poor_strat", "good_strat"}


@pytest.mark.asyncio
async def test_check_and_retire_suggest_mode_leaves_files_untouched(tmp_path):
    h = await _make_harness(tmp_path,
                            backtest_result=_bt_metrics(sharpe=-3.0, win_rate=5.0, trades=9),
                            ai_mode="suggest")
    h.loader.save(_strategy("poor_strat"))
    before = _snapshot(h.strategies_dir)

    await h.manager.check_and_retire()

    assert _snapshot(h.strategies_dir) == before, \
        "suggest mode must not rewrite strategy files"
    assert h.loader.load("poor_strat").enabled is True
    assert len(await _rows(h.db_path, "action=?", ("retired",))) == 1


# ── 3. generate_strategy ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_strategy_parses_valid_ai_json(tmp_path):
    h = await _make_harness(tmp_path, ai_response=VALID_AI_JSON)
    h.loader.save(_strategy("existing_strat"))
    before = _snapshot(h.strategies_dir)

    result = await h.manager.generate_strategy(target_symbols=["BTCUSDT"])

    assert result is not None
    assert result["name"] == "ai_valid_strategy"
    assert result["entry_conditions"]["long"] == ["rsi < 30"]
    assert h.deepseek.calls, "AI must have been consulted"
    assert "BTCUSDT" in h.deepseek.calls[0][1], "target symbols must reach the prompt"
    assert _snapshot(h.strategies_dir) == before, \
        "generate_strategy() alone must not write any strategy file"


@pytest.mark.asyncio
async def test_generate_strategy_tolerates_code_fenced_json(tmp_path):
    h = await _make_harness(tmp_path, ai_response=f"```json\n{VALID_AI_JSON}\n```")

    result = await h.manager.generate_strategy()

    assert result is not None and result["name"] == "ai_valid_strategy"


@pytest.mark.asyncio
async def test_generate_strategy_defaults_missing_timeframes_to_1h(tmp_path):
    payload = dict(VALID_AI_CONFIG)
    payload.pop("timeframes")
    h = await _make_harness(tmp_path, ai_response=json.dumps(payload))

    result = await h.manager.generate_strategy()

    assert result is not None
    assert result["timeframes"] == ["1h"], "missing timeframes must default to ['1h']"


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    None,                                   # AI call failed / no client
    "",                                     # empty body
    "not json at all {{{",                  # unparseable
    json.dumps({"name": "no_entries", "timeframes": ["1h"]}),  # no entry_conditions
])
async def test_generate_strategy_reports_failure_without_touching_files(tmp_path, response):
    h = await _make_harness(tmp_path, ai_response=response)
    h.loader.save(_strategy("existing_strat"))
    before = _snapshot(h.strategies_dir)

    result = await h.manager.generate_strategy()

    assert result is None, f"AI output {response!r} must be reported as a failure"
    assert h.loader.list_names() == ["existing_strat"]
    assert _snapshot(h.strategies_dir) == before, \
        "a failed generation must not corrupt existing strategy files"


@pytest.mark.asyncio
async def test_generate_strategy_is_rate_limited(tmp_path):
    h = await _make_harness(tmp_path, ai_response=VALID_AI_JSON)
    h.manager._last_generation_time = time.time()  # inside the 24h interval

    result = await h.manager.generate_strategy()

    assert result is None
    assert h.deepseek.calls == [], "rate-limited generation must not call the AI"


@pytest.mark.asyncio
async def test_generate_strategy_ai_exception_is_handled_and_keeps_files(tmp_path):
    """FIXED BUG regression: an AI/transport failure must not become a 500.

    `generate_strategy()` previously called `_call_deepseek()` without a guard, so
    any future change to the AI client (one that raised instead of returning None)
    would have turned `POST /api/strategy-lifecycle/generate` into an unhandled 500.
    It now catches the failure, logs it, and returns None like the other failure
    paths, leaving existing strategy files untouched.
    """
    h = await _make_harness(tmp_path, ai_error=RuntimeError("deepseek exploded"))
    h.loader.save(_strategy("existing_strat"))
    before = _snapshot(h.strategies_dir)

    result = await h.manager.generate_strategy()

    assert result is None, "an AI failure must be reported as a handled failure"
    assert _snapshot(h.strategies_dir) == before, "existing strategies must be untouched"


# ── 4. validate_and_deploy ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_validate_and_deploy_saves_strategy_on_pass(tmp_path):
    h = await _make_harness(tmp_path,
                            backtest_result=_bt_metrics(sharpe=1.2, win_rate=55.0, trades=40))
    h.loader.save(_strategy("existing_strat"))

    ok = await h.manager.validate_and_deploy(dict(VALID_AI_CONFIG))

    assert ok is True
    assert sorted(h.loader.list_names()) == ["ai_valid_strategy", "existing_strat"]
    saved = h.loader.load("ai_valid_strategy")
    assert saved.entry_conditions["short"] == ["rsi > 70"]
    assert "ai_valid_strategy" in h.engine._strategies, \
        "a deployed strategy must be reloaded into the engine"
    deployed = await _rows(h.db_path, "action=?", ("deployed",))
    assert [r["strategy_name"] for r in deployed] == ["ai_valid_strategy"]
    assert h.backtest.calls[0]["strategies"] == ["ai_valid_strategy"]


@pytest.mark.asyncio
async def test_validate_and_deploy_rejection_cleans_up_and_keeps_existing_files(tmp_path):
    h = await _make_harness(tmp_path,
                            backtest_result=_bt_metrics(sharpe=0.1, win_rate=20.0, trades=15))
    h.loader.save(_strategy("existing_strat"))
    before = _snapshot(h.strategies_dir)

    ok = await h.manager.validate_and_deploy(dict(VALID_AI_CONFIG))

    assert ok is False
    assert h.loader.list_names() == ["existing_strat"], \
        "a strategy that fails validation must not stay on disk"
    assert _snapshot(h.strategies_dir) == before, "existing strategies must be byte-identical"
    generated = await _rows(h.db_path, "action=?", ("generated",))
    assert [r["strategy_name"] for r in generated] == ["ai_valid_strategy"]
    assert "ai_valid_strategy" not in h.engine._strategies


@pytest.mark.asyncio
async def test_validate_and_deploy_removes_strategy_when_backtest_errors(tmp_path):
    h = await _make_harness(tmp_path, backtest_result={"error": "No historical data found"})
    h.loader.save(_strategy("existing_strat"))
    before = _snapshot(h.strategies_dir)

    ok = await h.manager.validate_and_deploy(dict(VALID_AI_CONFIG))

    assert ok is False
    assert h.loader.list_names() == ["existing_strat"]
    assert _snapshot(h.strategies_dir) == before


@pytest.mark.asyncio
async def test_validate_and_deploy_rejects_malformed_config_without_touching_files(tmp_path):
    h = await _make_harness(tmp_path,
                            backtest_result=_bt_metrics(sharpe=2.0, win_rate=70.0))
    h.loader.save(_strategy("existing_strat"))
    before = _snapshot(h.strategies_dir)

    # No "name" -> StrategyConfig validation fails before anything is written.
    ok = await h.manager.validate_and_deploy({"mode": "trend", "timeframes": ["1h"]})

    assert ok is False
    assert _snapshot(h.strategies_dir) == before, \
        "an invalid AI config must not corrupt the strategies directory"
    assert h.backtest.calls == [], "nothing should be backtested if the config is invalid"


# ── 5. isolation from production state ──────────────────────────────────

@pytest.mark.asyncio
async def test_lifecycle_never_touches_production_db(tmp_path):
    before = _fingerprint(PROD_DB)
    h = await _make_harness(tmp_path, ai_response=VALID_AI_JSON,
                            backtest_result=_bt_metrics(sharpe=1.2, win_rate=55.0))

    await h.manager.log_event("probe", "generated", "isolation check")
    await h.manager.generate_strategy()
    await h.manager.validate_and_deploy(dict(VALID_AI_CONFIG))
    await h.manager.check_and_retire()

    assert Path(h.db_path).parent == tmp_path
    assert Path(h.db_path).name == "lifecycle_test.db"
    assert Path(h.db_path).resolve() != PROD_DB.resolve()
    assert _fingerprint(PROD_DB) == before, \
        "the lifecycle manager must not write to data/binance_trader.db"
    assert len(await _rows(h.db_path)) >= 2, "events must land in the temp DB"
