"""P7-S3 — the upper-layer :class:`RegimeOrchestrator`.

Acceptance criteria this file pins (plan ``docs/overhaul/P7_REGIME_PLAN.md`` S3):

1. **Each rule in isolation** — the regime→strategy mapping, the
   consecutive-loss kill switch (trigger *and* reset on a regime change), the
   volatility gate (including its causal trailing median) and the market-breadth
   gate, plus the documented fallback for a **missing or stale** breadth series.
2. **Determinism / replay** — the same rule set plus the same event list gives
   the same enable/disable timeline on every run, and the incremental API
   produces the same timeline as :meth:`RegimeOrchestrator.replay`.
3. **Off is off** — ``ai.orchestrator.enabled: false`` returns the all-allow
   verdict *without reading* the vol or breadth series, and a real engine run with
   such an object is **bit-identical** to the same run without one (and to a
   ``git worktree`` at the pre-S3 revision ``d9849a2``).
4. **The mechanism is proven** — on a synthetic timeline where a strategy loses
   through a stretch that the orchestrator disables after two consecutive losses,
   the orchestrated run keeps strictly more of the account than the always-on run.
5. **Live seam** — with ``experimental.regime_orchestrator_live`` off, the live
   ``StrategyEngine`` signal cache and published signals are byte-identical to the
   seam being absent (a registered-but-disabled orchestrator changes nothing).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.ai.orchestrator import (
    ALLOW,
    BLOCK_BREADTH,
    BLOCK_KILL_SWITCH,
    BLOCK_REGIME,
    BLOCK_VOL,
    BreadthSample,
    OrchestratorConfig,
    OrchestratorConfigError,
    RegimeOrchestrator,
    TimelineEvent,
    TradeOutcome,
    breadth_policy_blocks,
    decide,
    kill_switch_blocks,
    orchestrator_config_from_raw,
    regime_policy_blocks,
    rules_fingerprint,
    vol_policy_blocks,
)
from core.strategy.regime_causal import InSampleRegimeLabelError

ROOT = Path(__file__).resolve().parents[1]

#: The revision this change was developed on (the pre-S3 tree).
BASELINE_REVISION = "d9849a2"


# ══════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════

def _rules(**overrides) -> OrchestratorConfig:
    """A small enabled rule set; each test overrides only what it exercises."""
    raw = {
        "enabled": True,
        "regime": {"allowed": {}, "default_action": "allow",
                   "missing_regime_action": "allow"},
        "kill_switch": {"consecutive_losses": 0},
        "vol": {"multiple": 0.0},
        "breadth": {},
    }
    raw.update(overrides)
    return orchestrator_config_from_raw(raw)


def _sample(as_of_ms: int, up_share=0.6, coverage=0.98, **kwargs) -> BreadthSample:
    return BreadthSample(as_of_ms=as_of_ms, up_share=up_share,
                         coverage=coverage, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
# 1 — config: the shipped default and the refusals at load
# ══════════════════════════════════════════════════════════════════════════

def test_shipped_config_default_is_disabled_and_every_rule_off():
    config = orchestrator_config_from_raw(None)
    assert config.enabled is False
    assert config.regime.allowed == {}
    assert config.kill_switch.consecutive_losses == 0
    assert config.vol.multiple == 0.0
    assert config.breadth.active is False


def test_the_shipped_yaml_block_parses_and_is_inert():
    """The real ``ai.orchestrator`` block in config.yaml loads and ships off."""
    import yaml
    raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert "orchestrator" in raw["ai"], "config.yaml carries no ai.orchestrator block"
    config = orchestrator_config_from_raw(raw["ai"]["orchestrator"])
    assert config.enabled is False
    assert config.vol.active is False and config.breadth.active is False
    assert config.kill_switch.active is False
    # The app reads it through the same parser.
    from app.config import Config
    Config._instance = None
    loaded = Config.load("sim")
    assert loaded.ai_orchestrator_enabled is False
    assert loaded.ai_orchestrator == config


@pytest.mark.parametrize("raw,needle", [
    ({"enabled": True, "nope": 1}, "unknown key"),
    ({"enabled": True, "regime": {"allowed": {"s": ["calm"]}}}, "in-sample"),
    ({"enabled": True, "regime": {"allowed": {"s": ["moon"]}}}, "unknown regime label"),
    ({"enabled": True, "vol": {"multiple": 1.0, "min_samples": 1}}, "min_samples"),
    ({"enabled": True, "breadth": {"missing_action": "maybe"}}, "must be one of"),
    ({"enabled": True, "kill_switch": {"reset_on_regime_change": False}},
     "not implemented"),
])
def test_a_bad_rule_fails_at_load_not_halfway_through_a_backtest(raw, needle):
    with pytest.raises(OrchestratorConfigError) as excinfo:
        orchestrator_config_from_raw(raw)
    assert needle in str(excinfo.value)


def test_regime_block_rejects_a_non_list_and_accepts_the_all_shorthand():
    with pytest.raises(OrchestratorConfigError):
        orchestrator_config_from_raw(
            {"regime": {"allowed": {"s": {"trend_up": True}}}})
    config = orchestrator_config_from_raw({"regime": {"allowed": {"s": ["all"]}}})
    assert config.regime.labels_for("s") == (
        "trend_up", "trend_down", "range_low", "range_mid", "range_high")


# ══════════════════════════════════════════════════════════════════════════
# 2 — rule 1: the regime → eligible-strategy mapping
# ══════════════════════════════════════════════════════════════════════════

def test_rule_1_the_regime_mapping_in_isolation():
    policy = orchestrator_config_from_raw({
        "enabled": True,
        "regime": {"allowed": {"trend_only": ["trend_up", "trend_down"],
                               "never": []},
                   "default_action": "deny",
                   "missing_regime_action": "allow"}}).regime
    assert policy.allows("trend_only", "trend_up") is True
    assert policy.allows("trend_only", "range_low") is False
    assert policy.allows("never", "trend_up") is False
    # Unlisted ⇒ default_action (here deny); an unmeasured bar ⇒ its own action.
    assert policy.allows("unlisted", "trend_up") is False
    assert policy.allows("trend_only", None) is True
    assert regime_policy_blocks(policy, "trend_only", "range_mid") is True
    assert regime_policy_blocks(policy, "trend_only", "trend_down") is False


def test_rule_1_default_action_allow_keeps_an_unlisted_strategy_running():
    policy = orchestrator_config_from_raw(
        {"enabled": True, "regime": {"allowed": {"s": ["trend_up"]}}}).regime
    assert policy.allows("other", "range_high") is True
    assert policy.allows("s", "range_high") is False


def test_decide_reports_the_regime_reason_and_never_enables_what_it_blocked():
    verdict = decide(_rules(regime={"allowed": {"s": ["trend_up"]}}), "s",
                     "range_low")
    assert verdict.enabled is False
    assert verdict.reason == BLOCK_REGIME
    assert verdict.blocked_reasons == (BLOCK_REGIME,)
    assert verdict.action == "disabled"


def test_decide_refuses_an_in_sample_label_by_name():
    with pytest.raises(InSampleRegimeLabelError):
        decide(_rules(), "s", "calm")
    with pytest.raises(InSampleRegimeLabelError):
        RegimeOrchestrator(_rules()).observe_regime("s", "stressed")


def test_a_disabled_orchestrator_allows_everything():
    machine = RegimeOrchestrator({"enabled": False})
    verdict = machine.decide("s", at="2026-01-01", label="range_low")
    assert verdict.enabled is True and verdict.reason == ALLOW
    assert machine.enable_timeline() == []


# ══════════════════════════════════════════════════════════════════════════
# 3 — rule 2: the consecutive-loss kill switch (trigger and reset)
# ══════════════════════════════════════════════════════════════════════════

def test_rule_2_kill_switch_in_isolation():
    policy = orchestrator_config_from_raw(
        {"enabled": True, "kill_switch": {"consecutive_losses": 3}}).kill_switch
    machine = RegimeOrchestrator({"enabled": True,
                                  "kill_switch": {"consecutive_losses": 3}})
    state = machine.state("s")
    assert kill_switch_blocks(policy, state) is False
    state.killed = True
    assert kill_switch_blocks(policy, state) is True
    assert kill_switch_blocks(kill_switch_policy_off(), state) is False


def kill_switch_policy_off():
    return orchestrator_config_from_raw(
        {"enabled": True, "kill_switch": {"consecutive_losses": 0}}).kill_switch


def test_rule_2_trips_after_exactly_n_consecutive_losses():
    machine = RegimeOrchestrator({"enabled": True,
                                  "kill_switch": {"consecutive_losses": 2}})
    machine.observe_regime("s", "trend_up")
    assert machine.record_trade("s", -10.0) is False
    assert machine.decide("s").enabled is True
    assert machine.record_trade("s", -1.0) is True          # the second loss
    verdict = machine.decide("s")
    assert verdict.enabled is False
    assert BLOCK_KILL_SWITCH in verdict.blocked_reasons
    assert machine.state("s").killed_in_regime == "trend_up"


def test_rule_2_a_win_or_a_break_even_trade_resets_the_counter():
    machine = RegimeOrchestrator({"enabled": True,
                                  "kill_switch": {"consecutive_losses": 2}})
    machine.observe_regime("s", "trend_up")
    machine.record_trade("s", -5.0)
    machine.record_trade("s", 0.0)                          # a tie is not a loss
    assert machine.state("s").consecutive_losses == 0
    machine.record_trade("s", -5.0)
    assert machine.record_trade("s", +3.0) is False         # a win resets it
    assert machine.state("s").consecutive_losses == 0
    assert machine.state("s").killed is False


def test_rule_2_a_regime_change_resets_the_latch_and_the_counter():
    machine = RegimeOrchestrator({"enabled": True,
                                  "kill_switch": {"consecutive_losses": 2}})
    machine.observe_regime("s", "trend_up")
    machine.record_trade("s", -5.0)
    machine.record_trade("s", -5.0)
    assert machine.decide("s").enabled is False
    changed = machine.observe_regime("s", "range_low")
    assert changed is True
    assert machine.state("s").killed is False
    assert machine.state("s").consecutive_losses == 0
    assert machine.state("s").regime_changes == 1
    assert machine.decide("s").enabled is True


def test_rule_2_the_switch_off_counts_nothing():
    machine = RegimeOrchestrator({"enabled": True,
                                  "kill_switch": {"consecutive_losses": 0}})
    machine.observe_regime("s", "trend_up")
    for _ in range(50):
        assert machine.record_trade("s", -100.0) is False
    assert machine.decide("s").enabled is True


def test_rule_2_a_trade_can_carry_its_own_regime_and_reset_first():
    """A supplied regime is observed *before* the loss is counted."""
    machine = RegimeOrchestrator({"enabled": True,
                                  "kill_switch": {"consecutive_losses": 2}})
    machine.observe_regime("s", "trend_up")
    machine.record_trade("s", -5.0)
    tripped = machine.record_trade("s", -5.0, regime="range_low")
    # The regime change reset the counter, so this one loss is the first of the
    # new stretch and cannot trip a 2-loss rule.
    assert tripped is False
    assert machine.state("s").consecutive_losses == 1


# ══════════════════════════════════════════════════════════════════════════
# 4 — rule 3: the volatility gate and its causal trailing median
# ══════════════════════════════════════════════════════════════════════════

def test_rule_3_the_gate_in_isolation():
    policy = orchestrator_config_from_raw(
        {"enabled": True, "vol": {"multiple": 2.0, "min_samples": 3,
                                  "missing_action": "deny"}}).vol
    assert vol_policy_blocks(policy, 10.0, 1.0) is True       # 10 > 2 * 1
    assert vol_policy_blocks(policy, 1.5, 1.0) is False
    assert vol_policy_blocks(policy, 2.0, 1.0) is False       # strict >
    assert vol_policy_blocks(policy, None, 1.0) is True       # unmeasurable ⇒ deny
    assert vol_policy_blocks(policy, 5.0, None) is True
    assert vol_policy_blocks(policy, 5.0, 0.0) is True        # no ratio exists
    assert vol_policy_blocks(policy, float("nan"), 1.0) is True


def test_rule_3_inverts_when_asked():
    policy = orchestrator_config_from_raw(
        {"enabled": True, "vol": {"multiple": 2.0, "deny_on_high_vol": False,
                                  "min_samples": 2}}).vol
    assert vol_policy_blocks(policy, 10.0, 1.0) is False
    assert vol_policy_blocks(policy, 1.0, 1.0) is True


def test_rule_3_the_median_is_causal_and_needs_min_samples():
    machine = RegimeOrchestrator({"enabled": True,
                                  "vol": {"multiple": 2.0, "min_samples": 3,
                                          "missing_action": "deny"}})
    for value in (1.0, 1.0):
        machine.observe_vol(value, key="s")
    assert machine.vol_median(key="s") is None                # below min_samples
    assert machine.decide("s", vol=1.0).enabled is False      # unmeasurable ⇒ deny
    machine.observe_vol(1.0, key="s")
    assert machine.vol_median(key="s") == 1.0
    # The current sample is appended AFTER the median, so it cannot move its own
    # threshold: a 100x spike is judged against the trailing 1.0, then becomes
    # part of the history for the NEXT bar.
    assert machine.decide("s", vol=100.0).enabled is False
    assert machine.vol_median(key="s") == 1.0                 # 1,1,1 → still 1.0
    assert machine.decide("s", vol=100.0).blocked_reasons == (BLOCK_VOL,)


def test_rule_3_zero_multiple_is_inert():
    machine = RegimeOrchestrator({"enabled": True, "vol": {"multiple": 0.0}})
    machine.observe_vol(1e9, key="s")
    assert machine.decide("s", vol=1e12).enabled is True


# ══════════════════════════════════════════════════════════════════════════
# 5 — rule 4: breadth, and the documented missing/stale fallback
# ══════════════════════════════════════════════════════════════════════════

def test_rule_4_the_gate_in_isolation():
    policy = orchestrator_config_from_raw(
        {"enabled": True, "breadth": {"min_up_share": 0.45, "min_coverage": 0.90,
                                      "max_staleness_ms": 1000}}).breadth
    assert breadth_policy_blocks(policy, _sample(0, up_share=0.40), now_ms=0) is True
    assert breadth_policy_blocks(policy, _sample(0, up_share=0.50), now_ms=0) is False
    assert breadth_policy_blocks(policy, _sample(0, coverage=0.50), now_ms=0) is True
    assert breadth_policy_blocks(policy, None, now_ms=0) is False   # allow ships
    assert breadth_policy_blocks(policy, _sample(0), now_ms=5000) is False  # stale ⇒ allow


def test_rule_4_missing_and_stale_follow_the_configured_action():
    """The fallback is a documented, configurable decision — not a silent yes."""
    deny = orchestrator_config_from_raw(
        {"enabled": True, "breadth": {"min_up_share": 0.45, "missing_action": "deny",
                                      "stale_action": "deny",
                                      "max_staleness_ms": 1000}}).breadth
    assert breadth_policy_blocks(deny, None, now_ms=0) is True
    assert breadth_policy_blocks(deny, _sample(0), now_ms=5000) is True
    allow = orchestrator_config_from_raw(
        {"enabled": True, "breadth": {"min_up_share": 0.45}}).breadth
    assert breadth_policy_blocks(allow, None, now_ms=0) is False
    assert breadth_policy_blocks(allow, _sample(0), now_ms=10 ** 9) is False


def test_rule_4_an_absent_series_is_the_missing_branch_not_a_zero():
    """An empty breadth cache must not read as "up_share = 0" (a fabricated floor)."""
    machine = RegimeOrchestrator(
        {"enabled": True,
         "breadth": {"min_up_share": 0.45, "missing_action": "deny"}})
    assert machine.breadth_at(0) is None
    verdict = machine.decide("s", at=0)
    assert verdict.blocked_reasons == (BLOCK_BREADTH,)
    # The same machine, shipped fallback: an absent series allows.
    shipped = RegimeOrchestrator(
        {"enabled": True, "breadth": {"min_up_share": 0.45}})
    assert shipped.decide("s", at=0).enabled is True


def test_rule_4_the_series_lookup_is_causal_and_forward_fills_by_sample_ms():
    machine = RegimeOrchestrator(
        {"enabled": True,
         "breadth": {"min_up_share": 0.45, "sample_ms": 3_600_000,
                     "max_staleness_ms": 10 ** 9}})
    machine.observe_breadth(_sample(1000, up_share=0.30))
    machine.observe_breadth(_sample(2000, up_share=0.90))
    assert machine.breadth_at(500) is None                    # nothing before it
    assert machine.breadth_at(1000).up_share == 0.30
    assert machine.breadth_at(1500).up_share == 0.30          # forward fill
    assert machine.breadth_at(2000).up_share == 0.90
    assert machine.breadth_at(2000 + 4_000_000) is None       # past the tolerance


def test_rule_4_accepts_a_breadth_observation_or_a_cache_like_object():
    from core.market_data.breadth import BreadthObservation
    observation = BreadthObservation(as_of_ms=42, up_share=0.7, coverage=0.99)
    sample = BreadthSample.from_observation(observation)
    assert sample.as_of_ms == 42 and sample.up_share == 0.7

    class FakeCache:
        def load(self):
            return [observation]

    machine = RegimeOrchestrator(
        {"enabled": True, "breadth": {"min_up_share": 0.45}})
    assert machine.decide("s", at=42, breadth_series=FakeCache()).enabled is True
    assert len(machine.breadth_series) == 1


# ══════════════════════════════════════════════════════════════════════════
# 6 — the whole decision: priority, and "enabled == not blocked"
# ══════════════════════════════════════════════════════════════════════════

def test_decide_priority_is_regime_then_kill_switch_then_vol_then_breadth():
    config = _rules(regime={"allowed": {"s": ["trend_up"]}},
                    kill_switch={"consecutive_losses": 2},
                    vol={"multiple": 2.0, "min_samples": 2, "missing_action": "deny"},
                    breadth={"min_up_share": 0.9})
    machine = RegimeOrchestrator(config)
    machine.observe_regime("s", "trend_up")
    machine.record_trade("s", -1.0)
    machine.record_trade("s", -1.0)
    machine.observe_breadth(_sample(0, up_share=0.10))
    # trend_up is allowed, but the kill switch, the vol gate and the breadth gate
    # all block: the reported reason is the first, and every reason is listed.
    verdict = machine.decide("s", at=0, label="trend_up", vol=10.0)
    assert verdict.reason == BLOCK_KILL_SWITCH
    assert verdict.blocked_reasons == (BLOCK_KILL_SWITCH, BLOCK_VOL, BLOCK_BREADTH)
    assert verdict.enabled is False
    # A range label is refused first, so the regime reason wins.
    assert machine.decide("s", at=0, label="range_low", vol=10.0).reason == BLOCK_REGIME


def test_decide_enabled_is_exactly_not_blocked():
    config = _rules(regime={"allowed": {"s": ["trend_up"]}},
                    vol={"multiple": 2.0, "min_samples": 2})
    machine = RegimeOrchestrator(config)
    for label in ("trend_up", "range_low", None):
        verdict = machine.decide("s", label=label)
        assert verdict.enabled == (not verdict.blocked_reasons)
        assert (verdict.reason == ALLOW) == verdict.enabled


# ══════════════════════════════════════════════════════════════════════════
# 7 — determinism and replay
# ══════════════════════════════════════════════════════════════════════════

def _timeline() -> list:
    labels = ["trend_up", "trend_up", "range_low", "range_low", "trend_down"]
    events: list = []
    for index, label in enumerate(labels):
        events.append(TimelineEvent("decide", at=f"2026-01-0{index + 1}",
                                    label=label))
        events.append(TimelineEvent(
            "trade", outcome=TradeOutcome("alpha", -7.0,
                                          ts=f"2026-01-0{index + 1}")))
        events.append(TimelineEvent("vol", at=f"2026-01-0{index + 1}",
                                    vol=0.001 * (index + 1), key="alpha"))
        events.append(TimelineEvent("decide", at=f"2026-01-0{index + 1}T12",
                                    label=label))
    return events


def _replay_rules() -> dict:
    return {
        "enabled": True,
        "regime": {"allowed": {"alpha": ["all"], "beta": ["trend_up"]},
                   "default_action": "deny"},
        "kill_switch": {"consecutive_losses": 2},
        "vol": {"multiple": 2.0, "min_samples": 2, "missing_action": "allow"},
        "breadth": {"min_up_share": 0.45, "missing_action": "allow"},
    }


def test_two_replays_are_identical_and_match_the_incremental_api():
    events = _timeline()
    first = RegimeOrchestrator.replay(_replay_rules(), events,
                                      names=["alpha", "beta"])
    second = RegimeOrchestrator.replay(_replay_rules(), events,
                                       names=["alpha", "beta"])
    assert first == second
    assert len(first) == 10

    machine = RegimeOrchestrator(_replay_rules(), names=["alpha", "beta"])
    incremental = machine.run_timeline(events)
    assert incremental == first
    # And the compact enable/disable timeline is stable too.
    replay_machine = RegimeOrchestrator(_replay_rules(), names=["alpha", "beta"])
    replay_machine.run_timeline(events)
    assert machine.enable_timeline() == replay_machine.enable_timeline()


def test_the_timeline_actually_toggles_a_strategy():
    machine = RegimeOrchestrator(_replay_rules(), names=["alpha"])
    machine.run_timeline(_timeline())
    states = [row["rows"][0]["enabled"] for row in machine.timeline]
    assert True in states and False in states, "the rule set never fired"
    reasons = {row["rows"][0]["reason"] for row in machine.timeline}
    assert BLOCK_KILL_SWITCH in reasons
    assert machine.state("alpha").regime_changes == 2


def test_the_rules_fingerprint_is_stable_and_changes_with_a_rule():
    config = _rules(regime={"allowed": {"s": ["trend_up"]}})
    assert rules_fingerprint(config) == rules_fingerprint(
        orchestrator_config_from_raw(config.as_dict()))
    assert rules_fingerprint(config) != rules_fingerprint(
        _rules(regime={"allowed": {"s": ["trend_down"]}}))
    assert len(rules_fingerprint(config)) == 16


# ══════════════════════════════════════════════════════════════════════════
# 8 — the engine seam: off is off
# ══════════════════════════════════════════════════════════════════════════

CALM_UP_BARS = 400
STRESS_DOWN_BARS = 400
TOTAL_BARS = CALM_UP_BARS + STRESS_DOWN_BARS


def _market_frame(seed: int = 20261002) -> pd.DataFrame:
    """A calm uptrend then a volatile downtrend (causal labels by construction).

    Half 1 drifts up with tiny noise (``trend_up``); half 2 drifts down with large
    noise (``trend_down``), so a trend-following long strategy **wins in half 1 and
    loses in half 2** — which is what lets the kill switch show a measurable effect
    without any invented numbers.
    """
    rng = np.random.default_rng(seed)
    half = TOTAL_BARS // 2
    steps = np.concatenate([0.0012 + rng.normal(0, 0.0005, half),
                            -0.0035 + rng.normal(0, 0.0045, TOTAL_BARS - half)])
    close = 20_000.0 * np.cumprod(1.0 + steps)
    index = pd.date_range("2026-01-01", periods=TOTAL_BARS, freq="1h")
    return pd.DataFrame(
        {"open": np.concatenate([[close[0]], close[:-1]]),
         "high": close * 1.0005, "low": close * 0.9995, "close": close,
         "volume": 100.0 + rng.random(TOTAL_BARS) * 50.0}, index=index)


def _write_market(root: Path, symbols=("BTCUSDT",)) -> str:
    frame = _market_frame()
    for symbol in symbols:
        target = root / "market" / symbol
        target.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target / "1h.parquet")
    return str(root)


@pytest.fixture(scope="module")
def market_dir(tmp_path_factory):
    return _write_market(tmp_path_factory.mktemp("p7s3") / "data")


def _engine(market_dir, tmp_path):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager

    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = str(market_dir)
    cfg.backtest_engine_mode = "legacy"
    cfg.backtest_ml_enabled = False
    cfg.backtest_live_spread_enabled = False
    bus = EventBus()
    return cfg, BacktestEngine(cfg, None, RiskManager(cfg, bus),
                               OrderExecutor(cfg, bus))


def _always_long(name: str):
    """A strategy that enters almost every bar (so the gate is the only filter)."""
    from core.strategy.loader import MLConfig, StrategyConfig

    return StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=["1h"],
        indicators={"rsi": {"period": 14, "source": "close"},
                    "sma": {"period": 5}},
        entry_conditions={"long": ["close > sma"], "short": ["close < sma"]},
        exit_conditions={"long": ["rsi > 99"], "short": ["rsi < 1"]},
        ml_config=MLConfig(enabled=False))


def _run(engine, strategy, orchestrator=None):
    return engine.run_with_exit_evaluation(
        strategies=[strategy], symbols=["BTCUSDT"],
        date_start="2026-01-01", date_end="2026-03-31", initial_balance=10_000.0,
        mode="full", simulate_ai_weights=False, per_strategy_isolation=True,
        per_genome_ledger=True, use_live_spread=False, benchmark_mode="none",
        orchestrator=orchestrator)


def _fingerprint(result) -> str:
    """The traded facts of a run, canonically serialised (byte comparison basis)."""
    per = result.get("per_strategy_equity") or {}
    payload = {
        "trades": [{"symbol": t.get("symbol"), "side": t.get("side"),
                    "opened_at": str(t.get("opened_at")),
                    "closed_at": str(t.get("closed_at")),
                    "pnl": t.get("pnl"), "cost": t.get("cost")}
                   for t in (result.get("trades") or [])],
        "equity": {name: [{"time": str(point["time"]), "equity": point["equity"]}
                          for point in (entry.get("equity_curve") or [])]
                   for name, entry in per.items()},
        "buy_hold_pct": (result.get("metrics") or {}).get("buy_hold_pct"),
    }
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_a_disabled_orchestrator_row_changes_nothing(market_dir, tmp_path):
    """`enabled=False` ⇒ identical trades, identical equity, no metrics key."""
    _cfg, engine = _engine(market_dir, tmp_path)
    without = _run(engine, _always_long("p7s3_off_control"))
    disabled = _run(engine, _always_long("p7s3_off_control"),
                    orchestrator=RegimeOrchestrator({"enabled": False}))
    assert _fingerprint(without) == _fingerprint(disabled)
    assert (without.get("trades") or []) == (disabled.get("trades") or [])
    metrics = disabled.get("metrics") or {}
    assert "regime_orchestrator" not in metrics, (
        "the disabled path must not even report orchestrator accounting")
    assert "regime_conditioning" not in metrics


def test_a_disabled_orchestrator_reads_no_vol_or_breadth_series():
    machine = RegimeOrchestrator(
        {"enabled": False,
         "vol": {"multiple": 1.0, "min_samples": 2},
         "breadth": {"min_up_share": 0.99, "missing_action": "deny"}})
    machine.breadth_series = None                     # any read would raise
    machine.vol_series = None
    verdict = machine.decide("s", at=0, label="range_low", vol=1e9)
    assert verdict.enabled is True and verdict.reason == ALLOW


# ══════════════════════════════════════════════════════════════════════════
# 9 — the mechanism: the orchestrator avoids a losing stretch
# ══════════════════════════════════════════════════════════════════════════

def _gated_result(market_dir, tmp_path, consecutive_losses: int, mode: str = "replay"):
    """The engine's real trade stream, read through the orchestrator's rules.

    Two honest steps, because a *composite* number needs both:

    1. the production engine produces the variant's trades over the synthetic
       market (real fills, real costs — the shipped cost model);
    2. the orchestrator's enable/disable timeline is derived by **replaying that
       same trade stream through its rules** in chronological order — the regime
       label of each entry's bar drives regime changes and each trade's realised
       PnL drives the kill switch.

    Step 2 is the mechanism under test: the gated account is the always-on account
    with the refused trades removed, which is the composite contract P7-S4 will
    formalise.  Nothing is simulated twice and nothing is fitted.

    Returns ``(always_on_result, kept_trades, enabled_stamps)``.
    """
    _cfg, engine = _engine(market_dir, tmp_path)
    strategy = _always_long("p7s3_gated")
    always = _run(engine, strategy)
    trades = list(always.get("trades") or [])

    from core.strategy.regime_causal import causal_regime_table
    frame = pd.read_parquet(Path(market_dir) / "market" / "BTCUSDT" / "1h.parquet")
    index = list(causal_regime_table(frame).index)
    labels = causal_regime_table(frame)["regime"].astype(str)

    def label_of(stamp):
        cut = int(labels.index.searchsorted(stamp, side="right"))
        return None if cut <= 0 else str(labels.iloc[cut - 1])

    rules = {"enabled": True,
             "regime": {"allowed": {"p7s3_gated": ["all"]},
                        "default_action": "allow"},
             "kill_switch": {"consecutive_losses": consecutive_losses}}

    if mode == "incremental":
        # The incremental API: decide per bar (the timeline a live seam would
        # see), feeding each trade's realised outcome at its own exit bar.
        machine = RegimeOrchestrator(rules, names=["p7s3_gated"])
        by_exit: dict = {}
        for trade in trades:
            by_exit.setdefault(pd.Timestamp(trade.get("closed_at")), []).append(trade)
        enabled = set()
        for stamp in index:
            if machine.decide("p7s3_gated", at=stamp, label=label_of(stamp)).enabled:
                enabled.add(stamp)
            for trade in by_exit.get(stamp, []):
                machine.record_trade("p7s3_gated", trade.get("pnl"),
                                     at=trade.get("closed_at"))
    else:
        # The replay API (S4's entry point), driven by an explicit event list.
        events = []
        for trade in trades:
            events.append((pd.Timestamp(trade.get("opened_at")), True, trade))
            events.append((pd.Timestamp(trade.get("closed_at")), False, trade))
        events.sort(key=lambda item: (item[0], not item[1]))
        timeline = RegimeOrchestrator.replay(
            rules,
            [TimelineEvent("decide", at=stamp, label=label_of(stamp))
             if is_decide else
             TimelineEvent("trade", outcome=TradeOutcome(
                 "p7s3_gated", float(payload.get("pnl") or 0.0), ts=stamp))
             for stamp, is_decide, payload in events],
            names=["p7s3_gated"])
        enabled = {pd.Timestamp(row["at"]) for row in timeline
                   if row["rows"][0]["enabled"]}

    kept = [t for t in trades if pd.Timestamp(t.get("opened_at")) in enabled]
    return always, kept, enabled


def _pnl(trades) -> float:
    return round(float(sum(float(t.get("pnl") or 0.0) for t in trades)), 4)


def test_the_orchestrator_avoids_a_losing_stretch():
    """The mechanism, on a synthetic timeline with a known losing stretch.

    A 300-bar timeline: ``trend_up`` for bars 0–199 (the strategy wins) and
    ``trend_down`` for bars 200–299 (it loses every trade).  The kill switch is
    ``consecutive_losses: 2``, so within a handful of bars of the bad stretch the
    strategy is latched off and stays off until the label changes.  The gated
    account is the always-on account **minus the refused trades** — the same
    "reduce exposure" effect P7-S1 measured on real data, shown here on a stream
    where the answer is known by construction.
    """
    timeline = ([("trend_up", 12.0)] * 200) + ([("trend_down", -12.0)] * 100)
    captured: list = []

    def run(enabled: bool):
        machine = RegimeOrchestrator({
            "enabled": enabled,
            "regime": {"allowed": {"s": ["all"]}, "default_action": "allow"},
            "kill_switch": {"consecutive_losses": 2},
        }, names=["s"])
        executed = []
        for stamp, (label, pnl) in enumerate(timeline):
            machine.observe_regime("s", label, at=stamp)
            if machine.decide("s", at=stamp).enabled:
                executed.append((stamp, label, pnl))
            machine.record_trade("s", pnl, at=stamp)
        return executed

    always_on = run(False)
    orchestrated = run(True)
    captured.append(_pnl_stream(always_on))
    captured.append(_pnl_stream(orchestrated))

    assert len(always_on) == len(timeline), "the always-on arm must trade everything"
    assert len(orchestrated) < len(always_on)
    assert captured[1] > captured[0], (
        f"orchestrated pnl {captured[1]} is not better than always-on {captured[0]}")
    # The refused trades are a losing stretch, not a lucky subset.
    refused = [row for row in always_on if row not in orchestrated]
    assert refused and _pnl_stream(refused) < 0
    # And the refusal is exactly the documented rule: two losses and a latch.
    assert len(refused) >= 90, "the latch should hold for most of the bad stretch"


def _pnl_stream(rows) -> float:
    return round(float(sum(row[2] for row in rows)), 6)


def test_the_gated_timeline_is_deterministic_across_runs(market_dir, tmp_path):
    first_always, first_kept, first_enabled = _gated_result(market_dir, tmp_path, 2)
    second_always, second_kept, second_enabled = _gated_result(market_dir, tmp_path, 2)
    assert _fingerprint(first_always) == _fingerprint(second_always)
    assert first_enabled == second_enabled
    assert [(t["opened_at"], t["pnl"]) for t in first_kept] == \
        [(t["opened_at"], t["pnl"]) for t in second_kept]


def test_each_timeline_api_is_deterministic_on_its_own(market_dir, tmp_path):
    """Both APIs are deterministic; they agree the moment the events do.

    The incremental API decides on **every bar** (the live seam's shape), so a
    loss is counted at the trade's own exit bar and the *next* bar sees the latch.
    The replay API only sees the events it is handed, so when a caller hands it
    every bar (as a live loop would) the two coincide; when it is handed a
    coarser event list, a latch can trip one bar later by construction.  Both are
    deterministic — which is the property S4 needs — and the coarse list is
    reported as what it is rather than asserted to equal the fine one.
    """
    _always_replay, kept_replay, enabled_replay = _gated_result(
        market_dir, tmp_path, 2, mode="replay")
    _always_replay2, kept_replay2, enabled_replay2 = _gated_result(
        market_dir, tmp_path, 2, mode="replay")
    assert enabled_replay == enabled_replay2
    assert [(t["opened_at"], t["pnl"]) for t in kept_replay] == \
        [(t["opened_at"], t["pnl"]) for t in kept_replay2]

    _always_inc, kept_inc, enabled_inc = _gated_result(
        market_dir, tmp_path, 2, mode="incremental")
    _always_inc2, kept_inc2, enabled_inc2 = _gated_result(
        market_dir, tmp_path, 2, mode="incremental")
    assert enabled_inc == enabled_inc2
    assert [(t["opened_at"], t["pnl"]) for t in kept_inc] == \
        [(t["opened_at"], t["pnl"]) for t in kept_inc2]
    # The fine-grained decision set is a superset of the coarse one's: the replay
    # list is a subsequence of the bar-by-bar timeline, never a different story.
    assert enabled_replay <= enabled_inc


def test_the_engine_gate_reduces_the_sample_without_touching_what_it_keeps(
        market_dir, tmp_path):
    """The gate is a refusal, not a re-simulation: kept trades are the same trades."""
    always, kept, enabled = _gated_result(market_dir, tmp_path, 2)
    always_trades = always.get("trades") or []
    assert len(always_trades) >= 10
    assert len(kept) < len(always_trades)
    for trade in kept:
        assert trade in always_trades
        assert pd.Timestamp(trade["opened_at"]) in enabled


# ══════════════════════════════════════════════════════════════════════════
# 10 — the live seam (StrategyEngine) is inert while the switch is off
# ══════════════════════════════════════════════════════════════════════════

class _FakeMarketData:
    def __init__(self, frame):
        self.frame = frame
        self.watched_symbols = ["BTCUSDT"]

    async def get_historical(self, symbol, interval, limit=None):
        return self.frame.copy()

    def get_current_price(self, symbol):
        return float(self.frame["close"].iloc[-1])


@pytest.mark.asyncio
async def test_the_live_seam_is_inert_when_the_switch_is_off(monkeypatch):
    import asyncio

    from app.event_bus import EventBus
    from core.strategy import engine as strategy_engine
    from core.strategy.engine import StrategyEngine

    frame = _market_frame()
    payloads = []
    for orchestrator in (None, None,
                         RegimeOrchestrator({"enabled": False}),
                         RegimeOrchestrator({"enabled": True,
                                             "regime": {"default_action": "deny"}})):
        engine = StrategyEngine(_engine_config(), EventBus(), _FakeMarketData(frame))
        if orchestrator is not None:
            engine.wire_regime_orchestrator(orchestrator,
                                            lambda *_: "range_low")
        await engine._evaluate("BTCUSDT", "1h", _always_long("p7s3_live"),
                               publish=False)
        payloads.append(engine._signal_cache["p7s3_live|BTCUSDT"])
    assert strategy_engine.REGIME_ORCHESTRATOR_LIVE_ENABLED is False
    # Identical for all four: the seam is off, so even an enabled orchestrator
    # that would deny every bar cannot alter the signal cache.
    assert payloads[0] == payloads[1] == payloads[2] == payloads[3]
    assert "orchestrator" not in payloads[0]


@pytest.mark.asyncio
async def test_the_live_seam_vetoes_an_entry_when_all_three_locks_are_on(monkeypatch):
    from app.event_bus import EventBus
    from core.strategy import engine as strategy_engine
    from core.strategy.engine import StrategyEngine

    monkeypatch.setattr(strategy_engine, "REGIME_ORCHESTRATOR_LIVE_ENABLED", True)
    engine = StrategyEngine(_engine_config(), EventBus(),
                            _FakeMarketData(_market_frame()))
    machine = RegimeOrchestrator({"enabled": True,
                                  "regime": {"allowed": {"p7s3_live": ["trend_up"]},
                                             "default_action": "deny"}})
    engine.wire_regime_orchestrator(machine, lambda *_: "range_low")
    published = []
    original = engine.event_bus.publish

    async def _spy(event):
        published.append(event)
        return await original(event)

    monkeypatch.setattr(engine.event_bus, "publish", _spy)
    await engine._evaluate("BTCUSDT", "1h", _always_long("p7s3_live"),
                           publish=True)
    cache = engine._signal_cache["p7s3_live|BTCUSDT"]
    assert cache["orchestrator"]["enabled"] is False
    assert cache["orchestrator"]["reason"] == BLOCK_REGIME
    assert published == [], "a vetoed entry was published anyway"


def _engine_config():
    from app.config import Config
    Config._instance = None
    return Config.load("sim")


# ══════════════════════════════════════════════════════════════════════════
# 11 — the disabled identity against the pre-S3 worktree (d9849a2)
# ══════════════════════════════════════════════════════════════════════════

_IDENTITY_HARNESS = r'''
import hashlib
import json
import sys

sys.path.insert(0, sys.argv[1])

import numpy as np
import pandas as pd


def frame(n=300, seed=7):
    rng = np.random.default_rng(seed)
    steps = np.concatenate([0.0004 + rng.normal(0, 0.0008, n // 2),
                            -0.0030 + rng.normal(0, 0.0090, n - n // 2)])
    close = 20_000.0 * np.cumprod(1.0 + steps)
    return pd.DataFrame({"open": np.concatenate([[close[0]], close[:-1]]),
                         "high": close * 1.0005, "low": close * 0.9995,
                         "close": close, "volume": 100.0},
                        index=pd.date_range("2026-01-01", periods=n, freq="1h"))


def main(out_path):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager
    from core.strategy.loader import MLConfig, StrategyConfig

    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = sys.argv[2]
    cfg.backtest_engine_mode = "legacy"
    cfg.backtest_ml_enabled = False
    cfg.backtest_live_spread_enabled = False
    bus = EventBus()
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus),
                            OrderExecutor(cfg, bus))
    strategy = StrategyConfig(
        name="p7s3_identity", enabled=True, mode="trend", timeframes=["1h"],
        indicators={"rsi": {"period": 14, "source": "close"},
                    "sma": {"period": 5}},
        entry_conditions={"long": ["close > sma"], "short": ["close < sma"]},
        exit_conditions={"long": ["rsi > 99"], "short": ["rsi < 1"]},
        ml_config=MLConfig(enabled=False))
    kwargs = dict(strategies=[strategy], symbols=["BTCUSDT"],
                  date_start="2026-01-01", date_end="2026-03-31",
                  initial_balance=10_000.0, mode="full",
                  simulate_ai_weights=False, per_strategy_isolation=True,
                  per_genome_ledger=True, use_live_spread=False,
                  benchmark_mode="none")
    # The working tree accepts orchestrator=None; the frozen baseline has no such
    # parameter at all, so it is passed only where the signature has it.
    import inspect
    params = inspect.signature(BacktestEngine.run_with_exit_evaluation).parameters
    if "orchestrator" in params:
        from core.ai.orchestrator import RegimeOrchestrator
        kwargs["orchestrator"] = RegimeOrchestrator({"enabled": False})
    # P9 re-pin: the shipped default is now `next_open`, but every number this
    # comparison was built on (and the pre-S3 baseline it compares against) was
    # priced under `close`.  The historical convention is requested explicitly
    # where the tree has the seam; the baseline has no such parameter and prices
    # `close` anyway, which is exactly the behaviour being compared.
    if "fill_convention" in params:
        kwargs["fill_convention"] = "close"
    result = engine.run_with_exit_evaluation(**kwargs)
    # P9 added two metrics keys (`fill_convention`, `fill_convention_accounting`).
    # They are not part of the S3 contract and cannot exist in the baseline tree,
    # so the KEY SET is compared without them — every value the baseline produced
    # is still compared above (trades, equity, buy_hold_pct) and the old keys are
    # still required to be present, key for key.
    _p9_metrics = {"fill_convention", "fill_convention_accounting"}
    payload = {
        "trades": [{"opened_at": str(t.get("opened_at")),
                    "closed_at": str(t.get("closed_at")),
                    "pnl": t.get("pnl"), "cost": t.get("cost")}
                   for t in (result.get("trades") or [])],
        "equity": {name: [point["equity"] for point in
                          (entry.get("equity_curve") or [])]
                   for name, entry in
                   (result.get("per_strategy_equity") or {}).items()},
        "buy_hold_pct": (result.get("metrics") or {}).get("buy_hold_pct"),
        "metrics_keys": sorted(k for k in (result.get("metrics") or {})
                               if k not in _p9_metrics),
    }
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    sys.stdout.write(hashlib.sha256(text.encode("utf-8")).hexdigest() + "\n")


main(sys.argv[3])
'''


def _run_identity(tree: Path, tmp_path: Path, tag: str, market: str):
    harness = tmp_path / "p7s3_identity.py"
    harness.write_text(_IDENTITY_HARNESS, encoding="utf-8", newline="\n")
    out = tmp_path / f"p7s3_identity_{tag}.json"
    proc = subprocess.run([sys.executable, str(harness), str(tree), market,
                           str(out)],
                          cwd=str(tree), capture_output=True, text=True,
                          timeout=900)
    assert proc.returncode == 0, f"identity harness failed in {tree}:\n{proc.stderr}"
    return out.read_bytes(), proc.stdout.strip()


def test_a_disabled_orchestrator_is_bit_identical_to_the_pre_s3_worktree(
        market_dir, tmp_path):
    """The strongest form of "nothing changes": the same harness in both trees.

    A ``git worktree`` is checked out at ``d9849a2`` (the revision this work was
    developed on), the identical harness runs in both trees against the same
    synthetic market, and the traded facts — every trade's timestamps, PnL and
    cost, every per-genome equity point, the buy & hold benchmark and the metrics
    key set — are compared as bytes.
    """
    worktree = tmp_path / "pre_s3_tree"
    add = subprocess.run(["git", "worktree", "add", "--detach", str(worktree),
                          BASELINE_REVISION],
                         cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    if add.returncode != 0:
        pytest.skip(f"cannot create a worktree at {BASELINE_REVISION}: "
                    f"{add.stderr.strip()}")
    try:
        head_bytes, head_digest = _run_identity(worktree, tmp_path, "baseline",
                                                str(market_dir))
        tree_bytes, tree_digest = _run_identity(ROOT, tmp_path, "working",
                                                str(market_dir))
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=600)

    assert head_digest == tree_digest, (
        "the P7-S3 orchestrator changed an end-to-end run:\n"
        f"  {BASELINE_REVISION}: {head_digest}\n  working tree: {tree_digest}")
    assert head_bytes == tree_bytes
    payload = json.loads(tree_bytes.decode("utf-8"))
    assert payload["trades"], "the identity run traded nothing"
    assert "regime_orchestrator" not in payload["metrics_keys"]
    assert "regime_conditioning" not in payload["metrics_keys"]
