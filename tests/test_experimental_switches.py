"""The config-driven experimental-feature switch layer (`experimental:`).

Before this layer, the P4 capacities were only reachable by editing module-level
Python constants (``core/strategy/engine.py`` ``P4_*``, ``core/strategy/regime.py``
``REGIME_*``, ``core/strategy/pairs.py`` ``PAIRS_ENABLED``, ``core/ml/meta.py``
``META_LABELING_ENABLED``, ``core/market_data/microstructure.py``
``MICROSTRUCTURE_ENABLED``), while the project's own rule is that **every config
key must have a reader**.  ``config/config.yaml`` now carries an ``experimental:``
block, one switch per constant, all ``False`` as shipped.

What is asserted here, in order:

1. every key in the block has a reader: the YAML keys, the pydantic fields, the
   ``EXPERIMENTAL_FLAG_TARGETS`` table, the per-key notes and the layer grouping
   are the same set, every named constant really exists in the named module, and
   the app startup path really calls ``apply_experimental_flags``;
2. the defaults are all off, and applying an all-false block leaves every module
   constant at ``False`` **and imports/touches nothing** (that is the "no default
   behaviour change" contract, proven by inspecting a fresh interpreter);
3. enabling exactly one switch flips exactly its own constant and nothing else —
   and the one switch that changes reachable behaviour on its own
   (``engine_regime_diagnostics``) adds the regime label while leaving the fused
   score and the indicator signal untouched;
4. with everything off, an **end-to-end signal + sizing run is bit-identical to a
   HEAD worktree** (the pre-change revision is checked out with ``git worktree``,
   the same harness runs in both trees, and the serialised results are compared
   byte-for-byte);
5. the startup notice lists exactly the enabled switches and reminds the operator
   that each is still gated by its own acceptance test;
6. an unknown key in the block is **reported, not silently ignored**.

Honest scope note: four of the eight switches (``regime_diagnostics``,
``pairs_enabled``, ``meta_labeling_enabled``, ``microstructure_enabled``) have no
production reader at all, so enabling them changes nothing reachable — the notes,
the YAML comments and README §4.4 say so rather than implying the capability
works.  This file pins that honesty with ``EXPERIMENTAL_KEY_NOTES``.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml
from loguru import logger

from app.config import (
    EXPERIMENTAL_FLAG_TARGETS,
    EXPERIMENTAL_KEY_NOTES,
    EXPERIMENTAL_LAYERS,
    ExperimentalConfig,
    apply_experimental_flags,
    experimental_from_raw,
    experimental_notices,
)

ROOT = Path(__file__).resolve().parents[1]

#: The revision the bit-identity proof compares against.  Frozen instead of
#: ``HEAD`` on purpose: once this change is committed, ``HEAD`` *contains* the
#: switch layer, so the baseline has to be the pre-change tree.  ``3a140cf`` was
#: the HEAD the switch layer was developed on; P7-S3 moved the default to
#: ``d9849a2`` (the revision the orchestrator work was developed on, which
#: already carries all eight P4/P6 switches *and* the P7-S1 causal-regime layer),
#: so the two compared trees differ only by this change.  Override with
#: ``BT_EXPERIMENTAL_BASELINE``.
BASELINE_REVISION = os.environ.get("BT_EXPERIMENTAL_BASELINE", "d9849a2")


@pytest.fixture
def restore_flags():
    """Restore every experimental module constant after a test that flips one."""
    snapshot = {key: getattr(importlib.import_module(module), attribute)
                for key, (module, attribute) in EXPERIMENTAL_FLAG_TARGETS.items()}
    yield
    for key, (module, attribute) in EXPERIMENTAL_FLAG_TARGETS.items():
        setattr(importlib.import_module(module), attribute, snapshot[key])


def _shipped_block() -> dict:
    raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert "experimental" in raw, "config/config.yaml carries no experimental: block"
    return raw["experimental"]


# ══════════════════════════════════════════════════════════════════════
# 1 — every key has a reader (the audit guard, like the ml: / risk.liquidity ones)
# ══════════════════════════════════════════════════════════════════════

def test_every_experimental_key_has_a_reader():
    """No ``experimental:`` key may exist without a module constant behind it."""
    block = _shipped_block()
    fields = set(ExperimentalConfig.model_fields)
    assert set(block) == fields, (
        "the experimental: block and the model fields disagree — add the reader "
        "first, then the key")
    assert fields == set(EXPERIMENTAL_FLAG_TARGETS), (
        "every switch must name the module constant it flips")
    assert fields == set(EXPERIMENTAL_KEY_NOTES), (
        "every switch must carry an operator-facing note (what it does / does not do)")
    layered = [key for keys in EXPERIMENTAL_LAYERS.values() for key in keys]
    assert sorted(layered) == sorted(fields), (
        "the layer grouping must partition the switches")

    config_source = (ROOT / "app" / "config.py").read_text(encoding="utf-8")
    main_source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    # The reader is mechanical: apply_experimental_flags is driven by the table...
    assert "EXPERIMENTAL_FLAG_TARGETS" in config_source
    assert "setattr(module, attribute, True)" in config_source
    # ...and the app startup path really calls it (a table nobody calls is a comment).
    assert re.search(r"^from app\.config import .*apply_experimental_flags",
                     main_source, re.M), "app/main.py must import apply_experimental_flags"
    assert "apply_experimental_flags(config)" in main_source, (
        "app/main.py must apply the switches at startup")

    for key, (module_name, attribute) in EXPERIMENTAL_FLAG_TARGETS.items():
        module = importlib.import_module(module_name)
        assert hasattr(module, attribute), (
            f"experimental.{key} claims {module_name}.{attribute}, which does not exist")
        path = ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
        assert path.exists(), f"{module_name} (reader of {key}) does not exist"
        text = path.read_text(encoding="utf-8")
        assert attribute in text, f"{path.name} never names {attribute}"
        # A real switch, not a mention: the module assigns it and it ships False.
        assert re.search(rf"^{re.escape(attribute)}\s*=\s*False", text, re.M), (
            f"{module_name}.{attribute} is not an `= False` module switch")
        assert getattr(module, attribute) is False, (
            f"{module_name}.{attribute} must ship False (no default behaviour change)")


def test_shipped_block_is_all_false_and_documented():
    """The shipped block is inert, and its comments state the gating honestly."""
    block = _shipped_block()
    assert block == {key: False for key in ExperimentalConfig.model_fields}
    text = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    for key, (module_name, attribute) in EXPERIMENTAL_FLAG_TARGETS.items():
        assert attribute in text, f"config.yaml documents no {attribute} for {key}"
        relative = module_name.replace(".", "/") + ".py"
        assert relative in text, f"config.yaml names no layer for {key} ({relative})"
    # The two facts an operator must not have to infer: gates can refuse, and some
    # switches currently change nothing reachable.
    assert "0/30" in text and "10/10" in text, "the measured gate verdicts must be stated"
    assert text.count("无生产读取者") >= 4, (
        "each reader-less switch must say so in the YAML")


# ══════════════════════════════════════════════════════════════════════
# 2 — defaults are off; an all-false apply touches nothing
# ══════════════════════════════════════════════════════════════════════

def test_defaults_are_all_off_and_the_flags_stay_off(restore_flags, monkeypatch):
    from app.config import Config

    assert all(getattr(ExperimentalConfig(), key) is False
               for key in EXPERIMENTAL_FLAG_TARGETS)
    for key, (module_name, attribute) in EXPERIMENTAL_FLAG_TARGETS.items():
        assert getattr(importlib.import_module(module_name), attribute) is False

    # Applying the shipped (all-false) block, exactly as app/main.py does.
    applied = apply_experimental_flags(
        types.SimpleNamespace(experimental=ExperimentalConfig()))
    assert set(applied) == set(EXPERIMENTAL_FLAG_TARGETS)
    assert set(applied.values()) == {False}
    for key, (module_name, attribute) in EXPERIMENTAL_FLAG_TARGETS.items():
        assert getattr(importlib.import_module(module_name), attribute) is False

    # And the loaded config object agrees with the file.
    monkeypatch.setattr(Config, "_instance", None)
    cfg = Config.load("sim")
    assert all(getattr(cfg.experimental, key) is False
               for key in EXPERIMENTAL_FLAG_TARGETS)


#: Probe run in a fresh interpreter: applying the all-false block must import none
#: of the five capability modules and write no constant.
_NO_IMPORT_PROBE = r'''
import os
import sys
import types

sys.path.insert(0, os.getcwd())
import app.config as config_module

WATCHED = ("core.strategy.engine", "core.strategy.regime", "core.strategy.pairs",
           "core.ml.meta", "core.market_data.microstructure")
before = set(sys.modules)
applied = config_module.apply_experimental_flags(
    types.SimpleNamespace(experimental=config_module.ExperimentalConfig()))
leaked = sorted(m for m in WATCHED if m in sys.modules and m not in before)
print("values=" + ",".join(sorted({str(v) for v in applied.values()})))
print("leaked=" + ",".join(leaked))
'''


def test_all_false_apply_imports_and_writes_nothing(tmp_path):
    """The default path must not even import the capability modules."""
    probe = tmp_path / "probe.py"
    probe.write_text(_NO_IMPORT_PROBE, encoding="utf-8", newline="\n")
    proc = subprocess.run([sys.executable, str(probe)], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines == ["values=False", "leaked="], proc.stdout


# ══════════════════════════════════════════════════════════════════════
# 3 — one switch flips exactly its own flag
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("key", sorted(EXPERIMENTAL_FLAG_TARGETS))
def test_enabling_one_switch_flips_exactly_its_own_flag(key, restore_flags):
    experiment = ExperimentalConfig(**{key: True})
    applied = apply_experimental_flags(types.SimpleNamespace(experimental=experiment))
    assert applied[key] is True
    assert [name for name, value in applied.items() if value] == [key]
    for other, (module_name, attribute) in EXPERIMENTAL_FLAG_TARGETS.items():
        value = getattr(importlib.import_module(module_name), attribute)
        assert value is (other == key), (
            f"enabling experimental.{key} moved {other} to {value}")


# ══════════════════════════════════════════════════════════════════════
# 3b — the one switch that IS reachable on its own changes diagnostics only
# ══════════════════════════════════════════════════════════════════════

class _FakeMarketData:
    """Deterministic stand-in for ``MarketDataProvider`` (no network, no disk)."""

    def __init__(self, n: int = 220, seed: int = 0):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        self.df = pd.DataFrame(
            {"open": close, "high": close + 0.5, "low": close - 0.5,
             "close": close, "volume": rng.random(n) * 10 + 1},
            index=pd.date_range("2026-01-01", periods=n, freq="1h"))
        self.watched_symbols = ["BTCUSDT"]

    async def get_historical(self, symbol, interval, limit=None):
        return self.df.copy()

    def get_current_price(self, symbol):
        return float(self.df["close"].iloc[-1])


@pytest.mark.asyncio
async def test_engine_regime_diagnostics_is_the_one_self_reachable_switch(restore_flags):
    """`engine_regime_diagnostics` adds the regime label — and nothing else.

    The docs claim it is the only switch that changes anything reachable without
    a caller registering a component (``_p4_regime`` is called from the live
    ``_evaluate``).  That claim is pinned here: the signal cache gains a ``p4``
    payload with a regime label, while the fused score and the indicator signal
    are untouched — and with the switch off no ``p4`` key appears at all.
    """
    from app.config import Config
    from app.event_bus import EventBus
    from core.strategy.engine import StrategyEngine
    from core.strategy.loader import StrategyConfig

    def _engine():
        Config._instance = None
        engine = StrategyEngine(Config.load("sim"), EventBus(), _FakeMarketData())
        strategy = StrategyConfig(
            name="exp_reach", enabled=True, mode="trend", timeframes=["1h"],
            indicators={"rsi": {"period": 14, "source": "close"}},
            entry_conditions={"long": ["close > 0"], "short": ["close < 0"]})
        return engine, strategy

    engine, strategy = _engine()
    await engine._evaluate("BTCUSDT", "1h", strategy)
    baseline = engine._signal_cache["exp_reach|BTCUSDT"]
    assert "p4" not in baseline

    apply_experimental_flags(types.SimpleNamespace(
        experimental=ExperimentalConfig(engine_regime_diagnostics=True)))
    engine, strategy = _engine()
    await engine._evaluate("BTCUSDT", "1h", strategy)
    entry = engine._signal_cache["exp_reach|BTCUSDT"]
    assert "p4" in entry, "the enabled diagnostic switch must attach the regime"
    assert entry["p4"]["regime"] and entry["p4"]["regime"]["regime"]
    # Diagnostics only: the tradeable numbers are identical.
    assert entry["final_score"] == baseline["final_score"]
    assert entry["indicator_signal"] == baseline["indicator_signal"]
    assert entry["threshold_met"] == baseline["threshold_met"]


# ══════════════════════════════════════════════════════════════════════
# 4 — bit-identical to a HEAD worktree with everything off
# ══════════════════════════════════════════════════════════════════════

#: The harness is run **unchanged** in both trees (it lives in the pytest tmp dir,
#: not in either checkout) and must therefore work on the pre-change revision too.
#: It mirrors the startup path: load the config, apply the switches if that
#: function exists, then evaluate two strategies end-to-end and size a grid of
#: positions.  The serialised result is what gets compared byte-for-byte.
_IDENTITY_HARNESS = r'''
import asyncio
import hashlib
import importlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd


def frame(n=220, seed=0):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": rng.random(n) * 10 + 1},
        index=pd.date_range("2026-01-01", periods=n, freq="1h"))


class FakeMarketData:
    def __init__(self):
        self.df = frame()
        self.watched_symbols = ["BTCUSDT"]

    async def get_historical(self, symbol, interval, limit=None):
        return self.df.copy()

    def get_current_price(self, symbol):
        return float(self.df["close"].iloc[-1])


def load_config():
    import app.config as config_module

    config_module.Config._instance = None
    cfg = config_module.Config.load("sim")
    apply = getattr(config_module, "apply_experimental_flags", None)
    if apply is not None:          # absent on the pre-change revision
        apply(cfg)
    return cfg


def flag_values(keys=None):
    """Read the module constants (the pre-change tree has no table).

    ``keys`` (optional) restricts the reading to the switches **both** compared
    revisions know about.  The bit-identity harness runs unchanged in two trees,
    and P7-S3 added one switch to the table: without the filter the flags block —
    and therefore the digest — would differ by that one *additive* key even though
    every signal and every size is byte-identical.  The comparison is still
    exhaustive for every pre-existing switch (and
    ``test_every_experimental_key_has_a_reader`` pins the full table).
    """
    import app.config as config_module

    targets = getattr(config_module, "EXPERIMENTAL_FLAG_TARGETS", None)
    if targets is None:
        targets = {
            "engine_regime_diagnostics": (
                "core.strategy.engine", "P4_REGIME_DIAGNOSTICS_ENABLED"),
            "engine_meta_filter": ("core.strategy.engine", "P4_META_FILTER_ENABLED"),
            "engine_pairs_signals": (
                "core.strategy.engine", "P4_PAIRS_SIGNALS_ENABLED"),
            "regime_gating": ("core.strategy.regime", "REGIME_GATING_ENABLED"),
            "regime_diagnostics": (
                "core.strategy.regime", "REGIME_DIAGNOSTICS_ENABLED"),
            "pairs_enabled": ("core.strategy.pairs", "PAIRS_ENABLED"),
            "meta_labeling_enabled": ("core.ml.meta", "META_LABELING_ENABLED"),
            "microstructure_enabled": (
                "core.market_data.microstructure", "MICROSTRUCTURE_ENABLED"),
        }
    if keys is not None:
        targets = {key: value for key, value in targets.items() if key in set(keys)}
    return {key: bool(getattr(importlib.import_module(mod), attr))
            for key, (mod, attr) in sorted(targets.items())}


async def signals():
    from app.event_bus import EventBus
    from core.strategy.engine import StrategyEngine
    from core.strategy.loader import StrategyConfig

    engine = StrategyEngine(load_config(), EventBus(), FakeMarketData())
    strategies = [
        StrategyConfig(
            name="exp_probe_long", enabled=True, mode="trend", timeframes=["1h"],
            indicators={"rsi": {"period": 14, "source": "close"}},
            entry_conditions={"long": ["close > 0"], "short": ["close < 0"]}),
        StrategyConfig(
            name="exp_probe_short", enabled=True, mode="trend", timeframes=["1h"],
            indicators={"rsi": {"period": 14, "source": "close"}},
            entry_conditions={"long": ["close < 0"], "short": ["close > 0"]}),
    ]
    out = {}
    for strategy in strategies:
        await engine._evaluate("BTCUSDT", "1h", strategy)
        out[strategy.name] = engine._signal_cache[strategy.name + "|BTCUSDT"]
    return out

def sizing():
    from core.risk.position_sizer import PositionSizer

    cfg = load_config()
    sizer = PositionSizer(cfg.hard_limits, cfg.soft_params, cfg.core_capital_pct,
                          cfg.satellite_capital_pct, cfg.risk_vol_targeting,
                          cfg.risk_liquidity)
    out = {}
    grid = ((10000.0, 100.0, False, None, None),
            (10000.0, 100.0, False, 0.45, None),
            (10000.0, 100.0, True, 0.45, None),
            (25000.0, 3.5, False, 0.9, 2000000.0),
            (4321.0, 65000.0, True, None, 750000.0))
    for kind in ("core", "satellite"):
        for balance, price, expanding, forecast, volume in grid:
            key = "%s|%.2f|%.4f|%s|%s|%s" % (kind, balance, price, expanding,
                                             forecast, volume)
            out[key] = sizer.calculate_position_size(
                balance, price, kind, expanding, forecast, volume, "BTCUSDT")
    return out


payload = {"flags": flag_values(sys.argv[2].split(",") if len(sys.argv) > 2 else None),
           "signals": asyncio.run(signals()),
           "sizing": sizing()}
text = json.dumps(payload, sort_keys=True, default=str, indent=1)
sys.stdout.write(hashlib.sha256(text.encode("utf-8")).hexdigest() + "\n")
with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)
'''


def _baseline_flag_keys(tree: Path) -> list[str]:
    """The experimental switch names the frozen baseline revision knows about.

    Run in the baseline worktree (not imported into this process — the two trees
    must not share a module cache), so the identity comparison covers exactly the
    switches that existed on both sides.
    """
    probe = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import app.config as config_module\n"
        "targets = getattr(config_module, 'EXPERIMENTAL_FLAG_TARGETS', None)\n"
        "if targets is None:\n"
        "    targets = {'engine_regime_diagnostics': 1, 'engine_meta_filter': 1,\n"
        "               'engine_pairs_signals': 1, 'regime_gating': 1,\n"
        "               'regime_diagnostics': 1, 'pairs_enabled': 1,\n"
        "               'meta_labeling_enabled': 1, 'microstructure_enabled': 1}\n"
        "print(','.join(sorted(targets)))\n")
    proc = subprocess.run([sys.executable, "-c", probe, str(tree)],
                          cwd=str(tree), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"cannot read the baseline switch table:\n{proc.stderr}"
    return [key for key in proc.stdout.strip().split(",") if key]


def _run_harness(tree: Path, tmp_path: Path, name: str,
                 flag_keys: set | None = None):
    """Run the identity harness with ``tree`` as the import root.

    ``flag_keys`` restricts the flags block to the switches both compared trees
    know about (see :func:`flag_values`); the signals and sizing blocks are never
    restricted.
    """
    harness = tmp_path / "identity_harness.py"
    harness.write_text(_IDENTITY_HARNESS, encoding="utf-8", newline="\n")
    out = tmp_path / f"{name}.json"
    argv = [sys.executable, str(harness), str(out)]
    if flag_keys is not None:
        argv.append(",".join(sorted(flag_keys)))
    proc = subprocess.run(argv, cwd=str(tree),
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"harness failed in {tree}:\n{proc.stderr}"
    return out.read_bytes(), proc.stdout.strip()


def test_signals_and_sizing_are_bit_identical_to_the_head_worktree(tmp_path):
    """Everything off ⇒ byte-for-byte the same signals and sizes as HEAD.

    A ``git worktree`` is checked out at the frozen pre-change revision, the same
    harness runs in both trees (the working tree has the ``experimental:`` block
    all-false; the worktree has no block at all), and the serialised signal cache
    plus 10 sizing results are compared as bytes.
    """
    worktree = tmp_path / "head_tree"
    add = subprocess.run(["git", "worktree", "add", "--detach", str(worktree),
                          BASELINE_REVISION],
                         cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    if add.returncode != 0:
        pytest.skip(f"cannot create a HEAD worktree at {BASELINE_REVISION}: "
                    f"{add.stderr.strip()}")
    try:
        # Both trees report the switches the BASELINE knows about, so a switch
        # added after the baseline cannot move the digest by itself (its own
        # inertness is pinned by test_every_experimental_key_has_a_reader and by
        # tests/test_p7_orchestrator.py's own worktree proof).
        head_keys = set(_baseline_flag_keys(worktree))
        head_bytes, head_digest = _run_harness(worktree, tmp_path, "head", head_keys)
        tree_bytes, tree_digest = _run_harness(ROOT, tmp_path, "working", head_keys)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=600)

    assert head_digest == tree_digest, (
        "the switch layer changed the end-to-end result:\n"
        f"  HEAD {BASELINE_REVISION}: {head_digest}\n  working tree: {tree_digest}")
    assert head_bytes == tree_bytes, "the harness output differs byte-for-byte"

    payload = json.loads(tree_bytes.decode("utf-8"))
    # The comparison is only meaningful if it really covered the seams: every
    # switch the BASELINE knows about is present and false.  (P7-S3's
    # `regime_orchestrator_live` is deliberately outside the compared set — it did
    # not exist at the baseline, and its own inertness is pinned by its reader
    # test and by tests/test_p7_orchestrator.py.)
    assert set(payload["flags"]) == set(head_keys)
    assert head_keys <= set(EXPERIMENTAL_FLAG_TARGETS)
    assert set(payload["flags"].values()) == {False}, payload["flags"]
    assert len(payload["signals"]) == 2
    assert len(payload["sizing"]) == 10
    for entry in payload["signals"].values():
        assert "p4" not in entry, "an experimental payload key leaked into the default cache"
        assert "orchestrator" not in entry, (
            "a P7-S3 payload key leaked into the default cache")
    assert hashlib.sha256(tree_bytes).hexdigest() == \
        hashlib.sha256(head_bytes).hexdigest()


# ══════════════════════════════════════════════════════════════════════
# 5 — the startup notice lists exactly the enabled switches
# ══════════════════════════════════════════════════════════════════════

def test_notice_is_silent_when_nothing_is_enabled():
    assert experimental_notices(ExperimentalConfig()) == []
    assert experimental_notices(None) == []


def test_notice_lists_exactly_the_enabled_switches():
    enabled = ("engine_pairs_signals", "microstructure_enabled")
    experiment = ExperimentalConfig(**{key: True for key in enabled})
    messages = experimental_notices(experiment)
    assert len(messages) == 1
    text = messages[0]
    assert f"EXPERIMENTAL FEATURES ENABLED (2 of {len(EXPERIMENTAL_FLAG_TARGETS)})" in text
    assert "gated by its OWN acceptance test" in text
    for key in enabled:
        assert f"experimental.{key}" in text
        assert EXPERIMENTAL_FLAG_TARGETS[key][1] in text      # the constant it flips
    for other in set(EXPERIMENTAL_FLAG_TARGETS) - set(enabled):
        assert f"experimental.{other}" not in text, (
            f"the notice named {other}, which is not enabled")

    all_on = experimental_notices(
        ExperimentalConfig(**{key: True for key in EXPERIMENTAL_FLAG_TARGETS}))
    assert (f"EXPERIMENTAL FEATURES ENABLED "
            f"({len(EXPERIMENTAL_FLAG_TARGETS)} of "
            f"{len(EXPERIMENTAL_FLAG_TARGETS)})") in all_on[0]
    for key in EXPERIMENTAL_FLAG_TARGETS:
        assert f"experimental.{key}" in all_on[0]
    # Every layer the YAML groups by is a heading in the notice, so the operator
    # can read which layer an enabled switch belongs to.
    for layer in EXPERIMENTAL_LAYERS:
        assert f"[{layer}]" in all_on[0]


def test_startup_logs_the_notice_and_the_reader_less_switches(monkeypatch):
    """`Config._load` emits the notice through loguru (the audit's own pattern)."""
    from app.config import Config

    captured: list[str] = []
    sink = logger.add(lambda message: captured.append(message.record["message"]),
                      level="WARNING")
    try:
        def _fake_load_yaml(self, relative_path):
            self._data = Config._deep_merge(self._data, {
                "experimental": {"engine_pairs_signals": True,
                                 "pairs_enabled": True,
                                 "not_a_switch": True}})

        monkeypatch.setattr(Config, "_instance", None)
        monkeypatch.setattr(Config, "_load_yaml", _fake_load_yaml)
        cfg = Config.load("sim")
    finally:
        logger.remove(sink)

    assert cfg.experimental.engine_pairs_signals is True
    assert cfg.experimental.pairs_enabled is True
    joined = "\n".join(captured)
    assert f"EXPERIMENTAL FEATURES ENABLED (2 of {len(EXPERIMENTAL_FLAG_TARGETS)})" in joined
    assert "experimental.engine_pairs_signals" in joined
    assert "experimental.pairs_enabled" in joined
    assert "NO production reader" in joined          # the honest note
    assert "experimental.not_a_switch" in joined     # the unknown key is reported
    assert cfg.experimental_notice_messages, "the notice must be kept on the config"


# ══════════════════════════════════════════════════════════════════════
# 6 — an unknown key is reported, never silently ignored
# ══════════════════════════════════════════════════════════════════════

def test_unknown_key_is_reported_not_ignored():
    experiment, messages = experimental_from_raw(
        {"engine_pairs_signals": True, "no_such_switch": True})
    assert experiment.engine_pairs_signals is True
    assert not hasattr(experiment, "no_such_switch")
    assert len(messages) == 1
    assert "experimental.no_such_switch is not a known switch" in messages[0]
    assert "IGNORED" in messages[0]
    assert "engine_pairs_signals" in messages[0]      # lists the known switches

    # Every unknown key gets its own line — none is dropped.
    _, messages = experimental_from_raw({"a": True, "b": True})
    assert len(messages) == 2
    assert "experimental.a is not a known switch" in messages[0]
    assert "experimental.b is not a known switch" in messages[1]

    # A block that is not a mapping is reported too (never a silent all-off).
    experiment, messages = experimental_from_raw(["engine_pairs_signals"])
    assert experiment == ExperimentalConfig()
    assert messages and "must be a mapping" in messages[0]

    # The shipped block parses clean: a known key is never reported as unknown.
    experiment, messages = experimental_from_raw(_shipped_block())
    assert messages == []
    assert experiment == ExperimentalConfig()

    # ...and a junk value inside a known key is coerced (documented _as_bool
    # semantics), not turned into an unknown-key error.
    experiment, messages = experimental_from_raw({"pairs_enabled": "yes"})
    assert messages == [] and experiment.pairs_enabled is True
    experiment, messages = experimental_from_raw({"pairs_enabled": 0})
    assert messages == [] and experiment.pairs_enabled is False
