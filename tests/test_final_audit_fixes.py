"""Final independent audit fixes (revision `fe11ccf`), findings F1, F3, F4, F5.

One section per finding, each asserting the *defect* is gone and the default
configuration is untouched:

* **F1** — the live volatility gap guard never fired.  ``RiskManager`` buffered
  bare float closes, so the fallback built ``pd.DataFrame({"close": closes})``
  with a **RangeIndex**; ``_series_has_gap`` then subtracted consecutive ints
  (``1 - 0 = 1`` "second") and always returned ``False``.  Measured before: 300
  buffered closes with a hidden 100-bar hole → index dtype ``int64``,
  ``gap_guard=False``, forecast **0.0698 %/bar**; with a ``DatetimeIndex`` →
  ``gap_guard=True`` (refused).
* **F3** — a missing t-statistic made the significance check *skip entirely*:
  ``credibility_gate({auc .60, n 5000, n_trades 300}, 0.004, t_stat=None,
  psr=None)`` returned ``allowed=True, reason="pass"``.  Also, the documented
  ``t > 2`` **or** ``PSR >= 0.95`` is effectively ``t >= 1.645`` under normality
  (``_norm_cdf(1.645) = 0.95``), so it is now an **AND**.
* **F4** — ``skip_ml_training`` loaded ``data/models/<SYM>_<strategy>_binary.pkl``
  with no gate re-check and no ``*_meta.json`` / schema-hash verification.
* **F5** — every ``ml:`` key has a reader, and ``default_meta_cost_pct(None, …)``
  is the same source as fills (0.26 % for ETHUSDT, not the 0.14 % fallback).

Nothing here writes ``data/``, ``strategies/`` or the live DB.
"""
from __future__ import annotations

import asyncio
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

FEATURE_HASH = "335e63360104"     # core.ml.features.feature_schema_hash(FEATURE_NAMES)
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    from app.config import Config
    yield
    Config._instance = None


def _config(tmp_path):
    from app.config import Config
    Config._instance = None
    cfg = Config.load("sim")
    cfg.db_path = str(tmp_path / "final_audit.db")
    cfg.backtest_ml_enabled = False
    cfg.backtest_live_spread_enabled = False
    return cfg


def _risk_manager(cfg):
    from app.event_bus import EventBus
    from core.risk.manager import RiskManager
    rm = RiskManager(cfg, EventBus())
    rm.update_balance(10_000.0)
    return rm


def _feed(rm, prices, *, times=None, interval="1h", symbol="BTCUSDT",
          close_time=True):
    """Publish ``MARKET_KLINE`` candles into the manager's live buffer."""
    from app.event_bus import Event, EventType
    for i, price in enumerate(prices):
        candle = {"close": float(price)}
        if close_time:
            stamp = (times[i] if times is not None
                     else pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i))
            candle["close_time"] = int(pd.Timestamp(stamp).timestamp() * 1000)
        asyncio.run(rm._on_kline(Event(EventType.MARKET_KLINE, {
            "symbol": symbol, "interval": interval, "candle": candle})))


def _walk(n=300, seed=20260930):
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0.0, 0.4, n))


# ══════════════════════════════════════════════════════════════════════
# F1 — the live volatility gap guard must actually run
# ══════════════════════════════════════════════════════════════════════

def test_int_index_fallback_cannot_silently_pass_the_gap_guard():
    """F1: the old fallback frame's index dtype made the guard inert.

    Documents the defect *and* proves it is unreachable now: the manager never
    builds a frame from bare floats, and a hand-built RangeIndex still cannot
    masquerade as a guarded series.
    """
    from core.risk.manager import _series_has_gap

    closes = list(_walk(300))
    bare = pd.DataFrame({"close": closes})              # the audited fallback
    assert bare.index.dtype == np.dtype("int64"), "the defect's precondition"
    assert _series_has_gap(bare.index, "1h") is False, (
        "a RangeIndex (0,1,2…) is compared as 1-second bars, so the guard never "
        "fires")

    # Same prices, real timestamps, one 100-bar hole: the guard fires.
    idx = pd.date_range("2026-01-01", periods=300, freq="1h")
    hidden = idx[:150].append(
        pd.date_range(idx[149] + pd.Timedelta(hours=100), periods=150, freq="1h"))
    assert _series_has_gap(pd.DatetimeIndex(hidden), "1h") is True
    # ... and a gap-free DatetimeIndex still passes.
    assert _series_has_gap(idx, "1h") is False


def test_live_kline_buffer_detects_a_hidden_gap_and_refuses_the_forecast(tmp_path):
    """F1: the buffered live series is timestamp-indexed and gap-guarded."""
    cfg = _config(tmp_path)
    cfg.risk_vol_targeting.enabled = True
    rm = _risk_manager(cfg)

    idx = pd.date_range("2026-01-01", periods=300, freq="1h")
    spliced = idx[:150].append(
        pd.date_range(idx[149] + pd.Timedelta(hours=100), periods=150, freq="1h"))
    # The last leg jumps 30 %, exactly the fake bar that would dominate an EWMA.
    prices = np.concatenate([_walk(300)[:150], _walk(300)[150:] * 1.3])
    _feed(rm, prices, times=spliced)

    frame = rm._buffered_frame("BTCUSDT", "1h")
    assert frame is not None and isinstance(frame.index, pd.DatetimeIndex)
    assert str(frame.index.dtype).startswith("datetime64")
    assert rm._market_data is None, "no REST source was involved"
    # The forecast is refused (sizing falls back to the fixed fraction).
    assert asyncio.run(rm.forecast_vol_pct("BTCUSDT", "1h")) is None
    assert asyncio.run(rm.resolve_forecast_vol_pct(
        {"symbol": "BTCUSDT", "timeframe": "1h"})) is None

    # A gap-free buffer of the same length still produces a forecast.
    rm2 = _risk_manager(cfg)
    _feed(rm2, _walk(200, seed=7))
    vol = asyncio.run(rm2.forecast_vol_pct("BTCUSDT", "1h"))
    assert vol is not None and 0.0 < vol < 5.0, (
        "a contiguous live buffer must still be sized by its forecast")


def test_timestampless_kline_is_refused_not_buffered(tmp_path):
    """F1: with no timestamp the guard cannot be applied, so the bar is dropped.

    The alternative (buffer the close, build a RangeIndex frame) is the audited
    defect.  A refusal is visible in the log and leaves the forecast unavailable.
    """
    cfg = _config(tmp_path)
    cfg.risk_vol_targeting.enabled = True
    rm = _risk_manager(cfg)
    _feed(rm, _walk(120), close_time=False)

    assert rm._kline_history == {}, "an un-timestamped candle must not be buffered"
    assert rm._recent_bars("BTCUSDT", "1h") is None
    assert rm._buffered_frame("BTCUSDT", "1h") is None
    assert asyncio.run(rm.forecast_vol_pct("BTCUSDT", "1h")) is None


def test_vol_targeting_off_is_bit_identical_for_sizing_and_stop(tmp_path):
    """F1 constraint: the default configuration (vol targeting off) is untouched."""
    from core.risk.position_sizer import PositionSizer

    cfg = _config(tmp_path)
    assert cfg.risk_vol_targeting.enabled is False
    rm = _risk_manager(cfg)
    # Nothing is buffered even if events arrive.
    _feed(rm, _walk(120))
    assert rm._kline_history == {}
    assert asyncio.run(rm.forecast_vol_pct("BTCUSDT", "1h")) is None

    signal = {"symbol": "BTCUSDT", "side": "long", "price": 50_000.0,
              "position_type": "satellite", "timeframe": "1h", "leverage": 2}
    result = asyncio.run(rm.check_signal(dict(signal)))
    assert result.approved, result.reason

    legacy = PositionSizer(cfg.hard_limits, cfg.soft_params,
                           cfg.core_capital_pct, cfg.satellite_capital_pct)
    qty, _ = legacy.calculate_position_size(10_000.0, 50_000.0, "satellite")
    stop = legacy.calculate_stop_loss(50_000.0, "long")
    assert result.adjusted_quantity == qty          # bit-identical, not approx
    assert result.adjusted_stop_loss == stop
    assert result.adjusted_quantity == 0.0048


# ══════════════════════════════════════════════════════════════════════
# F3 — missing significance evidence must refuse; the floors are an AND
# ══════════════════════════════════════════════════════════════════════

def test_missing_t_stat_or_psr_is_refused_with_a_named_reason():
    """F3: `t_stat=None` used to skip the check and return "pass"."""
    from core.ml.credibility import credibility_gate

    metrics = {"auc": 0.60, "n": 5000, "n_trades": 300, "cost_pct": 0.26}
    # The audited measurement: allowed=True, reason="pass".
    status = credibility_gate(dict(metrics), 0.004, t_stat=None, psr=None)
    assert status["allowed"] is False
    assert status["enabled"] is False
    assert "no significance evidence" in status["reason"]
    assert "t_stat" in status["reason"] and "psr" in status["reason"]
    assert status["t_stat"] is None and status["psr"] is None

    # Each half missing on its own is still a refusal naming that piece.
    only_t = credibility_gate(dict(metrics), 0.004, t_stat=2.6)
    assert only_t["allowed"] is False and "psr missing" in only_t["reason"]
    only_psr = credibility_gate(dict(metrics), 0.004, psr=0.99)
    assert only_psr["allowed"] is False and "t_stat missing" in only_psr["reason"]

    # A payload with the real numbers still passes — nothing else moved.
    good = credibility_gate(dict(metrics), 0.004, t_stat=2.6, psr=0.99)
    assert good["allowed"] is True and good["reason"] == "pass"


def test_significance_floors_are_a_conjunction_at_the_boundary():
    """F3 decision: **AND**, with the documented 2.0 / 0.95 values.

    Why AND and not OR: PSR is the one-sided normal probability, so
    ``PSR >= 0.95`` is exactly ``t >= 1.645`` under normality — an OR makes the
    documented 2.0 t floor dead (measured: ``t=1.65, PSR=0.9505`` → allowed).
    The floors are not redundant off the normal (PSR reads skew and kurtosis), and
    AND is strictly stricter than OR, so it can only refuse more.
    """
    from core.ml.credibility import credibility_gate

    metrics = {"auc": 0.60, "n": 5000, "n_trades": 300, "cost_pct": 0.26}

    def psr_normal(t: float) -> float:               # t = 1.645 → PSR = 0.95
        return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))

    assert psr_normal(1.645) == pytest.approx(0.95, abs=1e-3)
    assert psr_normal(1.60) < 0.95 and psr_normal(1.70) > 0.95

    below = credibility_gate(dict(metrics), 0.004,
                             t_stat=1.60, psr=psr_normal(1.60))
    above = credibility_gate(dict(metrics), 0.004,
                             t_stat=1.70, psr=psr_normal(1.70))
    assert below["allowed"] is False and "not significant" in below["reason"]
    assert above["allowed"] is False, (
        "t=1.70 clears the old effective 1.645 floor but not the documented 2.0")
    assert below["min_t_stat"] == pytest.approx(2.0)
    assert above["min_t_stat"] == pytest.approx(2.0)

    # Both floors cleared → allowed; the t floor alone at the boundary is not.
    assert credibility_gate(dict(metrics), 0.004, t_stat=2.01,
                            psr=psr_normal(2.01))["allowed"] is True
    assert credibility_gate(dict(metrics), 0.004, t_stat=2.0,
                            psr=0.99)["allowed"] is False       # ">" not ">="
    assert credibility_gate(dict(metrics), 0.004, t_stat=3.0,
                            psr=0.94)["allowed"] is False       # PSR floor binds


def test_legacy_gate_branch_cannot_enable_a_model_without_outer_stats():
    """F3: the legacy ``gate_from_evaluation`` branch (no ``*_oos`` keys)."""
    from core.ml.credibility import gate_from_evaluation

    legacy = {"metrics": {"auc": 0.62, "n": 5000}, "n_oos": 5000,
              "thresholds": {"expectancy": 0.004, "n_taken": 300}}
    refused = gate_from_evaluation(legacy)
    assert refused["allowed"] is False and refused["enabled"] is False
    assert "no significance evidence" in refused["reason"]

    with_stats = dict(legacy)
    with_stats["thresholds"] = {"expectancy": 0.004, "n_taken": 300,
                                "t_stat": 2.6, "psr": 0.99}
    assert gate_from_evaluation(with_stats)["allowed"] is True


def test_ml_gate_min_psr_is_read_by_the_predictor(tmp_path):
    """F5.2: `ml.gate_min_psr` used to be loaded and read nowhere."""
    from core.ml.predictor import MLPredictor

    cfg = _config(tmp_path)
    cfg.ml_gate_min_psr = 0.97
    cfg.ml_feature_list = None
    predictor = MLPredictor(cfg, None, None)
    gate_cfg = predictor._gate_config()
    assert gate_cfg["min_psr"] == pytest.approx(0.97)
    # ... and the value reaches the gate instead of being dropped.
    from core.ml.credibility import gate_from_evaluation
    status = gate_from_evaluation(
        {"metrics": {"auc": 0.62, "n": 5000}, "n_oos": 5000,
         "net_expectancy_oos": 0.004, "n_trades_oos": 300,
         "t_stat_oos": 3.0, "psr_oos": 0.96},
        min_trades=100, min_t_stat=2.0, min_psr=0.97, min_oos=100)
    assert status["allowed"] is False and "not significant" in status["reason"]


# ══════════════════════════════════════════════════════════════════════
# F4 — the web-backtest preload must verify the gate + sidecar + schema
# ══════════════════════════════════════════════════════════════════════

class _StubModel:
    def predict_proba(self, X):
        return np.tile(np.array([[0.4, 0.6]]), (len(X), 1))


def _write_artefact(models_dir: Path, stem: str, *, meta: dict | None):
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{stem}.pkl").write_bytes(pickle.dumps(_StubModel()))
    if meta is not None:
        (models_dir / f"{stem}_meta.json").write_text(
            json.dumps(meta), encoding="utf-8")
    return models_dir / f"{stem}.pkl"


def _engine(tmp_path):
    from core.backtest.engine import BacktestEngine
    cfg = _config(tmp_path)
    engine = BacktestEngine(cfg, None, None, None)
    engine._current_strategies = []
    return engine, cfg


def test_preload_refuses_an_ungated_pickle_and_loads_a_gated_one(tmp_path):
    """F4: the audited loop loaded every existing ``.pkl`` unconditionally."""
    from core.ml.features import FEATURE_NAMES, feature_schema_hash

    models_dir = tmp_path / "data" / "models"
    good_hash = feature_schema_hash(FEATURE_NAMES)

    # 1. no sidecar at all — the shipped data/models state (15 pkl, 0 meta)
    ungated = _write_artefact(models_dir, "BTCUSDT_alpha_binary", meta=None)
    # 2. a sidecar whose gate REFUSED the model
    refused = _write_artefact(models_dir, "BTCUSDT_beta_binary", meta={
        "feature_names": list(FEATURE_NAMES), "train_base_rate": 0.5,
        "feature_schema_hash": good_hash,
        "gate": {"allowed": False, "reason": "OOS AUC 0.4470 <= 0.55"}})
    # 3. a valid, gated artefact
    gated = _write_artefact(models_dir, "BTCUSDT_gamma_binary", meta={
        "feature_names": list(FEATURE_NAMES), "train_base_rate": 0.51,
        "feature_schema_hash": good_hash,
        "gate": {"allowed": True, "reason": "pass", "auc": 0.61,
                 "net_expectancy": 0.003, "n_oos": 3000}})
    # 4. a "gated" verdict with a stale feature schema
    stale = _write_artefact(models_dir, "BTCUSDT_delta_binary", meta={
        "feature_names": list(FEATURE_NAMES), "train_base_rate": 0.5,
        "feature_schema_hash": "deadbeefcafe",
        "gate": {"allowed": True, "reason": "pass"}})

    engine, cfg = _engine(tmp_path)
    cfg.data_dir = str(tmp_path / "data")
    engine.config = cfg

    from core.ml.trainer import MLTrainer
    trainer = MLTrainer(str(cfg.data_dir))

    # Before: the audited loop loaded every pickle on disk, gate or no gate.
    pre_fix = [p for p in sorted(models_dir.glob("*.pkl"))
               if trainer.load_model(str(p)) is not None]
    assert len(pre_fix) == 4, "all four pickles are loadable from disk"

    # After: only the verified one may be installed.
    loaded: dict[str, object] = {}
    for path, label in ((ungated, "no sidecar"), (refused, "gate refused"),
                        (gated, "valid"), (stale, "stale schema")):
        stem = path.stem
        ok, reason = engine._verify_ml_model_sidecar(stem, path, "binary")
        if ok:
            loaded[stem] = trainer.load_model(str(path))
        elif label == "no sidecar":
            assert "*_meta.json" in reason
        elif label == "stale schema":
            assert "schema hash mismatch" in reason
        else:
            assert "gate refused" in reason
    assert sorted(loaded) == ["BTCUSDT_gamma_binary"], (
        "exactly the gated, schema-verified pickle may load")


def test_preload_refuses_a_feature_contract_mismatch(tmp_path):
    """F4: a sidecar trained on a different column list is not scored positionally."""
    from core.ml.features import FEATURE_NAMES, feature_schema_hash

    models_dir = tmp_path / "data" / "models"
    short = list(FEATURE_NAMES)[:12]
    path = _write_artefact(models_dir, "BTCUSDT_alpha_binary", meta={
        "feature_names": short, "feature_schema_hash": feature_schema_hash(short),
        "gate": {"allowed": True, "reason": "pass"}})
    engine, cfg = _engine(tmp_path)
    cfg.data_dir = str(tmp_path / "data")
    engine.config = cfg
    ok, reason = engine._verify_ml_model_sidecar("BTCUSDT_alpha_binary", path, "binary")
    assert ok is False and "feature contract mismatch" in reason
    # A missing artefact is a refusal too, not an exception.
    ok2, reason2 = engine._verify_ml_model_sidecar(
        "BTCUSDT_nope_binary", models_dir / "BTCUSDT_nope_binary.pkl", "binary")
    assert ok2 is False and "no artefact" in reason2


def test_shipped_models_are_all_refused_by_the_preload_gate():
    """F4 on this checkout: 15 shipped pickles, 0 sidecars → 15 refusals."""
    models_dir = REPO / "data" / "models"
    pickles = sorted(models_dir.glob("*.pkl")) if models_dir.exists() else []
    if not pickles:
        pytest.skip("no shipped data/models in this checkout")
    sidecars = list(models_dir.glob("*_meta.json"))
    engine, cfg = _engine(Path(models_dir.parent.parent))
    ok_count = sum(1 for p in pickles
                   if engine._verify_ml_model_sidecar(p.stem, p, "binary")[0])
    assert ok_count == len(sidecars), (
        f"{len(pickles)} pickles / {len(sidecars)} sidecars: a pickle without a "
        f"gated sidecar must never verify")


# ══════════════════════════════════════════════════════════════════════
# F5 — no unread `ml:` key, and the meta cost is the fill source
# ══════════════════════════════════════════════════════════════════════

def test_every_ml_config_key_has_a_production_reader():
    """F5.2: the reader table for the `ml:` block must be empty of unread keys."""
    from app.config import Config

    data = yaml.safe_load((REPO / "config" / "config.yaml").read_text(encoding="utf-8"))
    keys = list((data.get("ml") or {}).keys())
    assert keys, "the ml: block must not be empty"

    config_src = (REPO / "app" / "config.py").read_text(encoding="utf-8")
    prod_files = [p for p in list((REPO / "core").rglob("*.py"))
                  + list((REPO / "app").rglob("*.py"))
                  + list((REPO / "web").rglob("*.py"))
                  + list((REPO / "scripts").rglob("*.py"))
                  + list((REPO / "tools").rglob("*.py"))
                  if p.name != "config.py" or p.parent.name != "app"]
    prod_text = {p: p.read_text(encoding="utf-8", errors="ignore") for p in prod_files}

    unread = []
    for key in keys:
        attr = f"ml_{key}"
        assert f"self.{attr}" in config_src, f"ml.{key} is not even loaded"
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(attr)}(?![A-Za-z0-9_])")
        if not any(pattern.search(text) for text in prod_text.values()):
            unread.append(key)
    assert unread == [], f"unread ml: keys (loaded but never read): {unread}"


def test_default_meta_cost_pct_matches_the_sim_fill_source():
    """F5.3: `default_meta_cost_pct(None, ...)` used to return the 0.14 % fallback."""
    from app.config import Config
    from core.ml.credibility import cost_pct_for
    from core.ml.meta import default_meta_cost_pct

    Config._instance = None
    cfg = Config.load("sim")
    eth_with = default_meta_cost_pct(cfg, symbol="ETHUSDT")
    eth_without = default_meta_cost_pct(None, symbol="ETHUSDT")
    assert eth_with == pytest.approx(0.26, abs=1e-9)
    assert eth_without == pytest.approx(eth_with), (
        "config=None must resolve the active config, not the 0.14 % fallback")
    assert eth_without == pytest.approx(
        cost_pct_for(cfg, symbol="ETHUSDT", order_type="market"))
    # The doc claim in docs/core-algorithms/12 line 27 is now true either way.
    assert default_meta_cost_pct(None, symbol="BTCUSDT") == pytest.approx(0.25)
