import os
import yaml
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent

#: Binance spot fee schedule (VIP0..VIP9), used as the *manual* fee-tier table of
#: the simulated account.  The trading host cannot read this machine's real
#: 30-day volume / BNB balance (mainnet account endpoints are unreachable), so the
#: tier is always a user selection — see ``docs/overhaul/TRADE_PAGE_API.md`` §五之二.
FEE_TIER_TABLE: list[dict] = [
    {"tier": "VIP0", "maker_pct": 0.1000, "taker_pct": 0.1000},
    {"tier": "VIP1", "maker_pct": 0.0900, "taker_pct": 0.1000},
    {"tier": "VIP2", "maker_pct": 0.0800, "taker_pct": 0.1000},
    {"tier": "VIP3", "maker_pct": 0.0420, "taker_pct": 0.0600},
    {"tier": "VIP4", "maker_pct": 0.0420, "taker_pct": 0.0540},
    {"tier": "VIP5", "maker_pct": 0.0360, "taker_pct": 0.0480},
    {"tier": "VIP6", "maker_pct": 0.0300, "taker_pct": 0.0420},
    {"tier": "VIP7", "maker_pct": 0.0240, "taker_pct": 0.0360},
    {"tier": "VIP8", "maker_pct": 0.0180, "taker_pct": 0.0300},
    {"tier": "VIP9", "maker_pct": 0.0120, "taker_pct": 0.0240},
]

#: ``use_bnb_discount`` multiplies the fee rate by this (25% off).
BNB_DISCOUNT_FACTOR = 0.75
BNB_DISCOUNT_PCT = int(round((1.0 - BNB_DISCOUNT_FACTOR) * 100))

#: Fallback half-spread (%) for a symbol that is not listed in the config block.
DEFAULT_SIM_SPREAD_PCT = 0.02

#: Port the web UI binds to when ``web_port`` is missing or unusable.
DEFAULT_WEB_PORT = 8899


class ConfigError(Exception):
    """A user-fixable configuration problem (malformed YAML, unusable values).

    Raised by :meth:`Config._load_yaml` instead of letting ``yaml.YAMLError``
    escape: the startup boundary in ``app/main.py`` logs ``str(exc)`` and exits
    with status 1, so a bad config file produces one actionable line rather than
    a parser traceback.
    """

FEE_TIER_NOTE = ("手续费等级为手动选择：本机无法读取币安账户的 30 天交易量与 BNB 持仓"
                 "（主网账户接口不可达），档位表为币安现货标准费率。")

#: ``system_config`` keys the manual tier selection is persisted under (survives
#: a restart; the YAML values are only defaults).
SIM_FEE_TIER_KEY = "sim.fee_tier"
SIM_USE_BNB_DISCOUNT_KEY = "sim.use_bnb_discount"


def fee_tier_entry(tier: str) -> Optional[dict]:
    """The maker/taker row for ``tier`` (case-insensitive) or ``None``."""
    key = str(tier or "").strip().upper()
    for row in FEE_TIER_TABLE:
        if row["tier"] == key:
            return row
    return None


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _yaml_error_message(relative_path: str, error: yaml.YAMLError) -> str:
    """One actionable line: ``<file> is not valid YAML (line N): <reason>``.

    ``yaml.YAMLError`` subclasses carry a ``problem_mark`` whose ``line`` is
    0-based; the message uses the editor's 1-based numbering.  When no mark is
    available the location is reported as "unknown line" rather than omitted, so
    the operator still knows it is a YAML syntax problem.
    """
    mark = (getattr(error, "problem_mark", None)
            or getattr(error, "context_mark", None))
    line = getattr(mark, "line", None)
    where = f"line {line + 1}" if isinstance(line, int) else "unknown line"
    reason = (getattr(error, "problem", None)
              or next((ln.strip() for ln in str(error).splitlines() if ln.strip()), "")
              or error.__class__.__name__)
    return f"{relative_path} is not valid YAML ({where}): {reason}"


def _coerce_web_port(value: Any, default: int = DEFAULT_WEB_PORT) -> int:
    """Validate ``web_port`` so it can never reach ``uvicorn`` unusable.

    A non-integer (``web_port: "abc"``) used to reach uvicorn and die with
    ``getaddrinfo failed`` *after* the 70-90 s warm-up; it now falls back to
    ``default`` with a warning.  An out-of-range integer is clamped into
    1..65535 with a warning (never silently, never a raw ``OverflowError``).
    Numeric strings (``web_port: "9001"``) are accepted.
    """
    port: Optional[int] = None
    if isinstance(value, bool):
        port = None
    elif isinstance(value, int):
        port = value
    elif isinstance(value, float):
        port = int(value) if float(value).is_integer() else None
    elif isinstance(value, str):
        text = value.strip()
        try:
            port = int(text, 10)
        except ValueError:
            try:
                as_float = float(text)
            except ValueError:
                port = None
            else:
                port = int(as_float) if as_float.is_integer() else None
    if port is None:
        logger.warning(f"Invalid web_port {value!r} — expected an integer between "
                       f"1 and 65535; using the default {default}")
        return default
    if not 1 <= port <= 65535:
        clamped = min(max(port, 1), 65535)
        logger.warning(f"web_port {port} is outside the valid range 1..65535 — "
                       f"clamping to {clamped}")
        return clamped
    return port

#: Public mainnet market-data mirror(s).  ``api.binance.com`` / ``stream.binance.com``
#: are unreachable from this deployment, while the ``.vision`` public mirrors answer
#: the full spot market (3716 symbols, klines back to 2017).  See
#: ``docs/overhaul/MARKET_PAGES_API.md`` §0.  Overridable from ``config/config.yaml``
#: under ``binance:`` — these are only *defaults*.
DEFAULT_MARKET_DATA_HOST = "https://data-api.binance.vision"
DEFAULT_MARKET_STREAM_HOST = "wss://data-stream.binance.vision"


class HardRiskLimits(BaseModel):
    max_daily_drawdown_pct: float = 5.0
    max_weekly_drawdown_pct: float = 10.0
    max_daily_loss_usdt: float = 500.0
    max_position_size_pct: float = 10.0
    max_leverage: int = 3
    min_stop_loss_distance_pct: float = 0.5
    max_open_trades: int = 8
    max_total_exposure_pct: float = 80.0
    max_consecutive_losses: int = 5
    circuit_breaker_action: str = "block_only"  # block_only | tighten_stops | close_all | close_worst
    trailing_stop_enabled: bool = True
    trailing_stop_distance_pct: float = 2.0
    emergency_stop_enabled: bool = True
    emergency_stop_threshold_pct: float = -5.0


class SoftRiskParams(BaseModel):
    risk_appetite: str = "balanced"
    position_size_pct: float = 5.0
    stop_loss_pct: float = 2.0
    take_profit_1_pct: float = 3.0
    take_profit_2_pct: float = 5.0
    take_profit_3_pct: float = 10.0
    leverage: int = 2


class SignalWeights(BaseModel):
    indicator: float = 0.5
    ml: float = 0.3
    news: float = 0.2


class VolTargetingConfig(BaseModel):
    """Volatility-targeting knobs (Phase P3) — ``risk.vol_targeting`` in YAML.

    Science: conditional volatility is predictable (ARCH/GARCH — Tsay,
    *Analysis of Financial Time Series*, ch. 3) while the *sign* of the next
    return is not, so the forecast is spent on risk rather than direction
    (``docs/core-algorithms/10-volatility-targeting.md``).

    **SAFE DEFAULT: ``enabled = False``.**  Every other field is inert until the
    switch is on, so the shipped behaviour of sizing, stops and barriers is
    bit-identical to the fixed-percentage implementation that P1/P2 measured.
    The whole block is opt-in precisely because the measured effect is
    regime-dependent and must be A/B-tested before it manages real money.

    Units: ``target_vol_pct`` and the ``stop_*``/``barrier_*`` widths are percent
    of price **per bar** (0.45 = 0.45 %/bar); ``lam``/``window`` configure the
    estimator in :mod:`core.ml.volatility`.
    """

    enabled: bool = False
    method: str = "ewma"
    lam: float = 0.94
    window: int = 500
    target_vol_pct: float = 0.45
    max_scale: float = 2.0
    min_scale: float = 0.25
    max_position_notional_pct: float = 10.0
    stop_vol_multiple: float = 3.0
    stop_min_pct: float = 0.5
    stop_max_pct: float = 6.0
    barrier_vol_multiple: float = 1.0
    barrier_min_pct: float = 0.004
    barrier_max_pct: float = 0.06


class Config:
    _instance = None

    def __new__(cls, mode: str = "sim"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    @classmethod
    def load(cls, mode: str = "sim") -> "Config":
        inst = cls(mode)
        if not inst._loaded:
            inst._load(mode)
        return inst

    def _load(self, mode: str):
        self.mode = mode
        self._data: dict = {}
        self._load_yaml("config/config.yaml")
        self._load_yaml("config/risk_params.yaml")

        secrets_path = PROJECT_ROOT / "config" / "secrets.yaml"
        if secrets_path.exists():
            self._load_yaml("config/secrets.yaml")

        self.binance_api_key = os.getenv("BINANCE_API_KEY", self._get_nested("binance", "api_key") or "")
        self.binance_api_secret = os.getenv("BINANCE_API_SECRET", self._get_nested("binance", "api_secret") or "")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", self._get_nested("deepseek", "api_key") or "")

        self.web_port = _coerce_web_port(self._get("web_port", DEFAULT_WEB_PORT))
        binance_cfg = self._get("binance", {})
        self.binance_testnet = binance_cfg.get("testnet", True) if isinstance(binance_cfg, dict) else True

        # Market data (REST + websocket) is served by the reachable mainnet public
        # mirror, NOT by the trading endpoint: `binance.testnet` keeps governing
        # only the trading/order client (orders, account, balances).
        md_host = (binance_cfg.get("market_data_host")
                   if isinstance(binance_cfg, dict) else None)
        self.market_data_host = str(md_host or DEFAULT_MARKET_DATA_HOST).rstrip("/")
        ms_host = (binance_cfg.get("market_stream_host")
                   if isinstance(binance_cfg, dict) else None)
        self.market_stream_host = str(ms_host or DEFAULT_MARKET_STREAM_HOST).rstrip("/")
        # Keep python-binance's socket manager off its hardcoded
        # `wss://stream.binance.com:9443/` default, which cannot be reached here.
        self.binance_stream_url = self.market_stream_host + "/"

        sw = self._get("signal_weights", {})
        self.signal_weights = SignalWeights(**sw) if sw else SignalWeights()

        core = self._get("core_position", {})
        self.core_max_symbols = core.get("max_symbols", 5) if isinstance(core, dict) else 5
        self.core_capital_pct = core.get("capital_pct", 0.7) if isinstance(core, dict) else 0.7

        sat = self._get("satellite_position", {})
        self.satellite_max_symbols = sat.get("max_symbols", 10) if isinstance(sat, dict) else 10
        self.satellite_capital_pct = sat.get("capital_pct", 0.3) if isinstance(sat, dict) else 0.3

        news = self._get("news", {})
        self.news_fetch_interval = news.get("fetch_interval_minutes", 30) if isinstance(news, dict) else 30
        self.news_max_articles = news.get("max_articles_per_symbol", 10) if isinstance(news, dict) else 10
        self.anomaly_threshold_pct = news.get("anomaly_threshold_pct", 3.0) if isinstance(news, dict) else 3.0
        self.volume_spike_multiplier = news.get("volume_spike_multiplier", 3.0) if isinstance(news, dict) else 3.0

        self.language = self._get("language", "zh")

        bt = self._get("backtest", {})
        self.backtest_engine_mode = bt.get("engine_mode", "auto") if isinstance(bt, dict) else "auto"
        if self.backtest_engine_mode not in ("auto", "hybrid", "legacy"):
            logger.warning(f"Invalid backtest.engine_mode '{self.backtest_engine_mode}', falling back to 'auto'")
            self.backtest_engine_mode = "auto"
        self.backtest_ml_enabled = bt.get("ml_enabled", False) if isinstance(bt, dict) else False

        # Cost model: trading fees + spread.  ``spread_pct`` is only the
        # *override* table now — a symbol that is not listed there is resolved
        # live from the public order book and finally falls back to
        # ``default_spread_pct`` (see core/backtest/cost_model.py).
        cost_cfg = bt.get("cost_model", {}) if isinstance(bt, dict) else {}
        if not isinstance(cost_cfg, dict):
            cost_cfg = {}
        self.backtest_cost_enabled = cost_cfg.get("enabled", True)
        self.backtest_taker_fee_pct = cost_cfg.get("taker_fee_pct", 0.04)
        self.backtest_spread_pct = dict(cost_cfg.get("spread_pct") or {})
        self.backtest_default_spread_pct = _as_float(
            cost_cfg.get("default_spread_pct"), 0.03)
        live_spread = cost_cfg.get("live_spread")
        if not isinstance(live_spread, dict):
            live_spread = {}
        # Live derivation is on by default for the real config; ``enabled: false``
        # restores the pure override/default behaviour (e.g. an offline machine).
        self.backtest_live_spread_enabled = _as_bool(live_spread.get("enabled"), True)
        self.backtest_live_spread_ttl = _as_float(
            live_spread.get("ttl_seconds"), 300.0)
        self.backtest_live_spread_timeout = _as_float(
            live_spread.get("timeout_seconds"), 3.0)
        self.backtest_market_data_host = str(
            cost_cfg.get("market_data_host") or self.market_data_host).rstrip("/")

        # ── GA evaluation settings (`ga:` in config.yaml) ──
        # These only affect how a genome is SCORED during GA/walk-forward:
        #   use_live_spread      — never price a historical fill from today's book
        #   min_champion_trades  — publication gate's trade-count floor
        #   alpha_weight         — weight of the DSR/Sharpe/buy&hold term
        #   evaluation_leverage  — 1.0 = cash model (the documented choice)
        ga_cfg = self._get("ga", {})
        if not isinstance(ga_cfg, dict):
            ga_cfg = {}
        self.ga_use_live_spread = _as_bool(ga_cfg.get("use_live_spread"), False)
        try:
            self.ga_min_champion_trades = max(
                int(ga_cfg.get("min_champion_trades", 30)), 1)
        except (TypeError, ValueError):
            self.ga_min_champion_trades = 30
        self.ga_alpha_weight = max(
            _as_float(ga_cfg.get("alpha_weight"), 1.0), 0.0)
        self.ga_evaluation_leverage = max(
            _as_float(ga_cfg.get("evaluation_leverage"), 1.0), 1.0)

        # Simulated-account cost model (docs/overhaul/TRADE_PAGE_API.md §五之二).
        # Deliberately separate from `backtest.cost_model` above: the backtest model
        # charges its costs at close time on top of the raw prices and must not
        # change behaviour, while the sim model moves the *fill price* itself.
        sim_cfg = self._get("sim", {})
        sim_cost = sim_cfg.get("cost_model", {}) if isinstance(sim_cfg, dict) else {}
        if not isinstance(sim_cost, dict):
            sim_cost = {}
        self.sim_cost_enabled = _as_bool(sim_cost.get("enabled"), True)
        self.sim_fee_tier = str(sim_cost.get("fee_tier", "VIP0") or "VIP0")
        if fee_tier_entry(self.sim_fee_tier) is None:
            logger.warning(f"Invalid sim.cost_model.fee_tier '{self.sim_fee_tier}', "
                           f"falling back to 'VIP0'")
            self.sim_fee_tier = "VIP0"
        self.sim_use_bnb_discount = _as_bool(sim_cost.get("use_bnb_discount"), False)
        self.sim_slippage_bps = _as_float(sim_cost.get("slippage_bps"), 2.0)
        self.sim_spread_pct = dict(sim_cost.get("spread_pct") or {})
        self.sim_cost_model = {
            "enabled": self.sim_cost_enabled,
            "fee_tier": self.sim_fee_tier,
            "use_bnb_discount": self.sim_use_bnb_discount,
            "slippage_bps": self.sim_slippage_bps,
            "spread_pct": dict(self.sim_spread_pct),
        }

        self.sim_default_spread_pct = _as_float(
            self.sim_spread_pct.get("default"), DEFAULT_SIM_SPREAD_PCT)

        # ── ML credibility block (docs/overhaul/ALGO_UPGRADE_PLAN.md §二 P2) ──
        # SAFE DEFAULT: `enabled` ships false.  The gate requires OOS AUC > 0.55
        # AND net-of-cost expectancy > 0; the measured live model has AUC
        # 0.396–0.447 (worse than the majority class), so ML stays off until a
        # newly trained, calibrated, gated model replaces it.
        ml = self._get("ml", {})
        if not isinstance(ml, dict):
            ml = {}
        self.ml_enabled = _as_bool(ml.get("enabled"), False)
        self.ml_model_type = str(ml.get("model_type", "lightgbm") or "lightgbm").lower()
        self.ml_calibration = str(ml.get("calibration", "isotonic") or "isotonic").lower()
        # `gate_auc_min` / `gate_net_expectancy_min` / `min_oos_rows` feed
        # `credibility_gate` through `MLPredictor._gate_config()`, and
        # `gate_min_trades` / `gate_min_t_stat` add the significance floor the
        # audit demanded (P2 #3).  `gate_enabled` and `gate_disable_url` were
        # deleted: the first was a switch that could not be honoured (an
        # unrecognised `true` must never be able to enable a refused model) and
        # the second pointed at `/api/ml`, which does not exist.
        self.ml_gate_auc_min = _as_float(ml.get("gate_auc_min"), 0.55)
        if not 0.5 <= self.ml_gate_auc_min <= 1.0:
            logger.warning(f"Invalid ml.gate_auc_min '{ml.get('gate_auc_min')}', "
                           f"falling back to 0.55")
            self.ml_gate_auc_min = 0.55
        self.ml_gate_net_expectancy_min = _as_float(ml.get("gate_net_expectancy_min"), 0.0)
        self.ml_gate_min_trades = int(_as_float(ml.get("gate_min_trades"), 100))
        self.ml_gate_min_t_stat = _as_float(ml.get("gate_min_t_stat"), 2.0)
        self.ml_gate_min_psr = _as_float(ml.get("gate_min_psr"), 0.95)
        self.ml_feature_list = list(ml.get("feature_list") or []) or None

        # Label / evaluation knobs.  `confidence_threshold` used to be evolved by
        # the GA and never read; it is now the *decision* threshold applied to the
        # calibrated P(up) (None → use the cost-aware threshold from the gate).
        self.ml_confidence_threshold = _as_float(ml.get("confidence_threshold"), None)
        self.ml_default_confidence_threshold = _as_float(
            ml.get("default_confidence_threshold"), 0.55)
        self.ml_forward_periods = int(_as_float(ml.get("forward_periods"), 4))
        self.ml_label_threshold = _as_float(ml.get("label_threshold"), 0.005)
        # Effective label threshold = max(label_threshold, k * round-trip cost).
        # 0.0 keeps the audited ±0.5 % threshold (40.5 % of BTC 1h bars tradeable);
        # 4.0 makes the target beat four times its own cost, which moves more mass
        # into the `flat` class.
        self.ml_label_cost_multiple = _as_float(ml.get("label_cost_multiple"), 0.0)
        self.ml_max_hold_bars = int(_as_float(ml.get("max_hold_bars"), 24))
        self.ml_label_params = dict(ml.get("label_params") or {})
        self.ml_barrier_atr_period = int(_as_float(ml.get("barrier_atr_period"), 14))
        self.ml_barrier_atr_multiple = _as_float(ml.get("barrier_atr_multiple"), 1.5)
        self.ml_barrier_min_pct = _as_float(ml.get("barrier_min_pct"), 0.004)
        self.ml_barrier_max_pct = _as_float(ml.get("barrier_max_pct"), 0.06)
        self.ml_embargo_bars = int(_as_float(ml.get("embargo_bars"), self.ml_forward_periods))
        self.ml_min_oos_rows = int(_as_float(ml.get("min_oos_rows"), 100))

        # Cost inputs.  All default to **None** so `core.ml.credibility` resolves
        # the round-trip cost through the `sim.cost_model` numbers — the same
        # source the fills pay (audit P2 #2: the old default fell back to
        # `backtest_taker_fee_pct` 0.04 %, half the real VIP0 taker fee, so the
        # gate gated on a cost 2× cheaper than reality).
        self.ml_taker_fee_pct = _as_float(ml.get("taker_fee_pct"), None)
        self.ml_half_spread_pct = _as_float(ml.get("half_spread_pct"), None)
        self.ml_slippage_bps = _as_float(ml.get("slippage_bps"), None)
        _bnb = ml.get("use_bnb_discount")
        self.ml_use_bnb_discount = None if _bnb is None else _as_bool(_bnb, False)
        self.ml_retrain_interval_hours = _as_float(ml.get("retrain_interval_hours"), 24.0)

        ai = self._get("ai", {})
        self.ai_mode = ai.get("mode", "semi_auto") if isinstance(ai, dict) else "semi_auto"
        self.ai_model = ai.get("model", "deepseek-chat") if isinstance(ai, dict) else "deepseek-chat"
        self.ai_base_url = ai.get("base_url", "https://api.deepseek.com") if isinstance(ai, dict) else "https://api.deepseek.com"
        ai_tasks = ai.get("tasks", {}) if isinstance(ai, dict) else {}
        self.ai_task_intervals = {
            "market_assessment": ai_tasks.get("market_assessment_minutes", 60) * 60,
            "coin_selection": ai_tasks.get("coin_selection_minutes", 240) * 60,
            "strategy_optimization": ai_tasks.get("strategy_optimization_minutes", 1440) * 60,
            "risk_adjustment": ai_tasks.get("risk_adjustment_minutes", 1440) * 60,
        }

        hard = self._get("hard_limits", {})
        if not hard:
            rp = self._get("risk_params", {})
            hard = rp.get("hard_limits", {}) if isinstance(rp, dict) else {}
        self.hard_limits = HardRiskLimits(**hard) if hard else HardRiskLimits()
        valid_actions = {"block_only", "tighten_stops", "close_all", "close_worst"}
        if self.hard_limits.circuit_breaker_action not in valid_actions:
            logger.warning(f"Invalid circuit_breaker_action '{self.hard_limits.circuit_breaker_action}', falling back to 'block_only'")
            self.hard_limits.circuit_breaker_action = "block_only"

        soft = self._get("soft_params", {})
        if not soft:
            rp = self._get("risk_params", {})
            soft = rp.get("soft_params", {}) if isinstance(rp, dict) else {}
        self.soft_params = SoftRiskParams(**soft) if soft else SoftRiskParams()

        # ── Volatility targeting (docs/overhaul/ALGO_UPGRADE_PLAN.md §二 P3) ──
        # `risk.vol_targeting` in config.yaml.  SAFE DEFAULT: disabled, so the
        # sizing / stop / barrier paths keep their pre-P3 fixed-percentage
        # behaviour unless an operator turns it on deliberately.
        risk_cfg = self._get("risk", {})
        vt_raw = risk_cfg.get("vol_targeting", {}) if isinstance(risk_cfg, dict) else {}
        if not isinstance(vt_raw, dict):
            vt_raw = {}
        try:
            self.risk_vol_targeting = VolTargetingConfig(**vt_raw)
        except Exception as e:  # a bad type must not stop trading
            logger.warning(f"Invalid risk.vol_targeting block ({e}) — using defaults "
                           f"(vol targeting disabled)")
            self.risk_vol_targeting = VolTargetingConfig()
        vt = self.risk_vol_targeting
        if str(vt.method).strip().lower() not in (
                "ewma", "realized_cc", "realized_parkinson",
                "realized_garman_klass", "garch11"):
            logger.warning(f"Invalid risk.vol_targeting.method '{vt.method}', "
                           f"falling back to 'ewma'")
            vt.method = "ewma"
        if not 0.0 <= vt.lam < 1.0:
            logger.warning(f"Invalid risk.vol_targeting.lam '{vt.lam}', falling back to 0.94")
            vt.lam = 0.94
        if vt.target_vol_pct <= 0.0:
            logger.warning("risk.vol_targeting.target_vol_pct must be > 0 — "
                           "vol targeting disabled")
            vt.enabled = False
        if vt.min_scale > vt.max_scale:
            logger.warning(f"risk.vol_targeting.min_scale {vt.min_scale} > max_scale "
                           f"{vt.max_scale} — swapped so the band is usable")
            vt.min_scale, vt.max_scale = vt.max_scale, vt.min_scale
        # Exposed for callers that only have `config` in hand (PositionSizer,
        # PositionGuard) without importing the model.
        self.vol_targeting = vt

        self.db_path = str(PROJECT_ROOT / "data" / "binance_trader.db")
        self.data_dir = str(PROJECT_ROOT / "data")
        self.strategies_dir = str(PROJECT_ROOT / "strategies")
        # Directory that settings endpoints persist to. Exposed as config so it can
        # be redirected (e.g. to a temp dir in tests) instead of mutating the
        # shipped YAML files — a mis-probed endpoint used to rewrite config.yaml
        # (including binance.testnet) as a side effect.
        self.config_dir = str(PROJECT_ROOT / "config")

        self._loaded = True

    def _load_yaml(self, relative_path: str):
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            # A missing config file is not fatal (code defaults apply), but it used
            # to be completely silent — the operator could not tell "the shipped
            # defaults are in effect" from "my file was never read".
            logger.warning(f"{relative_path} not found — using built-in defaults")
            return
        # Check file permissions for secrets files (POSIX only — Windows mode
        # bits do not express "readable by others", so a check there produced a
        # permanent false-positive SECURITY error on every startup).
        if "secrets" in relative_path and os.name == "posix":
            self._check_secrets_permissions(path, relative_path)
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            # A raw ``ParserError`` traceback (exit 2) named neither the file nor
            # the line; ``ConfigError`` is the clean, typed signal that the
            # startup boundary in app/main.py turns into a logged exit 1.
            raise ConfigError(_yaml_error_message(relative_path, e)) from e
        if not isinstance(data, dict):
            raise ConfigError(
                f"{relative_path} must contain a YAML mapping at the top level, "
                f"got {type(data).__name__}")
        self._data = self._deep_merge(self._data, data)

    @staticmethod
    def _check_secrets_permissions(path: Path, relative_path: str) -> None:
        """Warn when a POSIX secrets file is readable/writable by others."""
        import stat as _stat
        try:
            st = path.stat()
        except Exception:
            return  # filesystem may not support stat permissions
        if st.st_mode & (_stat.S_IROTH | _stat.S_IWOTH | _stat.S_IXOTH):
            logger.error(
                f"SECURITY: {relative_path} is readable/writable by others! "
                f"Fix with: chmod 600 {path}"
            )

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = Config._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def _get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def _get_nested(self, *keys) -> Any:
        d = self._data
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, {})
            else:
                return None
        return d if d != {} else None

    def update_soft_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.soft_params, k):
                setattr(self.soft_params, k, v)

    def update_signal_weights(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.signal_weights, k):
                setattr(self.signal_weights, k, v)


def _sim_settings_from_config(config: Config) -> dict:
    """Config-only part of the sim cost settings (no DB round-trip)."""
    return {
        "enabled": bool(getattr(config, "sim_cost_enabled", True)),
        "fee_tier": str(getattr(config, "sim_fee_tier", "VIP0") or "VIP0"),
        "use_bnb_discount": bool(getattr(config, "sim_use_bnb_discount", False)),
        "slippage_bps": _as_float(getattr(config, "sim_slippage_bps", 2.0), 2.0),
        "spread_pct": dict(getattr(config, "sim_spread_pct", {}) or {}),
        "default_spread_pct": _as_float(
            getattr(config, "sim_default_spread_pct", DEFAULT_SIM_SPREAD_PCT),
            DEFAULT_SIM_SPREAD_PCT),
    }


async def load_sim_cost_settings(db_path: str, config: Config) -> dict:
    """Effective sim cost settings: YAML defaults + the persisted manual selection.

    The tier is a *manual* choice (the machine cannot read a real Binance
    account's 30-day volume or BNB balance), so ``POST /api/fee/tier`` writes it
    to ``system_config``; this is where that override is picked up — hence the
    round trip on every fill and every estimate.
    """
    settings = _sim_settings_from_config(config)
    if not db_path:
        return settings
    try:
        import aiosqlite
        db = await aiosqlite.connect(db_path)
        db.row_factory = aiosqlite.Row
        try:
            cursor = await db.execute(
                "SELECT key, value FROM system_config WHERE key IN (?, ?)",
                (SIM_FEE_TIER_KEY, SIM_USE_BNB_DISCOUNT_KEY))
            stored = {r["key"]: r["value"] for r in await cursor.fetchall()}
        finally:
            await db.close()
    except Exception as e:  # a missing/corrupt DB must not stop trading
        logger.warning(f"Could not read persisted fee tier: {e}")
        return settings

    tier = stored.get(SIM_FEE_TIER_KEY)
    if tier and fee_tier_entry(tier) is not None:
        settings["fee_tier"] = str(tier).strip().upper()
    if SIM_USE_BNB_DISCOUNT_KEY in stored:
        settings["use_bnb_discount"] = _as_bool(
            stored[SIM_USE_BNB_DISCOUNT_KEY], settings["use_bnb_discount"])
    return settings


def sim_fee_pct(settings: dict, order_type: str = "market") -> float:
    """Fee rate in percent for the order type (taker for market, maker for limit)."""
    row = fee_tier_entry(settings.get("fee_tier", "VIP0")) or FEE_TIER_TABLE[0]
    pct = float(row["taker_pct"] if str(order_type).lower() != "limit"
                else row["maker_pct"])
    if settings.get("use_bnb_discount"):
        pct *= BNB_DISCOUNT_FACTOR
    return round(max(pct, 0.0), 6)


def sim_spread_pct(settings: dict, symbol: str) -> float:
    """Per-side half-spread (%) for ``symbol``."""
    spreads = settings.get("spread_pct") or {}
    return _as_float(spreads.get(str(symbol or "").upper()),
                     _as_float(settings.get("default_spread_pct"),
                               DEFAULT_SIM_SPREAD_PCT))


def sim_cost_quote(symbol: str, side: str, order_type: str, price: float,
                   quantity: float, settings: dict) -> dict:
    """Cost model for ONE fill — the single source of truth for sim fills AND
    for ``GET /api/fee/estimate``, so the estimate can never drift from reality.

    Formula (contract §五之二):
      * buy  (long):  ``fill = price × (1 + (spread/2 + slippage)/100)``
      * sell (short): ``fill = price × (1 − (spread/2 + slippage)/100)``
      * ``fee = quantity × fill × fee_pct/100`` (BNB discount ×0.75)
      * ``slippage_usdt = |fill − price| × quantity``  (spread/2 + slippage)

    Spread and slippage apply to **market** orders only: a limit order fills at
    its own limit price and pays the fee alone.  ``slippage_bps`` is basis
    points per side (1 bp = 0.01%), i.e. the same unit as ``spread_pct/2``.
    """
    symbol = str(symbol or "").upper()
    price = float(price or 0.0)
    quantity = float(quantity or 0.0)
    is_limit = str(order_type or "market").strip().lower() == "limit"
    side_l = str(side or "long").strip().lower()
    is_buy = side_l in ("long", "buy")

    fee_pct = sim_fee_pct(settings, order_type)
    slippage_bps = _as_float(settings.get("slippage_bps"), 2.0)
    spread_pct = sim_spread_pct(settings, symbol)

    enabled = bool(settings.get("enabled", True))
    # Limit fills get neither the spread nor the slippage — only the maker fee.
    edge_pct = 0.0 if (is_limit or not enabled) else spread_pct / 2.0 + slippage_bps / 100.0
    if not enabled:
        fee_pct = 0.0

    factor = (1.0 + edge_pct / 100.0) if is_buy else (1.0 - edge_pct / 100.0)
    fill_price = price * factor if price > 0 else 0.0
    notional = quantity * fill_price
    fee = notional * fee_pct / 100.0
    slippage = abs(fill_price - price) * quantity
    costs = fee + slippage
    # Cost priced into one unit of the base asset (fee + slippage), which is what
    # an old-style caller that only knows `pnl` needs in order to stay net.
    per_unit = costs / quantity if quantity else 0.0

    return {
        "enabled": enabled,
        "symbol": symbol,
        "side": side_l,
        "order_type": "limit" if is_limit else "market",
        "quoted_price": price,
        "fill_price": fill_price,
        "effective_price": price + (per_unit if is_buy else -per_unit),
        "quantity": quantity,
        "notional": notional,
        "fee_pct": fee_pct,
        "fee_usdt": fee,
        "slippage_bps": 0.0 if is_limit else slippage_bps,
        "spread_pct": 0.0 if is_limit else spread_pct,
        "edge_pct": edge_pct,
        "slippage_usdt": slippage,
        "cost_usdt": costs,
        "per_unit_cost": per_unit,
    }
