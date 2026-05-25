import os
import yaml
from pathlib import Path
from typing import Any
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent


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

        self.web_port = self._get("web_port", 8899)
        binance_cfg = self._get("binance", {})
        self.binance_testnet = binance_cfg.get("testnet", True) if isinstance(binance_cfg, dict) else True

        trading = self._get("trading", {})
        self.spot_enabled = trading.get("spot_enabled", True) if isinstance(trading, dict) else True
        self.futures_enabled = trading.get("futures_enabled", False) if isinstance(trading, dict) else False

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

        ai = self._get("ai", {})
        self.ai_mode = ai.get("mode", "semi_auto") if isinstance(ai, dict) else "semi_auto"
        self.ai_model = ai.get("model", "deepseek-chat") if isinstance(ai, dict) else "deepseek-chat"
        self.ai_base_url = ai.get("base_url", "https://api.deepseek.com") if isinstance(ai, dict) else "https://api.deepseek.com"
        self.ai_consult_interval = ai.get("consult_interval_minutes", 60) if isinstance(ai, dict) else 60
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

        soft = self._get("soft_params", {})
        if not soft:
            rp = self._get("risk_params", {})
            soft = rp.get("soft_params", {}) if isinstance(rp, dict) else {}
        self.soft_params = SoftRiskParams(**soft) if soft else SoftRiskParams()

        self.db_path = str(PROJECT_ROOT / "data" / "binance_trader.db")
        self.data_dir = str(PROJECT_ROOT / "data")
        self.strategies_dir = str(PROJECT_ROOT / "strategies")

        self._loaded = True

    def _load_yaml(self, relative_path: str):
        path = PROJECT_ROOT / relative_path
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self._data = self._deep_merge(self._data, data)

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
