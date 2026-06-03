import yaml
from pathlib import Path
from typing import Any
from pydantic import BaseModel


class MLConfig(BaseModel):
    enabled: bool = False
    confidence_threshold: float = 0.6
    features: list[str] = []
    weight: float = 0.3


class RiskExitConfig(BaseModel):
    """Standardized risk-based exit rules applied to every position.

    These exits are always active (stop-loss, trailing-stop, max-hold).
    Indicator-based exits in exit_conditions are OPTIONAL and can be
    disabled via use_indicator_exits=False — this lets GA focus purely
    on entry alpha while exits follow fixed risk rules.
    """
    stop_loss_pct: float = 2.0         # fixed stop loss distance from entry (%)
    trailing_stop_pct: float = 1.5     # trailing stop distance from best price (%)
    max_hold_hours: float = 48.0       # force close after N hours (0 = no limit)
    use_indicator_exits: bool = True   # if False, only risk exits are used


class StrategyConfig(BaseModel):
    name: str
    enabled: bool = True
    mode: str = "trend"
    timeframes: list[str] = ["1h"]
    symbols: list[str] = []  # empty = all symbols; otherwise restrict to these
    indicators: dict[str, Any] = {}
    entry_conditions: dict[str, list[str]] = {}
    exit_conditions: dict[str, list[str]] = {}
    reduce_conditions: dict[str, list[dict]] = {}
    ml_config: MLConfig | None = None
    risk_exit: RiskExitConfig | None = None  # standardized risk exits


class StrategyLoader:
    def __init__(self, strategies_dir: str):
        self.strategies_dir = Path(strategies_dir)

    def _normalize(self, name: str) -> str:
        import re
        result = name.lower().replace(" ", "_").replace("/", "_")
        # Keep Unicode word characters (including Chinese), strip only truly unsafe filename chars
        result = re.sub(r'[^\w\-.]', '_', result, flags=re.UNICODE)
        return re.sub(r'_+', '_', result).strip('_')

    def load(self, name: str) -> StrategyConfig:
        path = self.strategies_dir / f"{self._normalize(name)}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return StrategyConfig(**data)

    def load_all(self) -> list[StrategyConfig]:
        strategies = []
        if not self.strategies_dir.exists():
            return strategies
        for path in self.strategies_dir.glob("*.yaml"):
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            strategies.append(StrategyConfig(**data))
        return strategies

    def save(self, config: StrategyConfig):
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        path = self.strategies_dir / f"{self._normalize(config.name)}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, allow_unicode=True)

    def list_names(self) -> list[str]:
        if not self.strategies_dir.exists():
            return []
        return [p.stem for p in self.strategies_dir.glob("*.yaml")]

    def delete(self, name: str):
        path = self.strategies_dir / f"{self._normalize(name)}.yaml"
        if path.exists():
            path.unlink()
