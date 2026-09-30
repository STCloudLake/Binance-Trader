import yaml
from pathlib import Path
from typing import Any
from pydantic import BaseModel, field_validator


# ── Entry-structure ("condition_logic") contract ──────────────────────
#: Any condition met activates the side (historical behaviour, the default).
CONDITION_LOGIC_OR = "or"
#: Every condition must be met at the evaluated bar (GA-evolved, stricter).
CONDITION_LOGIC_AND = "and"
CONDITION_LOGIC_CHOICES = (CONDITION_LOGIC_OR, CONDITION_LOGIC_AND)


def normalize_condition_logic(value: Any, *, warn: bool = False) -> str:
    """Coerce *value* into a valid ``condition_logic`` (``"or"`` / ``"and"``).

    Anything else — ``None``, empty, a typo, a stray bool — falls back to
    ``"or"`` instead of raising, so pre-P1 / hand-written / AI-generated YAML
    always keeps loading.  With ``warn=True`` the fallback is logged so an
    invalid value is visible rather than silently ignored.
    """
    if value is not None:
        text = str(value).strip().lower()
        if text in CONDITION_LOGIC_CHOICES:
            return text
    if warn:
        from loguru import logger
        logger.warning(
            f"condition_logic={value!r} is not one of {CONDITION_LOGIC_CHOICES} — "
            f"falling back to '{CONDITION_LOGIC_OR}'")
    return CONDITION_LOGIC_OR


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
    #: How ``entry_conditions[side]`` combine: ``"or"`` (any condition, the
    #: historical behaviour) or ``"and"`` (all conditions — a GA-evolvable
    #: gene).  Persisted in the YAML, so a champion genome is traded with the
    #: exact structure it was scored under (:meth:`entry_sides` is the single
    #: evaluation path).
    condition_logic: str = CONDITION_LOGIC_OR
    exit_conditions: dict[str, list[str]] = {}
    reduce_conditions: dict[str, list[dict]] = {}
    ml_config: MLConfig | None = None
    risk_exit: RiskExitConfig | None = None  # standardized risk exits

    @field_validator("condition_logic", mode="before")
    @classmethod
    def _coerce_condition_logic(cls, value: Any) -> str:
        return normalize_condition_logic(value, warn=True)

    def entry_sides(self, df) -> tuple[bool, bool]:
        """``(long_active, short_active)`` for the last (closed) bar of *df*.

        The canonical entry-structure evaluator for any ``StrategyConfig``
        (used by the live engine's entry path — ``StrategyEngine._evaluate``):
        ``condition_logic == "or"`` delegates to the shared OR kernel
        (``evaluation_kernel.evaluate_entry_conditions``), while ``"and"``
        requires *every* condition to hold.  The GA/backtest entry path reads
        the same field with this identical rule, so a champion genome cannot be
        scored under one structure and traded under another.  The default
        ``"or"`` keeps the live/backtest parity contract unchanged for
        strategies that never set the field.

        Empty condition lists are inactive for both modes (matching the GA
        entry path), so a malformed strategy cannot enter unconditionally.
        """
        # Imported lazily: ``evaluation_kernel`` pulls in the ML calibration
        # module, and ``loader`` is imported by light-weight callers (routes,
        # workers) that never evaluate a condition.
        from core.strategy.evaluation_kernel import evaluate_entry_conditions
        from core.strategy.indicators import evaluate_condition

        if self.condition_logic != CONDITION_LOGIC_AND:
            return evaluate_entry_conditions(df, self.entry_conditions)

        sides: list[bool] = []
        for side in ("long", "short"):
            conditions = self.entry_conditions.get(side, [])
            active = bool(conditions)
            for cond in conditions:
                mask = evaluate_condition(df, cond)
                if not (hasattr(mask, "iloc") and bool(mask.iloc[-1])):
                    active = False
                    break
            sides.append(active)
        return sides[0], sides[1]


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
            try:
                with open(path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                strategies.append(StrategyConfig(**data))
            except Exception as e:
                from loguru import logger
                logger.warning(f"Failed to load strategy '{path.stem}': {e} — skipping")
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
