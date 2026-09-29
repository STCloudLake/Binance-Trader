from app.config import SoftRiskParams, HardRiskLimits


class PositionSizer:
    def __init__(self, hard_limits: HardRiskLimits, soft_params: SoftRiskParams,
                 core_capital_pct: float = 0.7, satellite_capital_pct: float = 0.3):
        self.hard = hard_limits
        self.soft = soft_params
        self.core_capital_pct = core_capital_pct
        self.satellite_capital_pct = satellite_capital_pct

    def calculate_position_size(self, account_balance: float, current_price: float,
                                 position_type: str = "satellite",
                                 volatility_expanding: bool = False) -> tuple[float, float]:
        """Calculate position size with optional volatility-based adjustment.

        When volatility is predicted to expand:
        - Position size reduced to 70% (tighten risk during turbulent periods)
        - This is grounded in the GARCH volatility clustering literature

        Args:
            account_balance: Current account balance in USDT.
            current_price: Entry price of the asset.
            position_type: "core" or "satellite" (capital pool allocation).
            volatility_expanding: If True, ML predicts vol will expand →
                reduce position size.

        Returns:
            (quantity, risk_amount_usdt) tuple.
        """
        if position_type == "core":
            capital_pool = account_balance * self.core_capital_pct
        else:
            capital_pool = account_balance * self.satellite_capital_pct

        effective_pct = max(self.soft.position_size_pct, 0.1)
        risk_per_trade = capital_pool * (effective_pct / 100)
        max_risk = account_balance * (self.hard.max_position_size_pct / 100)
        risk_per_trade = min(risk_per_trade, max_risk)

        # ── Volatility-based adjustment ──
        if volatility_expanding:
            risk_per_trade *= 0.7  # reduce position during high-vol regimes

        quantity = risk_per_trade / current_price if current_price > 0 else 0
        return quantity, risk_per_trade

    def calculate_stop_loss(self, entry_price: float, side: str,
                            volatility_expanding: bool = False) -> float:
        """Calculate stop-loss price with optional volatility-based widening.

        When volatility is predicted to expand, SL is widened to 130%
        to give the trade more room and avoid being prematurely stopped out.
        """
        sl_pct = max(self.soft.stop_loss_pct / 100,
                     self.hard.min_stop_loss_distance_pct / 100)
        if volatility_expanding:
            sl_pct *= 1.3  # wider stop during high-vol regimes
        if side == "long":
            return entry_price * (1 - sl_pct)
        else:
            return entry_price * (1 + sl_pct)

    def trailing_stop_distance_pct(self, strategy_risk_exit=None) -> float:
        """Trailing-stop distance in percent — single source of truth.

        Live trading, the legacy backtest engine and the hybrid engine all call
        this so a position is managed with identical trailing semantics.
        Order of precedence: per-strategy ``risk_exit.trailing_stop_pct`` →
        ``hard_limits.trailing_stop_distance_pct`` (when enabled) → disabled (0).
        """
        if strategy_risk_exit is not None:
            return float(getattr(strategy_risk_exit, "trailing_stop_pct", 0.0) or 0.0)
        if not getattr(self.hard, "trailing_stop_enabled", False):
            return 0.0
        return float(getattr(self.hard, "trailing_stop_distance_pct", 0.0) or 0.0)

    def calculate_take_profits(self, entry_price: float, side: str) -> list[tuple[float, float]]:
        levels = [
            self.soft.take_profit_1_pct / 100,
            self.soft.take_profit_2_pct / 100,
            self.soft.take_profit_3_pct / 100,
        ]
        tps = []
        for pct in levels:
            if side == "long":
                tp_price = entry_price * (1 + pct)
            else:
                tp_price = entry_price * (1 - pct)
            tps.append((tp_price, pct))
        return tps
