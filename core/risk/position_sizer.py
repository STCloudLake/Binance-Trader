from app.config import SoftRiskParams, HardRiskLimits


class PositionSizer:
    def __init__(self, hard_limits: HardRiskLimits, soft_params: SoftRiskParams,
                 core_capital_pct: float = 0.7, satellite_capital_pct: float = 0.3):
        self.hard = hard_limits
        self.soft = soft_params
        self.core_capital_pct = core_capital_pct
        self.satellite_capital_pct = satellite_capital_pct

    def calculate_position_size(self, account_balance: float, current_price: float,
                                 position_type: str = "satellite") -> tuple[float, float]:
        if position_type == "core":
            capital_pool = account_balance * self.core_capital_pct
        else:
            capital_pool = account_balance * self.satellite_capital_pct

        # Enforce a floor of 1% — AI or misconfiguration can't zero out trading
        effective_pct = max(self.soft.position_size_pct, 1.0)
        risk_per_trade = capital_pool * (effective_pct / 100)
        max_risk = account_balance * (self.hard.max_position_size_pct / 100)
        risk_per_trade = min(risk_per_trade, max_risk)

        quantity = risk_per_trade / current_price if current_price > 0 else 0
        return quantity, risk_per_trade

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        sl_pct = max(self.soft.stop_loss_pct / 100, self.hard.min_stop_loss_distance_pct / 100)
        if side == "long":
            return entry_price * (1 - sl_pct)
        else:
            return entry_price * (1 + sl_pct)

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
