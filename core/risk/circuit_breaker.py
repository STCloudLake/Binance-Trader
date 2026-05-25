import time
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    max_daily_drawdown_pct: float = 5.0
    max_weekly_drawdown_pct: float = 10.0
    max_daily_loss_usdt: float = 500.0
    max_consecutive_losses: int = 5

    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    consecutive_losses: int = 0
    daily_start_equity: float = 0.0
    week_start_equity: float = 0.0
    is_tripped: bool = False
    trip_reason: str = ""
    tripped_at: float = 0.0
    _last_alert_reason: str = field(default="", repr=False)

    def set_equity(self, equity: float):
        if self.daily_start_equity == 0:
            self.daily_start_equity = equity
        if self.week_start_equity == 0:
            self.week_start_equity = equity
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def add_trade_result(self, pnl: float):
        # Round to 2dp — same precision as trade history DB storage,
        # so the breaker's loss count matches what the user sees in history.
        pnl_rounded = round(pnl, 2)
        self.daily_pnl += pnl_rounded
        self.weekly_pnl += pnl_rounded
        if pnl_rounded < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def check(self) -> tuple[bool, str]:
        if self.is_tripped:
            return True, self.trip_reason

        if self.peak_equity > 0:
            daily_dd = (self.peak_equity - self.current_equity) / self.peak_equity * 100
            if daily_dd > self.max_daily_drawdown_pct:
                self._trip(f"Daily drawdown {daily_dd:.2f}% exceeds limit {self.max_daily_drawdown_pct}%")
                return True, self.trip_reason

        if abs(self.daily_pnl) >= self.max_daily_loss_usdt and self.daily_pnl < 0:
            self._trip(f"Daily loss ${abs(self.daily_pnl):.2f} exceeds limit ${self.max_daily_loss_usdt}")
            return True, self.trip_reason

        if self.consecutive_losses >= self.max_consecutive_losses:
            self._trip(f"Consecutive losses {self.consecutive_losses} >= limit {self.max_consecutive_losses}")
            return True, self.trip_reason

        return False, ""

    def _trip(self, reason: str):
        self.is_tripped = True
        self.trip_reason = reason
        self.tripped_at = time.time()

    def is_new_trip(self) -> bool:
        """Returns True only the first time check() trips on a given reason.
        Subsequent calls with same reason return False (dedup)."""
        if self.is_tripped and self.trip_reason != self._last_alert_reason:
            self._last_alert_reason = self.trip_reason
            return True
        return False

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.daily_start_equity = self.current_equity
        self.peak_equity = self.current_equity

    def reset_weekly(self):
        self.weekly_pnl = 0.0
        self.week_start_equity = self.current_equity
        self.peak_equity = self.current_equity

    def reset_trip(self):
        self.is_tripped = False
        self.trip_reason = ""
        self.tripped_at = 0.0
        self.consecutive_losses = 0
        self._last_alert_reason = ""
