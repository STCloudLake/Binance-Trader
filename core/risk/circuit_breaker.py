import time
from datetime import datetime
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
    week_peak_equity: float = 0.0
    current_equity: float = 0.0
    consecutive_losses: int = 0
    daily_start_equity: float = 0.0
    week_start_equity: float = 0.0
    is_tripped: bool = False
    trip_reason: str = ""
    tripped_at: float = 0.0
    _last_alert_reason: str = field(default="", repr=False)
    _last_check_date: str = field(default="", repr=False)  # ISO date for auto daily reset
    _last_check_week: str = field(default="", repr=False)  # ISO week for auto weekly reset

    def set_equity(self, equity: float):
        if self.daily_start_equity == 0:
            self.daily_start_equity = equity
        if self.week_start_equity == 0:
            self.week_start_equity = equity
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        if equity > self.week_peak_equity:
            self.week_peak_equity = equity

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
        # ── Auto daily/weekly reset based on date (independent of external timer) ──
        today = datetime.now().strftime("%Y-%m-%d")
        this_week = datetime.now().strftime("%Y-W%W")
        if self._last_check_date and self._last_check_date != today:
            self.reset_daily()
        if self._last_check_week and self._last_check_week != this_week:
            self.reset_weekly()
        self._last_check_date = today
        self._last_check_week = this_week

        if self.is_tripped:
            return True, self.trip_reason

        if self.peak_equity > 0:
            daily_dd = (self.peak_equity - self.current_equity) / self.peak_equity * 100
            if daily_dd > self.max_daily_drawdown_pct:
                self._trip(f"Daily drawdown {daily_dd:.2f}% exceeds limit {self.max_daily_drawdown_pct}%")
                return True, self.trip_reason

        # Weekly drawdown uses its own peak so the daily peak reset cannot mask a
        # multi-day slide. (Previously max_weekly_drawdown_pct was never checked:
        # the config advertised a weekly guard that did not exist.)
        if self.max_weekly_drawdown_pct > 0 and self.week_peak_equity > 0:
            weekly_dd = (self.week_peak_equity - self.current_equity) / self.week_peak_equity * 100
            if weekly_dd > self.max_weekly_drawdown_pct:
                self._trip(f"Weekly drawdown {weekly_dd:.2f}% exceeds limit {self.max_weekly_drawdown_pct}%")
                return True, self.trip_reason

        if self.daily_pnl < 0 and abs(self.daily_pnl) >= self.max_daily_loss_usdt:
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
        # NOTE: week_peak_equity is deliberately NOT reset here — a fresh daily
        # peak must not hide a cumulative weekly drawdown.

    def clamp_peak_to_current(self):
        """Only reset peak_equity to current — does NOT reset daily PnL.
        Used when peak was artificially inflated by stale position tracking."""
        if self.current_equity > 0 and self.peak_equity > self.current_equity:
            self.peak_equity = self.current_equity

    def reset_weekly(self):
        self.weekly_pnl = 0.0
        self.week_start_equity = self.current_equity
        self.week_peak_equity = self.current_equity
        self.peak_equity = self.current_equity

    def reset_trip(self):
        self.is_tripped = False
        self.trip_reason = ""
        self.tripped_at = 0.0
        self.consecutive_losses = 0
        self._last_alert_reason = ""
