# Circuit Breaker Auto-Response & AI Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable auto-response actions when circuit breaker trips (tighten stops, close all, close worst), deduplicate breaker alerts, and enable AI-driven breaker handling + recovery in full_auto mode.

**Architecture:** Extend the existing event-driven risk pipeline. RiskManager publishes critical RISK_BREACH on trip; a new handler in main.py executes the configured action. In full_auto mode, DeepSeekController overrides the action and runs a recovery loop that self-evaluates when to resume trading.

**Tech Stack:** Python 3.11+, asyncio, Pydantic, YAML, loguru, SQLite

---

## File Map

| File | Role |
|---|---|
| `config/risk_params.yaml` | New `circuit_breaker_action` default |
| `app/config.py` | `HardRiskLimits` gains `circuit_breaker_action` field |
| `core/risk/circuit_breaker.py` | `_last_alert_reason` field for dedup |
| `core/risk/manager.py` | Throttled breaker alerts; fire `_on_trip` callback |
| `app/main.py` | `_on_circuit_breaker_trip` handler, `_execute_breaker_action` |
| `core/ai/prompts.py` | `BREAKER_ACTION_PROMPT`, `BREAKER_RECOVERY_PROMPT` |
| `core/ai/deepseek_ctl.py` | `decide_breaker_action`, `_breaker_recovery_loop`, `_build_breaker_context` |
| `web/templates/settings.html` | Circuit breaker action dropdown |
| `web/server.py` | Pass/save `circuit_breaker_action` |

---

### Task 1: Config Layer

**Files:**
- Modify: `binance_trader/config/risk_params.yaml`
- Modify: `binance_trader/app/config.py:10-19`

- [ ] **Step 1: Add `circuit_breaker_action` to risk_params.yaml**

Add the new key under `hard_limits`:

```yaml
# risk_params.yaml — insert as last line under hard_limits:
  circuit_breaker_action: block_only
```

- [ ] **Step 2: Add field to HardRiskLimits model**

Edit `binance_trader/app/config.py:10-19`:

```python
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
```

- [ ] **Step 3: Verify config loads**

Run a quick Python check:

```python
python -c "from app.config import Config; c = Config.load('sim'); print(c.hard_limits.circuit_breaker_action)"
```

Expected: `block_only`

- [ ] **Step 4: Commit**

```bash
git add binance_trader/config/risk_params.yaml binance_trader/app/config.py
git commit -m "feat: add circuit_breaker_action to config and HardRiskLimits"
```

---

### Task 2: Circuit Breaker Alert Dedup

**Files:**
- Modify: `binance_trader/core/risk/circuit_breaker.py`

- [ ] **Step 1: Add `_last_alert_reason` and modify `_trip` to return whether it's a new trip**

Edit `binance_trader/core/risk/circuit_breaker.py`:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add binance_trader/core/risk/circuit_breaker.py
git commit -m "feat: add is_new_trip() dedup to CircuitBreaker"
```

---

### Task 3: RiskManager — Throttle Alerts + Fire Trip Callback

**Files:**
- Modify: `binance_trader/core/risk/manager.py`

- [ ] **Step 1: Add throttle state and trip callback to RiskManager**

Edit `binance_trader/core/risk/manager.py`:

In `__init__`, add after `self._account_balance: float = 0.0`:

```python
        self._account_balance: float = 0.0
        self._on_trip_callback: callable | None = None  # set by main.py
        self._last_breaker_alert_time: float = 0.0
```

Add method to register the trip callback:

```python
    def on_trip(self, callback: callable):
        """Register a callback invoked when breaker trips (async, receives trip reason dict)."""
        self._on_trip_callback = callback
```

In `check_signal`, replace the existing breaker trip block (lines 81-83) with:

```python
        tripped, reason = self.breaker.check()
        if tripped:
            if self.breaker.is_new_trip() and self._on_trip_callback:
                import asyncio
                asyncio.create_task(self._on_trip_callback({
                    "reason": reason,
                    "daily_drawdown_pct": (
                        (self.breaker.peak_equity - self.breaker.current_equity) / self.breaker.peak_equity * 100
                    ) if self.breaker.peak_equity > 0 else 0,
                    "daily_pnl": self.breaker.daily_pnl,
                    "consecutive_losses": self.breaker.consecutive_losses,
                    "open_positions": self._open_positions.copy(),
                }))
            return RiskResult(approved=False, reason=f"Circuit breaker tripped: {reason}")
```

Replace the alert block in `_on_signal` (lines 60-63) with throttled version:

```python
            is_critical = "Circuit breaker" in result.reason or "loss limit" in result.reason
            if is_critical:
                import time as _time
                now = _time.time()
                # Throttle: only alert once per 5 minutes for breaker rejections
                if now - self._last_breaker_alert_time >= 300:
                    self._last_breaker_alert_time = now
                    await self._log_risk_event("signal_rejected", "warning", result.reason, "RiskManager")
```

- [ ] **Step 2: Commit**

```bash
git add binance_trader/core/risk/manager.py
git commit -m "feat: add trip callback and alert throttling to RiskManager"
```

---

### Task 4: AI Prompts

**Files:**
- Modify: `binance_trader/core/ai/prompts.py`

- [ ] **Step 1: Append breaker action and recovery prompts**

Add to end of `binance_trader/core/ai/prompts.py`:

```python
BREAKER_ACTION_PROMPT = """You are a risk management expert. A circuit breaker has just tripped. Decide the immediate action.

Breaker state:
{context}

Available actions:
- block_only: Only block new entries, leave existing positions alone
- tighten_stops: Tighten stop-loss on all positions to 2% from current price
- close_all: Market-close ALL open positions immediately
- close_worst: Only close the position with the largest unrealized loss

Choose the action that best protects the portfolio given the current drawdown and positions.

Return ONLY valid JSON:
{{
  "action": "close_all",
  "rationale": "Brief explanation of why this action was chosen"
}}"""

BREAKER_RECOVERY_PROMPT = """You are a risk management expert. The circuit breaker is currently tripped. Evaluate whether it is safe to resume trading.

{context}

Consider:
1. Has enough time passed since the trip?
2. Is the market regime favorable for re-entry?
3. Have positions been resolved?

Return ONLY valid JSON:
{{
  "reset": true,
  "reason": "Brief explanation of decision"
}}"""
```

- [ ] **Step 2: Commit**

```bash
git add binance_trader/core/ai/prompts.py
git commit -m "feat: add breaker action and recovery AI prompts"
```

---

### Task 5: Breaker Response Handler in main.py

**Files:**
- Modify: `binance_trader/app/main.py`

- [ ] **Step 1: Add `_execute_breaker_action` function and handler**

In `main.py`, after `_on_position_reduce` handler (after line 97), add:

```python
    async def _execute_breaker_action(action: str, reason: str):
        """Execute the configured circuit breaker response action."""
        if action == "block_only":
            logger.info(f"Breaker action: block_only — {reason}")
            return

        open_positions = order_executor.get_open_positions()
        if not open_positions:
            logger.info(f"Breaker tripped but no open positions — {reason}")
            return

        if action == "close_all":
            logger.warning(f"Breaker action: close_all — closing {len(open_positions)} positions")
            for sym in list(open_positions.keys()):
                price = market_data.get_current_price(sym) or open_positions[sym].get("current_price", open_positions[sym]["entry_price"])
                result = await order_executor.close_position(sym, 100, price)
                if result.get("ok"):
                    invested_returned = result.get("invested_returned", 0)
                    trade_pnl = result.get("pnl", 0)
                    new_balance = await atomic_adjust_balance(invested_returned + trade_pnl, config.db_path)
                    risk_manager.update_balance(new_balance)
                    logger.info(f"Breaker close_all: {sym} PnL={trade_pnl:.2f} Balance={new_balance:.0f}")

        elif action == "close_worst":
            # Find position with worst unrealized PnL
            worst_sym = None
            worst_pnl = float("inf")
            for sym, pos in open_positions.items():
                entry = pos["entry_price"]
                qty = pos["quantity"]
                side = pos["side"]
                cur_price = market_data.get_current_price(sym) or pos.get("current_price", entry)
                unrealized = (cur_price - entry) * qty if side == "long" else (entry - cur_price) * qty
                if unrealized < worst_pnl:
                    worst_pnl = unrealized
                    worst_sym = sym

            if worst_sym:
                logger.warning(f"Breaker action: close_worst — closing {worst_sym} (uPnL={worst_pnl:.2f})")
                price = market_data.get_current_price(worst_sym) or open_positions[worst_sym]["entry_price"]
                result = await order_executor.close_position(worst_sym, 100, price)
                if result.get("ok"):
                    invested_returned = result.get("invested_returned", 0)
                    trade_pnl = result.get("pnl", 0)
                    new_balance = await atomic_adjust_balance(invested_returned + trade_pnl, config.db_path)
                    risk_manager.update_balance(new_balance)
                    logger.info(f"Breaker close_worst: {worst_sym} PnL={trade_pnl:.2f} Balance={new_balance:.0f}")

        elif action == "tighten_stops":
            logger.warning(f"Breaker action: tighten_stops — adjusting stops on {len(open_positions)} positions")
            for sym, pos in open_positions.items():
                price = market_data.get_current_price(sym) or pos.get("current_price", pos["entry_price"])
                side = pos["side"]
                # Set stop-loss 2% from current price
                new_sl = price * 0.98 if side == "long" else price * 1.02
                pos["stop_loss"] = round(new_sl, 2)
                await event_bus.publish(Event(EventType.POSITION_UPDATE, {
                    "symbol": sym, "side": side, "quantity": pos["quantity"],
                    "entry_price": pos["entry_price"], "current_price": price,
                    "stop_loss": pos["stop_loss"],
                    "position_type": pos.get("position_type", "satellite"),
                    "position_value": pos.get("amount_usdt", pos["quantity"] * pos["entry_price"]),
                    "amount_usdt": pos.get("amount_usdt", pos["quantity"] * pos["entry_price"]),
                    "trade_group": pos.get("trade_group", ""),
                    "closed": False, "pnl": 0,
                }))
                logger.info(f"Breaker tighten_stops: {sym} {side} SL→{new_sl:.2f}")

    async def _on_circuit_breaker_trip(event: Event):
        data = event.data
        reason = data.get("reason", "Unknown")
        logger.error(f"Circuit breaker TRIPPED: {reason}")

        if config.ai_mode == "full_auto" and deepseek_ctl.client:
            action = await deepseek_ctl.decide_breaker_action(data)
        else:
            action = config.hard_limits.circuit_breaker_action

        await _execute_breaker_action(action, reason)

        # In full_auto mode, start AI recovery loop
        if config.ai_mode == "full_auto" and deepseek_ctl.client:
            asyncio.create_task(deepseek_ctl._breaker_recovery_loop())

    event_bus.subscribe(EventType.RISK_BREACH, _on_circuit_breaker_trip, 
                         lambda e: e.data.get("event_type") == "circuit_breaker_trip")
```

Wait — the EventBus subscribe doesn't support filtering lambdas currently. Let me check if we need to modify approach.

Actually, let me check the EventBus subscribe signature to handle this properly.

- [ ] **Step 2: Check EventBus subscribe signature and adjust**

The subscriber receives ALL events of the subscribed type. We'll handle filtering inside the handler instead:

```python
    async def _on_risk_breach(event: Event):
        if event.data.get("event_type") != "circuit_breaker_trip":
            return
        data = event.data
        reason = data.get("detail", "Unknown")
        logger.error(f"Circuit breaker TRIPPED: {reason}")

        if config.ai_mode == "full_auto" and deepseek_ctl.client:
            action = await deepseek_ctl.decide_breaker_action(data)
        else:
            action = config.hard_limits.circuit_breaker_action

        await _execute_breaker_action(action, reason)

        if config.ai_mode == "full_auto" and deepseek_ctl.client:
            asyncio.create_task(deepseek_ctl._breaker_recovery_loop())

    event_bus.subscribe(EventType.RISK_BREACH, _on_risk_breach)
```

And in `risk_manager.py`, the trip callback publishes `RISK_BREACH` with `event_type: "circuit_breaker_trip"`:

The `_on_trip_callback` in RiskManager should publish:

```python
    async def _trip_callback(self, data: dict):
        await self.event_bus.publish(Event(EventType.RISK_BREACH, {
            "event_type": "circuit_breaker_trip",
            "level": "critical",
            "detail": data["reason"],
            "daily_drawdown_pct": data.get("daily_drawdown_pct", 0),
            "daily_pnl": data.get("daily_pnl", 0),
            "consecutive_losses": data.get("consecutive_losses", 0),
            "open_positions": data.get("open_positions", {}),
            "triggered_by": "CircuitBreaker",
        }))
```

Update RiskManager.__init__ to wire `_trip_callback`:

```python
        self._account_balance: float = 0.0
        self._last_breaker_alert_time: float = 0.0
```

And the `check_signal` trip block:

```python
        tripped, reason = self.breaker.check()
        if tripped:
            if self.breaker.is_new_trip():
                import asyncio
                drawdown = (
                    (self.breaker.peak_equity - self.breaker.current_equity) / self.breaker.peak_equity * 100
                ) if self.breaker.peak_equity > 0 else 0
                asyncio.create_task(self._trip_callback({
                    "reason": reason,
                    "daily_drawdown_pct": drawdown,
                    "daily_pnl": self.breaker.daily_pnl,
                    "consecutive_losses": self.breaker.consecutive_losses,
                    "open_positions": self._open_positions.copy(),
                }))
            return RiskResult(approved=False, reason=f"Circuit breaker tripped: {reason}")
```

This is cleaner — `_trip_callback` is a method on RiskManager itself that publishes the RISK_BREACH event directly.

- [ ] **Step 3: Update RiskManager with _trip_callback method**

In `binance_trader/core/risk/manager.py`, add the `_trip_callback` method after `check_signal`:

```python
    async def _trip_callback(self, data: dict):
        """Publish critical RISK_BREACH when breaker first trips (deduped by is_new_trip)."""
        await self.event_bus.publish(Event(EventType.RISK_BREACH, {
            "event_type": "circuit_breaker_trip",
            "level": "critical",
            "detail": data["reason"],
            "daily_drawdown_pct": data.get("daily_drawdown_pct", 0),
            "daily_pnl": data.get("daily_pnl", 0),
            "consecutive_losses": data.get("consecutive_losses", 0),
            "open_positions": data.get("open_positions", {}),
            "triggered_by": "CircuitBreaker",
        }))
```

And update `check_signal` breaker block to use it:

```python
        tripped, reason = self.breaker.check()
        if tripped:
            if self.breaker.is_new_trip():
                drawdown = (
                    (self.breaker.peak_equity - self.breaker.current_equity) / self.breaker.peak_equity * 100
                ) if self.breaker.peak_equity > 0 else 0
                asyncio.ensure_future(self._trip_callback({
                    "reason": reason,
                    "daily_drawdown_pct": drawdown,
                    "daily_pnl": self.breaker.daily_pnl,
                    "consecutive_losses": self.breaker.consecutive_losses,
                    "open_positions": self._open_positions.copy(),
                }))
            return RiskResult(approved=False, reason=f"Circuit breaker tripped: {reason}")
```

Add `import asyncio` at the top of manager.py.

- [ ] **Step 4: Commit**

```bash
git add binance_trader/core/risk/manager.py binance_trader/app/main.py
git commit -m "feat: add breaker response handler and trip callback"
```

---

### Task 6: DeepSeekController — AI Breaker Decision & Recovery

**Files:**
- Modify: `binance_trader/core/ai/deepseek_ctl.py`

- [ ] **Step 1: Add breaker methods to DeepSeekController**

After `wire()` method, add `_build_breaker_context`:

```python
    def _build_breaker_context(self, breaker_data: dict = None) -> str:
        """Build context string for breaker-related AI decisions."""
        parts = []
        if breaker_data:
            parts.append(f"Trip reason: {breaker_data.get('reason', 'Unknown')}")
            parts.append(f"Daily drawdown: {breaker_data.get('daily_drawdown_pct', 0):.2f}%")
            parts.append(f"Daily PnL: {breaker_data.get('daily_pnl', 0):.2f} USDT")
            parts.append(f"Consecutive losses: {breaker_data.get('consecutive_losses', 0)}")

        if self._risk_manager:
            try:
                bal = self._risk_manager._account_balance
                parts.append(f"Account balance: {bal:.0f} USDT")
                cb = self._risk_manager.breaker
                parts.append(f"Breaker tripped: {cb.is_tripped}")
                if cb.is_tripped:
                    parts.append(f"Trip reason: {cb.trip_reason}")
                    parts.append(f"Peak equity: {cb.peak_equity:.0f} USDT")
                    parts.append(f"Current equity: {cb.current_equity:.0f} USDT")
            except Exception:
                pass

        if self._executor:
            try:
                positions = self._executor.get_open_positions()
                if positions:
                    pos_list = []
                    for s, p in positions.items():
                        upnl = p.get("unrealized_pnl", 0)
                        pos_list.append(f"{s}: {p['side']} qty={p['quantity']:.4f} @ {p['entry_price']:.2f} uPnL={upnl:.2f}")
                    parts.append(f"Open positions ({len(positions)}): " + "; ".join(pos_list))
                else:
                    parts.append("Open positions: none")
            except Exception:
                pass

        if self._market_data:
            try:
                prices = []
                for sym in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]:
                    price = self._market_data.get_current_price(sym)
                    if price:
                        prices.append(f"{sym}={price:.2f}")
                parts.append("Current prices: " + ", ".join(prices))
            except Exception:
                pass

        return "\n".join(parts)
```

- [ ] **Step 2: Add `decide_breaker_action`**

```python
    async def decide_breaker_action(self, breaker_data: dict) -> str:
        """Ask DeepSeek which breaker action to take. Returns action string. Timeout 15s, fallback close_all."""
        from core.ai.prompts import BREAKER_ACTION_PROMPT
        context = self._build_breaker_context(breaker_data)
        prompt = BREAKER_ACTION_PROMPT.format(context=context)

        try:
            result = await asyncio.wait_for(
                self._call_deepseek(
                    "You are a risk management expert. Always respond in valid JSON.",
                    prompt
                ),
                timeout=15.0
            )
            if result:
                data = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
                action = data.get("action", "close_all")
                logger.info(f"AI breaker decision: {action} — {data.get('rationale', '')}")
                if action in ("block_only", "tighten_stops", "close_all", "close_worst"):
                    return action
        except asyncio.TimeoutError:
            logger.warning("AI breaker decision timed out, fallback to close_all")
        except Exception as e:
            logger.warning(f"AI breaker decision failed: {e}, fallback to close_all")

        return "close_all"
```

- [ ] **Step 3: Add `_breaker_recovery_loop`**

```python
    async def _breaker_recovery_loop(self):
        """Background task: periodically evaluate if breaker can be reset. Exits when breaker is no longer tripped."""
        from core.ai.prompts import BREAKER_RECOVERY_PROMPT
        await asyncio.sleep(120)  # Wait 2 minutes before first evaluation

        while self._running and self._risk_manager:
            try:
                cb = self._risk_manager.breaker
                if not cb.is_tripped:
                    logger.info("Breaker recovery loop: breaker already reset, exiting")
                    return

                context = self._build_breaker_context()
                prompt = BREAKER_RECOVERY_PROMPT.format(context=context)
                result = await self._call_deepseek(
                    "You are a risk management expert. Always respond in valid JSON.",
                    prompt
                )

                if result:
                    data = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
                    if data.get("reset"):
                        cb.reset_trip()
                        cb.reset_daily()
                        logger.info(f"AI recovery: breaker reset — {data.get('reason', '')}")
                        await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
                            "level": "info",
                            "type": "breaker_recovery",
                            "message": f"AI 已恢复交易: {data.get('reason', '自动恢复')}",
                        }))
                        await self._heartbeat("breaker_recovery", True)
                        return
                    else:
                        logger.info(f"AI recovery: keep breaker tripped — {data.get('reason', '')}")

                await self._heartbeat("breaker_recovery", True)
            except Exception as e:
                logger.warning(f"Breaker recovery evaluation failed: {e}")
                await self._heartbeat("breaker_recovery", False)

            # Re-check every 5 minutes
            await asyncio.sleep(300)
```

Add `import asyncio` to deepseek_ctl.py imports.

- [ ] **Step 4: Commit**

```bash
git add binance_trader/core/ai/deepseek_ctl.py
git commit -m "feat: add AI breaker decision and recovery loop to DeepSeekController"
```

---

### Task 7: Settings Page UI

**Files:**
- Modify: `binance_trader/web/templates/settings.html`
- Modify: `binance_trader/web/server.py`

- [ ] **Step 1: Add circuit_breaker_action dropdown to settings.html**

In `binance_trader/web/templates/settings.html`, after the `max_consecutive_losses` input block (after line 93), insert:

```html
            <div class="mb-3"><label class="text-sm text-slate-400">熔断自动操作</label>
                <select name="circuit_breaker_action" class="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white text-sm">
                    <option value="block_only" {{ 'selected' if hard_limits.circuit_breaker_action == 'block_only' }}>仅阻止新开仓</option>
                    <option value="tighten_stops" {{ 'selected' if hard_limits.circuit_breaker_action == 'tighten_stops' }}>收紧所有止损</option>
                    <option value="close_all" {{ 'selected' if hard_limits.circuit_breaker_action == 'close_all' }}>平掉所有仓位</option>
                    <option value="close_worst" {{ 'selected' if hard_limits.circuit_breaker_action == 'close_worst' }}>平仓亏损最大持仓</option>
                </select>
                <p class="text-xs text-slate-500 mt-1">半自动模式使用此设置；全自动模式由 AI 实时决定</p></div>
```

- [ ] **Step 2: Update web/server.py to pass and save circuit_breaker_action**

The settings page already passes `hard_limits` model, and since we added the field to the Pydantic model, it's automatically included in the template. No change needed for the GET endpoint.

For the POST endpoint `save_risk_settings`, add the new parameter:

Edit `binance_trader/web/server.py:1102-1106`, add `circuit_breaker_action` parameter:

```python
    @app.post("/api/settings/risk")
    async def save_risk_settings(max_daily_drawdown: float = Form(5.0), max_daily_loss: float = Form(500.0),
                                  max_open_trades: int = Form(8), max_position_size_pct: float = Form(10.0),
                                  max_leverage: int = Form(3), max_consecutive_losses: int = Form(5),
                                  circuit_breaker_action: str = Form("block_only"),
                                  spot_enabled: str = Form("1"), futures_enabled: str = Form("0")):
```

Then in the save logic (after line 1112), add:

```python
        config.hard_limits.circuit_breaker_action = circuit_breaker_action
```

And in the YAML persistence block (after line 1132), add:

```python
            hl["circuit_breaker_action"] = circuit_breaker_action
```

- [ ] **Step 3: Commit**

```bash
git add binance_trader/web/templates/settings.html binance_trader/web/server.py
git commit -m "feat: add circuit_breaker_action to settings UI and API"
```

---

### Task 8: Integration Verification

**Files:**
- No new files. Run the app and verify.

- [ ] **Step 1: Start the app in sim mode**

```bash
cd binance_trader && python -m app.main --mode sim --port 8899
```

- [ ] **Step 2: Verify settings page loads**

Open `http://127.0.0.1:8899/settings`. Confirm:
- "熔断自动操作" dropdown appears with 4 options
- "仅阻止新开仓" is selected by default
- The hint text "半自动模式使用此设置；全自动模式由 AI 实时决定" is visible

- [ ] **Step 3: Verify config persistence**

Change the dropdown to "平掉所有仓位", click "保存风险参数". Restart the app. Confirm the setting persisted.

- [ ] **Step 4: Verify no regressions**

Check that:
- Dashboard loads and shows positions
- Strategies page works
- Alerts page shows alerts
- Trades appear in trade history

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: verified circuit breaker auto-response integration"
```
