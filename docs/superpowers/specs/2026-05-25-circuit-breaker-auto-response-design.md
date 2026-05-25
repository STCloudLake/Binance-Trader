# Circuit Breaker Auto-Response & AI Recovery

Date: 2026-05-25

## Problem

1. **Alert storm**: Circuit breaker trips → every rejected signal triggers 2 alerts (RISK_BREACH + ALERT_TRIGGER), generating 100+ duplicate alerts with same reason.
2. **No auto-response**: When breaker trips, existing positions are left unmanaged. The system only blocks new entries.
3. **No AI recovery path**: In full_auto mode, the AI has no mechanism to handle breaker events or decide when to resume trading.

## Design

### 1. Config: circuit_breaker_action

New field in `risk_params.yaml` → `hard_limits`:

```yaml
hard_limits:
  circuit_breaker_action: block_only  # block_only | tighten_stops | close_all | close_worst
```

`HardRiskLimits` model gains `circuit_breaker_action: str = "block_only"`.

### 2. Alert Dedup

**CircuitBreaker**: new field `_last_alert_reason: str`. `_trip()` only fires if reason changed.
**RiskManager._on_signal()**: breaker rejection alerts are throttled to once per 5 minutes (track `_last_breaker_alert_time`).

### 3. Breaker Response Execution

When breaker trips, `RiskManager` publishes `RISK_BREACH` (level="critical"). A new handler in `main.py` subscribes and executes:

```
_on_circuit_breaker_trip(event)
  ├─ semi_auto: read config.hard_limits.circuit_breaker_action
  └─ full_auto: call deepseek_ctl.decide_breaker_action(data) → returns action
  └─ _execute_breaker_action(action, executor, risk_manager)
```

`_execute_breaker_action`:

| Action | Behavior |
|---|---|
| `block_only` | No-op (already blocked by breaker) |
| `tighten_stops` | For each open position: set stop_loss to current_price ± 2% (below for long, above for short). Publish POSITION_UPDATE per position. |
| `close_all` | Iterate all positions, call `executor.close_position(sym, 100, price)`. |
| `close_worst` | Sort positions by unrealized PnL, close the lowest. |

### 4. AI Full-Auto Integration

**DeepSeekController** new methods:

- `decide_breaker_action(data: dict) → str`: Build context with breaker reason, open positions, drawdown%. Call DeepSeek with a concise prompt. Returns one of the 4 action values. Timeout 15s, fallback to `close_all`.

- `_breaker_recovery_loop()`: Spawned as background task when breaker trips. Periodically (interval from config, default 5 min):
  1. Check if breaker is still tripped. If not, exit loop.
  2. Build recovery context (current drawdown, market state, time since trip).
  3. Call DeepSeek: "Should the circuit breaker be reset? Respond {\"reset\": true/false, \"reason\": \"...\"}".
  4. If `reset: true`: call `risk_manager.breaker.reset_trip()` + `reset_daily()`, publish "trading resumed" alert, exit loop.

- `_build_breaker_context() → str`: Portfolio state + breaker state formatted for AI prompt.

### 5. Settings Page UI

In `settings.html`, risk params card, below the breaker trip alert box:

```html
<div class="mb-3">
  <label class="text-sm text-slate-400">熔断自动操作</label>
  <select name="circuit_breaker_action" class="w-full bg-slate-800 border ...">
    <option value="block_only">仅阻止新开仓</option>
    <option value="tighten_stops">收紧所有止损</option>
    <option value="close_all">平掉所有仓位</option>
    <option value="close_worst">平仓亏损最大持仓</option>
  </select>
  <p class="text-xs text-slate-500 mt-1">半自动模式下熔断时自动执行的操作；全自动模式由 AI 实时决定</p>
</div>
```

Saved via existing `/api/settings/risk` endpoint (add the new field).

### 6. Files Changed

| File | Change |
|---|---|
| `config/risk_params.yaml` | Add `circuit_breaker_action` |
| `app/config.py` | Add field to `HardRiskLimits` |
| `core/risk/circuit_breaker.py` | Add `_last_alert_reason` dedup |
| `core/risk/manager.py` | Throttle breaker alerts; publish critical RISK_BREACH on trip |
| `app/main.py` | Add `_on_circuit_breaker_trip` handler + `_execute_breaker_action` |
| `core/ai/deepseek_ctl.py` | Add `decide_breaker_action`, `_breaker_recovery_loop`, `_build_breaker_context` |
| `core/ai/prompts.py` | Add `BREAKER_ACTION_PROMPT`, `BREAKER_RECOVERY_PROMPT` |
| `web/templates/settings.html` | Add circuit_breaker_action dropdown |
| `web/server.py` | Pass new field to template; save it from form POST |

### 7. Edge Cases

- **Breaker trips with no open positions**: `tighten_stops`/`close_all`/`close_worst` are no-ops. `block_only` is sufficient.
- **AI unavailable during full_auto trip**: Fallback to `close_all` (safest default).
- **Recovery loop while breaker manually reset**: Loop detects `is_tripped=False` and exits cleanly.
- **Multiple consecutive trips**: Each trip spawns a new recovery loop; old loops exit immediately because `is_tripped` is already True with the same reason (dedup means only first trip fires the handler).
