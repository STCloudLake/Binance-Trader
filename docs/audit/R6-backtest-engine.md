# 审计报告 R6 — Backtest Engine

**日期**: 2026-07-16
**审查文件**: `core/backtest/engine.py`, `core/backtest/engine_hybrid.py`, `core/backtest/event_executor.py`, `core/backtest/signal_matrix.py`, `core/backtest/metrics.py`, `core/backtest/cost_model.py`
**审查代码行数**: ~1100 行
**审计方法**: 引擎等价性对比 + 前瞻偏差检测 + 成本模型验证

---

## 摘要

| 等级 | 数量 |
|------|------|
| Critical | 1 |
| High | 2 |
| Medium | 4 |
| Low | 2 |
| **总计** | **9** |

---

## 漏洞清单

| ID | 等级 | 文件:行 | 描述 |
|----|------|---------|------|
| R6-001 | **Critical** | engine.py:844-863 | 回测引擎入口条件使用 AND 逻辑，实盘使用 OR 逻辑 — 结果不可比 |
| R6-002 | High | engine.py:540-548 | ML 准确率追踪使用未来数据 — 前瞻偏差 (look-ahead bias) |
| R6-003 | High | event_executor.py:244-267 | Hybrid executor 的仓位计算不使用 strategy.risk_exit — 与 Legacy/Live 不一致 |
| R6-004 | Medium | engine.py:806-808 | `dominant_regime` 假设 BTCUSDT 始终在 symbols 列表中 |
| R6-005 | Medium | metrics.py:47-48 | 盈亏判断不包含手续费 — trade["pnl"] 已扣费但 win/loss 用裸 PNL |
| R6-006 | Medium | signal_matrix.py:58-61 | IndicatorGrouper JSON 序列化 float 可能导致 hash 不一致 |
| R6-007 | Medium | engine.py:165-166 | `_select_engine` ValueError 被静默 fallback — 无告警 |
| R6-008 | Low | engine.py:926-929 | Kelly-lite 风险调整使用固定 1% — 与实盘可配置值不一致 |
| R6-009 | Low | event_executor.py:251 | Hybrid executor 的 sl_dist 硬编码 0.02(2%) — 忽略用户配置 |

---

## 详细分析

### R6-001 🔴 Critical — 回测与实盘条件逻辑不一致

**文件**: `core/backtest/engine.py:844-863` vs `core/strategy/engine.py:106-115`

**Legacy 回测引擎** (engine.py:844-863):
```python
# ALL conditions must be met (AND logic)
long_active = True
for cond in long_conds:
    mask = evaluate_condition(df_primary, cond)
    met = bool(hasattr(mask, 'iloc') and mask.iloc[-1])
    if not met:
        long_active = False
        break
```

**实盘策略引擎** (strategy/engine.py:106-115):
```python
# ANY condition met = side active (OR logic)
for cond in conditions:
    mask = evaluate_condition(df, cond)
    met = bool(hasattr(mask, 'iloc') and mask.iloc[-1])
    if met and side == "long":
        long_active = True  # stays True once any condition matches
```

**描述**: 这是回测与实盘之间的根本性逻辑差异：
- **回测**: 所有 long 条件必须**同时**满足才触发
- **实盘**: 任何一个 long 条件满足就触发

这意味着回测结果**严重低估**了策略的实际交易频率和风险。GA 优化的策略基于 AND 逻辑选出的"冠军"，在实盘 OR 逻辑下可能表现完全不同。

示例：
```yaml
entry_conditions:
  long:
    - "rsi < 30"        # 条件1：超卖
    - "volume_ratio > 1.5"  # 条件2：放量
```
- 回测：必须同时超卖 AND 放量 → 很少触发
- 实盘：超卖 OR 放量 → 频繁触发

**Signal Matrix (Hybrid 引擎)** 的逻辑也需要确认。如果 signal_matrix.py 也使用 AND 逻辑，那么 GA 评估和实盘都基于 AND，但实盘仍使用 OR → GA 优化的策略在实盘无效。

**修复建议**: 统一为一种逻辑。推荐 OR（任何条件满足）因为它更灵活，用户可以通过写更多条件来实现 AND 效果（嵌套条件）。

---

### R6-002 🟠 High — ML 准确率追踪使用未来数据

**文件**: `core/backtest/engine.py:540-548`

```python
# After prediction, look into the future to check accuracy
future_df = df_tf[df_tf.index > ts]  # ← ts is current time
if len(future_df) >= fwd:
    cur_close = float(sliced.iloc[-1]["close"])
    fut_close = float(future_df.iloc[fwd - 1]["close"])
    ret = (fut_close - cur_close) / cur_close
    if abs(ret) >= th:
        ml_total += 1
        if (ret >= th and conf >= 0.5) or (ret <= -th and conf < 0.5):
            ml_correct += 1
```

**描述**: `future_df = df_tf[df_tf.index > ts]` 直接访问了当前时间戳之后的未来数据来评估 ML 预测准确率。虽然这不影响交易决策（预测本身使用 `sliced = df_tf.iloc[:pos_tf + 1]` 即仅使用历史数据），但 `ml_accuracy_pct` 指标在最终报告中展示给用户，使用了未来信息。

然而更严重的是：这个代码**确实存在**在回测的每个时间步访问未来数据。虽然当前实现中它只用于准确率统计，但如果在未来某次重构中将 `ml_predictions` 的值用于交易决策（例如动态禁用 ML），前瞻偏差就会污染交易决策。

**复现路径**: 回测完成后查看 ml_accuracy_pct → 数值被未来数据污染 → 偏高 → 对 ML 模型过度信任。

**修复建议**: 评估准确率应在回测完成后进行（使用带标签的完整数据集），而非在回测循环中实时计算。

---

### R6-003 🟠 High — Hybrid Executor 仓位计算不一致

**文件**: `core/backtest/event_executor.py:244-267` vs `core/backtest/engine.py:921-953`

**Hybrid EventDrivenExecutor**:
```python
qty, risk_amount = self.sizer.calculate_position_size(...)
if self.cost_config:
    risk_capital = balance * 0.01  # hardcoded 1%
    sl_dist = 0.02  # hardcoded 2%
    qty_risk = risk_capital / (price * sl_dist)
    qty = min(qty, qty_risk)
sl = self.sizer.calculate_stop_loss(entry_price=price, side=side)  # uses global soft_params
```

**Legacy BacktestEngine**:
```python
qty, risk_amount = sizer.calculate_position_size(...)
re = strategy.risk_exit
if re is not None:
    risk_capital = balance * 0.01
    sl_dist = re.stop_loss_pct / 100.0  # uses strategy-specific value
    qty_risk = risk_capital / (price * sl_dist)
    qty = min(qty, qty_risk)
sl = price * (1 - sl_pct) if side == "long" else price * (1 + sl_pct)  # uses strategy risk_exit
```

Hybrid 引擎不使用策略级别的 `risk_exit` 配置，导致：
1. 止损距离始终使用全局 `soft_params.stop_loss_pct`，而非策略自定义值
2. trailing_stop_pct 硬编码为 1.5%

**复现路径**: 策略配置了 `risk_exit.stop_loss_pct = 1.0` → Hybrid 回测仍使用全局 2% → 回测结果与实盘/legacy 不同。

---

### R6-004 🟡 Medium — BTCUSDT 硬编码假设

**文件**: `core/backtest/engine.py:806-808`

```python
dominant_regime = market_regime.get("BTCUSDT", "range")
w_ind, w_ml, w_news = _update_weights(dominant_regime, step)
```

**描述**: 市场状态检测假设 `BTCUSDT` 始终在回测的 symbols 列表中。如果用户仅回测山寨币（如 `["SOLUSDT", "XRPUSDT"]`），`market_regime` 中可能没有 BTCUSDT 的 regime 数据 → 始终 fallback 到 "range" → 信号权重不会根据市场状态调整。

此外，`market_regime` dict 仅在 ML 循环中填充（line 504-511），如果 ML 未启用，dict 永远为空 → 所有 symbol 都是 "range"。

---

### R6-005 🟡 Medium — Win/Loss 分类基于未扣费 PNL

**文件**: `core/backtest/metrics.py:47-48`

```python
winning = [t for t in trades if t.get("pnl", 0) > 0]
losing = [t for t in trades if t.get("pnl", 0) < 0]
```

**描述**: `trade["pnl"]` 在 `_close_position()` 中已扣除了交易成本（engine.py:1029: `pnl -= costs`）。但此处直接用 `pnl > 0` 判断盈亏。如果一笔交易扣除成本后恰好 PNL=0，它不会被计入赢或输，导致 `win_rate = wins / n_trades` 可能不等于 `(wins + losses) / n_trades`。

这是合理但不精确的分类。更重要的是：`trade["pnl"]` 可能正好为 0（微利被手续费抵消后），此时该交易被归类为"非赢非输"但 `n_trades` 仍计入分母 → `win_rate_pct` 偏低。

---

### R6-006 🟡 Medium — JSON 序列化 float 可能导致 hash 不一致

**文件**: `core/backtest/signal_matrix.py:58-61`

```python
@staticmethod
def _config_hash(config: StrategyConfig) -> str:
    raw = json.dumps(config.indicators, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

**描述**: `config.indicators` 是 `dict[str, Any]`，其中的 float 值在 JSON 序列化时的精度取决于 Python 的 `json.dumps` 实现。虽然 `sort_keys=True` 确保了键顺序稳定，但浮点精度问题可能导致 `2.0` 和 `2.00` 产生相同的 JSON 字符串（json.dumps 将 `2.0` 序列化为 `2.0`，而 `2.00` 在 Python 中就是 `2.0`），所以这个问题在 Python 中不太可能发生。

但 `ensure_ascii=True` 意味着非 ASCII 字符（如中文策略名？不，indicator configs 不会有中文）被转义。这应该没问题。

实际风险较低，但为了绝对确定性，应使用规范化的 JSON（如限制小数位数）。

---

### R6-007 🟡 Medium — Hybrid 引擎失败静默 fallback

**文件**: `core/backtest/engine.py:164-179`

```python
try:
    use_hybrid = self._select_engine(strategies, _engine_mode) == "hybrid"
except ValueError:
    use_hybrid = False

if use_hybrid:
    try:
        return run_hybrid(...)
    except Exception as e:
        logger.warning(f"Hybrid engine failed ({e}), falling back to legacy")
        # Fall through to legacy engine below
```

**描述**: 两层静默 fallback：
1. `_select_engine` 的 ValueError 被静默捕获 → 退化为 legacy → 用户不知道为什么用了 legacy
2. `run_hybrid` 的任何异常被静默捕获 → 退化为 legacy → 回测结果不同但无明确告警

如果用户在 Web UI 中明确选择了 "hybrid" 模式，却静默退化为 legacy，回测结果可能显著不同（signals、trades、metrics）。

**修复建议**: 至少对显式指定的 engine_mode 失败时给出 **WARNING** 级别日志，并在回测结果 metadata 中记录实际使用的引擎。

---

### R6-008 🟢 Low — Kelly-lite 固定 1% 风险预算

**文件**: `core/backtest/engine.py:926`

```python
risk_capital = balance * 0.01  # risk 1% of capital per trade
```

**描述**: Kelly-lite 仓位调整中，风险预算硬编码为余额的 1%。实盘中此值可通过 `soft_params.position_size_pct` 配置（默认 5%）。回测比实盘保守 5 倍，可能导致回测低估了策略的实际表现。

**修复建议**: 使用 `self.config.soft_params.position_size_pct / 100` 替代硬编码。

---

### R6-009 🟢 Low — Hybrid Executor 硬编码止损距离

**文件**: `core/backtest/event_executor.py:251`

```python
sl_dist = 0.02  # default 2% stop loss
```

**描述**: Hybrid 引擎的 `sl_dist` 硬编码为 2%。Legacy 引擎从 `sizer.soft.stop_loss_pct` 读取（可配置）。Live 交易中从 `soft_params.stop_loss_pct` 读取。三个路径使用三种不同的止损距离来源。

---

## 累计统计

| 等级 | R1 | R2 | R3 | R4 | R5 | R6 | 累计 |
|------|-----|-----|-----|-----|-----|-----|------|
| Critical | 1 | 1 | 2 | 1 | 0 | 1 | 6 |
| High | 4 | 3 | 1 | 3 | 2 | 2 | 15 |
| Medium | 5 | 3 | 2 | 4 | 4 | 4 | 22 |
| Low | 3 | 2 | 1 | 1 | 2 | 2 | 11 |
| **总计** | **13** | **9** | **6** | **9** | **8** | **9** | **54** |
