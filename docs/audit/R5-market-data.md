# 审计报告 R5 — Market Data + Event Bus (深度)

**日期**: 2026-07-16
**审查文件**: `core/market_data/provider.py`, `core/market_data/ohlcv_cache.py`
**审查代码行数**: ~350 行
**审计方法**: WebSocket 生命周期分析 + 数据一致性验证 + 缓存策略审查

---

## 摘要

| 等级 | 数量 |
|------|------|
| Critical | 0 |
| High | 2 |
| Medium | 4 |
| Low | 2 |
| **总计** | **8** |

---

## 漏洞清单

| ID | 等级 | 文件:行 | 描述 |
|----|------|---------|------|
| R5-001 | High | provider.py:157-158 | 价格缓存在 isFinal 检查之前更新 — 使用未关闭 K 线的价格 |
| R5-002 | High | provider.py:143-145 | 固定 5 秒重连延迟 — 无指数退避，网络不稳定时频繁重连 |
| R5-003 | Medium | provider.py:278-279 | REST 降级使用倒数第二根 K 线 — 与已关闭的 WS K 线可能重复 |
| R5-004 | Medium | ohlcv_cache.py | 缓存无 LRU 淘汰 — 长时间运行内存无限增长 |
| R5-005 | Medium | provider.py:55-72 | `_prefetch_history` 的 MIN_CANDLES 跳过逻辑可能忽略增量更新 |
| R5-006 | Medium | provider.py:82-96 | `_prefetch_history` 多批次请求的预分配 `all_klines = []` 无容量限制 |
| R5-007 | Low | provider.py:204-208 | `_price_history` 每小时清理一次内存列表重建 — O(n) 每符号每秒 |
| R5-008 | Low | provider.py:242 | `get_current_price` 可能返回过时价格 — 无 TTL 检查 |

---

## 详细分析

### R5-001 🟠 High — 价格缓存使用未关闭 K 线价格

**文件**: `core/market_data/provider.py:157-158`

```python
# Always cache the latest price from every kline update
self._price_cache[symbol] = float(kline["c"])
if not kline["x"]:  # isFinal check comes AFTER price cache update
    return
```

**描述**: `_price_cache` 在所有 K 线更新时被更新（包括未关闭的 K 线），而 `MARKET_KLINE` 事件仅在 K 线关闭时才发布。这意味着：

1. 策略引擎的信号评估使用已关闭 K 线的 OHLCV 数据（正确）
2. **但 `get_current_price()` 返回的是未关闭 K 线的最新价**（近似正确，但语义不清晰）
3. 价格被缓存两次（line 157 和 line 169），line 157 总是更新，line 169 仅在 isFinal 时更新

实际影响：`get_current_price()` 用于计算入场价格。如果未关闭 K 线的中间价波动很大，入场信号可能基于一个快速变化的价格。这不是 bug，但需要明确文档。

**修复建议**: 区分 `_price_cache`（实时价格）和 `_closed_price_cache`（已关闭 K 线收盘价），在需要稳定的地方使用后者。

---

### R5-002 🟠 High — WebSocket 重连无指数退避

**文件**: `core/market_data/provider.py:143-145`

```python
except Exception as e:
    logger.warning(f"WebSocket connection error: {e}")
logger.info(f"WebSocket disconnected after {msg_count} msgs, reconnecting in 5s")
await asyncio.sleep(5)
```

**描述**: 每次 WebSocket 断开后固定等待 5 秒重连。在以下场景中存在问题：

1. **Binance 限频**: 如果因连接过多被限频，固定 5 秒重试会持续触发限频
2. **网络闪断**: 如果网络在 5 秒内未恢复，下一次连接也会失败
3. **无最大重试次数**: 如果凭据无效（API key 错误），会无限重连

没有 `max_reconnect_attempts` 和指数退避策略。

**复现路径**: API 凭据错误 → 每次连接失败 → 5秒后重试 → 永远循环 → 日志填满。

**修复建议**:
```python
backoff = 1
max_backoff = 60
while self._running:
    try:
        # ... connect ...
        backoff = 1  # reset on success
    except AuthenticationError:
        logger.error("Auth failed, not reconnecting")
        break
    except Exception:
        await asyncio.sleep(min(backoff, max_backoff))
        backoff *= 2
```

---

### R5-003 🟡 Medium — REST 轮询与 WS 的 K 线重复

**文件**: `core/market_data/provider.py:278-293`

```python
# Use second-to-last candle — guaranteed to be closed.
candle = {
    "close_time": int(df.index[-2].timestamp() * 1000),
    ...
}
await event_bus.publish(Event(EventType.MARKET_KLINE, {...}))
```

**描述**: REST 降级轮询发布倒数第二根 K 线。但如果在 WS 中断期间 WS 已经处理了该 K 线（在中断前），REST 轮询会重新发布同一根 K 线 → 策略引擎重复评估 → 可能重复产生信号。

虽然 RiskManager 的 `_pending_signals` 和同 symbol 检查会阻止重复下单，但这仍然浪费了计算资源并可能触发告警。

**修复建议**: 在发布前检查该 K 线的 `close_time` 是否已被 WS 处理（使用 `_last_kline_time` 中记录的时间戳）。

---

### R5-004 🟡 Medium — 缓存无限增长

**文件**: `core/market_data/ohlcv_cache.py`

**描述**: OHLCV 缓存是一个简单的 `dict[str, DataFrame]`，其中 key 为 `{symbol}_{interval}`。每根新 K 线通过 `append_candle()` 追加到 DataFrame。没有任何淘汰策略：

- 5 个 symbol × 5 个 interval = 25 个 DataFrame
- 每个 1m K 线 DataFrame 以每年 ~525,600 行增长
- 内存使用量随时间线性增长

虽然 Parquet 磁盘缓存每 5 分钟刷新，但内存中的 DataFrame 只会增大。

**复现路径**: 运行 1 个月 → 25 × 43,200 根 1m K 线 → 约 1M 行 → 数百 MB 内存。

**修复建议**: 实现滑动窗口，每个 DataFrame 仅保留最近 N 根 K 线（如 10,000 根 1m K 线 = 约 7 天）。

---

### R5-005 🟡 Medium — Prefetch 跳过已缓存数据忽略增量

**文件**: `core/market_data/provider.py:73-76`

```python
existing = self.cache.get(symbol, interval)
min_candles = MIN_CANDLES.get(interval, 200)
if existing is not None and len(existing) >= min_candles:
    continue  # already has enough data
```

**描述**: 如果磁盘上已有足够数据（通过 `download_history.py` 预下载），`_prefetch_history` 会跳过。但如果磁盘数据是 1 个月前下载的，本月的新数据不会被获取。

不过，WebSocket 启动后会自动补充新 K 线，所以影响有限。但对于不活跃的 interval（如 4h），可能需要等待数天才能积累足够数据。

---

### R5-006 🟡 Medium — prefetch 无数据量限制

**文件**: `core/market_data/provider.py:82-96`

```python
batches = BATCHES.get(interval, 1)
all_klines = []
for batch in range(batches):
    klines = await self.client.get_klines(**params)
    all_klines = klines + all_klines
```

**描述**: 对于 1m K 线，`BATCHES = 4`，每批 1000 根 = 最多 4000 根。但对于自定义的更长 interval，batches=1 意味着仅获取 1000 根。这在多数场景下是合理的，但没有对获取的总数据量进行上限检查。

---

### R5-007 🟢 Low — 价格历史内存效率低

**文件**: `core/market_data/provider.py:204-208`

```python
self._price_history[symbol].append((time.time(), price))
cutoff = time.time() - 3600
self._price_history[symbol] = [
    (t, p) for t, p in self._price_history[symbol] if t > cutoff
]
```

**描述**: 每秒对每个 symbol 执行一次列表重建（列表推导），复杂度 O(n)。对于 5 个 symbol，每秒创建 5 个新列表。虽然每小时仅保留 3600 个数据点，但每秒重建仍是不必要的开销。

**修复建议**: 使用 `collections.deque` 配合周期性清理，而非每秒重建列表。

---

### R5-008 🟢 Low — 价格无 TTL 检查

**文件**: `core/market_data/provider.py:242`

```python
def get_current_price(self, symbol: str) -> float | None:
    return self._price_cache.get(symbol)
```

**描述**: 如果 WebSocket 断开且 REST 轮询失败，`_price_cache` 可能包含数分钟甚至数小时前的过时价格。调用者（策略引擎、PositionGuard、Web UI）使用此价格进行交易决策，但没有时间戳检查。

**修复建议**: 返回 `(price, timestamp)` 元组，让调用者自行判断价格是否过时；或添加 TTL 参数。

---

## 累计统计

| 等级 | R1 | R2 | R3 | R4 | R5 | 累计 |
|------|-----|-----|-----|-----|-----|------|
| Critical | 1 | 1 | 2 | 1 | 0 | 5 |
| High | 4 | 3 | 1 | 3 | 2 | 13 |
| Medium | 5 | 3 | 2 | 4 | 4 | 18 |
| Low | 3 | 2 | 1 | 1 | 2 | 9 |
| **总计** | **13** | **9** | **6** | **9** | **8** | **45** |
