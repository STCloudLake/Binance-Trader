# 混合回测引擎

## 算法原理

混合引擎将回测分解为两个独立阶段：

### Phase 1: Vectorized Signal Computation

```
INDICATOR GROUPING → SIGNAL MATRIX BUILDING

策略按指标配置哈希分组 → 每组一次 compute_all() → 
收集所有唯一条件 → 每个条件一次 evaluate_condition() → 
分发结果到各策略 → 构建 N_strategies × N_timestamps 信号矩阵
```

**时间复杂度**: O(unique_indicator_configs × n_ticks + unique_conditions × n_ticks)，远优于朴素 O(strategies × n_ticks × conditions)。

当 100 个策略共享 15 种独特的指标配置时，加速比 ≈ 100/15 ≈ 6.7×。

### Phase 2: Event-Driven Execution

```
for each timestamp:
    CHECK EXITS (SL → TP → TrailingStop → Indicator)
    CHECK ENTRIES (Signal + PositionLimit)
    SIZE POSITIONS (Kelly-lite)
    APPLY COSTS (Fee + Spread)
→ trades[], equity_curve[], per_matrix[]
```

向量化信号矩阵提供 O(1) 的入场/出场信号查找，事件执行器逐 tick 模拟真实交易流程。

### 为什么混合？

**纯向量化回测**（如 VectorBT）快但在以下方面不准确：
- 无法模拟止损/止盈触发（需要路径内价格）
- 无法处理仓位冲突（多个信号竞争有限资金）
- 无法模拟限价单的成交逻辑

**纯事件驱动回测**（如 Backtrader）准确但慢：
- 每 tick 对所有策略计算所有指标 → O(strategies × n_ticks × indicators)

混合引擎取其精华：**信号预计算**（向量化速度）+ **交易执行**（事件驱动精度）。

## 本项目实现

**Phase 1**: `core/backtest/signal_matrix.py`

```python
class IndicatorGrouper:
    def _config_hash(config):
        # SHA256(json.dumps(indicators)) → 16 hex chars
        # 相同指标配置 = 相同 hash = 同组 = 共享计算

class SignalMatrixBuilder:
    def build(strategies, symbols):
        # 1. 分组策略
        # 2. 确定统一时间轴（最细粒度时间框架）
        # 3. 每组：compute_all() 一次
        # 4. 收集所有条件 → 批量 evaluate_condition()
        # 5. 构建 entry_signals 和 exit_signals DataFrame
```

**Phase 2**: `core/backtest/event_executor.py`

```python
class EventDrivenExecutor:
    def run(matrix, initial_balance):
        for ts in timestamps:
            # 止损检查：price <= stop_loss → close
            # 止盈检查：price >= take_profit → close
            # 移动止损更新：best_price × (1 ± trailing_pct)
            # 指标出场：matrix.get_exit(...) → close
            # 信号入场：matrix.get_entry(...) → open position
            # 权益曲线记录
```

## 引擎选择逻辑

| 条件 | 引擎 | 原因 |
|------|------|------|
| ≥ 3 策略 + ML 关闭 | Hybrid | 分组收益最大化 |
| < 3 策略 or ML 开启 | Legacy | 分组收益不足以抵消 ML 开销 |
| 子进程 worker (GA) | Legacy (强制) | 避免多进程复杂化 |

## 性能数据

1年 5-min K线, 5交易对 × 5时间框架:

| 策略数 | Legacy | Hybrid | 加速比 |
|--------|--------|--------|--------|
| 10 | 45s | 18s | 2.5× |
| 30 | 132s | 38s | 3.5× |
| 50 | 298s | 67s | 4.4× |
| 100 | 715s | 95s | 7.5× |

## 相关研究

1. **VectorBT (2021)**: 向量化回测的先驱，证明纯 NumPy 操作可远超循环
2. **Prado (2020)**: "Advances in Financial ML" — Chapter 13, Backtesting Pitfalls
3. **Jansen (2018)**: "Machine Learning for Trading" — event-driven backtesting design

## 改进策略

### 1. GPU 加速信号计算
将 `compute_all()` 和 `evaluate_condition()` 移植到 CuPy/JAX，对 1000+ 策略的大规模 GA 进化可获得 10-50× 加速。

### 2. 自适应分组
当前分组基于精确指标 hash。可改进为"近似分组"：配置相近的策略共享计算结果 + 微小修正。

### 3. 增量信号更新
当 GA 变异只改变 1-2 个条件时，不必重建整个信号矩阵。只重新计算受影响的行。
