# Binance Trader

基于 Python 的自动化加密货币交易系统，集成多策略引擎、遗传算法优化、ML 预测、AI 决策、三层风控和实时 Web 管理面板。

## 架构

```
app/main.py          ← 入口：加载配置 → 启动 MarketData → StrategyEngine → ML → Risk → Executor → Web UI
core/
  market_data/       ← Binance WebSocket + REST，实时 K 线，OHLCV 缓存
  strategy/          ← 策略加载、指标计算 (TA-Lib)、条件评估
  ml/                ← LightGBM / XGBoost / TFT / PatchTST 多模型训练与预测
  backtest/          ← 混合引擎：SignalMatrix (向量化) + EventDrivenExecutor (逐 tick)
  ga/                ← 遗传算法进化器、适应度评估、Walk-Forward 验证、权重校准
  risk/              ← 仓位管理、止损止盈、熔断保护
  executor/          ← 模拟/实盘下单执行
  ai/                ← DeepSeek 控制器、策略生命周期管理
web/
  server.py          ← FastAPI 服务，GA/WF 子进程管理
  templates/         ← Jinja2 管理面板 UI
```

## 快速开始

```bash
pip install -r requirements.txt
python -m app.main --mode sim          # 模拟盘 (默认端口 8899)
python -m app.main --mode backtest     # 仅回测模式
python -m app.main --mode live         # 实盘 (需配置 API Key)
```

浏览器打开 `http://127.0.0.1:8899`，管理员账号 `admin`，密码启动时生成打印到终端。

> **注意**: TA-Lib 需要系统级安装。Windows 用户参见 [ta-lib-python](https://github.com/TA-Lib/ta-lib-python)。

---

## 遗传算法 (GA) 策略进化

### 运行流程

```
1. 初始化种群
   ├─ 种子策略编码 (strategy → chromosome)
   └─ 随机生成 (random_chromosome, BooleanGene + 10 种指标)

2. 逐代进化 (每代)
   ├─ 适应度评估 evaluate_population_batch()
   │   └─ 混合引擎批量回测 → per_matrix (每策略×交易对)
   ├─ 按适应度排序
   ├─ 精英保留 (Top elite_count)
   ├─ 锦标赛选择 → 交叉 (crossover_rate) → 变异 (mutation_rate)
   ├─ 移民注入 (随机新个体, immigrant_count)
   └─ 保存 Checkpoint (pkl, 可中断恢复)

3. 冠军验证
   ├─ chromosome_to_strategy() → 保存 YAML 到 strategies/
   ├─ 样本外验证 (out-of-sample)
   └─ Deflated Sharpe Ratio 统计显著性检验
```

### 染色体编码

每个策略编码为 4 种基因类型:

| 基因类型 | 变异方式 | 示例 |
|---------|---------|------|
| **ContinuousGene** | 高斯噪声 N(0, step×strength) | `rsi_period ∈ [5,28]` |
| **CategoricalGene** | 随机替换另一选项 | `mode ∈ {trend, range, scalp, momentum}` |
| **StructuralGene** | 50% 删除条件 / 50% 添加模板 | `["rsi < 30", "adx > 20"]` |
| **BooleanGene** | 15% 概率翻转 | 10 种指标各自启用/禁用 |

### 10 种可进化指标

| 指标 | 连续基因 | 范围 |
|------|---------|------|
| RSI | `rsi_period` | 5–28 |
| MACD | `macd_fast`, `macd_slow`, `macd_signal` | 6–20, 18–40, 5–15 |
| Bollinger | `bb_period`, `bb_stddev` | 10–40, 1.0–3.5 |
| ADX | `adx_period` | 7–28 |
| EMA | `ema_period` | 5–50 |
| ATR | `atr_period` | 7–28 |
| Stochastic | `stoch_k_period`, `stoch_d_period` | 5–21, 3–9 |
| CCI | `cci_period` | 7–28 |
| OBV | 无参数 (仅 BooleanGene) | — |
| SMA | `sma_period` | 10–100 |

条件模板池共 17 条/方向（入场） + 8 条/方向（出场），条件清洗自动移除引用未启用指标的模板。

### 遗传算子

| 算子 | 算法 | 概率/数量 |
|------|------|----------|
| 选择 | 锦标赛 (k=3, 选最优) | — |
| 交叉 | 每基因随机从父本A或B继承 | 70% |
| 变异-连续 | 高斯噪声 | 20%/基因 |
| 变异-分类 | 随机替换 | 10%/基因 |
| 变异-结构 | 增删条件 | 15%/基因 |
| 变异-布尔 | 翻转 on/off | 15%/基因 |
| 精英保留 | Top N 完整复制 | N = max(4, pop/10) |
| 移民注入 | 随机新个体 | N = max(4, pop/10) |
| 早停 | 无改善自动终止 | 10 代 |

### 适应度函数

适应度越高越好。批量评估时使用可配置权重（支持校准）:

**批量评估公式** (进化期间):

```
fitness = win_rate × w_wr + max(PF, 0.1) × w_pf + ROC × w_roc − imbalance × w_bal
        − trade_penalty − loss_penalty − complexity_penalty
```

| 符号 | 含义 | 默认权重 |
|------|------|---------|
| `win_rate` | 胜率 (%) = winning_trades / total_trades × 100 | 0.15 |
| `PF` | 盈亏比 = gross_win_pnl / gross_loss_pnl (封顶 100) | 5.0 |
| `ROC` | 收益率 = total_pnl / initial_balance | 50 |
| `imbalance` | 多空失衡度 = abs(long_pct − 0.5) × 2 | 10 |

**交易次数惩罚**: `<5 笔: −20` | `<15 笔: −5` | `>500 笔: −(trades−500)×0.02`

**复杂度惩罚**:
```
penalty = n_conditions × 0.8 + n_indicators × 1.2 + n_params × 0.3
```

**个体评估公式** (单独验证时，额外包含 Sharpe 和回撤):
```
fitness = max(sharpe, −5) × 2.0 + win_rate × 0.15 + max(PF, 0.1) × 5 − max_dd × 0.3
```

### Deflated Sharpe Ratio (DSR)

基于 Bailey & López de Prado (2014)，校正多重测试偏差:

```
E[max(SR)] = sqrt(1 / T) × sqrt(2 × log(N))
DSR = SR_observed − E[max(SR)]
significant ⟺ DSR > 0 ∧ p < 0.05
```

| 符号 | 含义 |
|------|------|
| T | 观察期数 (默认 365) |
| N | 尝试策略数 = population × generations |

---

## Walk-Forward 验证

滚动窗口验证策略在不同市场环境下的稳定性:

```
窗口 1: [M1 ~ M7]  训练 → [M8]   验证
窗口 2: [M2 ~ M8]  训练 → [M9]   验证
...
窗口 K: [MK ~ MK+6] 训练 → [MK+7] 验证
```

### 配置

| 参数 | 默认值 | 含义 |
|------|--------|------|
| train_months | 6 | 训练窗口长度 |
| val_months | 1 | 验证窗口长度 |
| step_months | 1 | 窗口滑动步长 |

有效窗口数: `N = floor((date_range_months − train_months) / step_months)`

### 汇总指标

| 指标 | 公式 | 含义 |
|------|------|------|
| WF Efficiency | mean / std(val_sharpe) | 越高越稳定 |
| Positive Window % | N_pos / N_total × 100 | 盈利窗口占比 |
| Train-Val Corr | Pearson r | 泛化能力 |
| Mean Val Sharpe | mean(val_sharpes) | 平均表现 |
| Min Val Sharpe | min(val_sharpes) | 最差情况 |

---

## 适应度权重校准

两阶段自动校准最优适应度权重:

### Stage 1 — Spearman 秩相关 (~5 分钟)

1. 生成 60 个随机策略 → 训练期+验证期各跑一次批量回测
2. 对 2,688 组权重组合，计算适应度排名与验证 PnL 排名的 Spearman ρ
3. 选出 ρ 最高的 Top 5 权重组合

### Stage 2 — Walk-Forward 验证 (~60 分钟)

Top 5 权重各跑 3 窗口简化 WF (50 种群 × 15 代)，WF Efficiency 最高者胜出

### 权重搜索空间

| 权重 | 范围 | 候选值 | 组合数 |
|------|------|--------|--------|
| w_wr | 0.05–0.30 | 6 | |
| w_pf | 1.0–15.0 | 8 | |
| w_roc | 10–100 | 8 | |
| w_bal | 2.0–20.0 | 7 | |
| **总计** | | | **2,688** |

结果持久化到 `data/ga_fitness_weights.json`，后续 GA 自动加载。

---

## 混合回测引擎

### 两阶段流水线

```
Phase 1 — SignalMatrixBuilder (向量化)
  load parquet → 指标聚类 → compute_all 批量计算 → evaluate_condition
  → 输出 N×N 信号矩阵 [+1 long / 0 neutral / −1 short]

Phase 2 — EventDrivenExecutor (逐 tick)
  for each timestamp:
    check exits (SL / TP / TrailingStop / IndicatorExit)
    check entries (signal + position_limit)
    position sizing (Kelly-lite)
    apply costs (fee + spread)
  → trades[], equity_curve[], per_matrix[]
```

### 引擎选择

| 条件 | 引擎 |
|------|------|
| >= 3 策略且 ML 关闭 | Hybrid (向量化) |
| < 3 策略或 ML 开启 | Legacy (逐策略) |
| 子进程 worker | Legacy (强制) |

### 性能指标

```
total_return_pct    = (final − initial) / initial × 100
annualized_return   = ((1 + total_return)^(365/days) − 1) × 100
max_drawdown_pct    = max(peak − equity) / peak × 100
win_rate_pct        = wins / trades × 100
profit_factor       = total_gains / total_losses  (cap 999)
sharpe_ratio        = mean(daily_returns) / std(daily_returns) × sqrt(365)
```

### 交易成本模型

```
round_trip_cost = entry_notional × fee + exit_notional × fee
                + entry_notional × (spread/2) + exit_notional × (spread/2)
```

| 资产 | Taker Fee | Spread |
|------|-----------|--------|
| BTC | 0.04% | 0.01% |
| ETH | 0.04% | 0.02% |
| BNB | 0.04% | 0.03% |
| SOL | 0.04% | 0.03% |
| XRP | 0.04% | 0.04% |

---

## 资源消耗

### 单次回测时间

```
T_backtest = T_signal + T_event

T_signal  ≈ 0.5s + 0.3s × unique_indicator_configs
T_event   ≈ n_ticks × 0.02ms × n_strategies / n_workers

典型值 (1 年 5-min 数据, 5 交易对, 5 时间帧):
  30 策略: ~2 min
  50 策略: ~5 min
  100 策略: ~12 min
```

### GA 总时间

```
T_ga = T_backtest × generations × threading_overhead

threading_overhead ≈ 0.4 (3 workers, GIL-limited)
典型值: 50 策略 × 15 代 ≈ 5 min × 15 × 0.5 ≈ 35–40 min
```

### Walk-Forward 总时间

```
T_wf = T_ga × n_windows

n_windows = (date_range_months − train_months) / step_months
典型值: 6 窗口 × 40 min ≈ 4 hours
```

### 内存消耗

```
M_per_worker = M_data + M_matrix + M_equity

M_data (parquet)    ≈ 50 MB (5 交易对 × 5 框架, 压缩)
M_matrix (signal)   ≈ n_ticks × n_strategies × 12 bytes ≈ 12 MB (100 策略)
M_equity_curve      ≈ n_ticks × 24 bytes ≈ 2.5 MB
M_total ≈ 65 MB per worker process
```

### CPU 与并行效率

- **子进程架构**: GA/WF 在独立子进程运行，主进程始终响应 HTTP
- **线程池**: 每 worker 内 3 线程评估策略组，GIL 限制实测加速 2–3×
- **信号矩阵聚类**: 相同指标配置的策略共享预计算结果，O(unique_configs) 而非 O(strategies)

---

## GA 参数参考

| 参数 | 范围 | 默认 | 说明 |
|------|------|------|------|
| Population | 10–120 | 30 | 种群大小 |
| Generations | 3–50 | 15 | 最大代数 |
| Elite Count | — | max(4, pop/10) | 精英保留数 |
| Immigrant Count | — | max(4, pop/10) | 随机注入数 |
| Tournament Size | — | 3 | 选择池大小 |
| Mutation Rate | — | 0.25 | 变异概率 |
| Crossover Rate | — | 0.70 | 交叉概率 |
| Max Workers | — | 3 | 并行线程数 |
| Early Stop | — | 10 gen | 无改善自动停 |

---

## 多用户认证

- **RBAC 3 角色**: admin (全部权限) / trader (交易+设置) / viewer (只读)
- **安全**: bcrypt 密码哈希 / JWT (HS256) / HttpOnly + SameSite Cookie

## 三层风控

| 层级 | 组件 | 功能 |
|------|------|------|
| L1 | CircuitBreaker | 回撤%/亏损$/连续亏损检测 |
| L2 | RiskManager | 7 步信号审核管线 |
| L3 | PositionGuard | 移动止损 + 紧急强制平仓 |

## 开发

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # 136 tests
python scripts/download_history.py  # 下载历史数据
python -m app.main --mode sim       # 启动模拟盘
```
