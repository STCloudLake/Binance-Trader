# 核心算法文档 — 与代码差异勘误（ERRATA）

**日期**: 2026-09-29
**背景**: `01`–`09` 九篇算法文档撰写于 2026-07-16，之后代码经历 Shared Kernel 重构、Phase 4a–4d 升级
以及 2026-09 的修复重构，部分描述已与代码不一致。本文以**当前代码**为准，逐条列出差异。
文档用于理解设计意图，但**实现细节请以代码为准**。

| 文档 | 文档说法 | 代码现状（2026-09） | 证据 |
|------|---------|-------------------|------|
| 01 信号融合 | 融合公式位于 `core/strategy/engine.py:_evaluate()`；默认权重 0.5/0.3/0.2 | 公式实现在 `core/strategy/evaluation_kernel.py:fuse_signals()`；`engine.py` 仅调用。**函数签名默认值是 0.6/0.3/0.1**，实际生效的是 `config/config.yaml:signal_weights`（0.5/0.3/0.2） | `evaluation_kernel.py:82-147`、`engine.py:153`、`config/config.yaml` |
| 01 信号融合 | 未说明 | 两条重要不变量：(a) `ml_enabled=False` 时 ML 权重**仍留在除数**中（有意稀释纯指标信号）；(b) 多空条件同时满足 → `indicator_signal` 强制为 0（不交易） | `evaluation_kernel.py:119-129`、`engine.py:121-125` |
| 01 HTF 趋势 | 描述为"硬性阻止逆势入场" | 实为**置信度乘数**：EMA(50) 同侧 1.0 / 2% 内 0.6 / 更远 0.0，多个高时间框架取 `min`；且仅当 `indicator_signal != 0` 且策略含 >1 时间框架时生效 | `evaluation_kernel.py:152-193` |
| 02 风控管线 | 引用阈值 7.5% / 600 USDT / 15 笔 / 杠杆 3 | 管线顺序正确（7 步），但阈值来自 `config/risk_params.yaml`：日回撤 **7.5%**、日亏损 **600**、最大持仓 **15**、**杠杆 4**；`app/config.py` 的代码默认值（5.0/500/8/3）只在配置文件缺失时生效 | `core/risk/manager.py:98-183`、`config/risk_params.yaml` |
| 02 风控管线 | 未说明 | 敞口检查用 `>=`；挂单信号预留有 **60 秒**超时自动清理；拒绝告警有限流 | `manager.py:35,94,141,166` |
| 03 熔断器 | 三态状态机 NORMAL→TRIPPED→**RECOVERY** | 代码中**没有 RECOVERY 状态**，只有 `is_tripped` 布尔 + `reset_trip()/reset_daily()/reset_weekly()`；AI 恢复由 `deepseek_ctl._breaker_recovery_loop`（首次 120s，之后每 300s）驱动 | `core/risk/circuit_breaker.py`、`core/ai/deepseek_ctl.py:121` |
| 03 熔断器 | 周度限制 10% | **2026-09 修复**：此前 `max_weekly_drawdown_pct`/`weekly_pnl` 从不参与判定（死配置）。现新增 `week_peak_equity` 并真正判定，且日重置不再清空周峰值 | `circuit_breaker.py:check()`、`tests/test_risk_manager.py` |
| 03 熔断器 | 描述为"日回撤" | `daily_dd` 用 `peak_equity`（每日重置）对 `current_equity`，属"日内峰值回撤"而非自然日开盘回撤 | `circuit_breaker.py:61-65` |
| 04 头寸规模 | 标题为 Kelly；文档写下限 `max(pct, 1.0)` | 实为**固定分数**（非 Kelly）：`pool = balance × (0.7 core / 0.3 satellite)`，`risk = pool × pct%`，上限 `balance × max_position_size_pct%`；下限为 **`max(pct, 0.1)`** | `core/risk/position_sizer.py:36-46` |
| 04 头寸规模 | 未说明 | `volatility_expanding=True` 时仓位 ×0.7、止损距离 ×1.3（Phase 4b 引入） | `position_sizer.py:42-43,57-58` |
| 04 头寸规模 | 称 Kelly-lite 只有一处实现 | 存在两处（现已统一语义）：回测两引擎都用 `balance × 1% / 止损距离` 作为风险预算上限；混合引擎在 2026-09 前用的是 `position_size_pct`，已修正 | `core/backtest/engine.py`、`event_executor.py` |
| 05 混合引擎 | 执行顺序 SL→TP→Trailing→Indicator；SHA256 指标分组 | **与代码一致**（已核对）。另新增：入场语义与实时内核对齐、250 根 K 线预热、期末强平、行序按策略优先 | `signal_matrix.py`、`event_executor.py`、`engine_hybrid.py` |
| 05 混合引擎 | 引擎选择表 | 与 `_select_engine` 一致；`engine_mode=hybrid` 且启用 ML 会抛 `ValueError` | `core/backtest/engine.py:39-72` |
| 06 GA 进化 | 群体 30 / 15 代 / 3 workers | 实际默认 **80 群体 / 30 代 / 4 workers**（`fitness_calibrate` 另有 50×15） | `core/ga/evolver.py:34-43` |
| 06 适应度 | `win_rate×0.15 + PF×5 + ROC×50 − imbalance×10 …` | 实际为 `max(sharpe,−5)×2.0 + win_rate×0.15 + min(PF,100)×5 − max_dd×0.3`，另加 `total_return < −5` 惩罚与复杂度/笔数惩罚。**无 ROC、无 imbalance**（该公式只存在于校准网格） | `core/ga/fitness.py:63-99` |
| 07 DSR | 函数名 `deflated_sharpe(...)`，返回数值，用其比较 `if dsr <= 0` | 实际为 `deflated_sharpe_ratio(observed_sharpe, n_trials, observation_periods=365, variance_sharpe=1.0)`，**返回字典** `{dsr, p_value, significant}` | `fitness.py:134-183` |
| 07 DSR | 称完整 Bailey–López de Prado 实现 | 简化版：p 值用单侧正态检验，未纳入偏度/峰度；`SR≤0` 或 `N≤1` 直接返回中性结果 | `fitness.py:166-180` |
| 08 三重障碍 | `timeout_label=2.0` 默认 | 代码默认 **`None`**（超时行被过滤为 NaN）；`2.0` 只出现在文档示例调用中 | `core/ml/features.py:296-301` |
| 08 三重障碍 | 用 `close` 判断障碍、按索引先后处理 | 代码用 `high`/`low` 做盘中判断，且**先判上轨**，同柱双触时乐观取 label=1 | `features.py:333-354` |
| 09 追踪止损 | 实盘 `_update_trailing_stop`，距离 = `trailing_stop_distance_pct` | 实盘部分一致；**回测部分此前存在三套语义**（legacy 硬编码 1.5% / hybrid 用 `soft.stop_loss_pct` / 实盘用 hard 配置）。2026-09 已统一为 `PositionSizer.trailing_stop_distance_pct()`：策略 `risk_exit.trailing_stop_pct` 优先，否则用 `hard_limits.trailing_stop_distance_pct`（启用时 2.0%，未启用则关闭） | `position_sizer.py:trailing_stop_distance_pct`、`engine.py`、`event_executor.py` |
| 09 追踪止损 | 未说明 | 实盘有 ±1% 入场价上下限（floor/ceiling）与 0.01 棘轮保护；回测引擎无此保护（对齐点是"以最优价为基准"） | `core/risk/position_guard.py:141-178` |
| 全部文档 | 大量 Sharpe/回撤/胜率基准表与性能数字 | 仓库中**没有任何脚本或产物可复现**这些数字，应视为设计目标而非实测结果 | — |

## 维护约定

1. 修改算法实现时，同步更新对应文档；若来不及，请在本文件追加一行勘误。
2. 新增/调整配置项时，明确它是"代码默认值"还是"配置文件覆盖值"（本项目大量阈值由 `config/*.yaml` 覆盖）。
3. 文档中的公式必须能在代码中定位到唯一实现；若存在第二处实现，应视为重构对象（例如本文件记录的 Kelly-lite 与追踪止损）。
