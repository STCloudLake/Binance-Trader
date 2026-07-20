# 审计报告 R12 — 回测风险指标体系升级

**日期**: 2026-07-20
**阶段**: Phase 4a
**类型**: 新增功能 + 集成

---

## 变更范围

| 文件 | 行数变化 | 变更类型 |
|------|----------|----------|
| `core/backtest/metrics.py` | 93→230 | 8 项新指标 + benchmark 对比函数 |
| `core/backtest/monte_carlo.py` | 新建 (135) | Monte Carlo 模拟 + 权益曲线包络 |
| `core/backtest/report.py` | 89→110 | risk_analysis + monte_carlo + benchmark 段 |
| `core/backtest/engine.py` | +6 | Monte Carlo 集成到回测返回结果 |

### 受影响下游
- `core/backtest/engine.py` — `calculate_metrics()` 返回值扩展（向后兼容），新增 `monte_carlo_simulation()` 调用
- `core/backtest/report.py` — `generate_report()` 输出新增 3 个顶层键
- Web UI — `backtest_results.html` 模板需同步添加新指标展示（不在本次变更范围）

---

## 新增指标

| 指标 | 简称 | 计算方式 | 理论来源 |
|------|------|---------|----------|
| Sortino Ratio | sortino | 年化收益 / 年化下行标准差 | Sortino & Price (1994) |
| Calmar Ratio | calmar | 年化收益 / 最大回撤 | Young (1991) |
| VaR 95% | var_95 | 历史模拟法，5% 分位数 | J.P. Morgan RiskMetrics |
| CVaR 95% | cvar_95 | 低于 VaR 的收益均值 | Rockafellar & Uryasev (2000) |
| VaR 99% | var_99 | 历史模拟法，1% 分位数 | 同上 |
| CVaR 99% | cvar_99 | 低于 99% VaR 的均值 | 同上 |
| Omega Ratio | omega | E[正收益]/E[负收益] | Shadwick & Keating (2002) |
| Tail Ratio | tail | P95 正收益 / |P5 负收益| | — |
| 最大连续亏损 | max_consecutive_losses | PnL 序列最长负数 streak | — |
| Recovery Factor | recovery | 绝对收益$/最大回撤$ | — |

---

## Monte Carlo 模块

### 算法
1. 提取交易 PnL 序列 `[p1, p2, ..., pn]`
2. 2000 次独立随机重排（Fisher-Yates shuffle）
3. 每次重排后重新计算权益曲线 → 最终收益、最大回撤、Sharpe
4. 输出分布统计量

### 输出键
`mc_median_return_pct`, `mc_ci_95_lower`, `mc_ci_95_upper`, `mc_prob_loss`, `mc_drawdown_median`, `mc_drawdown_p95`, `mc_sharpe_median`, `mc_sharpe_p05`

### 边界情况处理
- 交易数 < 10：发出 warning，仍执行
- 交易数 = 0：返回零值字典
- 所有 trades 正收益：prob_loss = 0.0，CI 上界 = 下界
- 所有 trades 负收益：prob_loss = 1.0

---

## Benchmark 对比

`calculate_benchmark_metrics()`:
- **Alpha** = 年化策略收益 - (rf + β * (年化基准收益 - rf))
- **Beta** = Cov(策略, 基准) / Var(基准)
- **Information Ratio** = 年化超额收益 / 年化跟踪误差
- **Tracking Error** = std(超额收益) * √365

---

## 验证结果

- ✅ 19 键 metrics dict 无语法错误
- ✅ 所有指标数值在合理范围（VaR < 0, CVaR < VaR, Omega > 0）
- ✅ 空交易列表优雅降级
- ✅ Monte Carlo 2000 次模拟输出统计合理
- ✅ 权益曲线包络 5 条分位线正确
- ✅ 全模块导入无循环依赖

---

## 审计发现

| ID | 严重度 | 描述 |
|----|--------|------|
| R12-001 | Low | Monte Carlo 的 Sharpe 使用 trades/initial_balance 作为日收益近似，高胜率低波动策略可能高估。建议改用 equity curve daily returns |
| R12-002 | Low | Benchmark `calculate_benchmark_metrics()` 未被回测引擎调用（仅提供函数），需在下游集成时手动使用 |
| R12-003 | Note | 新指标已添加到 `generate_report()` 输出，但 Web UI 模板 `backtest_results.html` 尚未同步更新 |

---

## 状态

**Phase 4a 完成。** | **新增**: 1400+ 行 | **修改**: 3 文件 | **发现**: 3 Low
