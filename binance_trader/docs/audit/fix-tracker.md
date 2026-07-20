# 修复进度跟踪

**审计完成日期**: 2026-07-16
**修复完成日期**: 2026-07-16
**总漏洞数**: 90 | **已修复**: 34 | **已接受**: 56

## P0 — 紧急修复 (6/6 ✅)

| ID | 问题 | 修改文件 | 状态 |
|----|------|----------|------|
| R3-001 | 止损设置但从未enforce | position_guard.py | ✅ verified |
| R6-001 | 回测AND→OR逻辑统一 | engine.py (backtest) | ✅ verified |
| R4-001 | 余额操作事务化 | database.py | ✅ verified |
| R1-001 | pd.eval代码注入 | indicators.py | ✅ verified |
| R7-001 | AI参数上限 | deepseek_ctl.py | ✅ verified |
| R3-002 | AI floor覆盖 | position_sizer.py | ✅ verified |

## P1 — 高风险 (8/8 ✅)

| ID | 问题 | 修改文件 | 状态 |
|----|------|----------|------|
| R2-001 | 回撤跨日误触发 | circuit_breaker.py | ✅ verified |
| R2-003 | pending_signals超时 | risk/manager.py | ✅ verified |
| R4-003 | 仓位覆盖检测 | executor.py | ✅ verified |
| R8-001 | JWT secret持久化 | app/main.py | ✅ verified |
| R8-002 | WebSocket认证 | web/server.py | ✅ verified |
| R10-003 | SSRF防护 | news/fetcher.py | ✅ verified |
| R6-002 | ML前瞻偏差移除 | engine.py (backtest) | ✅ verified |
| R7-002 | API异常分类 | deepseek_ctl.py | ✅ verified |

## P2 — 稳定性 (9/9 ✅)

| ID | 问题 | 修改文件 | 状态 |
|----|------|----------|------|
| R2-002 | Exposure实时价格 | risk/manager.py | ✅ verified |
| R2-004 | 仓位数据统一 | risk/manager.py | ✅ verified |
| R4-002 | order_id UUID | executor.py | ✅ verified |
| R5-001 | 价格缓存TTL | market_data/provider.py | ✅ verified |
| R6-003 | Hybrid sizing统一 | event_executor.py | ✅ verified |
| R8-004 | 登录限速 | web/server.py | ✅ verified |
| R9-002 | Schema迁移版本化 | database.py | ✅ verified |
| R9-003 | Secrets权限检查 | app/config.py | ✅ verified |
| R10-004 | 跨模块原子性 | (P0-3/P1-2联动已覆盖) | ✅ verified |

## P3 — 代码质量 (11/67 ✅, 56 ⏭️ accepted)

| ID | 问题 | 状态 |
|----|------|------|
| R1-002 | 除零保护 | ✅ |
| R1-004 | Reduce counter加策略名 | ✅ |
| R1-005 | 私有属性→公共接口 | ✅ |
| R1-006 | 多空冲突WARNING | ✅ |
| R1-010 | load_all单独try/except | ✅ |
| R1-011 | STOCH参数从cfg读取 | ✅ |
| R1-012 | OBV SMA使用period | ✅ |
| R4-006 | 恢复仓位设置止损 | ✅ |
| R5-002 | WebSocket指数退避 | ✅ |
| R6-007 | Engine fallback日志 | ✅ |
| R7-005 | JSON提取用regex | ✅ |
| 其余56项 | Low/影响小/需大规模重构 | ⏭️ accepted |

## 修复文件汇总

| 文件 | 修改内容 |
|------|----------|
| `core/risk/position_guard.py` | +止损触发检查 +_stop_loss_close方法 |
| `core/backtest/engine.py` | AND→OR逻辑 + ML准确率后置 + 引擎fallback日志 |
| `db/database.py` | 余额事务化(BEGIN IMMEDIATE) + Schema版本化 |
| `core/strategy/indicators.py` | AST白名单验证 + 除零保护 + STOCH/OBV参数 |
| `core/ai/deepseek_ctl.py` | AI参数上限 + 异常分类 + JSON regex提取 |
| `core/risk/position_sizer.py` | floor 1%→0.1% |
| `core/risk/manager.py` | pending_signals超时 + Exposure实时价格 |
| `core/risk/circuit_breaker.py` | 日期自动重置 |
| `core/executor/executor.py` | order_id UUID + 仓位覆盖检测 + 恢复止损 |
| `core/market_data/provider.py` | 价格TTL + WS指数退避+认证检测 |
| `core/news/fetcher.py` | SSRF URL验证 |
| `core/strategy/loader.py` | load_all单文件try/except |
| `core/strategy/engine.py` | 多空冲突WARNING + reduce key+策略名 + 公共接口 |
| `core/backtest/event_executor.py` | 使用soft_params替代硬编码 |
| `app/main.py` | JWT secret持久化 + os/yaml导入 |
| `app/config.py` | secrets文件权限检查 |
| `web/server.py` | WS认证 + 登录限速 |
