# Binance Trader — 长盘修复与重构计划

**建立日期**: 2026-09-29
**依据**: [docs/HANDOVER.md](../HANDOVER.md) 的评估结论
**执行方式**: 分阶段推进，每阶段"改动 → 自测 → 独立审计"，审计发现回流为下一阶段任务

---

## 阶段划分与验收标准

| 阶段 | 内容 | 主要涉及文件 | 验收标准 |
|------|------|-------------|---------|
| **S0 安全网** | 建立基线、补测试基建、修 P0-1 | `core/strategy/engine.py`、`tests/`、`pytest.ini`、`requirements.txt` | 新增的实时链路测试由红转绿；`_signal_cache` 在真实 `_evaluate` 后可读 |
| **S1 语义统一** | 混合引擎与 Shared Kernel 对齐（P0-2） | `core/backtest/signal_matrix.py`、`event_executor.py`、`evaluation_kernel.py`、`engine.py` | `test_hybrid_equivalence` 全绿；GA 与实盘入场语义一致 |
| **S2 风控与安全** | 周度回撤落地、鉴权补齐、凭据治理、`/health` | `core/risk/circuit_breaker.py`、`web/server.py`、`app/main.py`、`app/config.py` | 新增安全测试全绿；viewer 无法做任何写操作；密钥不再落盘明文 |
| **S3 静默异常与重复逻辑** | 关键路径异常可见化、统一 `_close_position`/追踪止损 | `core/strategy/engine.py`、`core/backtest/*`、`app/main.py` | 静态扫描：关键路径无裸 `except: pass`；两引擎止损语义一致 |
| **S4 Web 层拆分** | `web/server.py`(2549行) → `web/routes/*` + `web/ws/*` | `web/` | 路由/模板行为不变（新旧路由与状态码逐一对比） |
| **S5 仓库卫生** | 孤儿目录/大日志/空 DB/失效 pycache、依赖补齐 | 仓库根、`.gitignore` | `git status` 干净；无 >100MB 未跟踪文件 |
| **S6 多轮审计** | 3 轮独立审计（正确性 / 安全 / 回归） | 全仓库 | 每轮发现均已修复或登记；最终 100% 测试通过 |
| **S7 文档交付** | README 校正、变更说明、运维手册 | `README.md`、`docs/` | 文档与代码一致（抽样核对公式/默认值/端点） |

---

## 不可动摇的约束

1. **每个阶段结束必须可运行**：全量测试不得比阶段开始时更差。
2. **禁止降低安全性**：任何"为了跑通"而放宽鉴权/风控/阈值的改动都不接受。
3. **写作用域分离**：同一文件同时只有一个写者；并行子代理不得写同一目录。
4. **审计独立**：审计子代理不得由本阶段的实现者兼任，且必须给出可复现证据（命令 + 输出）。
5. **不引入真实交易**：所有验证在 sim / 临时 DB / 假数据下进行，不连接真实下单接口。

---

## 审计轮次登记

| 轮次 | 范围 | 状态 | 主要发现 | 处置 |
|------|------|------|---------|------|
| A1 | S0–S1 正确性（实时链路 + 引擎等价） | 已完成 | 关键：同指标配置但不同时间框架的策略被**静默丢弃**；乐观：混合引擎套用制度阈值而 legacy 从不检测制度；`effective_entry_threshold` 对 bear 市场把顺势空单也按逆势阈值；混合引擎忽略 `risk_exit` 止损/追踪参数；`reduce_conditions` 混合引擎不支持；`per_strategy_isolation=True` 时 legacy 多数出场失效（GA 正走此路径）；swing_points 前瞻偏差；DataFeeder 空区间回退暴露预热数据 | **全部已修复**（见下）并新增 `tests/test_engine_parity_variants.py`（6 项变体门禁） |
| A2 | 安全（鉴权/注入/凭据/风控旁路） | 已完成 | 关键：`_validate_condition` 实为可绕过的正则黑名单，`pd.eval` 允许方法调用 → **任意文件写入**（已实测）；Jinja 未开启 autoescape → 存储型 XSS；4 条 trader 可达的风险弱化路径；未校验 `symbols` 进入 `pickle.load` 路径拼接；`config.py` 的 `logger` 局部导入导致启动崩溃 | **全部已修复**（见下）并新增 `tests/test_condition_security.py`（25 项）与 Web 鉴权回归 |
| A3 | 回归 + 资源 + 文档真伪 | 已完成 | **推翻**了"引擎已逐笔等价"的结论：多时间框架下 legacy 在**每个**时间框架评估出场、矩阵只建主周期行（单策略 178 vs 77、GA 形态 389 vs 207）；README 的 GA 适应度公式写错；门禁在无缓存数据时静默 skip；套件结果曾依赖可变的 `config/*.yaml`；77 处未使用导入、~25 个从未调用的函数；hybrid 比 legacy 快 ×4.1~×24（已证实） | 多时间框架出场已修并固化 3 项测试 + 新增合成行情门禁；README 公式已改正；`test_config.py` 改为校验"加载器忠实性"而非硬编码值；清理 12 处未用导入；其余登记为遗留（见 CHANGELOG §8） |

## 进度日志

- 2026-09-29：建立计划；确认基线 136 收集 / 135 通过 / 1 失败（`test_hybrid_equivalence`）。
- 2026-09-29 **S0 完成**：修 `ml_weight` NameError、修 reduce 计数器 key 不匹配、`evaluate_all_now` 异常改为 `logger.exception`、EventBus 记录订阅者异常；新增 `tests/test_strategy_live_path.py`（5 项实时链路回归测试）、`pytest.ini`、`requirements.txt` 补齐 `pytest-asyncio`。测试 141 通过。
- 2026-09-29 **S2 完成**：风控与安全加固。周度回撤真正生效（新增 `week_peak_equity`，日重置不再掩盖周内下滑，+4 项测试）；补齐 2 处鉴权缺口（`DELETE /api/backtest/{id}`、`POST /api/alerts/{id}/ack`）；交易所凭据（`/api/settings/binance`）、AI 凭据（`/api/settings/deepseek`）、风控阈值（`/api/settings/risk`）提升为 admin-only，并在 settings 模板中按角色隐藏/只读展示；新增匿名 `/health` 探针（含 DB 检查、熔断状态、持仓数，不泄露任何凭据）；修复 Windows 上 secrets 权限告警的永久误报；JWT secret 落盘时在 POSIX 上收紧为 0600 并提示生产环境改用环境变量。新增 `tests/test_web_security.py`（8 项）。测试 153 通过。
- 2026-09-29 **S1 完成**：混合引擎与 Shared Kernel 语义统一，达到**逐笔完全等价**（45/45 笔、数量/盈亏/退出价零差异、5 项头部指标完全一致）。修复内容：
  1. `signal_matrix` 入场逻辑由 AND 改为内核的 OR + 加权融合 + 制度感知阈值 + HTF 对齐（新增 `build_entry_signals` 等向量化内核函数）；
  2. 主时间框架改为"最短时间框架"（与 legacy 一致）；
  3. `DataFeeder` 增加 250 根预热 K 线（修复"回测首段指标为 NaN"的准确性问题）+ 交易时间线裁剪；
  4. legacy 入场循环 `break` → `continue`（原先会饿死后续交易对）；
  5. 退出价基准统一为持仓自身时间框架收盘价（原先 legacy 用 1m 起始价）；
  6. SL/TP 按触发价成交（原先 legacy 按收盘价成交，系统性偏差）；
  7. 追踪止损距离统一由 `PositionSizer.trailing_stop_distance_pct()` 提供（原先三套语义：1.5% / soft.stop_loss_pct / hard 配置）；
  8. legacy 增加期末强平（原先未平仓交易完全不计入 `trades`，胜率/PF/Sharpe 失真）；
  9. 混合引擎补齐 `max_hold_hours` / `use_indicator_exits` / 波动率缩放仓位 / 最大仓位上限 / Kelly-lite；
  10. 信号矩阵行序改为"策略优先、交易对次之"以匹配 legacy 的逐笔下单顺序。
  等价性测试加严为"逐笔 + 指标全等"，防止回归。
- 2026-09-29 **S3 完成**：重复逻辑与静默异常清理。
  - 新增 `core/backtest/trade_book.py:close_position`，两个引擎的 `_close_position` 由"逐字克隆"改为共享实现（解除再次漂移的风险）；等价性测试仍为 45/45 全等。
  - 关键路径静默异常改为可见：`evaluate_all_now`（logger.exception）、EventBus 订阅者异常、初始 ML 预测失败、告警规则求值失败、REST 轮询失败、AI 上下文各段落缺失（debug）。
  - `app/main.py` 两个长期后台循环由"fire-and-forget"改为持有句柄并在 shutdown 时 cancel + gather。
- 2026-09-29 **S4 完成（Web 层拆分）**：`web/server.py` 由 **2,588 行降至 104 行**，路由按域拆分为
  `web/routes/`（auth/pages/dashboard_partials/alerts/strategies/trading/settings/ai/backtest/lifecycle/ga/db_manager/users/health）
  与 `web/ws/alerts.py`，公共能力抽到 `web/context.py` / `web/deps.py` / `web/rendering.py`。
  验证：路由表与拆分前基线 `docs/overhaul/route-baseline.json` **完全一致（97/97，无增无减）**；
  关键页面渲染字节数不变（`/dashboard` 22,600 B、`/strategies` 27,767 B）；`/health` 返回 200；
  鉴权矩阵回归测试通过。
- 2026-09-29 **测试补强**：新增 `tests/test_strategy_live_path.py`(5)、`tests/test_web_security.py`(8)、
  `tests/test_trade_book.py`(7)、`tests/test_executor_live.py` 与 `tests/test_strategy_lifecycle.py`（实盘下单路径 + 策略生命周期，此前零覆盖）。
  测试总数 **194，全部通过**。
- 2026-09-29 **S7（进行中）**：README 校正（适应度公式、DSR、GA 默认值、引擎等价保证、预热、权限矩阵、
  测试命令）；新增 `docs/core-algorithms/00-ERRATA.md` 汇总 9 篇算法文档与代码的 20 处差异。
- 2026-09-29 **S5 完成（仓库卫生）**：删除孤儿目录 `binance_trader/`（92.4 MB）、`server.log`（295.8 MB）、`.pytest_cache`、全部 `__pycache__`、0 字节孤儿 DB（`data/sim_trades.db`、`data/trading.db`）、`.gitignore.bak`；工作区从 546 MB 降至 158 MB。`.gitignore` 不再忽略 `docs/superpowers/`（设计文档此前在每次 clone 时丢失）。补 `pytest.ini` 与 `pytest-asyncio` 依赖。

- 2026-09-29 **A3 发现已修复（多时间框架等价性）**：
  - `signal_matrix` 为策略的**每个时间框架**生成指标出场行，并用 `ffill` 对齐交易时间线
    （等价于 legacy 的 `df[df.index <= ts].iloc[-1]`）。
  - `event_executor` 按策略声明顺序逐时间框架检查出场，并以"≤ ts 的最后一根"取该周期收盘价成交
    （此前取持仓主周期收盘价，且对高时间框架用精确 `get_loc` 取价会失败）。
  - 新增 3 项多时间框架等价测试（1 策略 / 3 策略 GA 形态 / 3 策略 + 策略隔离，均为 15m+1h+4h），
    笔数、盈亏、Sharpe 全部一致（42/42、112/112、172/172）。
  - 新增**合成行情门禁**：用临时 parquet 合成 15m/1h/4h 数据，使等价性门禁在全新 clone
    （无 `data/market`）上也会真正执行，而不是静默 skip 后显示全绿。
  - README 的 GA 适应度公式说明改正为三套公式的真实归属（批量 / 单染色体 / 校准网格）。
  - `tests/test_config.py` 改为校验"加载器与 YAML 一致 + schema 默认值"，不再硬编码用户可改的风控值。
  - 清理本变更集涉及模块的 12 处未使用导入（web 层为主），`compileall`、路由 97/97 与全量测试均通过。
- 2026-09-29 **A1/A2 发现已全部修复**：
  - `signal_matrix`：指标按**分组内时间框架并集**计算（修复策略被静默丢弃）；行序与 legacy 一致。
  - 制度：legacy 在市场状态检测上**不再依赖 ML 分支**；`effective_entry_threshold` 修正为"基准 0.5 + 逆势侧 0.65"；与 hybrid 统一。
  - 混合引擎补齐 `risk_exit` 的 `stop_loss_pct` / `trailing_stop_pct`（Kelly-lite 分支因此恢复生效）。
  - `reduce_conditions` 存在时 `auto` 路由改用 legacy（不再静默少算交易）。
  - legacy 出场判定由 `sym not in positions` 改为 `pos_key not in positions`（修复 `per_strategy_isolation=True` 下 GA 路径出场失效）。
  - `swing_points` 检测结果前移 `lookback` 根（消除前瞻偏差）；`DataFeeder` 空区间不再回退暴露预热数据。
  - **条件表达式改为严格 AST 白名单求值器**（禁用属性访问/下标/lambda/推导式/字符串字面量/未白名单函数，列名仅从 DataFrame 解析）：实测封堵 `close.to_csv(...)` 任意文件写入。
  - Jinja 启用 autoescape；`/health` 未认证时只回状态；`/partials/user-list`、`/api/alerts/clear` 提升 admin；`/api/backtest/result/{id}` 需 trader；`per_page` 钳制 1..200。
  - 手动下单在 `risk_manager` 缺失时**失败关闭**（原先直接下单、完全绕过风控）；`position_guard` 追踪止损改用共享 `PositionSizer.trailing_stop_distance_pct()`。
  - `app/config.py` 顶部导入 logger（修复无效 `circuit_breaker_action` 导致启动 `UnboundLocalError`）。
  - `/api/settings/binance` 的 `testnet` 改为"未显式传参则保持不变"（此前任何空 POST 都会把系统切到实盘）。
  - `Config.config_dir` 可注入：设置持久化不再可能改写到仓库内的 YAML（新增守卫测试）。
---
