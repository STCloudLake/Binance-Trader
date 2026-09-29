# Binance Trader — 项目接手评估报告

**评估日期**: 2026-09-29
**评估范围**: 全部源码（90 个 Python 文件 / 20,091 行）、`docs/` 下 45 份文档（44 篇 md / 16,240 行 + 1 个 csv）、测试套件、Web 层、配置与数据目录
**方法**: 通读文档 + 通读关键代码 + 实际执行（pytest / compileall / Web TestClient / 定向探针）

> 标记约定：**【已验证】** = 我实际运行或读到代码确认；**【文档声明】** = 仅出现在文档中，未独立验证；**【存疑】** = 文档与代码冲突。

---

## 0. 一句话结论

这是一个**架构设计相当完整、文档量远超代码量、但当前版本实际上无法产生任何交易信号**的项目。

接手的第一优先级不是继续加功能，而是修两个 P0 缺陷：

1. **`core/strategy/engine.py:227` 引用了未定义变量 `ml_weight`**，导致实时策略评估 100% 抛 `NameError`，且被 `evaluate_all_now()` 的 `except Exception: pass` 静默吞掉 —— 信号缓存永远为空，`STRATEGY_SIGNAL` / `POSITION_EXIT` / `POSITION_REDUCE` 事件永不发布。**【已验证】**
2. **混合回测引擎（Hybrid）用 AND 逻辑，实时/传统引擎用 OR 逻辑 + 信号融合 + 0.5 阈值**，两者语义不同；而 GA 进化评估走的正是 Hybrid —— 即"用一套语义优化、用另一套语义实盘"。关键的等价性门禁测试当前**失败**（legacy 20 笔 vs hybrid 4 笔，0 笔匹配）。**【已验证】**

---

## 1. 项目概览

| 项目 | 内容 |
|------|------|
| 定位 | Python 自动化加密货币交易系统（Binance 现货/合约） |
| 运行方式 | `python -m app.main --mode {sim\|live\|backtest}`，默认 sim，Web UI `127.0.0.1:8899` |
| 技术栈 | asyncio 事件驱动 + FastAPI/uvicorn + aiosqlite + Jinja2/HTMX + TA-Lib + LightGBM/XGBoost/PyTorch + python-binance + DeepSeek(OpenAI SDK) |
| Python | 3.12.10（本机） |
| 规模 | 90 文件 / 20,091 行（core 12,065、web 3,636、tests 2,708、app 691、scripts 385、db 311、alerts 295） |
| 最大文件 | `web/server.py` 2,549 行、`core/backtest/engine.py` 1,311 行、`web/i18n.py` 1,087 行 |
| Git | 只有 2 个提交，`d6ba1f0`（根提交，一次性导入全部代码）+ `94a8b5d`（目录扁平化）。工作区干净；分支 `main` = `phase/3-4-architecture-ml-upgrade` |
| 数据 | `data/` 156 MB（parquet/模型/回测 JSON/562 个 GA YAML） |

> **注意**：项目没有可用历史。根提交一次性导入，因此无法用 `git log` 追溯任何设计决策；文档是唯一的历史来源。

---

## 2. 架构地图（已核对代码）

### 2.1 启动链路 — `app/main.py`（409 行，单文件 `async def main()`）

```
argparse(--mode, --port)
 → Config.load(mode)                     # app/config.py 单例，合并 config.yaml + risk_params.yaml + secrets.yaml
 → init_database(db_path)                # db/database.py，SQLite + 手写迁移
 → AuthManager + 无用户时自动创建 admin（随机密码，仅写 stderr）
 → EventBus.start()                      # app/event_bus.py，单 asyncio.Queue(maxsize=10000)
 → 实例化 9 个组件并手工 wire()          # market_data/strategy/ml/news/risk/executor/guard/ai/alerts
 → BacktestEngine + StrategyLifecycleManager
 → 订阅 POSITION_EXIT / POSITION_REDUCE / RISK_BREACH 处理器（含 4 种熔断动作）
 → asyncio.create_task(_circuit_breaker_reset_loop)   # L250，未保存句柄
 → 依次 start() 7 个组件 → 同步 sim 余额到风控
 → evaluate_all_now()                    # 首次评估
 → asyncio.create_task(_rest_polling_loop)            # L315，未保存句柄
 → 同步训练 5 个交易对的 ML 模型（阻塞！L318-327）
 → 用 REQUIRED_INDICATORS 特征预测并直接写 strategy_engine._ml_confidence
 → create_app() + 12 个 web_app.state.* 注入
 → uvicorn.Server(host=127.0.0.1, port=config.web_port).serve()
 finally: 逆序 stop() 9 个组件 + event_bus.shutdown()
```

**已识别问题**：两个 `create_task` 循环从未 cancel；ML 训练在 Web 绑定之前同步执行（启动数十秒到数分钟无 UI）；`main.py:345` 的裸 `except Exception: pass` 会静默吞掉所有 ML 预测失败。

### 2.2 信号→下单主链路（**当前在第一步之后即断裂**）

```
core/market_data/provider.py   Binance WS kline + REST 兜底 + OHLCV parquet 缓存
 → core/strategy/engine.py:_evaluate()                     ← 【断裂点】NameError: ml_weight
      ├ compute_all(df, strategy.indicators)                core/strategy/indicators.py（含 AST 白名单沙箱）
      ├ evaluate_entry_conditions()  ← OR 逻辑              core/strategy/evaluation_kernel.py:25
      ├ evaluate_exit_conditions()   ← OR 逻辑              evaluation_kernel.py:59
      ├ fuse_signals(indicator, ml, news, weights)          evaluation_kernel.py:82
      ├ check_higher_tf_trend()      ← EMA50 乘数          evaluation_kernel.py:152
      ├ 阈值 |score| >= 0.5
      └ publish STRATEGY_SIGNAL / POSITION_REDUCE / POSITION_EXIT
 → core/risk/manager.py:_on_signal() → check_signal()      # 7 步管线
 → publish ORDER_REQUEST
 → core/executor/executor.py:_on_order_request() → _execute_sim() / _execute_live()
 → core/risk/position_guard.py  移动止损 + 紧急止损 + 熔断回调
 → app/main.py 熔断动作处理（block_only / tighten_stops / close_all / close_worst）
```

### 2.3 模块清单（各包职责）

| 包 | 关键类/函数 | 职责 |
|----|------------|------|
| `core/market_data` | `MarketDataProvider`, `OHLVCache` | WS 多路复用、REST 兜底轮询、价格缓存(TTL 300s)、parquet 落盘 |
| `core/strategy` | `StrategyEngine`, `evaluation_kernel`, `indicators`, `StrategyLoader` | 策略评估、共享评估内核、指标计算、YAML 加载 |
| `core/ml` | `MLPredictor`, `MLTrainer`, `FeatureStore`, TFT/PatchTST | 方向/波动率双模型、40 维特征、在线增量、集成投票 |
| `core/backtest` | `BacktestEngine`(1,311行), `engine_hybrid`, `signal_matrix`, `event_executor`, `metrics`, `monte_carlo`, `data_feeder`, `report` | 传统逐 tick 引擎 + 两阶段混合引擎 + 风险指标 + MC |
| `core/ga` | `GAStrategyEvolver`, `genome`, `fitness`, `walkforward`, `fitness_calibrate` | 遗传进化、适应度、Walk-Forward、权重校准 |
| `core/risk` | `RiskManager`, `CircuitBreaker`, `PositionGuard`, `PositionSizer` | 三层风控（熔断 / 7 步管线 / 移动止损） |
| `core/executor` | `OrderExecutor` | 模拟/实盘下单、持仓恢复 |
| `core/ai` | `DeepSeekController`, `StrategyLifecycleManager`, `VibeTradingConnector`, `prompts` | AI 市场评估/选币/风控调整/熔断决策/策略生命周期 |
| `core/news` | `NewsAnalyzer`, `NewsFetcher`(SSRF 防护), `NewsSourceManager` | 新闻情绪 + 无源时的价格动量伪情绪 |
| `web` | `server.py`(2,549行，96 路由), `i18n.py`(1,087行), 10 个页面模板 + 15 个 partial | FastAPI + HTMX 管理面板，GA/WF/校准子进程管理 |
| `db` | `init_database`, `atomic_adjust_balance`, `SCHEMA` | 13 张表 + 9 索引，手写 schema_version 迁移 |
| `alerts` | `AlertManager`, `AlertRule` + 5 条默认规则 | 规则引擎 + WebSocket 推送 |

### 2.4 数据与持久化

- SQLite：`data/binance_trader.db`（路径硬编码于 `app/config.py:143`）
- 13 张表：`trades, orders, positions, alerts, ai_suggestions, risk_events, news_sources, news_articles, ml_models, system_config, users, backtest_records, strategy_lifecycle_events`
- 余额存在 `system_config('sim_balance')`，默认 10,000；`atomic_adjust_balance` 使用 `BEGIN IMMEDIATE` + 模块级 `asyncio.Lock`
- `data/models/*.pkl`（pickle）+ `*_tft.pt`/`*_patchtst.pt`（torch，`weights_only=False` → 反序列化即任意代码执行面）

---

## 3. 我实际验证过的事实（可直接引用）

| 验证项 | 命令/方式 | 结果 |
|--------|----------|------|
| 测试套件 | `python -m pytest tests/ -q` | **136 收集 / 135 通过 / 1 失败**（19.9s）。失败项：`tests/test_hybrid_equivalence.py::test_hybrid_matches_legacy_trade_for_trade` — `AssertionError: No common trades found between legacy (20) and hybrid (4) engines` |
| 语法完整性 | `python -m compileall app core web db alerts scripts` | 退出码 0，无错误 |
| 实时信号链路 | 构造假 MarketDataProvider 调用 `_evaluate()` | **`NameError: name 'ml_weight' is not defined`**；失败后 `_signal_cache` 为空；`evaluate_all_now(publish=True)` 同样静默失败，缓存仍为空 |
| 未定义名全量扫描 | 自写 symtable 扫描器（90 文件） | 仅 1 处真阳性：`core/strategy/engine.py:227 ml_weight`。另 3 处（`executor.py` 的 `SIDE_BUY/SIDE_SELL/ORDER_TYPE_MARKET`）是 `from binance.enums import *` 的误报，运行时正常 |
| Web 层可启动 | FastAPI TestClient + 临时 DB | 匿名 `/` → 302；登录 → 200；`/dashboard` 200（22.6 KB）；`/strategies` 200（27.8 KB） |
| 角色鉴权 | TestClient 以 viewer 身份探针 | `/api/trade`、`/api/backtest/run`、`/api/settings/{risk,binance,deepseek}`、`/api/strategy/*`、`/api/ga/evolve`、`/api/circuit-breaker/reset` 均 **403**（鉴权有效，与某份报告的说法相反） |
| 鉴权缺口 | 同上 | **确认 2 处**：`DELETE /api/backtest/{record_id}` 无角色校验（viewer 得 200 `{"ok":true}`）；`POST /api/alerts/{id}/ack` 无校验（影响很小） |
| 路由→鉴权映射 | 自写解析脚本 | 96 个路由；52 个处理器内无角色检查，其中 44 个是 GET 读接口 + login/logout/change-password，真正的写接口缺口只有上述 2 个 |
| 异步测试是否真跑 | 临时探针工程验证 pytest-asyncio 行为 | 项目 11 个 async 测试**均带 `@pytest.mark.asyncio`**，确实执行（`pytest-asyncio` 已安装，但 **未写入 requirements.txt**） |
| 文档声明 vs 实际 | `docs/audit`、README | README 称 136 tests → 实际 136（一致）；README 称"34 项关键修复"→ 见 §5 |

---

## 4. 文档体系与可信度评估

项目文档分四层，合计约 1.6 万行（44 篇 md），但**不能直接当作事实来源**。

### 4.1 `README.md` / `README_EN.md`（409 行）
描述架构成熟、公式详尽（GA 适应度、DSR、WF、混合引擎、成本模型、资源估算）。**已发现与代码不符之处**：

| README 声明 | 代码实际 |
|------------|---------|
| GA 群体默认 30、15 代 | `core/ga/evolver.py:34-35` = **80 群体 / 30 代** |
| Max Workers = 3 | `evolver.py:42` = **4** |
| 批量适应度 `win_rate×0.15 + PF×5 + ROC×50 − imbalance×10 − …` | `core/ga/fitness.py:63-84` = `max(sharpe,−5)×2 + win_rate×0.15 + PF×5 − max_dd×0.3`（**无 ROC、无 imbalance**；ROC/imbalance 公式只存在于 `fitness_calibrate.py:20` 的校准网格） |
| DSR 函数 `deflated_sharpe(...)`，返回数值 | 实为 `deflated_sharpe_ratio(...)`，**返回 dict** `{dsr, p_value, significant}`（`fitness.py:134,182`） |
| 13 种可进化指标 / 条件模板 | 与 `genome.py` 基本一致，此项可信 |
| "管理员密码启动时生成打印到终端" | 实际只写 stderr 且不写日志（更安全，文档滞后） |

### 4.2 `docs/audit/`（18 篇 md + 1 个 csv）
R1–R10 分模块审计 + R11 深度总结 + R12–R15（Phase 4a–4d 交付报告）+ `fix-tracker.md` + `severity-guidelines.md` + `cumulative-findings.csv`。

- 总量：**90 项发现**（Critical 10 / High 26 / Medium 35 / Low 19）
- 追踪结论：**34 项"已修复并验证"**、**56 项"accepted"（即搁置）**
- **关键可信度问题**：34 项修复全部是自我声明，`docs/audit` 中**没有任何测试产物或验证脚本**；56 项"accepted"里仍埋着 1 项 Critical（`R10-001` 信号链无补偿事务）、1 项未被追踪的 Critical（`R9-001`，与已修的 `R4-001` 同源）、7 项 High
- **tracker 已过期**：修复日期 2026-07-16，而 R12–R15 是 2026-07-20 的交付，其 16 项遗留问题全部未进 tracker
- **文档互相矛盾**：R11 把 8 项列为"待办 P2"，而 fix-tracker 把它们标为"✅ 已验证"；R12 自称"Phase 4a 完成"，同一文档 R12-003 又说 Web 模板未更新

### 4.3 `docs/core-algorithms/01–09`（9 篇）
算法说明文档，含公式、参数、默认值。**用于快速理解设计意图，但存在系统性漂移**（部分见 §4.1 对照表）。例如：
- `deflated_sharpe` 函数名错误（见上）
- 三重障碍标签：文档说 `timeout_label=2.0`，代码默认 `None`（NaN 行被过滤）；文档伪码用 `close`，代码用 `high/low` 且**同柱双触时乐观取上轨**
- 头寸下限：文档 `max(pct, 1.0)`，代码 `max(pct, 0.1)`（`position_sizer.py:36`）
- 熔断器：文档声称 NORMAL→TRIPPED→**RECOVERY** 三态状态机，代码只有 `is_tripped` 布尔 + `reset_trip/daily/weekly`，无 RECOVERY 状态
- 文档中所有 Sharpe/回撤/胜率基准表**无任何产物可复现**（仓库里没有对应脚本或数据）

### 4.4 `docs/superpowers/`（7 spec + 8 plan）
> ⚠️ **版本控制风险**：`.gitignore:71` 忽略了整个 `docs/superpowers/`，因此这 15 份设计规格与实施计划**只存在于本机工作区，未被 git 跟踪**（已跟踪的只有 `docs/audit/` 19 个文件、`docs/core-algorithms/` 9 个文件、`docs/development-roadmap.md`；全仓库共 156 个被跟踪文件）。任何新建 clone 都拿不到设计历史。建议尽快取消忽略或至少归档一份快照。

设计规格与实施计划，**315 个复选框，0 个被勾选**——复选框不承载任何完成信息，必须靠代码判断。基于代码核对结论：
- 2026-05-24 基础设计：**部分交付**（无 `web/routes/`、`web/ws/`、`core/market_data/ws_client.py`、`db/models.py`；计划本身在 Task 4.1 之后被截断）
- 2026-05-25 熔断自动响应：**已交付**
- 2026-05-26 告警中心重构：**已交付**
- 2026-05-26 认证 + DB 管理器：**已交付**（用户管理改成独立 `/users` 页，与计划不同）
- 2026-05-30 回测引擎 + AI 生命周期：**交付但缺 3 个接口 + 1 个测试文件**（`/api/backtest/compare`、`/api/strategy-lifecycle/retire|restore`、`tests/test_strategy_lifecycle.py`）
- 2026-06-03 混合引擎：**交付但缺 L3/L5 验证**（GA 排名一致性测试、e2e 对比 harness 未写）
- 2026-06-04 GA 质量改进：**交付但校准从未运行**（`data/ga_fitness_weights.json` 不存在 → 一直在用默认权重）
- 2026-07-16 审计：**已交付并超出计划**（R12–R15）

### 4.5 `docs/development-roadmap.md`
2026-07-16 撰写，提出 Phase 1–6 路线（P0/P1 修复 → 技术债 → Position Store/Shared Kernel → ML/回测 → 多交易所/分布式 → 策略市场/移动端）。
**重要**：其中的 Phase 1（周 1–2 资金安全修复，14 项）和 Phase 3（Position Store + Shared Kernel）**已被实现**（`evaluation_kernel.py` 存在、`R4-001` 事务已修），因此路线图的前两段已过期，接手时应从 Phase 4/5 续起。

---

## 5. 缺陷清单（按优先级，均已验证或标注来源）

### P0 — 阻断级（必须最先处理）

**P0-1 `ml_weight` 未定义 → 实时链路完全失效**
- 位置：`core/strategy/engine.py:227`（`_signal_cache[key]` 的 `"weights"` 字段）
- 原因：Shared Kernel 重构时把 `ml_weight` 改名为 `strategy_ml_weight`（L151），但漏改此处
- 影响：`_evaluate()` 每次执行都在写缓存时抛 `NameError`，位置在 publish 之前 → **永不发信号、永不发退出、永不减仓**。`evaluate_all_now()`（L375）与 `_on_kline` 链路的异常处理让它完全静默；Web UI 的策略监控因此永远是空的
- 为什么测试没抓到：`evaluate_sync()`（回测用，L331）不经过缓存字典；没有覆盖 `_evaluate` 的测试
- 建议修复（一行）：
  ```python
  "weights": {"indicator": w.indicator,
              "ml": strategy_ml_weight if ml_enabled else w.ml,
              "news": w.news},
  ```
- 顺带建议：给 `evaluate_all_now` 的 `except Exception: pass` 加 `logger.exception`，否则同类问题会再次隐形

**P0-2 混合引擎与实时/传统引擎信号语义不一致（AND vs OR）**
- 证据：`tests/test_hybrid_equivalence.py::test_hybrid_matches_legacy_trade_for_trade` **失败**（legacy 20 笔 / hybrid 4 笔 / 0 匹配）。该测试自述为"CRITICAL gate: if this test fails, the hybrid engine must not be used in production"
- 根因：
  - `core/backtest/signal_matrix.py:166-209` 对入场条件做 **AND**（"Long entry: AND all long entry conditions"），且**不做信号融合、不套 0.5 阈值、不套 HTF 乘数**
  - `core/strategy/evaluation_kernel.py:25` 与实时 `StrategyEngine`、传统 `BacktestEngine` 用 **OR** + `fuse_signals` + `|score|≥0.5` + HTF 乘数
- 波及面（**这是最需要警惕的一点**）：`core/ga/fitness.py:249-265` 的 GA 批量评估把每条染色体的 `ml_config.enabled` 强制置 False，再以 `per_strategy_isolation=True` 调 `run_with_exit_evaluation`；`_select_engine`（`engine.py:70-72`）在"策略数 ≥3 且 ML 关闭"时选 **Hybrid**。因此 **GA 进化适应度是在 AND 语义下算出来的，而实盘按 OR 语义下单** —— 冠军策略的排名与实盘表现没有保证的一致性
- 附带不一致：混合引擎的追踪止损用 `soft.stop_loss_pct`（2%）冒充追踪距离（`event_executor.py:267`），传统引擎硬编码 1.5%（`engine.py:904`），实时用 `trailing_stop_distance_pct`（2%）——**三套止损语义**（审计 R6-009 亦指出）
- 决策建议（三选一，需明确拍板）：
  1. 让 `SignalMatrixBuilder` 复用 Shared Kernel（OR + 融合 + 阈值）→ 保留"≥3 策略走混合"的加速，工作量中等，风险中
  2. 把 `backtest.engine_mode` 固定为 `legacy`（`config/config.yaml`）→ 立刻恢复语义一致，牺牲 GA 速度
  3. 明确接受 Hybrid 作为"独立的保守筛选器"并重写该门禁测试与文档 → 不推荐，会让 GA 结果不可信

### P1 — 高风险

**P1-1 实时凭据落盘且权限告警是空转**
`config/secrets.yaml` 含真实 Binance API key/secret、DeepSeek key、jwt_secret。已被 `.gitignore` 忽略（未被 git 跟踪，这点正确），但：`app/main.py:60-75` 启动会回写该文件；`app/config.py:152-164` 的权限检查在 Windows 上只会打印 `SECURITY: config/secrets.yaml is readable/writable by others!`（实测每次启动都报，且无实际效果）。**建议**：改用环境变量或 `dpapi`/凭据库，至少在文档中写明该告警在 Windows 上可忽略。

**P1-2 `max_weekly_drawdown_pct` 是死配置**
`core/risk/circuit_breaker.py` 中 `weekly_pnl` 与 `max_weekly_drawdown_pct`（配置值 10.0）从不参与 `check()` 判定；`app/main.py:246` 还会每天调用 `reset_weekly()`。配置显示"有周度保护"，实际没有。**要么实现，要么从配置中删除以免误导。**

**P1-3 鉴权缺口（2 处）+ 权限分级值得商榷**
- `DELETE /api/backtest/{record_id}`（`web/server.py:2034`）无角色校验 → viewer 可删除回测记录（实测 200）
- `POST /api/alerts/{id}/ack` 无校验（影响小）
- 权限分级：`/api/settings/binance`（写入交易所 API 密钥）、`/api/settings/risk`（改写风控阈值）只需 **trader** 角色而非 **admin**。这意味着任何 trader 可以替换交易所凭据或放宽风控。建议按最小权限原则改为 admin
- 另一个细节：路由的角色检查写在**处理器体内**，因此 FastAPI 请求体校验先于鉴权执行（实测空 body 会先返回 422 而非 403）。不构成漏洞，但会让未授权用户看到接口 schema

**P1-4 GA 适应度权重校准从未运行**
`data/ga_fitness_weights.json` 不存在，`data/ga_wf_state.json` 为空（`current_window: 0`）→ 一直在用 `DEFAULT_WEIGHTS`。README 与 spec 06-04 把"两阶段校准"当作已交付能力，实际未执行。

**P1-5 审计遗留 56 项，其中含未验证的 Critical**
`R10-001`（信号链无补偿事务，跨 main/strategy/risk/executor/db）仍开放；`R9-001` 与已修的 `R4-001` 同源但无追踪记录。此外 7 项 High 未修（含 `R1-003` 硬编码 `_TF_MIN` 与 2% 趋势阈值、`R3-003` 无最小名义额/步长校验、`R4-004` 模拟盘 PnL 不含手续费、`R7-003` AI 可改 signal_weights 无上限、`R7-004` 熔断超时在 semi_auto 下也会 close_all、`R8-003` 无会话吊销、`R10-002` ML 特征顺序不匹配）。
**注意**：`R7-003`/`R7-004` 直接关系到真金白银，接手后应优先复核。

**P1-6 风控配置取值偏激进，需与所有者确认**
`config/risk_params.yaml`（**YAML 覆盖代码默认值**，实际生效）：
- `max_position_size_pct: 50.0`（代码默认 10.0）→ 单仓上限达余额 50%
- `max_open_trades: 15`（默认 8）、`max_leverage: 4`（默认 3）
- 实践尺寸由 `soft_params.position_size_pct: 8.0` 决定，但硬上限 50% 给单笔失误留了很大空间
- `ai.mode: full_auto` + `config.yaml` 中的 `ai.model: deepseek-v4-pro`（代码默认 `deepseek-chat`）→ 启动即进入 AI 全自动；若该模型名无效，AI 调用会失败且部分被静默吞掉。**【存疑，需实测】**

### P2 — 质量与可维护性

| 项 | 证据 |
|----|------|
| Web 层是单文件巨石 | `web/server.py` 2,549 行、96 路由、全部闭包在 `create_app()` 内；`web/routes/`、`web/ws/` 是**空目录**（原计划的模块化从未落地） |
| 异常吞没 | 51 处 `except Exception:` 紧接 `pass`（实测统计，`web/server.py` 最重），无裸 `except:` |
| 未管理 asyncio 任务 | `main.py:250`（熔断日/周重置循环）与 `main.py:315`（REST 兜底轮询）的 `create_task` 无句柄、从不 cancel，依赖进程退出；`web/server.py` 内另有 `create_task(_monitor_ga())`。*（注：`scripts/ga_worker.py` 的进度上报守护线程经核实是正确停止的——`finally` 中置位标志并 `join(timeout=5)`，不存在泄漏。）* |
| 文件句柄泄漏 | `web/server.py:2251,2426` 把 `open(job_file+".log","w")` 直接传给 `Popen(stderr=...)`，从不关闭 |
| 反序列化风险 | `torch.load(..., weights_only=False)`（TFT/PatchTST）、`pickle.load`（GA checkpoint、回测直接加载 pkl） |
| 重复逻辑 | `EventDrivenExecutor._close_position` 是 `BacktestEngine._close_position` 的显式克隆（docstring 自认）；`main.py` 中余额调整块复制 5 次；入场/出场条件存在内核版与向量化版两套实现 |
| 测试基建缺失 | 无 `conftest.py`、无 `pytest.ini/pyproject.toml`；等价性门禁测试标了 `@pytest.mark.slow` 但**未注册该标记**（仅告警，不会跳过）；`pytest-asyncio` 未写入 `requirements.txt`（虽已安装） |
| 测试盲区 | 无 `tests/test_strategy_lifecycle.py`；**没有任何测试真正执行 `StrategyEngine._evaluate`**（P0-1 因此逃逸）；无 live 执行路径测试（`_execute_live` 完全未被覆盖） |
| 仓库卫生 | 根目录 `server.log` **295.8 MB**（已被 `.gitignore:58 *.log` 忽略，但仍在工作区占满磁盘）；孤儿目录 `binance_trader/data/`（25 个 parquet + 0 字节 db，约 90 MB，无任何代码引用，是扁平化提交 `94a8b5d` 的残留；已被 `**/market/`、`**/*.parquet`、`**/*.db` 规则忽略）；0 字节孤儿 `data/sim_trades.db`、`data/trading.db`；未被跟踪的备份 `data/binance_trader_backup_20260528_002727.db`(276 KB) |
| 缓存污染 | `tests/__pycache__` 中残留指向已删除路径 `binance_trader/tests/...` 的旧 `.pyc`，导致 pytest 报错栈行号指向不存在的文件（排查时会造成困惑） |
| 文档漂移 | 见 §4.1 / §4.3 的对照表；README 需要一次"以代码为准"的校订 |

---

## 6. 明确未完成的工作（来自计划与审计）

1. 计划 05-24：Task 4.1–6.2 从未被具体化（文档被截断）；`web/routes/`、`web/ws/manager.py`、`core/market_data/ws_client.py`、`alerts/notifications.py`、`db/models.py` 未实现；Telegram/Webhook 通知未实现
2. 计划 05-30：`POST /api/backtest/compare`、`POST /api/strategy-lifecycle/retire|restore`、`tests/test_strategy_lifecycle.py` 缺失
3. 计划 06-03：L3（跨引擎 GA 排名一致性）、L5（e2e 对比 harness）验证缺失
4. Spec 06-04：Walk-Forward UI/API 步骤自述"omitted for brevity"；`fitness_calibrate.py` 的 stage-2 为 stub；校准从未运行
5. 审计 R11 §6 架构建议全部未实现：Event Sourcing、单一 Position Store 写入者、**`/health` 端点**（路由表中确无）、配置版本化
6. 审计 56 项 accepted（2 Critical + 7 High + 30 Medium + 17 Low）+ R12–R15 的 16 项遗留（MC Sharpe 近似、基准函数未接入引擎、回测模板未更新新指标、Hurst O(n²)、FeatureStore 无 LRU、集成权重固定、在线学习灾难性遗忘风险、PatchTST 未进重训循环等）
7. 路线图 Phase 4（ML 管线完善）以后全部未启动：多交易所 Adapter、分布式、策略市场、移动端、合规审计

---

## 7. 建议的接手路线图

### 第 1 天（止血）
1. 修 P0-1（一行），并在 `evaluate_all_now` 的 except 中加日志
2. 补一个**真实的**回归测试：用假 MarketDataProvider 调 `_evaluate`，断言 `_signal_cache` 非空且 `STRATEGY_SIGNAL` 被发布（当前完全缺失，正是 bug 逃逸的原因）
3. 跑全量测试：修完 P0-1 后应只剩等价性门禁 1 项失败；P0-2 决策落地后目标 136/136 全绿
4. 就 P0-2 拍板：我建议**先选项 2**（`engine_mode: legacy`）立刻恢复语义一致，把"Hybrid 复用 Kernel"作为独立任务排期（选项 1），因为它涉及 GA 结果的可比性
5. 把 `docs/audit/fix-tracker.md` 的状态与代码对齐（或明确标注"自我声明、未经验证"），否则后续任何人都无法信任该文档

### 第 1 周
- 清理仓库：删 `binance_trader/` 孤儿目录、`server.log`(295 MB)、0 字节 db、失效 `__pycache__`；把 `pytest-asyncio` 补进 `requirements.txt`；加最小 `pytest.ini`（注册 `slow` 标记、固定 `asyncio_mode`）
- 安全：`secrets.yaml` 迁移到环境变量/凭据库；补 2 处鉴权缺口；把 `/api/settings/binance|risk` 提升为 admin-only
- 风控：实现或删除周度回撤；复核 `max_position_size_pct: 50`、`full_auto`、`deepseek-v4-pro` 三个配置是否是有意为之
- 复核 R7-003 / R7-004（AI 可无上限改权重、熔断超时强制平仓）

### 第 2–4 周
- Hybrid 与 Shared Kernel 统一（选项 1），补齐 L3/L5 验证；期间保持 GA 使用 legacy
- 拆 `web/server.py`：把 96 个路由按域迁到 `web/routes/`（目录已存在且为空，正好可用），用 app 级依赖做统一鉴权，消除"处理器内检查"的顺序问题
- 运行 GA 权重校准（Phase D），落地 `data/ga_fitness_weights.json`；跑一次完整 WF 验证闭环
- 补关键测试：`_evaluate` 实时链路、`_execute_live`（可用 mock client）、`tests/test_strategy_lifecycle.py`
- 消除重复：统一 `_close_position`、统一追踪止损语义、抽出 `main.py` 中重复的余额调整

### 第 2 个月起
按 `docs/development-roadmap.md` Phase 4 → 6 推进（ML 管线完善 → 多交易所/分布式 → 平台化），但**先重写路线图**，因为 Phase 1–3 已基本完成、文档已过期。

---

## 8. 运维与安全注意事项（接手即须知）

1. **不要提交 `config/secrets.yaml`**（已 gitignore，含真实密钥）；提交前 `git status` 确认
2. `binance.testnet: true` 但 `ai.mode: full_auto`：系统会以全自动 AI 决策运行，请先确认 testnet 边界
3. 启动会同步训练 5 个模型，Web UI 在此期间不可用；如需快速启动可先临时禁用该段
4. 无 `/health` 端点，无进程级健康检查；GA/WF/校准是子进程，日志在 `data/ga_jobs/*.log`
5. 回测结果 JSON 在 `data/backtest/`（约 60 MB），`data/ga_strategies/` 有 562 个 GA 策略 YAML，均被 gitignore
6. `strategies/` 下**只有 3 个 GA 冠军 YAML，没有人工编写的策略**；如需回测请先确认策略来源
7. 所有 `docs/audit` 的"已修复"结论均应视为**待复核**，不要据其判断风险已消除
8. **设计文档未入版本控制**：`docs/superpowers/` 被 `.gitignore` 忽略，clone 后设计历史全部丢失，接手第一步建议先归档

---

## 附：本次评估产出的可复用脚本（临时目录，未纳入仓库）

| 脚本 | 用途 |
|------|------|
| `bt_undefined_scan.py` | 基于 `symtable` 的全仓库未定义名扫描（可移植的 pyflakes 替代）→ 本次据此定位 P0-1 |
| `bt_route_authz.py` | 解析 `web/server.py`，输出"路由 → 是否有角色校验"映射 → 用于量化 P1-3 |
| `bt_probe_livepath.py` | 最小化 harness，复现 P0-1 并证明缓存为空 |
| `bt_web_smoke.py` / `bt_web_smoke2.py` / `bt_web_smoke3.py` | 用临时 DB 对 Web 层做端到端与鉴权探针 |

如需，我可以把前两个脚本整理进 `scripts/` 作为常驻的静态检查工具。
