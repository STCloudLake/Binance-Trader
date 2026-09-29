# 全项目审查：冗余功能 / 数据库冗余 / 非可替换化设计

**日期**: 2026-09-29
**方法**: 四路只读审计（耦合、冗余、数据库、账目）+ 独立复现。每条发现均带 `file:line` 与证据。

> ## 状态快照（as of this release）
>
> **本文是审计当时的快照，不是当前状态清单。** 下方条目已按下述状态标注：
>
> - **已完成**：§4.1 全部条目、§4.2 全部 6 项、§五 全部 2 项 —— 见各节内联的 `✅ 已完成` 标记与证据。
> - **本文不再维护**：新的清理项请以 `README.md` §11.3 与 `docs/overhaul/CHANGELOG.md` 为准。
>
> 本轮收敛的核验基线：测试 **606 passed / 0 failed**、`python -m compileall -q app core web db scripts` 退出码 0、
> 路由基线 **118 条**（拆分时 97 条）、账本恒等式在生产库上 `delta = 0.000000`（`python scripts/audit_db.py`）。

---

## 一、非可替换化设计（"焊接"清单）

规模：`BTCUSDT|ETHUSDT|BNBUSDT|SOLUSDT|XRPUSDT` 在 `*.py` 中命中 **346 处**；**17 份**重复的 5 币种列表（生产代码，不含 tests/data）。

### 1.1 高危

| 位置 | 焊死了什么 | 后果 | 最小解耦改法 |
|---|---|---|---|
| `core/executor/executor.py:300` | 实盘下单**恒发 `ORDER_TYPE_MARKET`**；调用方的 `order_type` 仅用于模拟盘（`:141`） | `/api/order` 接受 `limit`，**实盘静默按市价成交**；新增订单类型必须改执行器 | `ORDER_TYPES: dict[str, OrderSpec]` 注册表（DB CHECK 已允许 4 种） |
| `core/ai/deepseek_ctl.py:235-247` | `select_coins()` 结果只发建议，**从不写自选列表** | AI 选币是**死人功能**，改不动实际交易宇宙 | `validate_watchlist → save_watchlist` |
| `core/ai/strategy_lifecycle.py:278` | `all_symbols = [5 字面量]` 驱动 策略×币种 矩阵与 `:336` "删除亏损币种"写入 | 自动优化看不到 5 个之外的币种 | `await load_watchlist(db_path)`；另修 `:150/:221/:434` 的兜底字面量 |
| `web/routes/backtest.py:189-193,216` + `web/templates/partials/backtest_config.html:140-160` | 5 个具名表单字段 `spread_btc/_eth/_bnb/_sol/_xrp` | 第 6 个币种要改签名 + 10 处模板 | 单个 `spread_pct_json` 字段 |
| `core/backtest/signal_matrix.py:133` | `primary_sym = symbols[0]` 成为所有币种的时间轴 | 晚上市币种截断/损坏共享索引 | 取并集或显式 `timeline_symbol` |
| `core/backtest/engine.py:797` | `market_regime.get("BTCUSDT", "range")` | 不含 BTC 的回测**静默退化**为 range 制度 | 制度代理可配置，默认 `symbols[0]` |
| `core/backtest/cost_model.py:33` + `core/backtest/event_executor.py:73` | 两份 `spread_dict.get(sym, 0.03)`，默认 0.03 与模拟盘 `default: 0.02` 不一致 | **回测与模拟盘成本口径分歧** | 单一 `CostModel.from_settings()`，默认取模拟盘 default |

### 1.2 中危

- `core/executor/executor.py:286`：`round(qty, 5)` 写死；`SymbolInfo.step_size/tick_size/min_notional`（`core/market_data/universe.py:64-67,96-109`）**解析后只用于显示**，下单/风控 0 处消费 → 步长≠1e-5 的币种会被拒单。
- 周期面 5 处各自硬编码：`core/ml/predictor.py:97`（`not in ["1h","4h"]` 直接 return）、`core/ga/genome.py:159` 与 `:433`（`{"1m":1,...}[t]` 遇新周期 KeyError）、`core/market_data/provider.py:104-108`、`app/main.py:331,370`。
- `app/main.py:296`：新闻核心/卫星按**列表位置**切分（`[:3]` / `[3:]`）。
- `core/market_data/provider.py:4,160`、`core/executor/executor.py:5-6,297`：`binance` 直接 import 进 core，并用 monkeypatch 改 `bsm._get_stream_url` → 换交易所需重写两模块。
- `core/market_data/universe.py:244`、`web/routes/market.py:415,436,651`、`core/market_data/screener.py:757,896-897`：`quote="USDT"` 默认 + `symbol[:-len(USDT)]` 切片 → BTC/USDC 计价对不可见。
- `config/config.yaml:80-81` + `web/routes/settings.py:129,145`：`futures_enabled: true` **无任何合约代码**（配置撒谎）。

### 1.3 低危（但影响数据正确性）

- `trades.timeframe` 在 4 个写入点（`core/executor/executor.py:464,483`、`web/routes/market.py:887`、`web/routes/trading.py:65`）全部写死 `"1h"` → 非 1h 交易周期归属**永久错记**。
- 17 份重复币种列表（含 `web/routes/pages.py:12`、`app/main.py:287`、多个模板 `<option>`）。
- `core/executor/pending_orders.py:45` `TRACKED_SYMBOLS` 零引用（死常量）。
- `core/news/fetcher.py:70`：`BTCUSDT` 原样进新闻接口模板 → 按 "bitcoin" 检索的源永远匹配不到。

### 1.4 已确认解耦良好

`db/database.py` schema 与币种无关；`alerts/**`；`core/risk/**`（按 `position_type`+余额定价）；`core/strategy/engine.py:386` 与 `loader.py:33`；`core/ga/**` 四个入口本就接受 `symbols` 参数；`core/ml/**` 按币种命名模型文件（非枚举）；`core/market_data/data_client.py`（host 来自配置，覆盖 3716 对）；`web/i18n.py`。

> **结论**：`core/market_data/universe.py` 已有 `DEFAULT_WATCHLIST / load_watchlist / save_watchlist / validate_watchlist / SymbolInfo`，其余层都是"调用者忽略它"。可替换化主要是**删副本 + 单一默认解析器**，不是大重构。

---

## 二、数据库冗余与状态重复

真实库：1370 trades / 557 KB / `journal_mode=delete`；15 张表（13 在 SCHEMA + `pending_orders` + 孤儿 `test_tz`）。

### 2.1 死表（零写入零读取，或写入后无人读）

| 表 | 行数 | 证据 | 建议 |
|---|---|---|---|
| `orders` | 0 | 无 INSERT；仅 `settings.py:233` DELETE；`_orders` 是内存对象，`get_orders()` 只在测试用 | 删除 |
| `positions` | 0 | 无写入；12 列全空（`current_price`/`unrealized_pnl`/`take_profit_1..3`/`updated_at`） | 删除，或改造为真正的持仓表（见 2.3） |
| `risk_events` | 0 | 无写入无读取 | 删除 |
| `ml_models` | 0 | 无写入无读取（`engine.py` 里的 `ml_models` 是局部 dict；模型落盘） | 删除 |
| `news_articles` | 0 | 仅 `analyzer.py:199` 写入，**无任何读取**；`impact_level` 从未写 | 删除 |
| `test_tz` | 0 | 不在 SCHEMA，05-28 备份里就有，无代码引用 | 删除 |
| `news_sources` | 0 | 代码活着但表从未 seed；`api_key_encrypted` 从未读写 | seed 或删 |
| `ai_suggestions` | 182 | **写入侧已死**（无 INSERT；`deepseek_ctl.py:329` 只发事件，订阅者 `vibe_connector.py:33` 是研究用途）；全部 05-24/25 生成，`applied_at`/`rationale` 182/182 为 NULL | 重接写入或连同 3 个接口一起删 |

### 2.2 真正的 bug（不只是冗余）

1. **`/api/history/trades` 返回 685 条幽灵记录**：`close_position` 把 open 行也翻成 `status='closed'`（`executor.py:456/459`），而查询是 `WHERE status='closed'`（`market.py:1214`）→ 685/1370 行是 `exit_price NULL / pnl 0` 的幽灵行，**最近 50 条里有 24 条是幽灵**。
2. **每次重启都会重置止损**：`restore_positions:64` 读 `r.get("stop_loss")`，而 `trades` **没有该列** → 静默退回 2% 默认值；止盈从未持久化（`executor.py:173`）。
3. **重启后 PnL 基准错误**：`restore_positions:74` 把 `entry_price = fill_price`，而现金基准是 `qty×entry_price`（`executor.py:152-157`）→ 重启后平仓盈亏用错基准，直到一次减仓才被改写。
4. **`closed_at` 从未写入**（1370/1370 NULL）→ `/api/db/cleanup` 的 `trades` 清理条件 `closed_at < now-365d` **永远匹配 0 行**。
5. **余额存 3 份**：`system_config.sim_balance`（权威）、`app.state.balance`、`risk_manager._account_balance`；且 `save_sim_balance`（`database.py:331`）是**非原子读改写**，与 `atomic_adjust_balance` 竞争 → **实测偏差 19.878 USDT**（`10000+Σclose pnl = 9948.19` vs 存储 `9928.31`）。
6. **迁移与全新安装不一致**：v1 的 ALTER 给 `trader` 加列**没有 CHECK**（`database.py:40`），而全新 SCHEMA 有 `CHECK(trader IN ('manual','ai'))`（`:119`）；现网库里有 `trader='ft_9ef30e'`（真实用户名）→ **全新安装插入该值会 CHECK 失败**（已在 `:memory:` 复现）。`reduce_pct TEXT DEFAULT 0`（`:32`）造成数值列 TEXT 亲和性；`trades.action` 无 CHECK。
7. **`action='reduce'` 路径从未被走通且不自洽**：`executor.py:468-486` 改写数量但**不改 `entry_price`**，又从 `pos["fee"]` 里扣分摊买入费而 `entry_price` 仍含整笔买入成本。
8. **时区不一致**：DB 用 `CURRENT_TIMESTAMP`（UTC），`alerts/manager.py:81` 广播 localtime → **库与实时 UI 差 +8 小时**。
9. **热路径缺索引**：`trades(trade_group)`（`executor.py:456/475` EXPLAIN 显示 `SCAN trades`）、`trades(closed_at)`；`PRAGMA foreign_keys=0`（FK 声明形同虚设）。
10. **`journal_mode=delete` + 多短连接**（executor/matcher/alerts/web）→ 先坏的是锁竞争，然后是历史正确性。

---

## 三、冗余功能（"做了但没有实际作用"）

方法：`ast` 符号表（1401 个 def，0 解析失败）+ 全仓库名称出现扫描 + 路由↔模板匹配。

### 3.1 头条：**AI 面板整体是"结构性空转"**

`ai_suggestions` 表**从未被写入** —— 全仓库只有一个字符串引用 `db/database.py:167 CREATE TABLE`，**没有任何 `INSERT`**。生产者断了两次：

- `core/ai/deepseek_ctl.py:329 _publish_suggestion()` 只发 `EventType.AI_SUGGESTION` 事件，不写库；
- `AI_MARKET_STATE` **零订阅者**；`AI_SUGGESTION` 的唯一订阅者是 `core/ai/vibe_connector.py:43`，其函数体就是 `pass`，且该类**在生产代码中从未被实例化**（仅测试）。

于是以下内容**永远为空，不是偶发**：

| # | 严重度 | 块 | 证据 |
|---|---|---|---|
| 1 | 致命 | 建议卡片 `ai_panel.html:166-190` | `pages.py:135` 恒空 → 永远渲染"暂无待处理建议" |
| 2 | 致命 | 批准/拒绝按钮 `:178-181` | 行不存在，按钮仅在 `ai.py:97-101` 按行生成 → **永远不可能出现** |
| 3 | 致命 | `/api/market-state` 卡片 `:33` | `dashboard_partials.py:67` 读同一张空表 → 恒 `{"regime":"waiting"}`，**把服务端渲染的 `last_assessment` 覆盖掉**（`:35-39` 首次轮询后即死） |
| 4 | 高 | 心跳"今日建议 N 条" | `ai.py:168-172` 统计同一张空表 → 恒 0 |
| 5 | 高 | 策略引擎状态块 `:44-133` | 整块 JS 重实现，`trade.html` 已在轮询同一 `/api/strategy-monitor`（重复） |
| 6 | 中 | 初始渲染 vs 局部刷新 | `:169-185` 有置信度+200 字；`/partials/ai-suggestions`（`ai.py:93-107`）丢掉置信度且截断到 100 字 → 静默信息丢失 |
| 7 | 低 | `:146` | 同一 `<select>` 上有**两个 `class` 属性**（第二个被浏览器忽略） |
| 8 | 中 | `/api/news-sources` `ai.py:234` | 模板/JS 零引用 —— **它服务的 UI 块根本不存在** |

`/api/consult`（`ai.py:199` + `ai_panel.html:211`）是 AI 面板**唯一真正可用**的功能。

### 3.2 重复路由（运行时自删）

`GET /api/kline/{symbol}` 注册了**两次**（`dashboard_partials.py:133` 与 `market.py:746`），`market.py:784-787` 在运行时把它从 `app.router.routes` **删除** → `dashboard_partials.py:133-184` 与 `_market_data_provider():89` 全死。`_configured_client` 也在 `dashboard_partials.py:97` 与 `market.py:154` **逐字重复**。

> ✅ **已完成**：重复的 kline 路由与 `_market_data_provider` 已删除，`/api/kline/{symbol}` 只由 `web/routes/market.py:774` 提供。`_configured_client` 仍是两处（`market.py` 与 `dashboard_partials.py`）——两者都显式传 `testnet`，属**有意保留**的双客户端工厂，不是重复缺陷（README §4.2 已如实记录）。

### 3.3 死路由 25 条（非 `.claude` 零引用）

典型：`POST /api/ai-suggestions/{sid}/approve|reject`、`GET /api/news-sources`、`GET /api/alerts/filtered`、`POST /api/alerts/{id}/ack`（`alerts.acknowledged` 列存在但**没有任何按钮**）、`GET /api/trades`、`/api/backtest/running|history`、`/api/db/optimize|cleanup`、`DELETE /api/db/row`、`/api/ga/wf_stop|calibrate*`、`/api/strategy-lifecycle/*`。
另有一处"设计即坏"：`partials/alert_rules.html:11` 用 `hx-post` 切规则开关，但 `hx-target="#rules-list" hx-swap="outerHTML"` 会把整个规则列表替换成 `{"ok":true}` JSON。

> ✅ **已完成（除少数保留项）**：`GET /api/news-sources` 已删除；`/api/alerts/{id}/ack` 已补上按钮并接线（`alerts.py:27` + `partials/alert_list.html:19`）；`GET /api/trades`、`/api/backtest/running|history`、`/api/db/optimize|cleanup`、`DELETE /api/db/row`、`/api/strategy-lifecycle/*` 均已由页面/局部模板消费。`/api/ga/calibrate*` 随从未接线的校准搜索一并移除（仅保留权重加载）。

### 3.4 死配置（无人读取）

| 键 | 事实 |
|---|---|
| `config.yaml:40 mode: sim` | **从不被读取**！模式只来自 `--mode` CLI（`main.py:36,53`；`config.py:115-129` 把 mode 当参数）→ 改 YAML 无效 |
| `ai.consult_interval_minutes` | 仅 `pages.py:202` 重新显示表单；**没有任何调度**在用它 |
| `trading.spot_enabled` / `futures_enabled` | 只有 `pages.py:192-193`（显示）与 `settings.py:144-145,175-176`（回写）；**执行/风控无任何逻辑依赖** → UI 开关不起作用 |
| `sim.cost_model.fee_tier` / `use_bnb_discount` | 读作默认值但总被 `system_config` 覆盖（`:62-63` 注释说明是有意）→ 保留 |

> ✅ **已完成**：`mode:`、`ai.consult_interval_minutes`、`trading.spot_enabled` / `futures_enabled` 四个死键均已从 `config/config.yaml` 与 `app/config.py` 移除（模式由 `--mode` CLI 决定）。

### 3.5 死函数（36 个零调用者，全部为生产代码）

点名的候选**全部确认死亡**：`core/backtest/metrics.py:220 calculate_benchmark_metrics`、`monte_carlo.py:120 monte_carlo_equity_curve`、`report.py:118 report_to_json`、`deepseek_ctl.py:423 analyze_news`、`core/ai/vibe_connector.py` **全部方法与类本身**。
另有：`alerts/manager.py:142 acknowledge_alert`、`app/config.py:43 fee_tier_names`、`:335 sim_spread_for`、`auth.py:173 dispatch`、`core/backtest/engine.py:1315 _detect_market_regime`（活的是 `engine_hybrid.py:15 _detect_regimes`）、`screener.py:702 clear_cache`、`ttl_cache.py:45 peek`、`feature_store.py:245 has_data`、`features.py:261/363/426/459`、`predictor.py:284/292/336/392`、`trainer.py:83/240`、`news/analyzer.py:42 set_symbols`、`strategy/engine.py:333/343/346/372/469/473`、`indicators.py:343`。

> **已完成/已重分类**：上列死码绝大多数已删除或移入 `experimental/`。**注意两处本条曾误判**：
> `alerts/manager.py::acknowledge_alert` **不是死码**——它是 `POST /api/alerts/{id}/ack` 的单一写入点（`web/routes/alerts.py:35`）；`core/auth/auth.py::dispatch` 同样保留（`AuthMiddleware` 的框架入口）。另有本轮再次确认仍活着的：`create_regression_label`（`core/backtest/engine.py:1260`）、`build_features`（`core/ml/features.py:65`，测试用）、`_estimate_hurst`（`experimental/ml/features_extras.py:18` 导入并调用）。
> **`source_manager` 并非"仅测试"**：`NewsSourceManager.get_all_enabled()` 是**生产链路**（`core/news/analyzer.py:56` 与 `:152` 的周期抓取都调它）；只有写侧 CRUD（`add_source`/`update_source`/`delete_source`/`get_all`）目前仅由测试覆盖——**保留并记录**。

其中 **ML 预测子系统**（features/predictor/trainer 的那批）自成体系但**从未接线** → 需要产品决策，不应盲删。

### 3.6 重复实现与状态重复

- 币种/数量格式化：`toFixed(`/notional 在模板中 62×（trade.html）、19×（coin.html）、16×（ga_panel/audit.html）、14×（data.html）、9×（market.html）——**无共享 formatter**，易漂移。
- 心跳状态有两处来源：`system_config`（`ai.py:162-167`）与内存 `_last_run/_count`（`deepseek_ctl.py:202-205`）。
- 预警计数两种算法：`manager.py:131-140` 与 `/api/alerts/counts`。
- 有意保留（已注释说明）：标量/向量化双内核、`trade_book.close_position` 共享、两处 `_close_position`。

---

## 四、清理计划

### 4.1 可立即执行（低风险）—— ✅ 全部已完成

`core/ai/vibe_connector.py`（整文件）；6 个 backtest/report 死函数；`fee_tier_names`；`sim_spread_for`；`engine.py:_detect_market_regime`；`ttl_cache.peek`；`screener.clear_cache`；`feature_store.has_data`；`strategy/engine.py` 的 6 个访问器（先确认无插件调用）；`GET /api/news-sources`；`GET /api/trades`；`dashboard_partials.py:133-184`（重复 kline 路由）与其 `_market_data_provider`；`config.yaml` 的 `mode:` 键；死表 `orders`/`risk_events`/`ml_models`/`news_articles`/`test_tz`/`positions`（或改造）。
路由清理必须**同步删除或补上 UI 调用**，不能只删后端。

**已完成证据（逐项核验）**：

| 条目 | 当前状态 |
|---|---|
| `core/ai/vibe_connector.py` | 文件已删除（`core/ai/vibe_connector.py` 不存在） |
| `fee_tier_names`、`sim_spread_for` | 生产代码零引用（`app/config.py` 仅剩 `fee_tier` / `sim_cost_quote`） |
| `_detect_market_regime` | 已无该私有实现，制度检测统一走 `core/strategy/evaluation_kernel.py::detect_market_regime` |
| `ttl_cache.peek` | `TTLCache` 已无 `peek`（仅 `put` / `get`） |
| `screener.clear_cache` | 已删除 |
| `feature_store.has_data` | 整个 `core/ml/feature_store.py` 已移入 `experimental/ml/` |
| `GET /api/news-sources` | 路由已删除（`web/routes/*` 与模板中均无 `news-sources`） |
| 重复 kline 路由 | 只剩 `web/routes/market.py:774` 的 `/api/kline/{symbol}`；`dashboard_partials.py` 的副本与 `_market_data_provider` 已删 |
| `config.yaml` 的 `mode:` 键 | 已删除（`config/config.yaml` 无顶层 `mode:`） |
| 死表 `orders`/`risk_events`/`ml_models`/`news_articles`/`test_tz` | `db/database.py:231` 的 `DEAD_TABLES` 常量即这份清单，迁移时 DROP；`orders` 在代码/允许表清单/`reset-sim` 里已无残留引用。`positions` **保留**——它是活表（`OrderExecutor.restore_positions` 重启恢复 + `GET /partials/positions`） |
| `GET /api/trades` | 已接线并有 UI 消费（`web/routes/dashboard_partials.py` + `partials/trades`） |
| `strategy/engine.py` 访问器 | 仅保留在用的 `get_monitor_state` |

### 4.2 需产品决策 —— ✅ 6 项全部已决策并落地

1. `ai_suggestions` 生产者：**接线**（`_publish_suggestion` → INSERT）还是**删掉**建议面板/批准拒绝/心跳计数/3 个接口。
   → **已接线**：`core/ai/deepseek_ctl.py:469 _publish_suggestion` 写入 `INSERT INTO ai_suggestions`（`:524`），三个 AI 任务（选币/策略优化/风控调整）都会调用；`web/routes/ai.py:143-207` 的列表/批准/拒绝/局部模板均在。
2. ML 预测子系统（features/predictor/trainer 的 14 个未接线函数）：接入交易链路还是移入 `experimental/`。
   → **已移入 `experimental/ml/`**；留在 `core/ml/` 的是真正被调用的 `predict`/`train_model`/TFT/**PatchTST 训练器**与标签函数（`PatchTSTTrainer` 由 `core/backtest/engine.py:328` 调用，**是 LIVE 代码**）。
3. `spot_enabled`/`futures_enabled`/`ai.consult_interval`：接线还是从 UI 移除（避免"开关无用"）。
   → **已移除**：`app/config.py` 与 `config/config.yaml` 中均已无这三个键。
4. 预警确认（ack）功能：补按钮还是删接口与列。
   → **已补按钮并接线**：`web/routes/alerts.py:27 POST /api/alerts/{alert_id}/ack`（经 `AlertManager.acknowledge_alert` 单一写入点），`web/templates/partials/alert_list.html:19` 有 `hx-post` 按钮。
5. 策略生命周期面板（`/api/strategy-lifecycle/*`）：补 UI 还是删。
   → **已接线**：`web/routes/lifecycle.py` 提供 events/generate/optimize/partial，面板模板 `web/templates/partials/strategy_lifecycle.html` 消费。
6. `mode:` 配置键：删除并在文档中说明模式由 CLI/启动参数决定。
   → **已删除**（见 4.1），模式由 `--mode` CLI 决定。

---

## 五、待补章节 —— ✅ 两项均已补

- **解耦实施记录（GA 选币、动态 Spread、第一轮解耦 11 项）** —— **已落地**。GA 不再使用硬编码 5 币种：`web/routes/ga.py::resolve_ga_symbols` 校验用户选择的币种，缺省回落到持久化自选列表 / `core/market_data/universe.py::DEFAULT_WATCHLIST`；价差由 `core/backtest/cost_model.py` 统一解析（override → live → default），GA 与回测路由都不再自带默认价差表。回归门禁：`tests/test_ga_symbols.py`（21 项）、`tests/test_decoupling.py`（53 项）、`tests/test_cost_model.py`（26 项）。
- **账目 19.878 USDT 偏差根因与修复** —— **已修复并加固**。两个根因（买入侧成本被收两次、重复开仓孤立资金）写在 `tests/test_ledger_invariant.py` 文件头，并由该套件（9 项）逐条钉住；生产库的修复记录在 `ledger_reconciliation` 表（`2026-09-29T23:09:22`，`delta = 19.878128`），修复前备份为 `data/binance_trader_prerepair_20260929_230922.db`。当前 `python scripts/audit_db.py` 在生产库上输出 `delta = 0.000000`（恒等式成立）。

