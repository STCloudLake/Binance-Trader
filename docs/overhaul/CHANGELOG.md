# 修复与重构变更说明（2026-09-29）

本文件记录本轮"接手整备"实际做了什么、验证到什么程度、还留下什么。
配套阅读：[接手评估报告](../HANDOVER.md)、[执行计划](PLAN.md)、[Web 层拆分说明](WEB_SPLIT.md)、
[算法文档勘误](../core-algorithms/00-ERRATA.md)。

---

## 0. 一句话总结

把"设计完整但实际无法交易、且回测与实盘语义不一致"的系统，修成**可安全运行、双引擎逐笔等价、
关键安全漏洞已封堵、测试从 136 增至 231 项全绿**的状态；过程中由 3 个独立审计代理交叉验证，
其发现（含 2 个 Critical、6 个 High）已全部修复并补上回归测试。

---

## 1. 最重要的三个修复

### 1.0.2 新增：全币种行情体系 + 币种信息/市场数据/代币检测页 + K 线修复（2026-09-29 深夜）

**决定性发现**：本机到 `api.binance.com` / `stream.binance.com` 不通，但币安官方**公开行情镜像**
`https://data-api.binance.vision`（REST）与 `wss://data-stream.binance.vision`（WS）**可达**：
3716 个交易对、**496 个 USDT 可交易对**、1h 历史回溯到 2017-08-17、K 线无缺口。
因此行情全部切到该镜像（真实主网数据），下单仍走 testnet；"看到所有币种 + 数据对齐"由此成立。

**K 线问题的真正根因（已用浏览器实测证明）**：ECharts 蜡烛图把 5 元组按
`[open, close, low, high, extra]` 解析，而旧代码传的是 `[time, open, close, low, high]` ——
**时间戳被当成开盘价**，y 轴范围被拉到 `[0, 1.8e12]`，所有蜡烛被压成一条平线。
最小复现：空白 ECharts 页面下 5 元组 → extent `[0,1.8e12]`，4 元组 → `[83400,84600]`。
修复为 `[open, close, low, high]` + x 轴统一 category；并补齐秒→毫秒统一转换、1d/1w 周期、
dataZoom（inside+slider）与跨子图联动、富提示框（开高低收/涨跌幅/量/MA-EMA-BOLL 值）、
MA/EMA/BOLL 与 VOL/RSI/MACD 开关、"最近 N 根"、失败时清图并显示原因；修复了 3 面板纵向预算
溢出导致 RSI 被挤出画布、以及切换周期时缩放窗口串档的问题。

**新增页面**（路由由 Lead 加在 `web/routes/pages.py`）：
`/market` 全币种行情（搜索/排序/分页、自选管理、本地数据标记、行点击进交易页）、
`/coin/{symbol}` 币种详情（交易规则、24h、盘口、1d/7d/30d/90d 表现、风险评分与 flags、BTC 相关性）、
`/data` 全市场数据（涨跌幅/成交额/最活跃/波动率/价差榜 + BTC/ETH 占比）、
`/audit` 代币检测（启发式风险筛查，页面显著标注"非链上合约审计"）。

**去重**：`/dashboard` 与 `/` 改为 302 → `/trade`（仪表盘职能被现货页完全覆盖），
导航只保留一个「现货交易」，登录后跳 `/trade`。

**数据底座**（`core/market_data/`）：新增 `data_client.py`（httpx，15s 超时，按事件循环重绑连接池）、
`ttl_cache.py`（ticker24h/symbols 60s、coin 30s、depth/trades 2s、exchangeInfo 6h、错误不缓存）、
`universe.py`（exchangeInfo + 本地缓存 + 搜索/分页 + 上市日期 + 自选持久化）、`metrics.py`（多周期表现/波动/回撤/相关性/评分）；
`provider.py` 改为从数据镜像取 K 线并使用 `.vision` WS 主机、按持久化自选列表订阅。

**新接口**：`/api/market/symbols`、`/api/market/ticker24h`（3716 币种，60s 缓存）、
`GET/POST /api/market/watchlist`（上限 30，重启生效）、`/api/coin/{symbol}`、`/api/data/overview`、
`/api/audit/screen`、`/api/audit/{symbol}`、`POST /api/backtest/fetch-data`（按需下载任意币种数据，带数量上限）。

**回测/策略**：回测与策略页改为**全币种可搜索多选**（标记哪些币种已有本地 parquet），
`scripts/download_history.py` 重写为 `--symbols/--intervals/--start/--end/--data-dir` 任意组合、批量分页下载。

**验收（Lead 独立执行，真实主网数据）**：**18/18 通过** ——
`symbols` 496 个、`ticker24h` 3716 个（二次调用 0.05s 命中缓存）、`coin/SOLUSDT`（评分 92、相关性 0.856）、
`data/overview`（496 币种、97.3 亿 USDT、涨 314/跌 174）、`audit/screen` 与 `audit/BTCUSDT`、
五个页面全部 200 且标记齐全、`/dashboard` → `/trade`、viewer 写操作 403、
trader 自选更新成功、`fetch-data` 成功下载 ADAUSDT 1h（768 行）且超限请求被 400 拒绝。
全量测试 **422 通过**；路由基线相应扩展。

### 1.0.1 新增：Binance 风格现货交易页 `/trade`（2026-09-29 晚）

原交易界面过于基础、信息量少。新增一个信息密度对齐 Binance 现货的页面，并补齐其所需的数据与
限价单能力（接口契约见 [TRADE_PAGE_API.md](TRADE_PAGE_API.md)）。

**页面**（`web/templates/trade.html`，979 行；路由 `GET /trade`；导航栏新增「现货交易」入口）：

- 顶部行情条：交易对选择、最新价（大字）、24h 涨跌幅（红绿）、24h 高/低、24h 量（基础+计价）、实时时钟
- 左栏：**订单簿**（卖盘上、中间价差行、买盘下、深度背景条、买卖总量）+ **最新成交**（价格/数量/时间，按方向着色）
- 中栏：ECharts K 线 + 1m/5m/15m/1h/4h 切换 + MA7/25/99 + 可开关的 VOL/RSI/MACD 副图（复用 dashboard 的指标算法，无新依赖）
- 右栏：买入/卖出 → 市价/限价、金额输入 + 25/50/75/100% 可用余额快捷键、数量预览、止损 %、下单结果内联提示；
  账户卡片（余额/可用/冻结/权益/持仓市值/未实现盈亏/委托数）
- 底部标签页：**当前委托**（可撤单）/ **持仓** / **成交历史** / **订单历史**，含加载/空/错误态
- 轮询：深度与成交 2s、账户与委托 3s、K 线 5s；页面隐藏自动暂停；同 URL 请求不重叠；任一接口失败在该面板显示原因

**后端**（`web/routes/market.py` 555 行 + `core/executor/pending_orders.py` 420 行）：

- 新增 9 个接口：`/api/market/{ticker,depth,trades,overview}`、`/api/account`、`/api/orders`、
  `POST /api/order`、`POST /api/orders/{id}/cancel`、`/api/history/trades`；
  读接口登录即可，写接口要求 trader；全部走**配置感知的 testnet 客户端**（不再无参数直连主网）
- **限价单**：新表 `pending_orders`（DB 迁移 v2，可重启恢复），下单只冻结资金不扣现金
  （`available = balance - frozen`）；5 秒撮合循环按价格穿越成交，成交时**复用同一风控管线**
  （`RiskManager.check_signal → ORDER_REQUEST`），风控拒绝则标记 cancelled 并写入原因；
  应用关闭时优雅停止
- 行情接口带 2 秒 TTL 缓存与并发去重，避免轮询导致连接爆炸

**验收（Lead 独立执行）**：13/13 通过 —— 行情四接口返回真实 testnet 数据（BTC 84,344、深度 20/20、
价差 0.01、成交 30 条、概览 5 个交易对）、账户算术自洽（balance − available − frozen = 0，5 个持仓）、
`GET /trade` 200/59.5KB 且 12 项结构标记全中、viewer 下单 403、trader 限价挂单→出现在当前委托→撤单全链路成功；
路由基线由 97 扩至 107（拆分时的 97 条全部保留）。

**顺带修复**：`tests/test_event_executor.py` 的价格序列改为固定种子（原先未播种，
入场到出场的价差偶尔四舍五入为 0.00 盈亏，导致约 1/7 概率的假失败）；契约补充时间单位说明
（ticker/trades 为秒，`/api/kline` 为毫秒）。

### 1.0 第三轮审计（A3）追加修复：多时间框架等价性

A3 审计用一个更刁钻的变体**推翻**了我此前"已完成逐笔等价"的结论 —— 门禁只覆盖 1h 单周期，
而**多时间框架**下两引擎实际不等价：

- 根因一：legacy 会遍历 `strategy.timeframes` 中的**每一个时间框架**评估指标出场，而信号矩阵只为
  **主（最短）时间框架**生成出场行 → 审计实测单策略 178 vs 77 笔、3 策略 GA 形态 389 vs 207 笔。
- 根因二：出场成交价应取**触发该出场的时间框架**的收盘价，混合引擎原先取持仓自身（主）周期收盘价。
- 根因三：高时间框架在非整点 tick 上不存在对应 K 线，原实现按精确时间戳取价会失败。

修复：矩阵为策略的**每个时间框架**生成出场行（`ffill` 对齐到交易时间线，等价于 legacy 的
`df[df.index <= ts].iloc[-1]`）；执行器按策略**声明顺序**逐时间框架检查，并以"≤ts 的最后一根"
取该周期收盘价作为出场价。

验证：审计的原始反例（1 策略 / 3 策略 GA 形态 / 3 策略 + 策略隔离，均为 15m+1h+4h）现在
**笔数与盈亏完全一致**（42/42、112/112、172/172，PnL 与 Sharpe 全等）；三个形态已固化为常驻测试
`tests/test_engine_parity_variants.py::test_parity_multi_timeframe_*`。

同时新增**不依赖缓存数据的合成行情门禁**
（`test_parity_uses_synthetic_market_without_cached_data`）：`data/market/**` 被 gitignore，
全新 clone 上原来的门禁会**静默 skip 却显示全绿**；现在用临时 parquet 合成 15m/1h/4h 行情，
门禁在任何环境下都会真正执行。

A3 还指出 README 关于 GA 适应度公式的说明**有误**（已修正）与若干工程质量问题（处置见 §5）。

### 1.1 实时信号链路曾 100% 失效（P0，已修）

`core/strategy/engine.py:227` 引用了未定义变量 `ml_weight`（Shared Kernel 重构遗留），
每次策略评估都在写入信号缓存前抛 `NameError`，并被 `evaluate_all_now()` 的
`except Exception: pass` 静默吞掉 —— 缓存恒空、`STRATEGY_SIGNAL`/`POSITION_EXIT`/`POSITION_REDUCE`
永不发布，**系统不会开仓、不会平仓、不会减仓**。

- 证据：假 MarketDataProvider 直调 `_evaluate()` → `NameError: name 'ml_weight' is not defined`，`_signal_cache` 为空。
- 修复：改为记录"实际生效的 ML 权重"；`evaluate_all_now` 的异常改为 `logger.exception`；
  EventBus 的 `gather(return_exceptions=True)` 现在会把订阅者异常打出来（此前完全静默）。
- 新增 `tests/test_strategy_live_path.py`（5 项）断言 `_evaluate` 的可观测副作用（缓存 + 事件），
  这类缺陷以后无法再逃逸。

### 1.2 混合回测引擎与实盘/传统引擎语义不一致（P0，已修到逐笔等价）

门禁测试 `test_hybrid_matches_legacy_trade_for_trade` 原本**失败**（legacy 20 笔 vs hybrid 4 笔，0 笔匹配），
而 GA 进化恰好走 hybrid（≥3 策略且关闭 ML），等于"用一套语义优化、用另一套语义实盘"。

修复后达到**完全等价**：45 笔 vs 45 笔、逐笔数量/成交价/盈亏零差异、5 项头部指标完全一致。
为此修掉的不一致包括：

| # | 问题 | 处置 |
|---|------|------|
| 1 | hybrid 入场用 AND，内核/实盘用 OR | 新增向量化内核函数 `build_entry_signals`（OR + 加权融合 + 制度阈值 + HTF 对齐），hybrid 改用它 |
| 2 | 主时间框架取 `timeframes[0]` 而非"最短" | 两者统一取最短时间框架 |
| 3 | 回测数据从 `date_start` 截断 → 首段指标为 NaN | `DataFeeder` 预加载 250 根预热 K 线，交易时间线仍从 `date_start` 开始 |
| 4 | legacy 入场循环 `break` → 饿死后续交易对 | 改为 `continue`（BTCUSDT 不再永远压过 ETHUSDT） |
| 5 | legacy 出场价用 1m 起始价，hybrid 用主周期收盘价 | 统一为持仓自身时间框架收盘价 |
| 6 | 止损/止盈成交价：legacy 按收盘价、hybrid 按触发价 | 统一按触发价成交（滑点交给成本模型） |
| 7 | 追踪止损三套语义（1.5% / `soft.stop_loss_pct` / hard 配置） | 统一由 `PositionSizer.trailing_stop_distance_pct()` 提供 |
| 8 | legacy 期末不强平 → 未平仓交易完全不计入 `trades`（胜率/PF/Sharpe 失真） | 两引擎都期末强平 |
| 9 | hybrid 缺少波动率缩放仓位/最大仓位上限/Kelly-lite/`max_hold`/`use_indicator_exits` | 全部补齐 |
| 10 | 行序不同导致同一 tick 内建仓顺序不同 → 仓位规模系统性差异 | 矩阵行序改为"策略优先、交易对次之" |

### 1.3 策略条件表达式可被用于任意文件写入（Critical，已修）

`core/strategy/indicators.py` 所谓"AST 白名单"其实是**可绕过的正则黑名单**，且用
`pd.eval(condition, engine="python")` 求值 —— 允许任意属性/方法调用。审计代理实测：
条件 `close.to_csv(r'<任意路径>', header=['im'+'port o'+'s;o'+'s.syste'+'m(chr(99))'], ...) is not None`
通过了校验并真的写出了文件（首行是可被 `.pth` 利用的 import 语句）。任何 **trader** 账号都能通过
`POST /api/strategy` 写入这种条件，AI 策略生成同样可以。

- 修复：改为**严格 AST 白名单求值器**：只允许 名称/常量/比较/布尔/算术/一元运算/白名单函数调用；
  **完全禁止**属性访问、下标、lambda、推导式、f-string、字符串字面量、dunder 名称与未白名单函数；
  列名只从当前 DataFrame 解析，未知列返回全 False（不抛异常）。
- 新增 `tests/test_condition_security.py`（25 项），内含该真实利用载荷的回归测试。

---

## 2. 安全加固（全部来自独立审计，已修并加测试）

| 级别 | 问题 | 处置 |
|------|------|------|
| 关键 | 条件表达式任意方法调用 → 任意文件写 | 真实 AST 白名单（见 §1.3） |
| 高 | Jinja 未开 autoescape → 新闻/AI/告警文本可存储型 XSS | 启用 `select_autoescape`；已实测 `<img onerror>` 被转义 |
| 高 | trader 可 `/api/ai-mode` 切 `full_auto`、开/关熔断、改信号权重、AI 风险建议的 `stop_loss_pct` 无上限 | 三者提升 admin；AI 建议审批路径补齐与 `deepseek_ctl` 相同的钳制 |
| 高 | 未校验的 `symbols` 参与 `pickle.load` 路径拼接 | 退化为只接受合法交易对符号；模型路径做包含性校验 |
| 高 | `app/config.py` 局部导入 logger → 非法 `circuit_breaker_action` 会让**启动崩溃** | 顶部统一导入（审计已复现该崩溃） |
| 中 | `/partials/user-list` 无角色校验 | 提升 admin |
| 中 | `/api/backtest/result/{id}` 是无鉴权的**写**接口 | 提升 trader |
| 中 | `LIMIT {per_page}` 未钳制（负数 → 全表导出） | 钳制 1..200 |
| 中 | 匿名 `/health` 泄露持仓/熔断/策略数 | 匿名只返回 status/database/uptime，细节需登录 |
| 中 | 手动下单在 `risk_manager` 缺失时**绕过全部风控** | 改为失败关闭（拒绝下单） |
| 低 | 清空告警日志（审计轨迹）只需 trader | 提升 admin |
| 低 | 设置页把凭据塞给 trader 模板上下文 | 仅 admin 注入 |
| 低 | 追踪止损在 `position_guard` 里另有一套数学、忽略 `risk_exit` | 改用共享 helper |
| — | 设置端点可被改写仓库内 `config/*.yaml` | 新增可注入 `Config.config_dir`，并加"仓库配置永不被测试改写"守卫测试 |
| — | `/api/settings/binance` 空 POST 会把 `testnet` 静默翻成 false（= 实盘） | 未显式传参则保持原值 |

---

## 3. 结构重构

- **Web 层拆分**：`web/server.py` **2,588 → 90 行**；96 个路由按域拆到 `web/routes/`（14 个模块），
  WebSocket 到 `web/ws/alerts.py`，公共设施抽到 `web/context.py` / `web/deps.py` / `web/rendering.py`。
  验证：路由表与拆分前基线**完全一致（97/97）**，页面渲染字节数不变，388 次四角色黑盒探测通过。
- **共享交易记账**：新增 `core/backtest/trade_book.py:close_position`，两个引擎各自的
  `_close_position` 克隆（文档自认"Identical logic"）合并为一处；审计用 300 次随机差分验证算术完全一致。
- **共享评估内核**：入场/出场/融合/HTF/制度阈值集中在 `core/strategy/evaluation_kernel.py`，
  并新增向量化版本供 hybrid 使用（`build_entry_signals` 等），彻底消除"两套语义"。
- **纪律**：关键路径的静默异常改为可见（EventBus、策略评估、初始 ML 预测、告警规则、REST 轮询）；
  `app/main.py` 的 2 个长期后台循环改为持有句柄并在 shutdown 时取消。

---

## 4. 独立审计发现的额外问题（已修）

这些是**审计推翻了原计划中的乐观假设**后补修的：

1. **策略被静默丢弃（Critical）**：同指标配置、不同时间框架的策略拿不到条件结果 → 完全不出现在矩阵里
   （0 笔交易，GA 记为适应度 −20）。修复：指标按**分组内时间框架并集**计算。
2. **制度阈值不对称（High）**：hybrid 套用制度阈值而 legacy 从不检测制度（检测代码被埋在 ML 分支内，
   ML 关闭时永不执行）；且 `effective_entry_threshold("long", "bear")` 会让**顺势空单**也被按 0.65 阈值卡住。
   修复：legacy 每 tick 独立检测制度；阈值改为"基准 0.5 + 仅逆势侧 0.65"。
3. **`risk_exit` 被 hybrid 忽略（High）**：止损距离、追踪距离、Kelly-lite 分支全部失效。已补齐并纳入变体门禁。
4. **`reduce_conditions` 语义缺失（High）**：hybrid 无分批减仓路径（legacy 89 笔 vs hybrid 34 笔）。
   修复：`auto` 路由遇到含 `reduce_conditions` 的策略集改用 legacy。
5. **`per_strategy_isolation=True` 下 legacy 多数出场失效（High）**：判定用 `symbol` 而键是 `strategy|symbol`
   —— 而 GA 正是用这个模式评估适应度。已改为 `pos_key`。
6. **swing_points 前瞻偏差（High）**：中心窗口检测却把结果写在中心点，导致 T 时刻用到了 T+5 的数据。
   修复：检测结果整体后移 `lookback` 根（仅在确认后才可用）。
7. **`DataFeeder` 空区间回退**：请求区间无数据时会把预热数据当作可交易区间（回测区间完全错位）。已删除该回退。

---

## 5. 测试与验证现状

| 项目 | 结果 |
|------|------|
| 全量测试 | **234 项全部通过**（起始基线：136 项 / 1 项失败） |
| 新增测试文件 | `test_strategy_live_path.py`(5)、`test_web_security.py`(11)、`test_trade_book.py`(7)、`test_executor_live.py`(12)、`test_strategy_lifecycle.py`(25)、`test_condition_security.py`(25)、`test_engine_parity_variants.py`(9，含 1 项不依赖缓存数据的合成行情门禁) |
| 引擎等价 | 逐笔 + 5 项指标全等；变体覆盖：单/多时间框架（15m+1h+4h）、同配置异时间框架、`risk_exit` 覆盖、仅风险出场、策略隔离、reduce 路由、合成行情（无需缓存） |
| GA 引擎路由（审计 A3 指出，已核实） | 进化批量评估走 hybrid，而 `ga_worker.py` 的 Walk-Forward/校准子进程**强制 legacy**；此前二者语义不同（等于用一套语义优化、另一套验证）。经本轮统一后两引擎已逐笔等价，因此该路由差异不再造成语义错配，但仍是应简化的历史设计 |
| 路由完整性 | 97/97 与拆分前一致；四角色 × 97 路由黑盒探测 |
| 端到端启动 | 真实 `app.main` 启动 75 秒（临时 DB、禁用 AI 与交易所凭据）：组件全部拉起、风控管线与模拟撮合实际运行、`/health` 正常、优雅退出 |
| 编译 | `python -m compileall app core web db alerts scripts` 通过 |
| 测试基建 | 新增 `pytest.ini`（注册 `slow` 标记、固定 `asyncio_mode=strict`）、`pytest-asyncio` 补入 `requirements.txt` |

### 规模验证（5 交易对，2026-03-01 ~ 2026-06-30，`per_strategy_isolation=True`）

| 策略数 | legacy 耗时 / 交易数 / 峰值内存 | hybrid 耗时 / 交易数 / 峰值内存 | 加速比 |
|--------|-------------------------------|-------------------------------|--------|
| 3 | 31.48s / 1006 / 13.4 MB | 5.82s / 1006 / 11.2 MB | ×5.41 |
| 10 | 12.36s / 189 / 19.1 MB | 5.15s / 189 / 10.7 MB | ×2.40 |
| 30 | 13.46s / 189 / 34.3 MB | 9.20s / 189 / 12.1 MB | ×1.46 |

结论：①**交易数逐一致**（1006/1006、189/189、189/189），说明等价性在远超门禁测试的规模与时间跨度上成立；
②hybrid 在所有规模下都更快且更省内存，"≥3 策略走 hybrid"的取舍仍然成立。
（注：第一行为冷启动，含一次性模型/缓存加载，故 legacy 3 策略耗时高于 10 策略。）

---

## 6. 仓库与文档

- 清理：删除孤儿目录 `binance_trader/`（92 MB）、`server.log`（296 MB）、`.pytest_cache`、全部 `__pycache__`、
  0 字节孤儿 DB、`.gitignore.bak`；工作区 **546 MB → 158 MB**。
- `.gitignore` 不再忽略 `docs/superpowers/`（15 份设计规格/计划此前每次 clone 都会丢失）。
- 文档：README 按代码校正（适应度公式、DSR 名称与返回类型、GA 默认 80/30/4、引擎等价保证、250 根预热、
  权限矩阵、测试命令、`--mode backtest` 的真实行为）；新增 `docs/core-algorithms/00-ERRATA.md`
  汇总 9 篇算法文档与代码的 20 处差异。

---

## 7. 交付物索引

| 文件 | 内容 |
|------|------|
| `docs/HANDOVER.md` | 接手评估报告（原始缺陷清单与证据） |
| `docs/overhaul/PLAN.md` | 分阶段计划、验收标准、审计轮次登记、逐条进度日志 |
| `docs/overhaul/CHANGELOG.md` | 本文件（最终变更说明） |
| `docs/overhaul/WEB_SPLIT.md` | Web 层拆分的模块图与验证记录 |
| `docs/overhaul/route-baseline.json` | 拆分前 97 条路由基线（用于后续回归比对） |
| `docs/core-algorithms/00-ERRATA.md` | 算法文档勘误 |

---

## 8. 已知遗留（本轮**未**解决，需后续排期）

1. **K 线标签口径**：缓存 parquet 的索引多为**开盘时间**，而引擎用 `index <= ts` 取"截至 ts 的最后一根"，
   等于用到了 ts 时刻尚未收盘的那根（最多 1 根时间框架的前瞻）。属全项目历史遗留，需统一为"收盘时间"
   或在加载时右移一根；影响面覆盖实盘与回测，改动需专门评审。本轮修掉了同类的 swing_points 前瞻偏差。
2. **`reduce_conditions` 在混合引擎中仍不支持**：目前靠路由回退到 legacy 规避，未实现真正等价的分批减仓。
3. **JWT/会话吊销**：改密码不会使旧 token 失效；cookie `secure=False`（本地 HTTP 部署的取舍）。
   未做 per-user token version。
4. **GA 适应度权重校准从未执行**（`data/ga_fitness_weights.json` 不存在），一直在用默认权重；
   完整 Walk-Forward 闭环也未跑过。
5. **`pickle.load` / `torch.load(weights_only=False)`**：无上传入口，但模型与 checkpoint 属可写路径，
   长期建议改为 `safetensors` 或带校验的加载。
6. **回测结果页未同步 Phase 4a 新指标**（审计 R12-003 遗留）；`/api/backtest/compare`、
   `/api/strategy-lifecycle/retire|restore` 仍未实现。
7. **`web/routes` 的鉴权仍是处理器内检查**（非 FastAPI 依赖），因此请求体校验先于鉴权执行；
   拆分代理已留 `TODO(authz)` 标记。
8. **GA/WF/校准状态字典为模块级**（非 per-app），同一进程内创建第二个 app 实例会串状态。
9. 审计中 56 项"accepted"发现（含未修的 `R10-001` 信号链无补偿事务、`R7-003`/`R7-004`）
   仍未逐条处置——其中 `R7-*` 涉及真金白银，建议优先复核。
10. **死代码与重复实现（A3 指出）**：清理了本变更集涉及模块中的 12 处未使用导入，
    但仓库内仍有约 25 个从未被调用的函数（`metrics.calculate_benchmark_metrics`、
    `monte_carlo.monte_carlo_equity_curve`、`report.report_to_json`、`deepseek_ctl.analyze_news`、
    `vibe_connector` 全部方法等）——其中多数是"Phase 4 计划但未接线"的能力，删除前应确认产品意图。
    `evaluation_kernel` 同时保留标量版与向量版函数属**有意设计**（前者服务实盘/legacy，后者服务 hybrid），
    已由 `test_engine_parity_variants.py` 与审计的属性测试证明两者等价。
11. **GA 进化与 Walk-Forward 走不同引擎**（进化 hybrid / 子进程 legacy）——现已等价，但建议统一为显式配置。
12. `data/backtest/*.json` 单文件最大 11.3 MB（已 gitignore）；`data/market|models|ml_training`
    与 SQLite 库同样不入库，因此**全新 clone 无法直接运行 App**，需要先 `scripts/download_history.py`
    并让首次启动重建数据库——建议在 README 的快速开始中显式写出这一步（当前仅有隐式提示）。

---

## 9. 回滚方式

本轮未做任何 git 提交（全部改动留在工作区），因此：

```bash
git status                 # 查看 27 个修改 + 18 个新增
git stash -u               # 需要整体回滚时
git diff <file>            # 单文件审查
```

注意：被删除的 4 类文件（孤儿目录、大日志、0 字节 DB、`.gitignore.bak`）中，`.gitignore.bak` 是唯一
曾被 git 跟踪的文件，可用 `git checkout -- .gitignore.bak` 恢复；其余均为生成物，删除不影响运行。
