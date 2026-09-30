# Binance Trader

面向币安现货的 Python 3.12 自动化交易系统。以 asyncio 事件总线为骨架，把**行情 → 策略信号 → 风控 → 下单 → 持仓守护**串成一条可观测的流水线；行情来自币安主网公开镜像、下单走 testnet；配套一个 FastAPI + aiosqlite + ECharts 的 Web 控制台（现货交易页、全币种行情、币种信息、市场数据总览、代币启发式筛查、策略监控、回测/GA、AI 面板、预警中心、设置、DB 管理）。模拟盘带真实成本模型（手续费分档 + 盘口半价差 + 滑点），账本恒等式有专门的回归测试守护；回测有 legacy 与 hybrid 两套引擎，共用同一个评估内核，并有逐笔等价门禁。

> **状态**: `VERSION` **2.0.1** · 本 README 核对于 **2026-09-30** · 测试 **1073 passed, 0 failed**（见 [§10 测试](#10-测试与质量)；干净克隆另见 §10.1 的前提说明）
> **文档索引**：详细变更史见 [`docs/overhaul/CHANGELOG.md`](docs/overhaul/CHANGELOG.md)；本 README 只陈述**当前代码事实**，与陈旧描述冲突时**以代码为准**（见 [§11.3](#113-残余耦合与陈旧之处当前审计清单)）
> **运行环境**: Python 3.12 · Windows / Linux · 默认只监听 `127.0.0.1:8899`
> **一句话前提**: 本机 **无法访问 `api.binance.com`**，因此**行情**走 `data-api.binance.vision`（主网公开镜像），**下单**走 `testnet.binance.vision`。见 [§3](#3-关键环境约束必读)。

---

## 目录

1. [TL;DR 快速开始](#1-tldr-快速开始)
2. [关键环境约束（必读）](#3-关键环境约束必读)
3. [架构](#4-架构)
4. [目录地图](#5-目录地图)
5. [配置参考](#6-配置参考)
6. [Web UI 与 API](#7-web-ui-与-api)
7. [数据与回测](#8-数据与回测)
8. [运维](#9-运维)
9. [测试与质量](#10-测试与质量)
10. [已知限制与诚实说明](#11-已知限制与诚实说明)
11. [故障排查](#12-故障排查)
12. [后续路线与许可](#13-后续路线与许可)

> 编号说明：可达性表并入 §3.0，因此正文从 §3 起编号（§2 号不再使用），章节内的小节编号保持不变。

---

## 1. TL;DR 快速开始

```bash
# 1) 克隆
git clone https://github.com/STCloudLake/Binance-Trader.git
cd Binance-Trader

# 2) Python 3.12 虚拟环境
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 3) 依赖（TA-Lib 需要系统级预编译库，见下）
pip install -r requirements.txt

# 4) 写入凭据（不要提交这个文件）
copy config\secrets.yaml.example config\secrets.yaml   # Linux: cp

# 5) 放入至少一个策略 YAML（见下方说明；仓库里没有任何策略文件）
#    例：把 <你的策略>.yaml 拷进 strategies/

# 6) 启动模拟盘
python -m app.main --mode sim
```

浏览器打开 <http://127.0.0.1:8899> → 未登录会 302 到 `/login` → 用下面打印的账号登录。

> **你必须自己提供策略 YAML。** 仓库里**一个策略文件都没有被跟踪**：`git ls-files strategies/` 输出为空，磁盘上只有被 gitignore 的 `strategies/ga_champion_*.yaml`（由 GA 生成）。因此**干净克隆启动后策略数为 0**：`GET /api/strategy-monitor` 会走到空列表分支并返回 `{"strategies": [], "active_count": 0, "total_count": 0}`（`web/routes/dashboard_partials.py:104-108`）；`POST /api/backtest/run` 传 `rsi_reversal` 之类的名字时，引擎层返回的错误是 `{"error": "Strategy 'rsi_reversal' not found: Strategy file not found: <strategies/rsi_reversal.yaml>"}`（`core/backtest/engine.py:213-218` + `core/strategy/loader.py:53-56`），Web 层只会先回进度片段、稍后在 `GET /api/backtest/progress/{run_id}` / `result/{run_id}` 上带出这个错误。要跑起来，请按 `core/strategy/loader.py::StrategyConfig` 的字段（pydantic schema 即策略 schema）自己写 YAML 放进 `strategies/`，或让 GA/策略生命周期生成。见 [§11.4](#114-未做的事)。

### 首次登录的 admin 账号

首次启动时，如果 `users` 表为空，`app/main.py` 会**自动创建 admin 并生成随机密码**：

- 账号 `admin`，密码由 `AuthManager.generate_random_password()` 生成；
- 密码**只打印到 stderr**（终端），**不写入日志文件**，日志里只有 `=== DEFAULT ADMIN CREATED: username=admin (password not logged) ===`：
  ```
  ============================================================
  DEFAULT ADMIN: admin / <随机密码>
  ============================================================
  ```
- **请立刻在 `/settings` 或用户管理页修改密码**。密码用 bcrypt 哈希存库，明文不可恢复；忘记只能删除该用户或直接改库。

> 从 `--mode live` 切回模拟盘不会重建 admin（`users` 表非空即跳过）。

### JWT 会话密钥

`auth.jwt_secret` 为空时，启动过程会生成一个随机密钥并写入 `config/secrets.yaml`（POSIX 上 `chmod 600`），这样 token 在重启后仍有效。生产环境建议改用环境变量 `JWT_SECRET`，避免密钥落盘。

### CLI 参数（`app/main.py` 的 argparse 全量）

```bash
python -m app.main [--mode {sim,live,backtest}] [--port N] [--db PATH] [--data-dir DIR] [--config-dir DIR]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode {sim,live,backtest}` | `sim` | 运行模式。见下方说明 |
| `--port N` | `8899`（`config.web_port`） | 覆盖 Web UI 端口 |
| `--db PATH` | `<project>/data/binance_trader.db` | SQLite 路径；**优先级高于 `--data-dir`** |
| `--data-dir DIR` | `<project>/data` | 运行时数据根目录；未显式给 `--db` 时 db 变成 `<data-dir>/binance_trader.db` |
| `--config-dir DIR` | `<project>/config` | 设置/密钥持久化目录（`secrets.yaml`、`config.yaml`）。**用于冒烟测试，避免改写仓库内配置** |

**`--mode` 的真实语义**（读 `core/executor/executor.py:372-376`）：

| 模式 | 行为 |
|---|---|
| `sim` | 事件 `ORDER_REQUEST` → `_execute_sim()`：本地成交，带完整成本模型，不连交易所下单 |
| `live` | 事件 `ORDER_REQUEST` → `_execute_live()`：**用 testnet 客户端真实下单**（`config.binance_testnet` 决定打到哪里） |
| `backtest` | **不进入任何下单分支**：`_on_order_request` 只处理 `sim`/`live`。也就是说 `--mode backtest` 的行情/策略/风控都照跑，但订单被静默丢弃 |

> 真正的回测**不走 `--mode backtest`**：走 Web 的 `/backtest` 页或 `POST /api/backtest/run`，由 `BacktestEngine` 选引擎。
> `config/config.yaml` **没有** `mode:` 键，模式**只来自 CLI**（`app/config.py:125`，`Config._load` 的 `self.mode = mode`）；改 YAML 无效。

---

## 3. 关键环境约束（必读）

### 3.0 本机可达性（先看这一张表）

实测（2026-09-29，`urllib` 直连，8 秒超时）：

| 主机 | 结果 | 用途 |
|---|---|---|
| `https://data-api.binance.vision` | ✅ 可达 | **所有公开行情**：klines / ticker24h / depth / trades / exchangeInfo |
| `wss://data-stream.binance.vision` | ✅ 可达 | **实时 K 线流**（`provider.py` 把 socket 工厂指向它） |
| `https://testnet.binance.vision` | ✅ 可达 | **下单 / 账户 / 余额**（当前唯一的交易端点） |
| `https://api.binance.com`（及 api1–api4） | ❌ **超时** | **不可用** |
| `wss://stream.binance.com` | ❌ 不可用 | 不可用 |

### 3.1 绝不要调用 `api.binance.com`

这不是偏好，是硬约束。任何直连主网交易域名（`api.binance.com`）的调用都会**挂到超时**，把请求处理器一起拖死。**行情必须走 `config.market_data_host`**（默认 `https://data-api.binance.vision`），由 `core/market_data/data_client.py::MarketDataClient` 统管（15s 硬超时、连接池复用、失败抛 `MarketDataError`）。

### 3.2 绝不要用裸 `AsyncClient.create()`

```python
# ❌ 不要这样写：python-binance 默认打 api.binance.com，本机必然超时
client = await AsyncClient.create()

# ✅ 交易客户端：显式传 testnet（web/routes/market.py::_configured_client）
client = await AsyncClient.create(api_key=..., api_secret=..., testnet=config.binance_testnet)
```

`MarketDataProvider.start()` 里那次 `AsyncClient.create(...)` 是**交易**客户端（传 `testnet=config.binance_testnet`），失败只会记一条 warning 并以离线模式继续（`client = None`），**不会**用于任何 K 线读取。**所有 K 线读取都走 `self.data_client`**（`provider.py:120-131`），它绑定的是行情镜像。

**为什么这条最重要**：主网公开镜像能给出**真实主网数据**——3716 个交易对、**496 个 `USDT`/`TRADING` 现货对**、1h 历史回溯到 2017-08-17（数字与 `docs/overhaul/MARKET_PAGES_API.md` §0 一致）。而 testnet 只挂几个测试对、历史是合成的。用错主机 → 看到不全的币种、K 线缺失、回测无数据。

### 3.3 实时流的 socket URL

python-binance 的 `BinanceSocketManager` 硬编码 `wss://stream.binance.com:9443/`。`provider.py:240-248` 在创建 `bsm` 后**覆写** `bsm._get_stream_url`，把它指向 `config.binance_stream_host`（由 `config.binance_stream_url` 提供）。新增任何 socket 用法时不要绕过这一步。

### 3.4 Web UI 只绑本地

`app/main.py:572-574` 是 `uvicorn.Config(host="127.0.0.1", ...)`，**没有 `--host` 参数**。要从别的机器访问，请用 SSH 端口转发或反向代理 + HTTPS，不要把 host 硬改成 `0.0.0.0` 暴露到公网。

---

## 4. 架构

### 4.1 组件数据流

```
                    ┌──────────────────────────────────────────────┐
                    │  core/market_data/provider.py                │
   data-api.vision  │  MarketDataProvider                          │
   ──── REST ──────►│   • data_client (MarketDataClient, 15s 超时) │
   data-stream.vision│  • bsm 覆写 socket URL → .vision WS         │
   ──── WS ────────►│   • OHLVCache → data/market/<SYM>/<tf>.parquet│
                    │   • _price_cache(最大 300s 陈旧即拒用)        │
                    └───────────────────┬──────────────────────────┘
                                        │ Event(MARKET_KLINE)
                                        ▼
       ┌──────────────────────────────────────────────────────────────┐
       │  app/event_bus.py   EventBus (asyncio.Queue, maxsize=10000)  │
       │  单个消费协程；订阅者异常**打印 error 而不是吞掉**              │
       └───┬───────────────┬───────────────┬───────────────┬──────────┘
           │               │               │               │
           ▼               ▼               ▼               ▼
   ┌───────────────┐ ┌───────────┐ ┌─────────────┐ ┌──────────────┐
   │ StrategyEngine│ │MLPredictor│ │NewsAnalyzer │ │ AlertManager │
   │ core/strategy │ │ core/ml   │ │ core/news   │ │ alerts/      │
   │  /engine.py   │ │           │ │             │ │ (rules.json) │
   └───────┬───────┘ └─────┬─────┘ └──────┬──────┘ └──────────────┘
           │               │              │
           │  ML_PREDICTION│  NEWS_UPDATE │
           └───────┬───────┴──────────────┘
                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  core/strategy/evaluation_kernel.py  ← **唯一评估内核**         │
   │   evaluate_entry_conditions / evaluate_exit_conditions         │
   │   fuse_signals / check_higher_tf_trend / resolve_entry_side     │
   │   ENTRY_THRESHOLD=0.5, COUNTER_TREND_THRESHOLD=0.65             │
   │   指标计算: core/strategy/indicators.py (TA-Lib + AST 白名单)   │
   └───────────────────────────┬───────────────────────────────────┘
                               │ Event(STRATEGY_SIGNAL)  |score| >= 0.5
                               ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  core/risk/manager.py   RiskManager.check_signal()             │
   │   1 熔断器 2 总敞口 3 仓位规模 4 杠杆 5 止损 6 同币去重 7 最大笔数│
   │  core/risk/circuit_breaker.py   日/周回撤·日亏损·连亏          │
   │  core/risk/position_sizer.py    固定比例 + 硬上限（非真 Kelly）│
   └───────────────────────────┬───────────────────────────────────┘
                               │ Event(ORDER_REQUEST)
                               ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  core/executor/executor.py   OrderExecutor                     │
   │   sim  : sim_cost_quote() 改写**成交价**（spread/2 + 滑点）     │
   │   live : LOT_SIZE/NOTIONAL 校验 → 按 step_size 向下取整        │
   │          同一 clientOrderId 重试 3 次（-2010 视为已受理）       │
   │  core/executor/pending_orders.py  限价单撮合（~5s）            │
   │  core/market_data/universe.py     Shared Universe (tick/step)  │
   └───────────────────────────┬───────────────────────────────────┘
                               │ ORDER_UPDATE / POSITION_UPDATE
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
 ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
 │ PositionGuard│      │  db/database │      │  web/ (FastAPI)  │
 │ 止损/追踪/紧急│      │  SQLite WAL  │      │  routes/ + ws/   │
 │ 15s 轮询      │      │  v4 schema   │      │  Jinja2 + HTMX   │
 └──────┬───────┘      └──────────────┘      └──────────────────┘
        │ Event(POSITION_EXIT / POSITION_REDUCE)
        ▼
 app/main.py 的处理器 → executor.close_position()
                      → atomic_adjust_balance()  ← **账本唯一写入点**
```

### 4.2 职责归属与接缝

| 关注点 | 唯一归属模块 | 接缝（扩展点） |
|---|---|---|
| 公开行情 REST | `core/market_data/data_client.py::MarketDataClient` | 换行情源只改 `config.market_data_host` |
| 实时 K 线 | `core/market_data/provider.py::MarketDataProvider` | WS 主机在 `binance.market_stream_host` |
| 币种宇宙（tick/step/min_notional） | `core/market_data/universe.py::Universe` | 由 `app/main.py` 预加载一次，`order_executor.wire_universe(universe)` 与 `web_app.state.universe` 共用**同一实例** |
| 周期（interval）注册表 | `core/market_data/provider.py::INTERVAL_SPEC` | 新增周期只改这一张表（`minutes/min_candles/batches/poll_secs/ml_enabled`） |
| 信号评估 | `core/strategy/evaluation_kernel.py` | 实时与两套回测引擎**都调它**；`ENTRY_THRESHOLD` / `COUNTER_TREND_THRESHOLD` 是唯一阈值来源 |
| 指标与条件沙箱 | `core/strategy/indicators.py` | 条件表达式走 **AST 白名单**，不是正则黑名单 |
| 策略 YAML | `core/strategy/loader.py::StrategyLoader` → `strategies/*.yaml` | `StrategyConfig`（pydantic）字段即策略 schema |
| 仓位规模 / 止损距离 | `core/risk/position_sizer.py::PositionSizer` | `trailing_stop_distance_pct()` 是实盘 + 两套回测**唯一**的追踪止损距离来源 |
| 熔断 | `core/risk/circuit_breaker.py` | 动作由 `hard_limits.circuit_breaker_action` 决定 |
| 成交与成本 | `core/executor/executor.py` + `app/config.py::sim_cost_quote` | 成本公式只有一份，`GET /api/fee/estimate` 与真实成交调**同一个函数** |
| 账本 | `db/database.py::atomic_adjust_balance` | `BEGIN IMMEDIATE` + 模块级 `asyncio.Lock` |
| 事件 | `app/event_bus.py` | `EventType` 枚举（17 种） |
| 交易/账户客户端 | `web/routes/market.py::_configured_client`、`web/routes/dashboard_partials.py::_configured_client` | 两个客户端工厂都**显式传 `testnet=config.binance_testnet`**（绝不裸 `AsyncClient.create()`）；此外 `core/executor/executor.py:94`、`core/market_data/provider.py:168` 也各读一次该标志 |

### 4.3 两套回测引擎

两者共用 `core/strategy/evaluation_kernel.py` 与 `core/backtest/trade_book.py::close_position`（以前是两份逐字克隆的平仓记账，已被抽成共享实现，避免再次漂移）。

| | **legacy**（`core/backtest/engine.py`，逐 tick） | **hybrid**（`core/backtest/engine_hybrid.py`，向量化两阶段） |
|---|---|---|
| 结构 | 单循环遍历时间线，逐根 K 线判断 | `SignalMatrixBuilder` 预生成信号矩阵 → `EventDrivenExecutor` 回放 |
| 速度 | 基准 | 显著更快（规划文档记录 ×4.1 ~ ×24） |
| ML | 支持 LightGBM / TFT / PatchTST | **不支持**（`_select_engine` 直接抛 `ValueError`） |
| 部分减仓 `reduce_conditions` | 支持 | **不支持** |
| 适用 | 任意策略集 | 纯指标策略、≥3 条策略的 GA 批量评估 |

**`backtest.engine_mode: auto`（默认）的选择逻辑**（`engine.py::_select_engine`）：

1. `legacy` → 永远 legacy。
2. `hybrid` → 若任一策略启用 ML，**抛错**（提示改用 legacy 或关掉 `ml_enabled`）；否则 hybrid。
3. `auto` → 策略数 **≥ 3** 且无 ML 且**没有** `reduce_conditions` → hybrid；否则 legacy。
4. hybrid 运行期若抛异常，`run_with_exit_evaluation` 会记 warning 并**回退 legacy**（结果可能与 hybrid 模式有差异，日志里会说明）。

`scripts/ga_worker.py` 显式设 `config.backtest_engine_mode = "legacy"`（子进程没有完整引擎栈）。

### 4.4 成本模型：两套，故意不合并

| | **sim 成本模型** | **backtest 成本模型** |
|---|---|---|
| 代码 | `app/config.py::sim_cost_quote` | `core/backtest/cost_model.py::apply_trading_costs` |
| 作用点 | **改写成交价本身**（买贵、卖便宜） | 在**平仓时**叠加成本 |
| 为什么 | 模拟盘的余额、持仓成本、PnL 要自洽 | 回测不得污染入场价，否则止损/止盈价位会级联变化 |
| 明细 | maker/taker 分档、BNB 折扣、盘口半价差、滑点 bps | taker 费率、按 symbol 解析的价差 |
| 价差来源 | `sim.cost_model.spread_pct` → `default` → `0.02` | override → live depth → `default_spread_pct`（0.03） |

---

## 5. 目录地图

| 目录 / 文件 | 一句话职责 |
|---|---|
| `app/` | 入口与配置：`main.py`（argparse、组件装配与 wire、后台循环、优雅关闭）、`config.py`（单例配置 + 成本模型）、`event_bus.py`（事件总线） |
| `core/market_data/` | 行情层：`provider.py`（`INTERVAL_SPEC` 周期注册表）、`data_client.py`、`universe.py`、`ohlcv_cache.py`、`ttl_cache.py`、`metrics.py`、`screener.py`（启发式筛查） |
| `core/strategy/` | 策略层：`engine.py`、`evaluation_kernel.py`（共享内核）、`indicators.py`（含 AST 白名单）、`loader.py` |
| `core/risk/` | 风控层：`manager.py`（7 步管线）、`circuit_breaker.py`、`position_sizer.py`、`position_guard.py`（止损/追踪/紧急） |
| `core/executor/` | 执行层：`executor.py`（sim/live 下单与平仓）、`pending_orders.py`（限价单撮合） |
| `core/backtest/` | 回测：`engine.py`（legacy）、`engine_hybrid.py` + `signal_matrix.py` + `event_executor.py`（hybrid）、`cost_model.py`、`data_feeder.py`、`trade_book.py`、`metrics.py`、`monte_carlo.py`、`report.py` |
| `core/ga/` | 遗传算法：`evolver.py`、`genome.py`、`fitness.py`、`fitness_calibrate.py`、`walkforward.py` |
| `core/ml/` | 生产中真正接线的 ML：`predictor.py`、`trainer.py`、`features.py`、`tft_trainer.py`/`tft_model.py`、`patchtst_trainer.py`/`patchtst_model.py` |
| `core/ai/` | DeepSeek 控制器（`deepseek_ctl.py`）、策略生命周期（`strategy_lifecycle.py`）、提示词（`prompts.py`） |
| `core/news/` | 新闻情绪：`analyzer.py`、`fetcher.py`（带 SSRF 防护）、`source_manager.py` |
| `core/auth/` | `auth.py`：bcrypt + JWT + 内存会话 + 鉴权中间件 |
| `db/` | `database.py`：单一 DDL 源、v4 迁移链、余额原子操作 |
| `web/` | `server.py`（应用工厂）、`routes/*`（按域拆分的 118 条路由）、`ws/alerts.py`、`templates/`、`static/`、`i18n.py`、`deps.py`、`rendering.py`、`context.py` |
| `alerts/` | `manager.py` + `rules.py`：规则引擎、冷却、WebSocket 推送 |
| `strategies/` | 策略 YAML（`StrategyConfig` 的 pydantic 字段即 schema）。**仓库里没有跟踪任何策略文件**：`git ls-files strategies/` 为空，磁盘上只有被 gitignore 的 `ga_champion_*.yaml`。**干净克隆必须先自己放入策略 YAML**，见 [§1](#1-tldr-快速开始) |
| `scripts/` | `download_history.py`（历史下载 CLI）、`ga_worker.py`（GA 子进程）、`audit_db.py`（只读账本核查 + 表/索引清单，见 [§9.4](#94-账本核查对账)）、`verify_ai_panel_e2e.py` |
| `data/` | 运行时数据：`binance_trader.db`、`market/<SYM>/<tf>.parquet`、`models/`、`backtest/`、`ga_jobs/`、`symbols.json`。**`data/` 整体被 gitignore，一个文件都没跟踪**，所以干净克隆里既没有历史数据也没有模型 |
| `experimental/` | **不在交易链路上**的死代码存档，见 `experimental/ml/README.md` |
| `config/` | `config.yaml`、`risk_params.yaml`、`secrets.yaml`（被 gitignore）、`secrets.yaml.example`、`alert_rules.json` |
| `docs/` | `HANDOVER.md`、`development-roadmap.md`、`overhaul/*`、`core-algorithms/*`、`audit/*`、`superpowers/*`（设计史） |
| `tests/` | **45** 个 `.py`（含 `__init__.py` 与 `conftest.py`；即 **43** 个 `test_*.py`），见 [§10](#10-测试与质量) |
| `run_logs/` | 手工冒烟测试的重定向输出（已 gitignore） |
| `requirements.txt` | 运行 + 测试依赖（含 `python-multipart`——starlette 无条件导入它，缺了 FastAPI 起不来） |
| `.gitignore` | 忽略 `data/`、`strategies/ga_champion_*.yaml`、`secrets.yaml`、`run_logs/`，以及本地 DB 副本 `*.bak` / `*.db.bak` / `*.db.pre_restore` |
| `pytest.ini` | `testpaths=tests`、`asyncio_mode=strict`、`slow` marker |
| `VERSION` | `2.0.1` |

---

## 6. 配置参考

配置来源（`app/config.py::_load`）：`config/config.yaml` → `config/risk_params.yaml` → `config/secrets.yaml`，深合并；**环境变量优先于 YAML**。

### 6.1 环境变量

| 变量 | 覆盖 |
|---|---|
| `BINANCE_API_KEY` | `binance.api_key` |
| `BINANCE_API_SECRET` | `binance.api_secret` |
| `DEEPSEEK_API_KEY` | `deepseek.api_key` |
| `JWT_SECRET` | `auth.jwt_secret`（生产建议用它，而不是让程序落盘生成） |

### 6.2 `config/config.yaml`

| 键 | 默认 | 含义 | 读取处 |
|---|---|---|---|
| `web_port` | `8899` | Web UI 端口 | `config.py:208` |
| `language` | `zh` | UI 语言（`zh`/`en`） | `config.py:242`、`web/i18n.py` |
| `binance.testnet` | `true` | **下单/账户**客户端是否打 testnet。**不控制行情** | `config.py:210`、`executor.py:94` |
| `binance.market_data_host` | `https://data-api.binance.vision` | 公开行情 REST 主机 | `config.py:215-217` |
| `binance.market_stream_host` | `wss://data-stream.binance.vision` | 行情 WS 主机 | `config.py:218-223` |
| `auth.jwt_secret` | `''` | JWT 密钥；空则生成并落盘 | `main.py:198-227` |
| `auth.session_hours` | `24` | 会话与 JWT 有效期（小时） | `main.py:228` |
| `signal_weights.indicator` | `0.5` | 指标信号权重 | `config.py:225-226`、`engine.py:148` |
| `signal_weights.ml` | `0.3` | ML 权重（策略自带 `ml_config.weight` 时以策略为准） | 同上 |
| `signal_weights.news` | `0.2` | 新闻情绪权重 | 同上 |
| `core_position.max_symbols` | `5` | 「核心」分类的币种数上限 —— **仓位分类**，不是"前 3 个列表项" | `config.py:228-229`、`main.py:416` |
| `core_position.capital_pct` | `0.7` | 核心资金池比例 | `config.py:230` |
| `satellite_position.max_symbols` | `10` | 「卫星」分类上限 | `config.py:232-233` |
| `satellite_position.capital_pct` | `0.3` | 卫星资金池比例 | `config.py:234` |
| `news.fetch_interval_minutes` | `30` | 新闻抓取间隔 | `config.py:237` |
| `news.max_articles_per_symbol` | `10` | 每币种最大文章数 | `config.py:238` |
| `news.anomaly_threshold_pct` | `3.0` | 触发紧急抓取的价格异动阈值（%） | `config.py:239` |
| `news.volume_spike_multiplier` | `3.0` | 成交量放大倍数阈值 | `config.py:240` |
| `ai.mode` | 文件值 `full_auto` | `semi_auto` / `full_auto`；full_auto 下熔断动作由 AI 决定并触发恢复循环。**代码回退默认是 `semi_auto`**（`config.yaml` 显式写 `full_auto`） | `config.py:305`、`main.py:357-365` |
| `ai.model` | 文件值 `deepseek-v4-flash` | 模型名。**代码回退默认是 `deepseek-chat`**（`config.yaml` 显式覆盖） | `config.py:306` |
| `ai.base_url` | `https://api.deepseek.com` | OpenAI 兼容 base URL | `config.py:307` |
| `ai.tasks.market_assessment_minutes` | `60` | 市场评估周期（分钟→秒） | `config.py:308-314` |
| `ai.tasks.coin_selection_minutes` | `240` | 选币周期。**选币结果会写自选列表**（`deepseek_ctl.py:137-151`），校验后经 `save_watchlist` 落库 | 同上 |
| `ai.tasks.strategy_optimization_minutes` | `1440` | 策略优化周期 | 同上 |
| `ai.tasks.risk_adjustment_minutes` | `1440` | 风控调整周期 | 同上 |

### 6.3 `backtest.*`（回测成本与引擎）

| 键 | 默认 | 含义 |
|---|---|---|
| `backtest.engine_mode` | `auto` | `auto` / `hybrid` / `legacy`；非法值回退 `auto` 并 warning（`config.py:245-248`） |
| `backtest.ml_enabled` | `false` | 回测是否训练/使用 ML。为 false 时会**就地关掉**策略的 `ml_config.enabled`（`engine.py:159-171`） |
| `backtest.cost_model.enabled` | `true` | 总开关 |
| `backtest.cost_model.taker_fee_pct` | `0.04` | taker 费率（%） |
| `backtest.cost_model.spread_pct` | 见 YAML（BTC 0.01 / ETH 0.02 / BNB·SOL 0.03 / XRP 0.04） | **按 symbol 的价差覆盖表**；`default`/`*` 键代表"对所有 symbol 的显式覆盖" |
| `backtest.cost_model.default_spread_pct` | `0.03` | 最后兜底（无覆盖、无实时盘口时） |
| `backtest.cost_model.live_spread.enabled` | `true` | 是否从 `{market_data_host}/api/v3/depth?limit=5` 实时推导价差 |
| `backtest.cost_model.live_spread.ttl_seconds` | `300` | 实时价差缓存 TTL；**失败结果不缓存** |
| `backtest.cost_model.live_spread.timeout_seconds` | `3` | 实时价差单次超时 |
| `backtest.cost_model.market_data_host` | 继承 `binance.market_data_host` | 价差推导用的主机 |

**回测价差解析顺序**（`core/backtest/cost_model.py`）：

```
1. override   ← 调用方传入（回测/GA 表单里手输）或 config 的 spread_pct 表
2. live       ← (best_ask − best_bid) / mid × 100，来自公开盘口，缓存 TTL 300s
3. default    ← backtest.cost_model.default_spread_pct（0.03）
```

### 6.4 `sim.*`（模拟盘成本模型 —— 会改成交价）

| 键 | 默认 | 含义 |
|---|---|---|
| `sim.cost_model.enabled` | `true` | `false` → 回到"按报价成交、零费用"的旧行为 |
| `sim.cost_model.fee_tier` | `VIP0` | 默认档位。**这是默认值**：`POST /api/fee/tier` 会把选择写进 `system_config`（键 `sim.fee_tier` / `sim.use_bnb_discount`），此后每次成交与估价都以 DB 为准（`load_sim_cost_settings`） |
| `sim.cost_model.use_bnb_discount` | `false` | `true` → 所有费率 ×0.75（BNB 抵扣 25%） |
| `sim.cost_model.slippage_bps` | `2` | **每边**滑点，单位 bps（1 bp = 0.01%）。**只有市价单滑点** |
| `sim.cost_model.spread_pct.<SYM>` | BTC 0.01 / ETH 0.02 / BNB·SOL 0.03 / XRP 0.04 | **每边半价差**（%）。`BTCUSDT: 0.01` = 半价差为价格的 0.01% |
| `sim.cost_model.spread_pct.default` | `0.02` | 未列出的 symbol 用这个 |

**公式**（`app/config.py::sim_cost_quote`，唯一实现，`GET /api/fee/estimate` 与真实成交共用）：

```
edge_pct = 0                                  (限价单 或 enabled=false)
         | spread_pct/2 + slippage_bps/100     (市价单)

买入(long/buy):  fill = price × (1 + edge_pct/100)
卖出(short/sell): fill = price × (1 − edge_pct/100)

fee          = quantity × fill × fee_pct/100      fee_pct 由档位取 maker/taker，BNB 折扣 ×0.75
slippage_usdt = |fill − price| × quantity
cost_usdt     = fee + slippage_usdt
per_unit_cost = cost_usdt / quantity
effective_price = price ± per_unit_cost
```

- **费率分档**：`FEE_TIER_TABLE`（`config.py:14-25`）是币安现货 VIP0–VIP9 标准表。**档位是手动选择**——本机无法读取真实账户的 30 天交易量与 BNB 持仓，所以程序不会自己判断档位。
- 市价单用 `taker_pct`，限价单用 `maker_pct`（`sim_fee_pct`）。

### 6.5 `config/risk_params.yaml`

`hard_limits`（硬约束，`HardRiskLimits`）与 `soft_params`（软参数，`SoftRiskParams`）。文件里给的值会覆盖 `app/config.py` 里的类默认值。

| 键 | 文件值 | 类默认 |
|---|---|---|
| `max_daily_drawdown_pct` | `7.5` | `5.0` |
| `max_weekly_drawdown_pct` | `10.0` | `10.0` |
| `max_daily_loss_usdt` | `600.0` | `500.0` |
| `max_position_size_pct` | `50.0` | `10.0` |
| `max_leverage` | `4` | `3` |
| `min_stop_loss_distance_pct` | `0.5` | `0.5` |
| `max_open_trades` | `15` | `8` |
| `max_total_exposure_pct` | `80.0` | `80.0` |
| `max_consecutive_losses` | `10` | `5` |
| `circuit_breaker_action` | `block_only` | `block_only`（可选 `tighten_stops` / `close_all` / `close_worst`；非法值回退 `block_only`） |
| `trailing_stop_enabled` | `true` | `true` |
| `trailing_stop_distance_pct` | `2.0` | `2.0` |
| `emergency_stop_enabled` | `true` | `true` |
| `emergency_stop_threshold_pct` | `-5.0` | `-5.0` |
| `soft_params.position_size_pct` | `8.0` | `5.0` |
| `soft_params.stop_loss_pct` | `2.0` | `2.0` |
| `soft_params.leverage` | `2` | `2` |
| `take_profit_{1,2,3}_pct` | `3.0 / 5.0 / 10.0` | 同 |

### 6.6 自选列表（watchlist）

**不在 YAML 里**，而是持久化在数据库 `system_config` 表的 `watchlist_symbols`（JSON 数组）：

| 项 | 值 | 来源 |
|---|---|---|
| 键名 | `watchlist_symbols` | `core/market_data/universe.py:40` |
| 兜底默认 | `["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"]` | `universe.py:34` |
| 上限 | `WATCHLIST_MAX = 30` | `universe.py:37` |
| 读取 | `load_watchlist(db_path, default)` | `universe.py:399-428` |
| 写入 | `save_watchlist(db_path, symbols)`（去重保序 + 截断到 30） | `universe.py:431-440` |
| 校验 | `validate_watchlist()`：必须在 exchangeInfo 中且 `status=TRADING` | `universe.py:443-463` |

生效时机：`MarketDataProvider.start(symbols=None)` 读持久化列表（`provider.py:149-160`），再据此订阅 WS 流。**改动在下次重启生效**——`POST /api/market/watchlist` 的响应里明确带 `"restart_required": true`（`web/routes/market.py:519`）。AI 选币也会写这个键，同样下次重启生效。

读取顺序：`system_config` 存在 → 用它；不存在/非法 JSON → 用 `DEFAULT_WATCHLIST`。

### 6.7 `config/secrets.yaml`

```yaml
binance:
  api_key: "your_binance_api_key"
  api_secret: "your_binance_api_secret"
deepseek:
  api_key: "your_deepseek_api_key"
auth:            # 首次启动自动写入（也可由 JWT_SECRET 环境变量提供）
  jwt_secret: "..."
```

已 gitignore。`/settings` 的表单保存逻辑（`web/routes/settings.py:170-211`，`save_binance_settings`）**把空字段解释为"保持不变"，绝不擦除凭据**；`testnet` 只在显式提交时才改变（字段缺省时保持原值，`settings.py:178-181`）。

---

## 7. Web UI 与 API

### 7.1 页面

| 路径 | 用途 | 最低角色 |
|---|---|---|
| `/` | 302 → `/trade` | 登录 |
| `/dashboard` | 302 → `/trade`（仪表盘职能已被现货页完全覆盖，保留重定向让旧书签可用） | 登录 |
| `/trade` | **Binance 风格现货交易页**：行情条、订单簿 + 最新成交、ECharts K 线（1m/5m/15m/1h/4h + MA/EMA/BOLL/VOL/RSI/MACD）、买/卖下单（市价/限价 + 25/50/75/100% 快捷键 + 止损%）、账户卡片、当前委托/持仓/成交历史/订单历史四个标签页 | 登录 |
| `/market` | 全币种行情：搜索 / 排序 / 分页、自选管理、本地数据标记、行点击进交易页 | 登录 |
| `/coin/{symbol}` | 币种信息：交易规则、24h、盘口、1d/7d/30d/90d 表现、启发式风险评分与 flags、BTC 相关性 | 登录 |
| `/data` | 全市场数据总览：涨跌榜、成交额榜、最活跃、波动率榜、价差榜 | 登录 |
| `/audit` | 代币检测：启发式风险筛查（**页面显著标注"非链上合约审计"**） | 登录 |
| `/strategies` | 策略监控与 CRUD、信号权重、风控参数展示、AI 策略推荐 | 登录（写操作需 trader） |
| `/backtest` | 回测配置、运行、进度、结果、历史列表（引擎模式、成本模型、多币种多选） | **页面需 trader**（非 trader 302 → `/dashboard`） |
| `/ai` | AI 面板：建议卡片、市场评估、AI 心跳、`POST /api/consult` 手动咨询、策略生命周期事件 | 登录（审批需 trader） |
| `/alerts` | 预警中心：告警列表、筛选、规则开关 | 登录 |
| `/settings` | 设置：AI 模式、AI/新闻参数、Binance 凭据与 testnet 开关、风控阈值、熔断状态与重置、重启服务 | **页面需 trader**（非 trader 302 → `/trade`）；凭据/风控/重启等写接口需 **admin** |
| `/db-manager` | DB 管理：表浏览、行删除、备份/恢复/优化/清理、CSV 导出 | **admin**（否则 302 → `/trade`） |
| `/users` | 用户管理 | **admin**（否则 302 → `/trade`） |
| `/login` | 登录页 | **公开** |

### 7.2 路由总数

**118 条**（HTTP + WebSocket），由 AST 解析 `web/routes/*.py` + `web/ws/*.py` 的装饰器得出，并用运行期 `app.routes` 交叉校验（**两边完全一致，无差集**）。脚本：`%TEMP%\bt_routes_ast.py`，输出 `%TEMP%\bt_routes_ast.json`。

授权级别分布：**公开 4 / 登录 58 / trader 35 / admin 21**（4 + 58 + 35 + 21 = 118）。

- 公开 = `AuthMiddleware` 白名单：`/login`、`/api/auth/login`、`/api/auth/logout`、`/health`，以及 `/static/*` 与 `/ws/*` 前缀（`core/auth/auth.py:225`）。
- 登录 = 只要求有效会话（cookie `bt_session` 或 `Authorization: Bearer <JWT>`）。`/ws/alerts` 也在这一档：中间件按前缀放行，但 handler 自己校验 `?token=`，缺 token 或校验失败会以 `4001` 关闭连接（`web/ws/alerts.py:10-23`）。
- trader = handler 内 `_require_trader()`（`admin` 与 `trader` 角色都通过，`core/auth/auth.py:22-23`）。
- admin = handler 内 `_require_admin()` 或等价的内联 `user.is_admin` 判断（`web/routes/users.py`、`web/routes/db_manager.py`、`pages.py` 的 `/db-manager`、`/users`）。

> **两个计数口径别混**：上面的 118 是 `web/routes/*` + `web/ws/*` 里的**业务路由**，与 `docs/overhaul/route-baseline.json` 的 118 条一致。运行期 `app.routes` 会多一条 `GET /openapi.json`（`/docs`、`/redoc` 被显式关掉，但 OpenAPI schema 本身仍然挂着），所以 `len(app.routes)` 是 120（另有 `/static` 这个 Mount）。
>
> 另外 `/backtest` 页本身就要 trader（非 trader 会被 302 回 `/dashboard`），不只是"运行需要 trader"。

> 授权检查目前是**在 handler 内部**做的（`web/deps.py` 与若干 handler 上有 `TODO(authz)` 标记），返回 `{"error":"Forbidden"}` + 403。

### 7.3 端点（按模块分组）

#### 健康与鉴权（`health.py` / `auth.py`）

| 方法 | 路径 | 角色 |
|---|---|---|
| GET | `/health` | 公开。未认证只返回 `status`/`database`/`uptime_seconds`；认证后追加熔断状态、持仓数、策略数、mode |
| GET | `/login` | 公开 |
| POST | `/api/auth/login` | 公开。按 IP 限流：5 分钟窗口内 10 次，超出 429 |
| POST | `/api/auth/logout` | 公开 |
| POST | `/api/auth/change-password` | 登录 |

#### 用户（`users.py`）— 全部 admin

`GET /api/users`、`POST /api/users`、`POST /api/users/{uid}`、`DELETE /api/users/{uid}`、`POST /api/users/{uid}/toggle`、`GET /partials/user-list`

#### 行情 / 交易页 API（`market.py`）

| 方法 | 路径 | 角色 |
|---|---|---|
| GET | `/api/market/ticker` | 登录 |
| GET | `/api/market/depth` | 登录 |
| GET | `/api/market/trades` | 登录 |
| GET | `/api/market/overview` | 登录 |
| GET | `/api/market/ticker24h` | 登录。全市场 ticker，60s 缓存 |
| GET | `/api/market/symbols` | 登录。搜索/排序/分页，`total=496`（USDT/TRADING） |
| GET | `/api/market/watchlist` | 登录 |
| POST | `/api/market/watchlist` | **trader**。校验 TRADING + 上限 30，返回 `restart_required: true` |
| GET | `/api/coin/{symbol}` | 登录 |
| GET | `/api/data/overview` | 登录 |
| GET | `/api/kline/{symbol}` | 登录 |
| GET | `/api/account` | 登录。余额以 **DB** 为准 |
| POST | `/api/order` | **trader** |
| GET | `/api/orders` | 登录 |
| POST | `/api/orders/{order_id}/cancel` | **trader** |
| GET | `/api/fee/tier` | 登录 |
| POST | `/api/fee/tier` | **trader**。非法档位 → 400 |
| GET | `/api/fee/estimate` | 登录。与真实成交**共用** `sim_cost_quote` |
| GET | `/api/history/trades` | 登录 |

#### 交易（`trading.py`）

`POST /api/trade`（**trader**）、`POST /api/trade/close/{symbol}`（**trader**）、`GET /partials/stats`（登录）、`GET /partials/positions`（登录）

#### 仪表盘片段（`dashboard_partials.py`）

`GET /api/trades`、`GET /partials/trades`、`GET /api/market-state`、`GET /api/strategy-monitor`、`GET /api/price/{symbol}`（均登录）

#### 策略（`strategies.py`）

`GET /api/strategy/{name}`（登录）、`POST /api/strategy`（trader）、`PUT /api/strategy/{name}`（trader）、`POST /api/strategy/{name}/toggle`（trader）、`POST /api/strategy-symbols/{name}`（trader）、`DELETE /api/strategy/{name}`（trader）、`POST /api/strategy/reload`（trader）、`POST /api/strategy-recommend`（trader）

#### 回测（`backtest.py`）

`GET /backtest`（**trader**，非 trader 302 → `/dashboard`；`web/routes/backtest.py:193-196`）、`POST /api/backtest/run`（**trader**，`backtest.py:203` + `_require_trader`）、`GET /partials/backtest-active`、`GET /api/backtest/progress/{run_id}`、`GET /api/backtest/spreads`、`GET /partials/backtest-config`、`GET /partials/backtest-list`（以上登录）、`GET /api/backtest/{record_id}`（**trader**——它返回策略内部与成交历史，只读 viewer 不得枚举；`web/routes/backtest.py:494-503`）、`GET /api/backtest/result/{run_id}`（trader）、`DELETE /api/backtest/{record_id}`（trader）、`POST /api/backtest/fetch-data`（**trader**，按需下载历史数据）

#### GA / Walk-Forward（`ga.py`）

`POST /api/ga/evolve`（trader）、`GET /api/ga/status`（登录）、`POST /api/ga/stop`（trader）、`POST /api/ga/walkforward`（trader）、`GET /api/ga/wf_status`（登录）、`GET /partials/ga-panel`（登录）

#### 策略生命周期（`lifecycle.py`）

`GET /api/strategy-lifecycle/events`、`GET /partials/strategy-lifecycle`（登录）、`POST /api/strategy-lifecycle/generate`、`POST /api/strategy-lifecycle/optimize`（trader）

#### AI（`ai.py`）

`GET /api/ai-suggestions`、`GET /partials/ai-suggestions`、`GET /api/deepseek-models`、`GET /api/ai-heartbeat`（登录）、`POST /api/ai-suggestions/{sid}/approve`、`POST /api/ai-suggestions/{sid}/reject`、`POST /api/consult`（trader）

#### 预警（`alerts.py` + `ws/alerts.py`）

`GET /api/alerts`、`GET /api/alerts/filtered`、`GET /api/alerts/counts`、`GET /api/alert-rules`、`GET /partials/alerts`、`GET /partials/alerts-filtered`、`GET /partials/alert-rules`（登录）、`POST /api/alerts/{alert_id}/ack`、`POST /api/alert-rules/{index}/toggle`、`POST /api/alert-rules/{index}/remove`（trader）、`POST /api/alerts/clear`（**admin**）、`WS /ws/alerts`

#### 设置（`settings.py`）

`POST /api/ai-mode`、`POST /api/signal-weights`、`POST /api/settings/ai-news`、`POST /api/circuit-breaker/reset`（trader）、`POST /api/settings/deepseek`、`POST /api/settings/risk`、`POST /api/settings/binance`、`POST /api/settings/reset-sim`、`POST /api/settings/restart`（**admin**）

#### DB 管理（`db_manager.py`）— 全部 admin

`GET /api/db/table/{table}`、`GET /api/db/backup`、`POST /api/db/restore`、`POST /api/db/optimize`、`POST /api/db/cleanup`、`GET /api/db/export/{table}`、`DELETE /api/db/row/{table}/{row_id}`

> `db_manager.py` / `pages.py` 的允许表清单**已经清理干净**（`orders` 不再出现），`settings.py` 的 `reset-sim` 只 `DELETE FROM trades` / `DELETE FROM positions`。旧审计记录的"清单里仍有 `orders`"已不再成立。

#### 代币检测（`audit.py`）

`GET /api/audit/screen`、`GET /api/audit/{symbol}`（登录）

### 7.4 前端与移动端

- **Jinja2 + HTMX + Tailwind CDN + ECharts**（`web/templates/base.html:9-11`）。
- 响应式是**真实存在**的：`viewport` meta、`md:hidden` 汉堡菜单 + `#mobile-nav` 下拉（`base.html:63-91`）、导航/内边距用 `md:`/`sm:` 断点。桌面端与移动端导航项一致。
- `web/rendering.py` 的 Jinja 环境已开启 **autoescape**（XSS 加固）。
- 页面里的 K 线图按 ECharts 的 `[open, close, low, high]` 顺序传值——这是历史 bug 的根因（曾把时间戳当成开盘价，导致 y 轴被拉到 `[0, 1.8e12]`，蜡烛压成一条直线）。

---

## 8. 数据与回测

### 8.1 历史数据下载 CLI

```bash
python scripts/download_history.py \
    --symbols BTCUSDT,SOLUSDT \
    --intervals 1h,4h \
    --start 2024-01-01 \
    --end 2024-03-01 \
    [--data-dir DIR] [--data-host URL] [--timeout 15] [--concurrency 4] [--merge]
    [--list-intervals]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--symbols` | **必填** | 逗号分隔，如 `BTCUSDT,SOLUSDT` |
| `--intervals` | `1h` | 逗号分隔。合法值：`1s,1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M`（`--list-intervals` 可打印） |
| `--start` | **必填** | `YYYY-MM-DD`，**含当日** |
| `--end` | 现在 | `YYYY-MM-DD`，**含当日**（`end_of_day=True`，即下载完整结束日） |
| `--data-dir` | `config.data_dir` | 落地到 `<data-dir>/market/<SYMBOL>/<interval>.parquet` |
| `--data-host` | `config.market_data_host` | 行情主机（**不是** `api.binance.com`） |
| `--concurrency` | `4` | 并行下载的 symbol 数（信号量控制） |
| `--merge` | 关 | 与已有 parquet 合并（而非覆盖），按索引去重保序 |

每页最多 1000 根（`KLINES_MAX_LIMIT`），页间隔 0.12s 以避免触发限频。parquet 结构：

```
index  : close_time (datetime64, UTC — 每行一根已收盘 K 线)
columns: open, high, low, close, volume   (float64)
```

同一个布局也是 `MarketDataProvider._prefetch_history()` 与 `core/backtest/data_feeder.py::DataFeeder` 读写的那一份——`BacktestEngine` 从 `<data_dir>/market` 读。

### 8.2 回测怎么跑

1. **下载数据**（Web：回测页的"获取数据"按钮 → `POST /api/backtest/fetch-data`；或上面的 CLI）。
2. 在 `/backtest` 选策略、币种、日期区间、初始资金；价差可按 symbol 覆盖。
3. `POST /api/backtest/run` → `BacktestEngine.run_with_exit_evaluation()`：
   - 先用 `freeze_run_spreads(symbols, config, spread_overrides)` 把**本次用到的**每个 symbol 的价差解析一次并冻结（override → live → default），所以交易循环里不再做 I/O；
   - `_select_engine()` 选 legacy / hybrid（见 [§4.3](#43-两套回测引擎)）；
   - 结果写 `backtest_records` 表 + `data/backtest/*.json`。
4. 进度：轮询 `GET /api/backtest/progress/{run_id}`；结果：`GET /api/backtest/result/{run_id}`。

### 8.3 成本如何计入回测

回测成本**在平仓时按往返一次性扣减**（`core/backtest/cost_model.py::apply_trading_costs`），**不改入场价**——否则止损/止盈价位会随成本级联变化，回测就失真了。

```
pnl(毛) = (exit − entry) × qty            (long)
costs   = taker_fee_pct 与 spread_pct 按往返计算
pnl(净) = pnl(毛) − costs
```

这与 sim 盘的成本模型**故意不同**（sim 盘改的是成交价），`config.yaml` 里有注释明确说明不要把两者合并。

---

## 9. 运维

### 9.1 日志

loguru 默认输出到 **stderr**，项目**没有配置文件落盘 sink**。要留档就自己重定向：

```powershell
python -m app.main --mode sim *>> run_logs\service.log      # PowerShell
```
```bash
python -m app.main --mode sim >> run_logs/service.log 2>&1   # bash
```

`run_logs/` 已被 gitignore。关键日志行：

| 日志 | 含义 |
|---|---|
| `Watchlist in effect: ...` | 本次实际监控的币种（来自持久化自选列表） |
| `WebSocket connecting: N streams` / `WS msg #N` | 实时流状态 |
| `REST poll: N symbol/interval pairs (WebSocket has not delivered yet for M)` | REST 兜底轮询生效中，WS 尚未送到 |
| `SIGNAL: <strategy> LONG <SYM> @ price score=x` | 发出的策略信号 |
| `Circuit breaker TRIPPED: ...` | 熔断触发 |
| `Refusing duplicate open for <SYM>` | 重复开仓被拒（保护账本） |
| `Breaker action: ...` | 熔断响应动作执行 |

启动顺序是**先 web 就绪、后 ML 训练**吗？不是——ML 训练在 `server.serve()` 之前串行执行（`app/main.py:488-498`；`uvicorn.Server(...).serve()` 在 `main.py:575-580`），所以**启动后有一段时间端口未监听**。训练期间 `Exception` 只记 warning，不阻塞启动。

### 9.2 关机语义

`Ctrl+C` → `finally` 块按 **逆序** 关停：后台循环（`task.cancel()` + `gather`）→ 限价单撮合器 → AlertManager → PositionGuard → DeepSeek → Executor → RiskManager → NewsAnalyzer → MLPredictor → StrategyEngine → MarketData → EventBus。持仓与余额**不做任何落盘动作**，因为它们本来就实时写在 DB 里。

### 9.3 DB 备份 / 恢复 / 清理

| 操作 | 方式 |
|---|---|
| 备份（下载文件） | `GET /api/db/backup`（admin）。先 `shutil.copy2` 到 `<db>_backup_<YYYYmmdd_HHMMSS>.db`，再以附件返回。文件名 `binance_trader_backup_<ts>.db` |
| 恢复 | `POST /api/db/restore`（admin，上传文件）。**先校验前 16 字节必须是 `SQLite format 3\0`**，然后把当前库复制成 `<db>.pre_restore`，再用上传内容覆盖 |
| 优化 | `POST /api/db/optimize`（admin）：`VACUUM` + `REINDEX`，返回前后字节数 |
| 清理 | `POST /api/db/cleanup`（admin）：删除 90 天前的 `alerts`、90 天前的 `ai_suggestions`、以及 `action IN ('close','reduce') AND closed_at IS NOT NULL AND closed_at < datetime('now','-365 days')` 的 `trades`（`web/routes/db_manager.py::RETENTION_PREDICATE`；刻意不用 `COALESCE`，否则 SQLite 会放弃 `idx_trades_closed_at`）；然后 `VACUUM` |
| 导出 | `GET /api/db/export/{table}`（admin）→ CSV |
| 删行 | `DELETE /api/db/row/{table}/{row_id}`（admin） |
| 手工备份（推荐做法） | 先停服务，再复制 `data/binance_trader.db` **连同 `-wal` / `-shm`**（WAL 模式） |

### 9.4 账本核查（对账）

**没有**专门的对账 HTTP 端点，也没有自动对账任务。核查工具是只读脚本：

```bash
python scripts/audit_db.py [path\to\binance_trader.db]
```

它先把库**复制到临时目录**（连同 `-wal`/`-shm`）再以 `mode=ro` 打开（**绝不打开也不写生产库**），输出：账本恒等式的逐项计算与 `delta`、`trades` 的 action 分布与 `closed_at IS NULL` / `exit_price IS NULL` 计数、每张表的行数、索引清单。

**退出码语义**（`scripts/audit_db.py`）：

| 情况 | 输出 | 退出码 |
|---|---|---|
| identity 成立（`abs(delta) < 1e-6`） | `RESULT: IDENTITY HOLDS` | `0` |
| identity 不成立 | `RESULT: IDENTITY BROKEN` + `delta` | 非 0（drift） |
| **空库 / 没有 `system_config.sim_balance` 行** | `RESULT: FRESH DATABASE — NO LEDGER YET (no system_config.sim_balance row); nothing to compare, no drift.  Exit 0.` | `0` |
| 库文件不存在 | `ERROR: database not found: <path>` | 非 0 |

也就是说：**空库不是故障**——没有 `sim_balance` 行时脚本明确说"还没有账本可比"，以 0 退出；只有"有账本但恒等式不成立"才会以非 0 退出。所以可以把它放进 CI/定时任务里当漂移告警，而不必担心全新安装误报。

### 9.4.1 `ledger_reconciliation` 审计表

`ledger_reconciliation` 记录每一次**人工账本修复**：`balance_before` / `balance_after` / `delta` / `open_notional` / `realised_pnl` / `reason` / `detail` / `backup_path` / `created_by`。

- **全新安装**：表由 `db/database.py::LEDGER_RECONCILIATION_DDL` 直接建出。
- **历史库**：由 v4 迁移 `_migration_v4_ledger_reconciliation` **收敛**（旧库形状不同会重建并搬运可映射的列），所以升级后的库同样有这张表与 `idx_ledger_recon_at`。
- 当前生产库里有一行：`2026-09-29T23:09:22`，`delta = 19.878128`，`reason` 为 *P0 ledger drift repair: buy-side cost double charge + duplicate-open capital stranding; only sim_balance was adjusted*——即那次 −19.87 USDT 漂移的修复记录（修复备份：`data/binance_trader_prerepair_20260929_230922.db`）。

需要手工核对的恒等式：

```
10000 − Σ(open 行: quantity × entry_price) + Σ(close 行: pnl) == system_config.sim_balance
```

（等价写法：`SELECT SUM(quantity*entry_price) FROM trades WHERE status='open'` 与 `SELECT SUM(pnl) FROM trades WHERE status='closed'`——即 `scripts/audit_db.py` 与 `tests/test_ledger_invariant.py` 使用的谓词。）

**执行点**（生产代码里保证它成立的地方）：

1. `core/executor/executor.py:440-450` — 开仓时扣掉的现金 = `qty × fill_price + fee + slippage`，并且把 `trades.entry_price` **定义**为 `amount_usdt / qty`，于是 `qty × entry_price` 恰好等于被扣掉的现金；
2. `core/executor/executor.py::close_position()`（`executor.py:727`）— `invested_returned` 就是这份现金基准；`pnl` 只减去**卖出侧**的成本；
3. `app/main.py`（`_on_position_exit` / `_on_position_reduce` / 熔断动作）与 `core/risk/position_guard.py` — 平仓后统一 `atomic_adjust_balance(invested_returned + pnl)`；
4. `db/database.py:894` `atomic_adjust_balance` — `BEGIN IMMEDIATE`（`database.py:908`）+ 模块级 `balance_lock()`（`database.py:19-43`；按事件循环分锁，避免 `asyncio.Lock` 绑死循环），读改写在同一事务里。

**测试守护**：`tests/test_ledger_invariant.py`（恒等式；文件头记录了历史上 −19.87 USDT 漂移的两个根因：买入侧成本被收两次、重复开仓孤立资金）与 `tests/test_fees.py`（成本模型与恒等式协同）。

### 9.5 重启语义

| 事实 | 细节 |
|---|---|
| 持仓 | 从 `positions` 表恢复（`OrderExecutor.restore_positions`），含 basis、`stop_loss`、`entry_stop_loss` |
| 余额 | `system_config.sim_balance` 是权威；`app.state.balance` 只是缓存 |
| 挂单 | `pending_orders` 表恢复，撮合器每 ~5s 跑一次 |
| 自选列表 | 重启后从 `system_config.watchlist_symbols` 生效 |
| 手续费档位 | 持久化在 `system_config`，重启保留 |
| JWT 密钥 | 持久化在 `secrets.yaml`，重启后旧 token 仍有效 |
| 会话 | **在内存里**（`AuthManager._sessions`），重启后所有会话失效，需重新登录 |
| 策略 | 每次启动从 `strategies/*.yaml` 重新加载 |
| 优雅重启 | `POST /api/settings/restart`（admin） |

### 9.6 安全设施

| 设施 | 实现 |
|---|---|
| 认证 | bcrypt 哈希 + JWT(HS256) + 内存会话；cookie `bt_session`（`httponly`、`samesite=lax`） |
| 授权 | `AuthMiddleware` + handler 内 `_require_trader` / `_require_admin`；三层角色 `admin`/`trader`/`viewer`（`User.is_trader` 对 admin 也返回 True） |
| 登录限流 | 每 IP 5 分钟 10 次，超出 429 |
| 默认拒绝 | 未认证访问 `/api/*` 或 `/partials/*` → 401 JSON；其他路径 → 302 到 `/login` |
| XSS | Jinja `autoescape` 开启 |
| 策略条件注入 | **AST 白名单**求值器（禁用属性访问、下标、lambda、推导式、字符串字面量、未白名单函数；列名只能从 DataFrame 解析）；`evaluate_condition` 对非法输入返回全 False。见 `tests/test_condition_security.py` |
| 凭据 | 交易所/AI 密钥、风控阈值、重启接口均为 **admin-only**；`testnet` 只在显式提交时改变；空表单字段解释为"保持不变" |
| SSRF | `core/news/fetcher.py` 有防护 |
| 实盘前置 | `_execute_live` 只在 `config.mode == "live"` 时才可能被调用；`--mode sim`（默认）下实盘路径不可达 |

**没有实现的东西（不要误以为有）**：没有 CSRF token，防护只靠 `samesite=lax` cookie；没有 HTTPS/反向代理配置；没有审计日志表。

---

## 10. 测试与质量

### 10.1 怎么跑

```bash
# 全量
python -m pytest tests/ -q -p no:cacheprovider

# 单文件
python -m pytest tests/test_ledger_invariant.py -q -p no:cacheprovider

# 单个用例
python -m pytest "tests/test_fees.py::test_balance_identity_holds_after_a_round_trip" -q -p no:cacheprovider

# 跳过慢测
python -m pytest tests/ -q -m "not slow"
```

`pytest.ini`：`testpaths=tests`、`asyncio_mode=strict`、`markers=slow`、忽略 DeprecationWarning。**没有安装 `pytest-timeout`**，所以 `--timeout=` 会直接报参数错误。

**收集数：1073 项**（`python -m pytest tests/ --collect-only -q -p no:cacheprovider` 末行，本轮核验快照），**一次完整运行的结果**（本轮两次全绿之一；`216.61s` 那一次有 1 个失败，见下）：

```
1073 passed, 3 warnings in 271.07s (0:04:31)
```

本轮连续跑了 **2** 次全绿（`1073 passed` / `1073 passed`，退出码均 0；墙钟 `277 s` / `271 s`，本机同时有别的负载，故只报通过数不报秒数）；两次之间把 `tests/` 的两处守卫与四份文档改到一致；此前闭环审计轮连续跑过 **3** 次全绿（`221.15 s` / `220.94 s` / `218.41 s`，退出码均 0），其中最后一次是在**运行中的实盘进程重写了 `data/market/XRPUSDT/1m.parquet` 之后**（该文件 18:16:08 被改写，其 20 根 1m 窗口的成交额从 2 395 029.65 变为 **227.03** USDT —— 参与度上限从 23 950 掉到 2.27），套件仍为 `1070 passed`（当轮收集数）。

全部通过（0 failed）。`tests/` 里 68 个 `.py`（含 `__init__.py` 与 `conftest.py`，即 66 个 `test_*.py`）。核验期间仓库仍在被并行改动（本 README 只保证数字来自核验当次运行），但**没有任何已知失败用例**：曾经的 `tests/test_database.py::test_init_database_creates_tables`（它要求 v4 迁移**已经故意删除**的 `orders` 表存在）断言已修正，现在校验 `pending_orders`/`positions` 等真实表；`orders` 表在全仓库（代码、允许表清单、`reset-sim` 语句）里已无残留引用。曾被记录为"跟随 `test_ledger_invariant.py` 之后会失败"的 `test_money_concurrency.py::test_concurrent_opens_of_different_symbols_still_run_in_parallel` **在完整串行运行里通过**（`tests/conftest.py` 的 autouse fixture 在每个用例后还原进程级 `db.database.DB_PATH`，消除了跨文件状态污染）。

#### 干净克隆的测试结果（重要）

整个 `data/` 都在 `.gitignore` 里（`git ls-files data` 为空），所以**刚克隆下来直接跑全量测试并不是全绿**：回测等价性用例需要 `2026-05-25..2026-05-31` 的 **BTCUSDT + ETHUSDT** 缓存历史。核验时在两个**隔离副本**（当前 revision 的源码 + `.git`，但 `data/` 按下面两种状态准备）上各跑一次全量：

| 副本状态 | 实测结果 | 失败集合 |
|---|---|---|
| **完全无 `data/`**（真·干净克隆，本轮复测于当前 revision） | `8 failed, 1021 passed, 44 skipped in 250.94s` | 上述 7 个 `test_engine_parity_variants.py` 变体 + **1 个人为约定**：`tests/test_reaudit_fixes.py::test_evidence_index_reports_the_current_revision_and_a_current_chain`（副本没有 `.git`，见下） |
| 有 `data/`（models/backtest 等）但**无 `data/market/`** | 同上 7 个（此前该行记的 `8 failed` 含下面那个已消失的用例） | 上述 7 个。**该行曾把 `tests/test_microstructure.py::test_live_snapshot_is_optional_and_correct_when_available` 列为第 8 个失败**：那是旧版本在整跑里**真实抓取**币安盘口/成交时才会遇到的**实时行情时序**抖动（同文件的 docstring 记录：40 次单独实时抓取中有 3 次 `arrival_rate_hz == 0.0`，样例丢弃 3/90/100 笔；与执行顺序无关，README 此处原写的"顺序相关"是错的），不是"无数据"造成的。该用例现已改为**只用测试内构造的 Binance 形状快照**（不碰网络，`test_microstructure.py` 的 20 项全部如此）；本轮实测：把该文件单独放进一个**没有 `data/`** 的副本里运行 → `20 passed in 0.14s`，不再出现在任何无数据失败集合中 |

> 上表第一行的口径：把源码树整份复制到一个**没有 `data/`、也没有 `.git`** 的临时目录（复制 `alerts app config core db docs experimental scripts strategies tests tools web` + `pytest.ini`，`PYTHONPATH` 指向副本），全量一次得到 `8 failed, 1021 passed, 44 skipped in 250.94s`：7 个失败是需要 `data/market/` 历史的 `test_engine_parity_variants.py` 变体，第 8 个是证据索引守卫。**第 8 个不是环境产物、而是本轮刻意收紧的行为**：旧守卫把 `git rev-parse HEAD` 的失败吞掉后继续断言固定哈希表，所以副本里它"通过"；新守卫（见 `tests/test_reaudit_fixes.py::test_evidence_index_reports_the_current_revision_and_a_current_chain`）要求运行期重算 HEAD 与提交链，`git` 在 PATH 上但副本没有 `.git` 时**判定为失败**（只有 `git` 完全未安装才 skip，见该用例 docstring），因此干净克隆的失败集合从 7 变成 8。在仓库内该用例单跑为 `1 passed`。"有 `data/` 但无 `data/market/`"一列未在本轮重跑，沿用同一失败集合（多出的 `data/models` 等不改变这 7 个用例的输入）。

失败全部是"没有缓存历史"这一类，错误文本是那段唯一的 canonical 文案 `NO_MARKET_DATA_MESSAGE`（`core/backtest/signal_matrix.py:22-26`，legacy / hybrid / signal-matrix 三条路径共用同一句）：`No historical market data found for the selected symbols and date range. Download candles first with ...`；底层也会出现 `No timestamps found in data feeder`（见 `core/backtest/signal_matrix.py` 里对它的注释）：

- `tests/test_engine_parity_variants.py` 的 **7** 个真实数据变体：`test_parity_multi_timeframe_strategies`、`test_parity_multi_timeframe_single_strategy`、`test_parity_multi_timeframe_with_isolation`、`test_parity_shared_indicator_config_different_timeframes`、`test_parity_risk_exit_overrides`、`test_parity_risk_only_exits`、`test_parity_per_strategy_isolation`——它们用 `DATE_START/DATE_END = 2026-05-25/2026-05-31` 与 `SYMBOLS = ["BTCUSDT", "ETHUSDT"]`；
- `tests/test_hybrid_equivalence.py::test_signal_matrix_summary_statistics` 在无缓存时走 **skip** 路径（`tests/test_hybrid_equivalence.py:88`，同一句 `NO_MARKET_DATA_MESSAGE`），因此它在上面两次整跑里都不是失败；另 1 个 skipped 是 `tests/test_hybrid_equivalence.py:179`（"No trades in either engine -- not enough data variation"）。

同一文件里不受影响的用例：`test_engine_parity_variants.py::test_parity_uses_synthetic_market_without_cached_data`（自己往临时目录写合成 parquet）与 `::test_reduce_conditions_route_to_legacy`，以及 `test_hybrid_equivalence.py::test_no_lookahead_bias`。下好历史后重跑即全绿（**1073 passed**）：

```bash
python scripts/download_history.py --symbols BTCUSDT,ETHUSDT --intervals 1h --start 2026-05-25 --end 2026-05-31
```

> 因此：**上面的"全绿"数字都附带隐含前提——本机 `data/market/` 里已经有对应区间的缓存**。CI 或新机器上请在跑测试前先下历史，或接受那 7 个（真·干净克隆）/ 7 个（有 `data/` 但无 `data/market/`，本轮起不再有第 8 个）失败。

### 10.2 重点回归套件

| 套件 | 数量 | 守护什么 |
|---|---|---|
| `tests/test_ledger_invariant.py` | 15 | **账本恒等式**：`10000 − Σ(qty×entry_price) + Σpnl == sim_balance`，覆盖开仓/平仓/减仓/重启恢复/重复开仓 |
| `tests/test_fees.py` | 20 | sim 成本模型（档位、BNB 折扣、限价 vs 市价、滑点）+ 与恒等式协同 |
| `tests/test_cost_model.py` | 26 | 回测价差解析顺序（override → live → default）、实时深度推导、缓存与失败不缓存 |
| `tests/test_condition_security.py` | 25 | **条件表达式 AST 白名单**：19 个恶意载荷必须被拒；已知的 `close.to_csv(...)` 任意文件写入必须不可能 |
| `tests/test_hybrid_equivalence.py` | 3 | legacy 与 hybrid **逐笔等价**门禁 |
| `tests/test_engine_parity_variants.py` | 9 | 单/多时间框架、GA 形态、策略隔离下的等价性变体 |
| `tests/test_engine_router.py` | 5 | `auto`/`hybrid`/`legacy` 选择逻辑 |
| `tests/test_db_v4.py` | 29 | v4 迁移链（恒等重放、`trades` 重建、死表删除、`closed_at` 回填） |
| `tests/test_web_security.py` | 12 | 角色矩阵：viewer 不能做任何写操作；设置持久化不碰仓库内 YAML |
| `tests/test_market_api.py` | 61 | 交易页 API 契约（市场/账户/订单/历史） |
| `tests/test_universe_api.py` | 42 | 自选列表、`validate_watchlist`、币种宇宙 / 行情页 API |
| `tests/test_screener.py` | 59 | 启发式筛查（缺失数据为 None、flag 带证据） |
| `tests/test_executor_live.py` | 12 | 实盘下单路径：step_size 取整、LOT_SIZE/NOTIONAL 拒单、重复 clientOrderId、记账在重试循环之外 |
| `tests/test_strategy_live_path.py` | 5 | 实时信号链路端到端（防回归 `ml_weight` 那类 NameError） |
| `tests/test_trade_book.py` | 7 | 两引擎共享的平仓记账 |
| `tests/test_decoupling.py` | 53 | 解耦不变量（自选列表、GA 币种、spread、迁移） |
| `tests/test_ga_symbols.py` | 21 | GA 走真实自选列表而非硬编码 5 币种 |
| `tests/test_security_hardening.py` | 14 | 安全审计回归：存储型/客户端 XSS、凭据回显、上游扇出与限流、会话加固、**周期取自注册表而非硬编码 `"1h"`** |
| `tests/test_money_concurrency.py` | 15 | 资金并发与 admin 路由：同币种并发平仓/开仓只记一行、`/api/db/cleanup` 删除已实现 PnL 时必须同步调整余额、`reset-sim` 原子清空挂单 |
| `tests/test_ai_panel_wiring.py` | 31 | AI 面板接线（建议卡片的生产者→消费者链路） |

### 10.3 其它质量手段

```bash
python -m compileall -q app core web db scripts alerts     # 语法完整性
```

- 路由基线：`docs/overhaul/route-baseline.json`（拆分时 97 条，现为 118 条；与 `web/routes/*` + `web/ws/*` 的实测 118 条一致）。
- `docs/core-algorithms/00-ERRATA.md` 汇总算法文档与代码的 **22** 处差异（数一下表里的数据行即可）——**文档看公式前先看这份勘误**。

---

## 11. 已知限制与诚实说明

### 11.1 环境

- **主网交易 API 不可达。** `api.binance.com` / `api1-4` 从本机**超时**。所以：真实资金交易在本机不可能完成；`--mode live` 实际打到的是 **testnet**（由 `binance.testnet` 决定）。行情是真实主网数据，但**下单不是**。
- **手续费档位是手动设置。** 本机读不到真实账户的 30 天交易量与 BNB 持仓，所以 VIP 档位永远由用户选择（`POST /api/fee/tier`），程序不会自己判断。
- **`backtest` 模式不是回测。** `--mode backtest` 不进入下单分支，只是把模式字符串传下去（`config.mode = "backtest"` 时 `_on_order_request` 两个分支都不匹配）。

### 11.2 模型与启发式的边界

- **sim 滑点是模型，不是事实。** 固定 `slippage_bps` + 每 symbol 固定半价差，不随订单大小、盘口深度、波动率变化。大单的真实滑点会明显高于这个模型。
- **回测价差在无覆盖且盘口不可达时用常量兜底**（0.03%），与 sim 盘的兜底（0.02%）**不一致**——这是已知的口径分歧。
- **代币检测是启发式的，不是链上审计。** `core/market_data/screener.py` 只读取交易所公开行情（24h、盘口、K 线），不读合约、持仓分布、mint 权限、转账税、代理升级位、LP 锁。它**无法**发现蜜罐、rug pull、冻结/黑名单函数、隐藏增发。干净的评分只意味着"这个交易对的**交易所市场**看起来正常"，**不意味着这个币安全**。两个接口都返回 `DISCLAIMER`，页面也显著标注。
- **仓位管理不是真 Kelly。** `core/risk/position_sizer.py` 是固定比例法：`capital_pool × max(position_size_pct, 0.1)%`，再被 `max_position_size_pct` 截断。文档 `04-position-sizing-kelly.md` 里的 Kelly 公式**只在回测的 Kelly-lite 分支**部分体现；`docs/core-algorithms/` 是研究性描述，**不要当成实现说明**。
- **ML 不是信号的主导者。** `core/ml/` 只保留真正被调用的部分：`MLPredictor.predict` / `train_model`、`TFTTrainer` 与 **`PatchTSTTrainer` 都是 LIVE 代码**（`core/ml/patchtst_trainer.py`，由回测引擎在 `backtest.engine_mode=legacy` + `ml_engine=patchtst` 时调用：`core/backtest/engine.py:327-329`）以及标签函数。特征存储、集成投票、增量重训、回归头等**已移入 `experimental/ml/`，不在交易链路上**（见该目录 README，其中也把 `TFTTrainer`/`PatchTSTTrainer` 标为仍在使用）。回测的 hybrid 引擎完全不支持 ML。

### 11.3 残余耦合与陈旧之处（当前审计清单）

- **hybrid 与 fitness 的价差分发路径。** hybrid 分支为了让子模块读到本次推导出的价差，会**临时改写 `config.backtest_spread_pct`** 再在 `finally` 里还原（`core/backtest/engine.py:179-204`）。这是运行期对配置对象的副作用，属于已知的将就做法。
- ~~**`web/routes/db_manager.py` 与 `pages.py` 的允许表清单里还有 `orders`**，而 v4 迁移已删除该表；`settings.py` 的 `reset-sim` 里还执行 `DELETE FROM orders`。~~ **已修复（本条曾为真，现为历史）**：三处允许表清单均不含 `orders`（`db_manager.py:50/223/252`、`pages.py:296`），`web/routes/settings.py:229-230` 只 `DELETE FROM trades` / `DELETE FROM positions`。
- **`periods`/币种字面量在模板 `<option>` 等处仍有重复**；后端已收敛：`overrides`/默认价差只由 `core/backtest/cost_model.py` 解析，回测兜底币种取自 `core/market_data/universe.py::DEFAULT_WATCHLIST`，GA 不再自带价差表。`docs/overhaul/REFACTOR_AUDIT.md` 记录的 17 份重复 5 币种列表正在逐处收敛。
- **`trades.timeframe` 的写入点**若来自不传 `timeframe` 的调用方，会落到默认 `"1h"`。
- **授权是 handler 内联的**，不是 FastAPI 依赖；`web/deps.py` 与相关 handler 上有 `TODO(authz)`。
- **`ai_suggestions` / AI 面板**正在被清理通道重接线，本 README 只描述路由与页面的存在，**不断言建议卡片在每种路径下都有数据**。
- **`experimental/` 里的代码不会被任何生产模块导入**，删除它不影响交易行为。
- **`Nonexistent` 之类的一次性产物**：仓库根有一个 `nonexistent.db`（0 字节级），属于历史遗留。
- `docs/HANDOVER.md`、`docs/overhaul/REFACTOR_AUDIT.md` 描述的是**本轮大改之前的**状态（例如"实时链路断裂""AI 面板空转"），其中的问题**大部分已修**；查阅时把它们当历史记录，**以代码为准**。

### 11.4 未做的事

- 没有 CSRF token；没有 HTTPS；没有**谁改了什么**这类操作审计日志（`ledger_reconciliation` 只记录账本修复，见 [§9.4.1](#941-ledger_reconciliation-审计表)）；没有自动对账任务；没有多用户级别的自选列表（自选是全局的）；派生的交互式 OpenAPI 文档页被显式关掉（`docs_url=None, redoc_url=None`，所以 `/docs`、`/redoc` 是 404——但 `/openapi.json` 仍然可访问）。
- **干净克隆里没有历史数据，也没有策略。** `data/` 与 `strategies/` 的全部内容都在 `.gitignore` 里，`git ls-files strategies/` 输出为空：
  - **策略数为 0**：`GET /api/strategy-monitor` → `{"strategies": [], "active_count": 0, "total_count": 0}`；`POST /api/backtest/run` 用任何内置名字（如 `rsi_reversal`）都会在引擎层返回 `Strategy 'rsi_reversal' not found: Strategy file not found: …`。必须先自己写 `strategies/*.yaml`（schema 见 `core/strategy/loader.py::StrategyConfig`）。仓库**不附带示例策略**。
  - **回测/等价性测试无数据可用**：干净克隆跑全量测试会得到 `8 failed, 1021 passed, 44 skipped`（7 个真实数据变体 + 证据索引守卫，后者按设计在没有 `.git` 的副本里失败；详见 [§10.1](#101-怎么跑)），历史要自己用 `scripts/download_history.py` 下。

---

## 12. 故障排查

### ① 页面能开、但 K 线/行情全空，日志刷 `REST kline fetch failed`

**原因**：行情主机不可达或配置被改回了 `api.binance.com`。

```bash
# 先确认主机可达（应 1–2 秒返回 {"serverTime":...}）
python -c "import urllib.request,json;print(json.load(urllib.request.urlopen('https://data-api.binance.vision/api/v3/ping',timeout=8)))"
```

- 检查 `config/config.yaml` 的 `binance.market_data_host` / `market_stream_host`，确认是 `.vision` 域名。
- 再看日志里有没有 `WebSocket connecting: N streams`；没有 `Kline #1: ...` 说明流没通，此时**REST 兜底轮询**每 30s 会补，日志会出现 `REST poll: N symbol/interval pairs`。
- 币种不在 `.vision` 上（拼错/已下架）也会空：用 `GET /api/market/symbols?q=XXX` 确认它在 496 个 USDT/TRADING 对里。

### ② 打开任何页面被踢回 `/login`，接口返回 401 `{"error":"Unauthorized"}`

**原因**：会话失效。会话**存在内存里**，重启进程即全部失效；`session_hours`（默认 24h）到期也会失效。

- 重新登录即可。**注意**：重启后 JWT 仍有效（`secrets.yaml` 里持久化了密钥），但 **cookie 里的会话 token 已不在内存**，所以浏览器仍然要重新登录。
- 如果是 API 客户端：用登录返回的 `token` 加 `Authorization: Bearer <token>`，这条路径不依赖内存会话。

### ③ 下单被拒

按错误码/提示分别处理：

| 现象 | 原因 | 处理 |
|---|---|---|
| `403 {"error":"Forbidden"}` | 当前角色是 `viewer` | 用 `admin`/`trader` 账号；或让 admin 在 `/users` 提升角色 |
| `Position already open for X` | 风控去重（`check_signal` 第 6 步），或 executor 的重复开仓保护 | 先平掉该 symbol，或换一个 |
| `Max open trades N reached` | `hard_limits.max_open_trades`（文件值 15） | 平掉部分持仓或调高该值 |
| `Total exposure X% exceeds limit` | `max_total_exposure_pct`（80%） | 降低敞口或调高阈值 |
| `Circuit breaker tripped: ...` | 日/周回撤、日亏损或连亏达标 | `POST /api/circuit-breaker/reset`（trader），或等下一小时的自动日/周重置 |
| `order_below_lot_size` / `quantity ... rounds to 0` | 数量按 `step_size` 向下取整后为 0 | 加大下单金额 |
| `order_below_min_notional` | 名义金额小于该 symbol 的 `minNotional` | 加大下单金额 |
| `Insufficient balance for position sizing` | 可用余额不足 | 检查 `/api/account` 的 `available`（= 余额 − 冻结） |
| 限价单挂了但一直不成交 | 撮合器每 ~5s 按价格穿越判定 | 正常；用 `POST /api/orders/{id}/cancel` 撤单 |

### ④ 端口被占用

```
ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8899)
```

```powershell
# 找出占用者
Get-NetTCPConnection -LocalPort 8899 -State Listen | Select-Object OwningProcess
Get-Process -Id <PID> | Select-Object ProcessName,Path
# 或者直接换端口
python -m app.main --mode sim --port 8900
```
```bash
lsof -i :8899        # Linux/macOS
python -m app.main --mode sim --port 8900
```

### ⑤ `database is locked` / `sqlite3.OperationalError: database is locked`

**原因**：WAL 模式 + 多短连接（executor / 撮合器 / alerts / web）同时写；或**同一个 DB 被两个进程打开**（比如两份服务、或 pytest 与运行中的服务共用 DB）。

- 确认只有一个进程在写这个 DB（`--db` 指向别处时最容易踩）。
- 不要在网络盘/NFS 上跑 SQLite。
- 出现后重试通常能过；持续锁死就停服务 → 复制备份（含 `-wal`/`-shm`）→ `POST /api/db/optimize`。
- 冒烟测试请用 `--db` / `--data-dir` / `--config-dir` 指到临时目录，不要碰 `data/binance_trader.db`。

### ⑥ 启动后一段时间端口不通 / 没有密码

- ML 训练在 uvicorn 之前**串行**跑（`app/main.py:488-498`，`server.serve()` 在 `main.py:580`），端口要等训练完才监听。日志停在没有 `Web UI starting at` 就是还在训练。
- 没看到 `DEFAULT ADMIN`：说明 `users` 表非空（已经有账号了），不会重复创建。

---

## 13. 后续路线与许可

### 下一步（按价值排序）

1. **配置/审计卫生**：~~清掉 `db_manager.py`、`pages.py`、`settings.py` 里对已删除 `orders` 表的引用；修 `tests/test_database.py` 的陈旧断言。~~ **已完成**：三处 `orders` 引用与陈旧断言均已清理，套件 **1073 passed / 0 failed**（见 [§10.1](#101-怎么跑)）。
2. **统一价差兜底**：让回测的 `default_spread_pct`（0.03）与 sim 盘的 `default`（0.02）取同一个常量，消除口径分歧。
3. **拆掉 hybrid 的价差临时改写**：给 `run_hybrid` 传一个显式的 `spread_pct` 参数，不再副作用式改 `config`。
4. **授权依赖化**：把 `_require_trader` / `_require_admin` 从 handler 内联改成 FastAPI 依赖（`HTTPException(403)`），并补 CSRF token。
5. **真 Kelly / 波动率自适应仓位**：把研究文档里的公式落地并加回归测试（当前只有固定比例 + 截断）。
6. **可观测性**：加结构化日志 sink 与一个只读的 `/api/ledger/reconcile`，把恒等式核对变成一键操作。
7. **审计日志表**：记录谁在什么时候改了策略/风控/凭据。

### 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/overhaul/CHANGELOG.md`](docs/overhaul/CHANGELOG.md) | **详细变更史**：本轮修了什么、验收到什么程度、还留下什么（按 1.0 / 1.0.1 / 1.0.2 … 分节） |
| [`docs/HANDOVER.md`](docs/HANDOVER.md) | 接手评估报告：架构地图、接手时的问题清单与验证记录（**描述的是改动前的状态**） |
| [`docs/overhaul/PLAN.md`](docs/overhaul/PLAN.md) | 长盘修复计划 S0–S7、不可动摇的约束、3 轮审计（A1/A2/A3）的发现与处置、进度日志 |
| [`docs/overhaul/REFACTOR_AUDIT.md`](docs/overhaul/REFACTOR_AUDIT.md) | 四路只读审计：非可替换化"焊接"清单、数据库冗余与真 bug、冗余功能、清理计划（**同上，改动前快照**） |
| [`docs/overhaul/TRADE_PAGE_API.md`](docs/overhaul/TRADE_PAGE_API.md) | `/trade` 现货页的接口契约（冻结版），含 sim 成本模型 §五之二 |
| [`docs/overhaul/MARKET_PAGES_API.md`](docs/overhaul/MARKET_PAGES_API.md) | 行情/币种/数据/代币检测页的接口契约（冻结版），含 §0 主机可达性前置事实 |
| [`docs/overhaul/WEB_SPLIT.md`](docs/overhaul/WEB_SPLIT.md) | `web/server.py` 从 2366 行（编辑器口径 2588 行）单文件 → `web/routes/*` 的拆分记录（**历史快照**；现为 **115 行**，路由基线 **118 条**）、保留的不变量、已知差异 |
| [`docs/core-algorithms/*.md`](docs/core-algorithms/) | 9 篇算法说明（信号融合、风控管线、熔断、仓位、hybrid 回测、GA、DSR、三重障碍 ML、追踪止损） |
| [`docs/core-algorithms/00-ERRATA.md`](docs/core-algorithms/00-ERRATA.md) | **算法文档与代码的 22 处差异勘误**——看公式前先读它 |
| [`docs/development-roadmap.md`](docs/development-roadmap.md) | 开发路线图（后续方向的规划文档，**规划≠现状**，与代码冲突时以代码为准） |
| [`docs/audit/`](docs/audit/) | 15 轮历史审计分报告（R1–R15）与严重度分级 |
| [`docs/superpowers/`](docs/superpowers/) | 设计规格与实施计划，项目唯一的设计史来源 |
| [`experimental/ml/README.md`](experimental/ml/README.md) | 被移出生产链路的 ML 代码清单、为什么它们是死码、如何接回来 |
| [`docs/overhaul/route-baseline.json`](docs/overhaul/route-baseline.json) | Web 层拆分时的路由基线（拆分时 97 条；**当前 118 条**，见 [§7.2](#72-路由总数)），用于路由奇偶校验 |
| [`README_EN.md`](README_EN.md) | **陈旧的英文快照**：仍在描述旧目录结构（`binance_trader/…`）与拆分前的架构，**没有同步本 README 的任何本轮修订**。以中文 [`README.md`](README.md) 为准 |

### 许可与致谢

本项目未附带显式开源许可证文件（`LICENSE` 缺失）；如需使用/分发，请先与仓库所有者 `STCloudLake` 确认授权方式。

依赖与数据来源：**[python-binance](https://github.com/sammchardy/python-binance)**（交易所客户端）、**[FastAPI](https://fastapi.tiangolo.com/)** + **[uvicorn](https://www.uvicorn.org/)**（Web）、**[aiosqlite](https://github.com/omnilib/aiosqlite)**（数据库）、**[ECharts](https://echarts.apache.org/)** + **[HTMX](https://htmx.org/)** + **[Tailwind CSS](https://tailwindcss.com/)**（前端）、**[TA-Lib](https://github.com/TA-Lib/ta-lib-python)** / **[LightGBM](https://github.com/microsoft/LightGBM)** / **[XGBoost](https://github.com/dmlc/xgboost)** / **[PyTorch](https://pytorch.org/)**（指标与 ML）、**[loguru](https://github.com/Delgan/loguru)**（日志）、**[DeepSeek](https://www.deepseek.com/)**（LLM）。行情与交易数据来自 **Binance 公开 API**（`data-api.binance.vision` / `testnet.binance.vision`）。

> **风险提示**：加密货币交易存在本金全部损失的风险。本系统默认运行模拟盘；任何切换到真实资金交易的决定、参数设置与后果都由使用者自行承担。`--mode live` 在当前环境下只会打到 Binance **testnet**，但这并不构成对任何未来配置变更的安全保证。
