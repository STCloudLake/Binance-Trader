# Binance Trader

面向币安现货的 Python 3.12 自动化交易系统。asyncio 事件总线把 **行情 → 策略信号 → 风控 → 下单 → 持仓守护** 串成一条可观测流水线，配套 FastAPI + aiosqlite + ECharts 的 Web 控制台。模拟盘带真实成本模型（手续费分档 + 半价差 + 滑点），账本恒等式有专门的回归测试守护；回测有 legacy / hybrid 两套引擎，共用同一个评估内核。

**架构一段话**：`core/market_data/provider.py` 从 `data-api.binance.vision`（REST）与 `data-stream.binance.vision`（WS）取行情，落 `data/market/<SYM>/<tf>.parquet` 并抛 `MARKET_KLINE`；`app/event_bus.py` 的单个消费协程把事件分发给 `StrategyEngine`（可选 `MLPredictor` / `NewsAnalyzer` / `AlertManager`）；**所有**信号评估走唯一的 `core/strategy/evaluation_kernel.py`（实时与两套回测共用同一内核与同一组阈值），产出 `STRATEGY_SIGNAL`；`core/risk/manager.py::check_signal()` 按 熔断 → 总敞口 → 仓位规模 → 杠杆 → 止损 → 同币去重 → 最大笔数 **七步**过滤后发 `ORDER_REQUEST`；`core/executor/executor.py` 按 `--mode` 走 sim（本地成交，成本模型改写成交价）或 live（testnet 真实下单）；`core/risk/position_guard.py` 以 15s 轮询执行止损/追踪/紧急平仓，`db/database.py::atomic_adjust_balance()` 是账本**唯一**写入点。

```
data-api.binance.vision (REST) / data-stream.binance.vision (WS) → MarketDataProvider
  ─ MARKET_KLINE → EventBus → StrategyEngine / ML / News → evaluation_kernel
  ─ STRATEGY_SIGNAL → RiskManager (7 步) → OrderExecutor (sim | live)
  ─ ORDER_UPDATE → PositionGuard + SQLite + Web → POSITION_EXIT → atomic_adjust_balance()
```

> **状态**：`VERSION` **2.0.1** · Python **3.12**（实测 3.12.10）· Windows / Linux · 默认只监听 `127.0.0.1:8899`
> **测试**：**1318** 项收集（`python -m pytest tests/ --collect-only -q`）；全量基线要求 `1318 passed, 0 failed`。见 [§5.1](#51-测试)。
> 本 README 只写**当前代码事实**，每个数字旁给出发它的命令；与代码冲突时以代码为准。

---

## 目录

1. [环境约束（必读）](#1-环境约束必读)
2. [安装与启动](#2-安装与启动)
3. [Web 控制台](#3-web-控制台)
4. [算法层](#4-算法层)
5. [运维](#5-运维)
6. [已知限制与残余](#6-已知限制与残余)
7. [文档索引](#7-文档索引)

---

## 1. 环境约束（必读）

### 1.1 主机可达性（实测 2026-09-30，`Invoke-WebRequest -TimeoutSec 8` 直连）

| 主机 | 结果 | 用途 |
|---|---|---|
| `https://data-api.binance.vision` | ✅ HTTP 200 | **所有公开行情**：klines / ticker24h / depth / trades / exchangeInfo |
| `wss://data-stream.binance.vision` | 配置目标（未做 socket 握手实测） | **实时 K 线流**；`provider.py` 把 socket 工厂指向它 |
| `https://testnet.binance.vision` | ✅ HTTP 200 | **下单 / 账户 / 余额**——当前唯一的交易端点 |
| `https://api.binance.com` | ❌ 8s 超时 | **不可用**（主网交易域名一律不可达） |

复现（三条命令只换 URL，结果分别为 200 / 200 / 超时）：

```powershell
Invoke-WebRequest -Uri https://data-api.binance.vision/api/v3/ping -TimeoutSec 8 -UseBasicParsing
Invoke-WebRequest -Uri https://testnet.binance.vision/api/v3/ping   -TimeoutSec 8 -UseBasicParsing
Invoke-WebRequest -Uri https://api.binance.com/api/v3/ping          -TimeoutSec 8 -UseBasicParsing
```

**为什么这是硬约束**：`api.binance.com` 的调用会挂到超时，把请求处理器一起拖死。行情**必须**走 `config.binance.market_data_host`（默认 `https://data-api.binance.vision`），由 `core/market_data/data_client.py::MarketDataClient` 统管（15s 硬超时、连接池复用、失败抛 `MarketDataError`）。这个镜像给的是**真实主网数据**；testnet 只有少量测试对、历史是合成的，用它做行情会看到不全的币种与缺失的 K 线。

### 1.2 绝不要用裸 `AsyncClient.create()`

```python
# ❌ python-binance 默认打 api.binance.com，本机必然超时
client = await AsyncClient.create()
# ✅ 交易/账户客户端必须显式传 testnet（见 web/routes/market.py::_configured_client）
client = await AsyncClient.create(api_key=..., api_secret=..., testnet=config.binance_testnet)
```

所有 K 线读取都走 `data_client`（绑定行情镜像）。`MarketDataProvider.start()` 里那次 `AsyncClient.create(...)` 是**交易**客户端，失败只记 warning 并以离线模式继续。

### 1.3 实时流的 socket URL 被刻意覆写，Web 只绑本地

python-binance 的 `BinanceSocketManager` 硬编码 `wss://stream.binance.com:9443/`（该主机不可用），`provider.py` 在创建 `bsm` 后覆写 `bsm._get_stream_url` 指向 `config.binance.market_stream_host`——**新增任何 socket 用法时不要绕过这一步**。另外 `app/main.py` 里是 `uvicorn.Config(host="127.0.0.1", ...)`，**没有 `--host` 参数**；要从别的机器访问请用 SSH 端口转发或反向代理 + HTTPS，不要把 host 改成 `0.0.0.0` 暴露到公网。

---

## 2. 安装与启动

### 2.1 依赖与凭据

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell；Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt     # TA-Lib 需要系统级预编译库，见该库文档
copy config\secrets.yaml.example config\secrets.yaml   # Linux: cp
```

`requirements.txt` 含运行 + 测试依赖；`python-multipart` 是必需的（starlette 无条件导入它，缺了 FastAPI 起不来）。`config/secrets.yaml` 已 gitignore，可写 `binance.api_key` / `api_secret`、`deepseek.api_key`；`auth.jwt_secret` 留空时启动会生成随机密钥并写回该文件（POSIX 上 `chmod 600`），因此重启后旧 token 仍有效。也可以用环境变量（**优先级高于 YAML**）：`BINANCE_API_KEY`、`BINANCE_API_SECRET`、`DEEPSEEK_API_KEY`、`JWT_SECRET`。

**你必须自己提供策略 YAML**——仓库**没有跟踪任何策略文件**（`git ls-files strategies/` 输出为空；磁盘上只有被 gitignore 的 `strategies/ga_champion_*.yaml`，由 GA 生成）。干净克隆的**策略数是 0**：`GET /api/strategy-monitor` 返回空列表，`POST /api/backtest/run` 用任何名字都会得到 `Strategy '<name>' not found: Strategy file not found: ...`。策略 schema 就是 `core/strategy/loader.py::StrategyConfig` 的 pydantic 字段；自己写 YAML 放进 `strategies/`，或让 GA / 策略生命周期生成。

### 2.2 数据与数据库

| 路径 | 内容 |
|---|---|
| `data/binance_trader.db` | SQLite（WAL，v4 schema）。`--db` / `--data-dir` 可改 |
| `data/market/<SYM>/<tf>.parquet` | K 线缓存（`index = close_time`(UTC)，列 `open, high, low, close, volume`，P6-B 后另加 `quote_volume` / `trade_count`）；历史用 `scripts/download_history.py` 下 |
| `data/models/`、`data/backtest/`、`data/ga_jobs/`、`data/ga_strategies/` | ML 模型产物、回测结果、GA 任务与冠军 |
| `config/config.yaml`、`config/risk_params.yaml` | 运行配置与风控阈值（深合并，`risk_params.yaml` 覆盖代码类默认值） |
| `system_config` 表 | **数据库侧**设置：自选列表 `watchlist_symbols`、手续费档位 |

`data/` 整体被 gitignore，一个文件都没跟踪：干净克隆里既没有历史数据也没有模型。

### 2.3 启动

```bash
python -m app.main --mode sim
```

浏览器打开 <http://127.0.0.1:8899> → 未登录 302 到 `/login`。**首次启动**若 `users` 表为空，会自动创建 `admin` 并生成随机密码，密码**只打印到 stderr**（不写日志、bcrypt 存库、明文不可恢复），请立刻在 `/settings` 或用户管理页改掉；`users` 表非空时不会重建。ML 训练在 `uvicorn` 之前**串行**执行，所以启动后可能有一段时间端口尚未监听（训练抛异常只记 warning，不阻塞启动）。

### 2.4 CLI 参数（`python -m app.main --help` 全量）

```
usage: main.py [-h] [--mode {sim,live,backtest}] [--port PORT] [--db DB]
               [--data-dir DATA_DIR] [--config-dir CONFIG_DIR]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode {sim,live,backtest}` | `sim` | 模式**只来自 CLI**（`config.yaml` 没有 `mode:` 键）。语义见 `executor.py::_on_order_request`：`sim` 本地成交（完整成本模型，不连交易所）；`live` 用 testnet 客户端真实下单；**`backtest` 不是回测**——不进入任何下单分支、订单被静默丢弃，真正的回测走 `/backtest` 页或 `POST /api/backtest/run` |
| `--port N` | `8899` | Web 端口 |
| `--db PATH` | `<project>/data/binance_trader.db` | SQLite 路径；优先级高于 `--data-dir`（后者默认 `<project>/data`） |
| `--config-dir DIR` | `<project>/config` | 设置/密钥持久化目录。**冒烟测试请用它 + `--db` 指到临时目录，不要碰生产库** |

### 2.5 关键配置项

`app/config.py::_load` 依次深合并 `config/config.yaml` → `config/risk_params.yaml` → `config/secrets.yaml`，环境变量最高。

| 键 | 默认 | 含义 |
|---|---|---|
| `web_port` / `language` | `8899` / `zh` | Web 端口与 UI 语言（`zh`/`en`） |
| `binance.testnet` | `true` | **下单/账户**客户端是否打 testnet。**不控制行情** |
| `binance.market_data_host` / `market_stream_host` | `.vision` 两个域名 | 行情 REST / WS 主机 |
| `ai.mode` / `ai.model` | `full_auto` / `deepseek-v4-flash` | `semi_auto` / `full_auto`；LLM 模型名 |
| `signal_weights.*` | `indicator 0.5 / ml 0.3 / news 0.2` | 信号融合权重 |
| `backtest.engine_mode` / `ml_enabled` | `auto` / `false` | 引擎选择（非法值回退 `auto`）；false 时会就地关掉策略的 `ml_config.enabled` |
| `sim.cost_model.*` | `enabled: true`、`fee_tier: VIP0`、`slippage_bps: 2` | 模拟盘成本模型（见 §4.2） |
| `risk.*` | 全部 `enabled: false` | 波动率目标化 / 流动性，默认惰性（见 §4.3） |
| `experimental.*` | 全部 `false` | P4 实验性能力开关（engine 缝 / regime / pairs / meta / microstructure，见 §4.4） |

`config/risk_params.yaml` 给出 `hard_limits`（日/周回撤、日亏损、最大敞口、最大笔数、杠杆、追踪止损、紧急止损等）与 `soft_params`（仓位比例、止损、杠杆、三档止盈）；**以文件里的值为准**，本文不复制这张表。自选列表**不在 YAML 里**，存在数据库 `system_config.watchlist_symbols`（兜底 `BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT`，上限 30），改动**下次重启生效**（`POST /api/market/watchlist` 的响应带 `restart_required: true`）；手续费档位同理——本机读不到真实账户的 30 天交易量与 BNB 持仓，只能手动选择并持久化。

---

## 3. Web 控制台

### 3.1 页面（与 `docs/overhaul/route-baseline.json` 的 GET 页面路由逐条对应）

| 路径 | 用途 | 最低角色 |
|---|---|---|
| `/`、`/dashboard`、`/health` | 302 → `/trade`；`/health` 是 JSON 健康检查（认证后追加熔断/持仓/策略/mode） | 登录（`/health` 公开） |
| `/trade` | **现货交易页**：行情条、订单簿 + 最新成交、ECharts K 线、买/卖下单（市价/限价、比例快捷键、止损%）、账户卡片、当前委托/持仓/成交历史/订单历史 | 登录 |
| `/market` | 全币种行情：搜索 / 排序 / 分页、自选管理、本地数据标记 | 登录 |
| `/coin/{symbol}` | 币种信息：交易规则、24h、盘口、周期表现、启发式风险评分与 flags、BTC 相关性 | 登录 |
| `/data` | 全市场数据总览：涨跌榜、成交额榜、最活跃、波动率榜、价差榜 | 登录 |
| `/audit` | 代币检测：启发式风险筛查（页面显著标注**非链上合约审计**） | 登录 |
| `/strategies` | 策略监控与 CRUD、信号权重、风控参数展示、AI 策略推荐 | 登录（写操作 trader） |
| `/backtest` | 回测配置 / 运行 / 进度 / 结果 / 历史列表 | **trader**（否则 302 → `/dashboard`） |
| `/ai` | AI 面板：建议卡片、市场评估、AI 心跳、手动咨询、策略生命周期事件 | 登录（审批 trader） |
| `/alerts` | 预警中心：告警列表、筛选、规则开关 | 登录 |
| `/settings` | AI 模式与参数、Binance 凭据与 testnet 开关、风控阈值、熔断状态与重置、重启服务 | **trader**（否则 302 → `/trade`）；凭据/风控/重启写接口 **admin** |
| `/db-manager`、`/users` | 表浏览/行删除/备份恢复/优化清理/CSV 导出；用户管理 | **admin**（否则 302 → `/trade`） |
| `/manual`、`/manual/{doc_path}` | **手册**：`docs/**` 与根 README 的站内文档浏览器（目录树 / 渲染 / 跳转），见 §3.3 | 登录（只读，viewer 可读） |
| `/login` | 登录页 | 公开 |

授权检查目前在 handler 内部完成（`web/deps.py` 的 `_require_trader` / `_require_admin` 与等价内联判断），三层角色 `admin` / `trader` / `viewer`（admin 也是 trader）；`TODO(authz)` 标记了将来改成 FastAPI 依赖的位置。前端是 Jinja2 + HTMX + Tailwind CDN + ECharts，Jinja 环境开启 autoescape，导航在移动端真实可用。

### 3.2 路由总数：121

```bash
python scripts/regen_route_baseline.py --check
# old routes: 121 / new routes: 121 / added: 0 / removed: 0
```

121 条 = **120 个 HTTP 路由**（GET 72 / POST 43 / DELETE 4 / PUT 1）+ **1 个 WebSocket**（`/ws/alerts`）。按模块分组的端点清单不必在 README 里维护：**`docs/overhaul/route-baseline.json` 是权威清单**，上面的脚本可从运行期路由表重新生成（`--check` 只报告不写；有路由消失时退出 1）。注意 `/docs`、`/redoc` 被显式关掉（404），但 `/openapi.json` 仍可访问。

### 3.3 手册 `/manual`（站内文档浏览器）

把仓库里已有的 markdown 文档做成站内手册，不再需要翻原始文件：`GET /manual`（落地页：简介 + 目录树 + 主要文档清单）、`GET /manual/{doc_path}`（单篇文档，`doc_path` 是**仓库相对路径**，如 `/manual/docs/overhaul/PLAN.md`）、`GET /api/manual/tree`（JSON 目录树，供侧栏筛选与工具使用）。三条路由与其他运营页同源鉴权（`AuthMiddleware`）：未登录访问页面 302 → `/login`，访问 `/api/manual/tree` 得 401。

| 关注点 | 行为 |
|---|---|
| **收录规则**（代码与文档同一条） | 仅 `docs/**/*.md`（递归）+ 根 `README.md` / `README_EN.md`；路径任一段以 `.` 开头、符号链接、非 markdown、以及 `docs/` 之外的 markdown（如 `experimental/ml/README.md`）都不收录。因此 `docs/overhaul/route-baseline.json` 是**文档而非手册页面** |
| **数据来源** | 请求时直接读取文件（不复制、不落库）；仅缓存渲染结果，缓存键为 `(路径, mtime_ns, size)`，改文件立即失效 |
| **路径安全** | 请求路径先归一化并拒绝 `..` / 绝对路径 / 盘符 / NUL / 空段 / 百分号编码（含双重编码）穿越，再要求它属于上面的收录集合，最后校验 `resolve()` 后仍在允许根内且不是符号链接；其余一律 404 |
| **渲染** | markdown-it-py（CommonMark + table 规则，`html=False`，文档里的原始 HTML 只当文本）+ KaTeX CDN 渲染 `$$…$$` / `$…$`（CDN 不可达时回退显示原始 LaTeX 源码） |
| **导航** | 侧栏目录树按目录分组、可折叠、当前文档高亮、支持标题/路径筛选；页内目录（`##` 以下）、面包屑、上一篇/下一篇；文档内相对链接（同目录 / 上级 / 跨目录 / 目录链接 / `#锚点`）重写为手册路由，目标不在手册内时渲染为标记过的死链接（不跳转、不 404） |

对应测试 `tests/test_manual_route.py`（收录完整性、200+标题、404、穿越拒绝、链接重写、鉴权、JSON 形状）。

---

## 4. 算法层

公式与研究性描述**不在本 README**：算法总纲见 [`docs/research/CORE_ALGORITHMS.md`](docs/research/CORE_ALGORITHMS.md)；按子系统拆分的 **16** 篇（`00-ERRATA` + `01`–`15`，命令 `(Get-ChildItem docs/core-algorithms -Filter *.md).Count`）见 [`docs/core-algorithms/`](docs/core-algorithms/)。看公式前先读 [`00-ERRATA.md`](docs/core-algorithms/00-ERRATA.md)：它汇总 **22** 处文档与代码的差异（该表 24 行 = 表头 + 分隔行 + 22 条）。

### 4.1 两套回测引擎，一个评估内核

| | **legacy**（`core/backtest/engine.py`，逐 tick） | **hybrid**（`core/backtest/engine_hybrid.py`，向量化两阶段） |
|---|---|---|
| 结构 | 单循环遍历时间线 | `SignalMatrixBuilder` 预生成信号矩阵 → `EventDrivenExecutor` 回放 |
| ML | 支持 LightGBM / TFT / PatchTST | **不支持**（`_select_engine` 直接抛 `ValueError`） |
| 部分减仓 `reduce_conditions` | 支持 | **不支持** |

两者共用 `core/strategy/evaluation_kernel.py` 与 `core/backtest/trade_book.py::close_position`。`backtest.engine_mode: auto`（默认）下：策略数 ≥ 3 且无 ML、无 `reduce_conditions` → hybrid，否则 legacy；hybrid 运行期抛异常会记 warning 并**回退 legacy**（结果可能与 hybrid 模式有差异）。legacy / hybrid / signal-matrix 三条路径在缺数据时报同一句 canonical 文案 `NO_MARKET_DATA_MESSAGE`。

### 4.2 成本模型：两套，故意不合并

| | **sim 成本模型** | **backtest 成本模型** |
|---|---|---|
| 代码 | `app/config.py::sim_cost_quote` | `core/backtest/cost_model.py::apply_trading_costs` |
| 作用点 | **改写成交价本身**（买贵、卖便宜），`pnl` 存入即已扣净 | **平仓时**按往返一次性叠加成本，**不动入场价**（否则止损/止盈价位会级联变化） |
| 价差来源 | `sim.cost_model.spread_pct.<SYM>` → `default` → `0.02` | override → live depth → `default_spread_pct`（`0.03`） |

`spread_pct.<SYM>` 的语义是**完整买卖价差**（%，`BTCUSDT: 0.01` = 1 bp 盘口），成本模型**每边收一半**（`spread/2`）；sim 与 backtest 用同一约定。`sim_cost_quote` 是唯一实现，`GET /api/fee/estimate` 与真实成交共用：

```
edge_pct = 0                                     # 限价单，或 enabled=false
         | spread_pct/2 + slippage_bps/100       # 市价单
买入 fill = price × (1 + edge_pct/100)；卖出 fill = price × (1 − edge_pct/100)
fee = quantity × fill × fee_pct/100              # 档位取 maker/taker，BNB 折扣 ×0.75
slippage_usdt = |fill − price| × quantity；cost_usdt = fee + slippage_usdt
```

回测价差解析顺序固定为 `override → live（公开盘口，缓存 300s，失败结果不缓存）→ default`，并在每次运行开始时**冻结**本次用到的每个 symbol 的价差，交易循环里不再做 I/O。**两套兜底常量不一致**（0.03 vs 0.02）是已知的刻意口径分歧，见 §6。

### 4.3 默认关闭的能力（不要以为它们在生效）

| 能力 | 开关（当前值） | 为什么默认关 |
|---|---|---|
| **ML 门控** | `config.yaml` `ml.enabled: false`；门在 `core/ml/credibility.py::credibility_gate`（OOS AUC > 0.55 **且** 净成本期望 > 0），由 `core/ml/predictor.py::_gate_config` 消费 | 实测线上模型**差于多数类**（accuracy 0.41–0.47 vs 0.54–0.67，OOS AUC 0.396–0.447）且反校准（预测 0.91 → 实际 0.22），开启是**负贡献** |
| **波动率目标化** | `risk.vol_targeting.enabled: false` | 完全 opt-in；关闭时仓位与止损宽度与固定比例实现**逐位一致**。该块里 `barrier_vol_multiple` / `barrier_min_pct` / `barrier_max_pct` 是 **RESERVED/惰性**键（无生产调用方，设成非默认值只在启动时打 WARNING），不要当成生效配置 |
| **`risk.liquidity`（成交量/冲击）** | `enabled: false`、`impact_k: 0.0` | 关闭时 `PositionSizer` 不调用参与度钩子，回测成本保持审计过的 `fees + spread/2` 逐位不变（`tests/test_liquidity.py` 守护）。`impact_k` 在配置里明确标注 **ILLUSTRATIVE, NOT CALIBRATED** |
| **P4 新能力** | 模块级常量全为 `False`：`META_LABELING_ENABLED`（`core/ml/meta.py`）、`PAIRS_ENABLED`（`core/strategy/pairs.py`）、`REGIME_GATING_ENABLED`（`core/strategy/regime.py`）、`MICROSTRUCTURE_ENABLED`（`core/market_data/microstructure.py`） | 已实现且有独立测试，但**未接入实时链路**（默认惰性、关闭时全放行）；在真实 1h 主流币数据上配对与 meta-label 门**拒绝**全部被测一级规则——这是有效结论，不是缺陷。这些常量现在有配置开关，见 §4.4 |

`core/` 里仍有 GA、AI（DeepSeek 控制器 + 策略生命周期）、新闻情绪、代币启发式筛查（`core/market_data/screener.py`）在链路上；`experimental/` 是**不在交易链路上**的死代码存档（见 [`experimental/ml/README.md`](experimental/ml/README.md)）。

### 4.4 实验性开关（`config.yaml` 的 `experimental:`，默认全关）

P4 那批能力此前只能改 Python 常量；现在一个能力一个开关，`app/config.py::apply_experimental_flags` 由 `app/main.py` 启动路径显式调用（配置全关时**不导入能力模块、不写任何常量**，信号与仓位与 HEAD **逐位一致**，`tests/test_experimental_switches.py` 用 `git worktree` 在 `3a140cf` 上逐字节比对）。

| 开关（默认 `false`） | 常量 → 实际改动 |
|---|---|
| `engine_regime_diagnostics` | `P4_REGIME_DIAGNOSTICS_ENABLED`：把 regime 标签附进信号缓存（**只诊断，不改方向/仓位**）；这是唯一"单开即可达"的开关 |
| `engine_meta_filter` / `engine_pairs_signals` | `P4_META_FILTER_ENABLED` / `P4_PAIRS_SIGNALS_ENABLED`：已注册的 MetaLabeler / pairs provider 可过滤或替换信号，但生产链路**尚未注册**（`wire_meta_filter` / `wire_pairs_provider` 无调用者），单开无效 |
| `regime_gating` / `regime_diagnostics` | `REGIME_GATING_ENABLED` 切到因果 HMM 并启用门；`REGIME_DIAGNOSTICS_ENABLED` **无生产读取者** |
| `pairs_enabled` / `meta_labeling_enabled` / `microstructure_enabled` | `PAIRS_ENABLED` / `META_LABELING_ENABLED` / `MICROSTRUCTURE_ENABLED`：**均无生产读取者**（microstructure 连**接线缝隙都没有**），打开不改变任何可达行为 |

每个开关只打开**那道缝**，能力自身的验收门仍会拒绝：真实 1h 主流币配对 **0/30** 通过协整检验、meta 门 **10/10** 拒绝一级规则、regime 只接受因果 HMM 标签。**诚实预期**：这些开关提高的是**可测量性与纪律**（能否复现、能否 A/B），**不增加预测优势**。启动时会打一条 WARNING 逐项列出已开启的开关（`experimental_notices`，与 `inert_barrier_key_warnings` 同型），未知键也会被点名而不是静默忽略。

---

## 5. 运维

### 5.1 测试

```bash
python -m pytest tests/ --collect-only -q -p no:cacheprovider   # 末行: 1318 tests collected
python -m pytest tests/ -q -p no:cacheprovider                  # 全量
python -m pytest tests/ -q -m "not slow"                        # 跳过慢测
```

`pytest.ini`：`testpaths=tests`、`asyncio_mode=strict`、`markers=slow`。**没有安装 `pytest-timeout`**，所以 `--timeout=` 会直接报参数错误。

唯一容易被机器负载误报的是 `tests/test_ml_credibility.py::test_feature_pipeline_cost_is_bounded`：它断言特征流水线耗时 `< 3.0 s`，并发压力下会超时失败（实测压力下整跑 1 failed / 1216 passed，同一用例单独运行 1.74s 通过）。看到只有这一条失败时，先单独重跑它再判断。

**干净克隆的隐含前提**：`data/` 全部 gitignore，所以没有任何缓存历史的克隆直接跑全量**不是全绿**——`tests/test_engine_parity_variants.py` 的 7 个真实数据变体依赖 `2026-05-25..2026-05-31` 的 BTCUSDT + ETHUSDT 1h 缓存（`DATE_START` / `DATE_END` / `SYMBOLS` 就在该文件头部），缺数据时以同一句 `NO_MARKET_DATA_MESSAGE` 失败（`tests/test_hybrid_equivalence.py` 同类用例会 skip）。CI 或新机器先下这段历史即可（下面这条会写 `data/market/`，本次未执行）：

```bash
python scripts/download_history.py --symbols BTCUSDT,ETHUSDT --intervals 1h --start 2026-05-25 --end 2026-05-31
```

### 5.2 审计与完整性脚本（全部只读，可放进定时任务）

```bash
python scripts/audit_db.py [path\to\binance_trader.db]        # 账本对账
python scripts/check_data_integrity.py                        # K 线缓存缺口
python scripts/regen_route_baseline.py --check                # 路由基线奇偶校验
python -m compileall -q app core web db scripts alerts        # 语法完整性
```

- **`audit_db.py`**：把库复制到临时目录再以 `mode=ro` 打开（**绝不写生产库**），输出账本恒等式逐项计算与 `delta`、`trades` 的 action 分布、每张表行数、索引清单。恒等式是 `10000 − Σ(open 行: quantity × entry_price) + Σ(close 行: pnl) == system_config.sim_balance`。退出码：`0` = 等式成立（`RESULT: IDENTITY HOLDS`）**或**空库/无 `sim_balance` 行（`FRESH DATABASE — NO LEDGER YET`，空库不是故障）；非 0 = 真有漂移或库文件不存在。
- **`check_data_integrity.py`**：逐个 parquet 报 bar 数、跨度与日历缺口，超过阈值（默认 1.5 × bar 长度）即 `GAP`；`--strict` 在有缺口时退出 1。实测 **29** 个缓存文件里 **25** 个带缺口（2026-09-30 快照：`RESULT: 25/29 file(s) carry a gap beyond 1.5 x bar length`）。带缺口的序列会被波动率路径的 splice guard 拒绝，修法是按输出末尾给出的 `download_history.py ... --merge` 回填。
- **`regen_route_baseline.py`**：用生产同构的 `create_app`（指向一次性临时 DB 与空 config 目录）枚举路由并与基线对比；`--check` 只报告不写。

### 5.3 数据刷新

```bash
# 下载历史（这两个脚本的 --symbols / --intervals 是逗号分隔）
python scripts/download_history.py --symbols BTCUSDT,SOLUSDT --intervals 1h,4h \
    --start 2024-01-01 --end 2024-03-01 [--merge] [--concurrency 4] [--data-host URL]
python scripts/download_history.py --backfill --symbols BTCUSDT   # 补 P6-B 的 quote_volume / trade_count 列（可续跑）
python scripts/download_history.py --list-intervals               # 合法周期：1s…1M

# ML 可信度测量（--symbols / --intervals 是 nargs="*"，必须空格分隔）
python scripts/ml_credibility_measure.py --symbols BTCUSDT ETHUSDT --intervals 1h
python tools/p6_volume_bars_experiment.py all --symbols BTCUSDT ETHUSDT SOLUSDT
```

> 两种 `--symbols` 写法**不能混**：`download_history.py` / `check_data_integrity.py` 收逗号分隔的字符串；`ml_credibility_measure.py` / `tools/p6_volume_bars_experiment.py` 是 `nargs="*"`，写成逗号会被当成**一个** symbol，随后 `FileNotFoundError`。

下载落地到 `<data-dir>/market/<SYMBOL>/<interval>.parquet`，每页最多 1000 根，页间隔 0.12s 避免限频。Web 端等价入口是回测页的「获取数据」按钮（`POST /api/backtest/fetch-data`，trader）。

### 5.4 备份 / 恢复 / 清理

| 操作 | 方式 |
|---|---|
| 备份（下载文件） | `GET /api/db/backup`（admin）：先 `shutil.copy2` 到 `<db>_backup_<YYYYmmdd_HHMMSS>.db` 再以附件返回 |
| 恢复 | `POST /api/db/restore`（admin，上传文件）：**校验前 16 字节必须是 `SQLite format 3\0`**，把当前库复制成 `<db>.pre_restore` 后再覆盖 |
| 清理 | `POST /api/db/cleanup`（admin）：删 90 天前的 alerts / ai_suggestions 与 365 天前已实现 PnL 的 close/reduce 成交，然后 `VACUUM` |
| 优化 / 导出 / 删行 | `POST /api/db/optimize`（`VACUUM` + `REINDEX`）、`GET /api/db/export/{table}` → CSV、`DELETE /api/db/row/{table}/{row_id}`（均 admin） |
| **手工备份（推荐）** | **先停服务**，再复制 `data/binance_trader.db` **连同 `-wal` / `-shm`**（WAL 模式） |

`ledger_reconciliation` 表记录每一次**人工账本修复**（修复前/后余额、delta、原因、备份路径、操作者）。

### 5.5 日志与重启

loguru 默认输出到 **stderr**，项目**没有配置文件 sink**；要留档就自己重定向（`run_logs/` 已 gitignore）：

```bash
python -m app.main --mode sim >> run_logs/service.log 2>&1   # PowerShell 用 *>> run_logs\service.log
```

`Ctrl+C` 按逆序关停各组件；持仓与余额本来就在 DB 里，关机时不需要落盘。重启后的状态：持仓从 `positions` 表恢复（含 basis 与止损）、余额以 `system_config.sim_balance` 为权威、挂单从 `pending_orders` 恢复（撮合器每 ~5s 跑一次）、自选列表与手续费档位从 `system_config` 生效、**会话在内存里所以全部失效需重新登录**（JWT 因密钥落盘仍有效）。优雅重启入口是 `POST /api/settings/restart`（admin）。

### 5.6 常见故障

| 现象 | 原因与处理 |
|---|---|
| 页面能开但 K 线/行情全空，日志刷 `REST kline fetch failed` | 行情主机不可达或被改回 `api.binance.com`。确认 `market_data_host` / `market_stream_host` 是 `.vision` 域名；日志里没有 `Kline #1:` 说明 WS 没通，此时 REST 兜底轮询每 30s 会补 |
| 打开任何页面被踢回 `/login`，接口 401 | 会话在内存里，重启进程即失效（`session_hours` 默认 24h）。重新登录；API 客户端可用登录返回的 token 加 `Authorization: Bearer <token>` |
| `403 {"error":"Forbidden"}` | 当前角色是 `viewer`，写操作需 `trader` / `admin` |
| `Position already open for X` / `Max open trades N reached` / `Total exposure X% exceeds limit` / `Circuit breaker tripped` | 风控 7 步管线命中；平仓、调高阈值（`config/risk_params.yaml`），或 `POST /api/circuit-breaker/reset`（trader） |
| `order_below_lot_size` / `order_below_min_notional` / `quantity ... rounds to 0` | 数量按 `step_size` 向下取整后为 0，或名义金额小于该 symbol 的 `minNotional`；加大下单金额 |
| `database is locked` | WAL + 多短连接，或**同一个 DB 被两个进程打开**（含 pytest 与运行中的服务共用）。确认只有一个写进程，不要放网络盘；持续锁死就停服务 → 备份（含 `-wal`/`-shm`）→ `/api/db/optimize` |
| 端口被占用（`Errno 10048`） | `python -m app.main --mode sim --port 8900`，或先找出占用进程 |

---

## 6. 已知限制与残余

- **主网交易 API 不可达。** `api.binance.com` 从本机超时，所以**真实资金交易在本机不可能完成**：`--mode live` 实际打到 testnet（打到哪由 `binance.testnet` 决定），行情是真实主网数据但下单不是。**手续费档位也只能手动设置**——本机读不到真实账户的 30 天交易量与 BNB 持仓。
- **`--mode backtest` 不是回测**，订单被静默丢弃（见 §2.4）。
- **sim 滑点是模型，不是事实。** 固定 `slippage_bps` + 每 symbol 固定半价差，不随订单大小、盘口深度、波动率变化；大单真实滑点会明显更高。且**两套价差兜底常量不一致**：回测 `default_spread_pct` = 0.03，sim `default` = 0.02。
- **回测/GA 是现金模型，不含杠杆。** `leverage` 在 `core/ga/**` 与 `core/backtest/engine.py` 里出现 **0** 次（命令：`Get-ChildItem core/ga -Recurse -Filter *.py | Select-String '\bleverage\b'`），而 `risk_params.yaml` 的 `max_leverage: 4` / `soft_params.leverage: 2` 只作用于实盘——回测/GA 的收益与风险指标不能直接与带杠杆的实盘对比。
- **流动性冲击系数未标定**：`risk.liquidity.impact_k` 默认 0（能力整体关闭，见 §4.3），配置里明确标注 ILLUSTRATIVE, NOT CALIBRATED；要开必须来自实测的冲击研究并注明来源。
- **实盘波动率路径只能看到最近 600 根 bar。** live forecast path 的 splice guard 只能拒绝这个窗口内的缺口（如 −100、−300），更早的洞（如 −900）它看不见；全文件检查靠 `scripts/check_data_integrity.py`。**缓存仍有历史缺口**：实测 29 个 parquet 里 25 个带缺口（§5.2，2026-09-30 快照）——这是形状 + 时间戳，不是固定值。
- **代币检测是启发式的，不是链上审计。** `core/market_data/screener.py` 只读交易所公开行情，不读合约、持仓分布、mint 权限、转账税、代理升级位、LP 锁；**无法**发现蜜罐、rug pull、冻结/黑名单函数、隐藏增发。干净评分只说明这个交易对的**交易所市场**看起来正常。
- **仓位管理不是真 Kelly。** `core/risk/position_sizer.py` 是固定比例法（资本池 × 百分比，再被 `max_position_size_pct` 截断）；`docs/core-algorithms/04-position-sizing-kelly.md` 的 Kelly 公式只在回测的 Kelly-lite 分支部分体现，那份文档是研究性描述，**不要当成实现说明**。
- **安全面**：没有 CSRF token（防护只靠 `samesite=lax` cookie）、没有 HTTPS/反代配置、没有"谁改了什么"的操作审计日志；授权是 handler 内联的（`TODO(authz)`）。
- **`experimental/` 不被任何生产模块导入**，删除它不影响交易行为；仓库根可能残留一次性产物（如 `nonexistent.db`）。
- **`docs/HANDOVER.md`、`docs/overhaul/REFACTOR_AUDIT.md` 描述的是大改之前的状态**（"实时链路断裂""AI 面板空转"等问题大部分已修）；把它们当历史记录，**以代码为准**。

---

## 7. 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/research/CORE_ALGORITHMS.md`](docs/research/CORE_ALGORITHMS.md) · [`docs/core-algorithms/`](docs/core-algorithms/) | 算法总纲；按子系统拆分的 16 篇（先看 [`00-ERRATA.md`](docs/core-algorithms/00-ERRATA.md) 的 22 处勘误） |
| [`docs/HANDOVER.md`](docs/HANDOVER.md) · [`docs/development-roadmap.md`](docs/development-roadmap.md) | 接手评估报告（**改动前快照**）；开发路线图（**规划≠现状**） |
| [`docs/overhaul/PLAN.md`](docs/overhaul/PLAN.md) · [`CHANGELOG.md`](docs/overhaul/CHANGELOG.md) | 长盘修复计划 S0–S7 与三轮审计的发现与处置；详细变更史（改了什么、验收程度、剩余项） |
| [`docs/overhaul/ALGO_UPGRADE_PLAN.md`](docs/overhaul/ALGO_UPGRADE_PLAN.md) · [`ALGO_UPGRADE_EVIDENCE.md`](docs/overhaul/ALGO_UPGRADE_EVIDENCE.md) | P1–P4 算法升级的验收标准与逐阶段实测证据、未闭环清单 |
| [`docs/overhaul/REFACTOR_AUDIT.md`](docs/overhaul/REFACTOR_AUDIT.md) · [`WEB_SPLIT.md`](docs/overhaul/WEB_SPLIT.md) | 四路只读审计；`web/server.py` 拆分记录与保留的不变量 |
| [`docs/overhaul/TRADE_PAGE_API.md`](docs/overhaul/TRADE_PAGE_API.md) · [`MARKET_PAGES_API.md`](docs/overhaul/MARKET_PAGES_API.md) | `/trade` 与行情/币种/数据/代币检测页的接口契约（冻结版），后者含主机可达性前置事实 |
| [`docs/overhaul/route-baseline.json`](docs/overhaul/route-baseline.json) | **路由权威清单**（当前 118 条），用于奇偶校验 |
| [`docs/audit/`](docs/audit/) · [`docs/superpowers/`](docs/superpowers/) · [`experimental/ml/README.md`](experimental/ml/README.md) | 历史审计分报告 R1–R15 与严重度分级；设计规格与实施计划（唯一的设计史来源）；被移出生产链路的 ML 死码清单 |
| `/manual`（站内手册） | 上表的站内渲染版本（§3.3）：目录树 + 筛选、页内目录与面包屑、相对链接重写、KaTeX 公式；只收录 `docs/**` 与根 README，`route-baseline.json` 等非 markdown 文件不在其中 |

### 许可与致谢

本项目**未附带显式开源许可证文件**（`LICENSE` 缺失）；如需使用/分发，请先与仓库所有者 `STCloudLake` 确认授权方式。

依赖与数据来源：python-binance（交易所客户端）、FastAPI + uvicorn（Web）、aiosqlite（数据库）、ECharts + HTMX + Tailwind CSS（前端）、TA-Lib / LightGBM / XGBoost / PyTorch（指标与 ML）、loguru（日志）、DeepSeek（LLM）。行情与交易数据来自 Binance 公开 API（`data-api.binance.vision` / `testnet.binance.vision`）。

> **风险提示**：加密货币交易存在本金全部损失的风险。本系统默认运行模拟盘；任何切换到真实资金交易的决定、参数设置与后果都由使用者自行承担。`--mode live` 在当前环境下只会打到 Binance **testnet**，但这不构成对任何未来配置变更的安全保证。
