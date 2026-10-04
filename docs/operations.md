# 运维参考（Operations）

这份文档承载 README 原先内联的运维细节：CLI、配置键、GA job 字段、实验开关、路由清单、
数据与数据库操作、故障处置。README 只保留"是什么、能做什么、装起来怎么跑"。

**本文描述当前代码事实**，每个数字旁给出发它的命令；与代码冲突时以代码为准。

---

## 1. 环境约束（必读）

### 1.1 主机可达性（实测 2026-09-30，`Invoke-WebRequest -TimeoutSec 8` 直连）

| 主机 | 结果 | 用途 |
|---|---|---|
| `https://data-api.binance.vision` | HTTP 200 | **所有公开行情**：klines / ticker24h / depth / trades / exchangeInfo |
| `wss://data-stream.binance.vision` | 配置目标（未做 socket 握手实测） | **实时 K 线流**；`provider.py` 把 socket 工厂指向它 |
| `https://testnet.binance.vision` | HTTP 200 | **下单 / 账户 / 余额**——当前唯一的交易端点 |
| `https://api.binance.com` | 8s 超时 | **不可用**（主网交易域名一律不可达） |

复现（三条命令只换 URL，结果分别为 200 / 200 / 超时）：

```powershell
Invoke-WebRequest -Uri https://data-api.binance.vision/api/v3/ping -TimeoutSec 8 -UseBasicParsing
Invoke-WebRequest -Uri https://testnet.binance.vision/api/v3/ping   -TimeoutSec 8 -UseBasicParsing
Invoke-WebRequest -Uri https://api.binance.com/api/v3/ping          -TimeoutSec 8 -UseBasicParsing
```

**为什么这是硬约束**：`api.binance.com` 的调用会挂到超时，把请求处理器一起拖死。行情**必须**走
`config.binance.market_data_host`（默认 `https://data-api.binance.vision`），由
`core/market_data/data_client.py::MarketDataClient` 统管（15s 硬超时、连接池复用、失败抛
`MarketDataError`）。这个镜像给的是**真实主网数据**；testnet 只有少量测试对、历史是合成的，用它做
行情会看到不全的币种与缺失的 K 线。

### 1.2 绝不要用裸 `AsyncClient.create()`

```python
# ❌ python-binance 默认打 api.binance.com，本机必然超时
client = await AsyncClient.create()
# ✅ 交易/账户客户端必须显式传 testnet（见 web/routes/market.py::_configured_client）
client = await AsyncClient.create(api_key=..., api_secret=..., testnet=config.binance_testnet)
```

所有 K 线读取都走 `data_client`（绑定行情镜像）。`MarketDataProvider.start()` 里那次
`AsyncClient.create(...)` 是**交易**客户端，失败只记 warning 并以离线模式继续。

### 1.3 实时流的 socket URL 被刻意覆写，Web 只绑本地

python-binance 的 `BinanceSocketManager` 硬编码 `wss://stream.binance.com:9443/`（该主机不可达），
`provider.py` 在创建 `bsm` 后覆写 `bsm._get_stream_url` 指向 `config.binance.market_stream_host`——
**新增任何 socket 用法时不要绕过这一步**。另外 `app/main.py` 里是 `uvicorn.Config(host="127.0.0.1", ...)`，
**没有 `--host` 参数**；要从别的机器访问请用 SSH 端口转发或反向代理 + HTTPS，不要把 host 改成
`0.0.0.0` 暴露到公网。

---

## 2. CLI 与启动

```
usage: main.py [-h] [--mode {sim,live,backtest}] [--port PORT] [--db DB]
               [--data-dir DATA_DIR] [--config-dir CONFIG_DIR]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode {sim,live,backtest}` | `sim` | 模式**只来自 CLI**（`config.yaml` 没有 `mode:` 键）。语义见 `executor.py::_on_order_request`：`sim` 本地成交（完整成本模型，不连交易所）；`live` 用 testnet 客户端真实下单；**`backtest` 不是回测**——不进入任何下单分支、订单被静默丢弃，真正的回测走 `/backtest` 页或 `POST /api/backtest/run` |
| `--port N` | `8899` | Web 端口 |
| `--db PATH` | `<project>/data/binance_trader.db` | SQLite 路径；优先级高于 `--data-dir` |
| `--data-dir DIR` | `<project>/data` | 运行数据目录 |
| `--config-dir DIR` | `<project>/config` | 设置/密钥持久化目录。**冒烟测试请用它 + `--db` 指到临时目录，不要碰生产库** |

**首次启动**若 `users` 表为空，会自动创建 `admin` 并生成随机密码，密码**只打印到 stderr**
（不写日志、bcrypt 存库、明文不可恢复），请立刻在 `/settings` 或用户管理页改掉；`users` 表非空时
不会重建。ML 训练在 `uvicorn` 之前**串行**执行，所以启动后可能有一段时间端口尚未监听（训练抛异常
只记 warning，不阻塞启动）。

---

## 3. 配置键

`app/config.py::_load` 依次深合并 `config/config.yaml` → `config/risk_params.yaml` →
`config/secrets.yaml`，环境变量最高。

| 键 | 默认 | 含义 |
|---|---|---|
| `web_port` / `language` | `8899` / `zh` | Web 端口与 UI 语言（`zh`/`en`） |
| `binance.testnet` | `true` | **下单/账户**客户端是否打 testnet。**不控制行情** |
| `binance.market_data_host` / `market_stream_host` | `.vision` 两个域名 | 行情 REST / WS 主机 |
| `ai.mode` / `ai.model` | `full_auto` / `deepseek-v4-flash` | `semi_auto` / `full_auto`；LLM 模型名 |
| `signal_weights.*` | `indicator 0.5 / ml 0.3 / news 0.2` | 信号融合权重 |
| `backtest.engine_mode` / `ml_enabled` | `auto` / `false` | 引擎选择（非法值回退 `auto`）；false 时会就地关掉策略的 `ml_config.enabled` |
| `backtest.fill_convention` | `close` | **成交口径**：`close` = 信号那根 bar 自己的收盘价（零执行延迟，出厂默认）；`next_open` = 信号不变、成交价（**开仓与平仓同时**）取同一序列上一根 bar 的 `open`。非法值在**配置加载时**抛具名 `UnknownFillConventionError`。**翻成 `next_open` 会让所有历史回测/冠军数字作废**（实测差额见 `docs/overhaul/P9_FILL_CONVENTION_EVIDENCE.md`）。窗口最后一根 bar 无下一根时：开仓拒绝并计数、平仓回落收盘价并计数（`metrics["fill_convention_accounting"]`）；hybrid 引擎不支持 `next_open`，会具名报错 |
| `ga.benchmark_mode` | `exposure_matched` | 发布门消费哪个基准，见 §5.4 |
| `ga.keep_checkpoint` | `true` | 干净完成后是否保留检查点，见 §5.5 |
| `sim.cost_model.*` | `enabled: true`、`fee_tier: VIP0`、`slippage_bps: 2` | 模拟盘成本模型 |
| `ml.*` | `enabled: false` | ML 门控与训练参数（`gate_*` 是可信度门阈值） |
| `risk.*` | 全部 `enabled: false` | 波动率目标化 / 流动性，默认惰性（见 §6） |
| `experimental.*` | 全部 `false` | P4 实验性能力开关（见 §6） |

`config/risk_params.yaml` 给出 `hard_limits`（日/周回撤、日亏损、最大敞口、最大笔数、杠杆、追踪止损、
紧急止损等）与 `soft_params`（仓位比例、止损、杠杆、三档止盈）；**以文件里的值为准**，本文不复制这张表。

自选列表**不在 YAML 里**，存在数据库 `system_config.watchlist_symbols`（兜底
`BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT`，上限 30），改动**下次重启生效**
（`POST /api/market/watchlist` 的响应带 `restart_required: true`）；手续费档位同理——本机读不到真实
账户的 30 天交易量与 BNB 持仓，只能手动选择并持久化。

**凭据**：`config/secrets.yaml` 已 gitignore，可写 `binance.api_key` / `api_secret`、
`deepseek.api_key`；`auth.jwt_secret` 留空时启动会生成随机密钥并写回该文件（POSIX 上 `chmod 600`），
因此重启后旧 token 仍有效。也可以用环境变量（**优先级高于 YAML**）：`BINANCE_API_KEY`、
`BINANCE_API_SECRET`、`DEEPSEEK_API_KEY`、`JWT_SECRET`。

**策略 YAML 必须自己提供**——仓库**没有跟踪任何策略文件**（`git ls-files strategies/` 输出为空；
磁盘上只有被 gitignore 的 `strategies/ga_champion_*.yaml`，由 GA 生成）。干净克隆的**策略数是 0**：
`GET /api/strategy-monitor` 返回空列表，`POST /api/backtest/run` 用任何名字都会得到
`Strategy '<name>' not found: Strategy file not found: ...`。策略 schema 就是
`core/strategy/loader.py::StrategyConfig` 的 pydantic 字段；自己写 YAML 放进 `strategies/`，
或让 GA / 策略生命周期生成。

---

## 4. 路由清单（121）

```bash
python scripts/regen_route_baseline.py --check
# old routes: 121 / new routes: 121 / added: 0 / removed: 0
```

121 条 = **120 个 HTTP 路由**（GET 72 / POST 43 / DELETE 4 / PUT 1）+ **1 个 WebSocket**（`/ws/alerts`）。
权威清单是 `docs/overhaul/route-baseline.json`，上面的脚本可从运行期路由表重新生成（`--check` 只报告
不写；有路由消失时退出 1）。注意 `/docs`、`/redoc` 被显式关掉（404），但 `/openapi.json` 仍可访问。

页面路由与最低角色：

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
| `/manual`、`/manual/{doc_path}` | **手册**：`docs/**` 与根 README 的站内文档浏览器 | 登录（只读，viewer 可读） |
| `/login` | 登录页 | 公开 |

授权检查目前在 handler 内部完成（`web/deps.py` 的 `_require_trader` / `_require_admin` 与等价内联判断），
三层角色 `admin` / `trader` / `viewer`（admin 也是 trader）；`TODO(authz)` 标记了将来改成 FastAPI
依赖的位置。前端是 Jinja2 + HTMX + Tailwind CDN + ECharts，Jinja 环境开启 autoescape。

### 4.1 手册 `/manual` 的收录与渲染规则

三条路由：`GET /manual`（落地页）、`GET /manual/{doc_path}`（单篇，`doc_path` 是**仓库相对路径**，
如 `/manual/docs/overhaul/PLAN.md`）、`GET /api/manual/tree`（JSON 目录树）。未登录访问页面 302 →
`/login`，访问 `/api/manual/tree` 得 401。

| 关注点 | 行为 |
|---|---|
| **收录规则** | 仅 `docs/**/*.md`（递归）+ 根 `README.md` / `README_EN.md`；路径任一段以 `.` 开头、符号链接、非 markdown、以及 `docs/` 之外的 markdown（如 `experimental/ml/README.md`）都不收录 |
| **数据来源** | 请求时直接读取文件（不复制、不落库）；仅缓存渲染结果，缓存键为 `(路径, mtime_ns, size)` |
| **路径安全** | 请求路径先归一化并拒绝 `..` / 绝对路径 / 盘符 / NUL / 空段 / 百分号编码穿越，再要求它属于收录集合，最后校验 `resolve()` 后仍在允许根内且不是符号链接；其余一律 404 |
| **渲染** | markdown-it-py（CommonMark + table 规则，`html=False`）+ KaTeX CDN 渲染 `$$…$$` / `$…$`（CDN 不可达时回退显示原始 LaTeX 源码） |
| **导航** | 侧栏目录树分组、可折叠、当前文档高亮、支持筛选；页内目录、面包屑、上一篇/下一篇；文档内相对链接重写为手册路由，目标不在手册内时渲染为死链接（不跳转、不 404） |

对应测试 `tests/test_manual_route.py`。

---

## 5. GA job 字段

`POST /api/ga/evolve` 与 `POST /api/ga/walkforward` 接受以下 job 字段。**字段不存在 = 旧行为**，
每一项都有逐位一致的回归测试守护。

### 5.1 `timeframe_pool` — 周期白名单

周期是**基因组的一部分**，不设限的 GA 会把大部分预算花在 `1m` 基因上：3 个月 × 3 币的 1m 回测每币
**~390 000** 根 bar，是 15m 的 **15×**、1h 的 **60×**（实测：pop 20 / 3 币 / 12 workers 的任务 25 分钟
只评估完 20 个基因组中的 11 个）。

| 关注点 | 约定 |
|---|---|
| **缺省** | 字段不存在 = **不限制**，与加入该字段之前逐位一致（`1m/5m/15m/1h/4h` 全部仍可选） |
| **校验** | 取值必须来自唯一 interval registry `core.market_data.provider.INTERVAL_SPEC`（`1s`/`1M` 因无 bar 长度/ML 规格被拒绝）；非法周期 → 接口 **HTTP 400** 或 worker **加载即失败**（`UnknownTimeframeError`） |
| **基因约束** | `core/ga/genome.py` 的 `timeframes` 分类基因：随机初始化、变异、交叉/精英/`resume` 检查点、解码全都被限制在白名单内；解出的策略 `timeframes` **非空且 ⊆ 白名单** |
| **可审计** | 启动日志 `GA timeframe_pool=15m,1h,4h`（不限制时为 `unrestricted`）、progress JSON、result 与冠军 YAML 的 `provenance.timeframe_pool`；`scripts/ga_job_status.py` 的 `job params` 行打印 `tf_pool=` |

面板「Timeframes (GA gene pool)」默认勾选 `15m/1h/4h`；全部取消勾选 = 不写该字段 = 不限制。
测试：`tests/test_ga_timeframe_pool.py`。

### 5.2 `symbol_mode` — 逐币种独立进化（P7-S2）

池化 GA 在**整篮子**上评一个种群，一个只在某个币上有效的信号会被其他币的表现平均掉。
`symbol_mode: "per_symbol"` 让 GA 变成**每个币一套独立种群**：每个候选**只在它自己的币上**评分，
每个币各出一个冠军。

| 关注点 | 约定 |
|---|---|
| **开关是 job 字段，不是配置键** | `"pooled"`（缺省）/ `"per_symbol"`；`config/config.yaml` 里**没有** `ga.symbol_mode`，`Config` 也没有该属性（`tests/test_p7_symbol_mode.py::test_there_is_no_config_key_for_the_mode` 守护）——任何配置改动都不会改变 GA 的搜索形状 |
| **缺省逐字节一致** | 字段缺失或 `"pooled"` 时与旧行为逐位一致：同 seed 下种群哈希、每代试错计数、冠军基因、发布门结论与 result 载荷全部相同（三种口径都做过逐字节比对） |
| **校验** | `core/ga/evolver.py::parse_symbol_mode`；未知取值抛具名 `UnknownSymbolModeError` ⇒ **接口 HTTP 400**、worker **job 加载即失败**；缺失/空白 = `pooled` |
| **冠军** | **每币一个冠军** YAML（`ga_champion_<币>_<时间戳>`），且 `champion_config.symbols == [该币]`；`provenance.symbol_mode / champion_symbol / n_symbols`；result 是 `champions[]` **加**顶层镜像 |
| **试错计数（DSR 诚实性）** | per-symbol 一轮实际执行 `len(symbols) × population × generations` 次评估，所有臂共享同一本搜索账；`evolve()` 在**所有臂跑完后只算一次** `n_trials`，所以**每个**冠军的 DSR 都用**整轮**真正试过的变体数 |
| **检查点身份** | 检查点记录 `symbol_mode` 与 `arm_symbol`；**跨形状续跑**抛具名 `CheckpointSymbolModeMismatchError`；per-symbol 续跑只续检查点所属的那个臂 |
| **成本** | 墙钟时间≈不变：每个 per-symbol 候选只在 1 个币上评分（实测 200.8 s vs 226.1 s） |

实测结论（BTC+ETH 1h，样本外 2026-02~06，exposure-matched 基准）：pooled 冠军 alpha 中位 **−0.526pp** /
147 笔；per_symbol **−0.3054pp** / 81.5 笔；**两臂 DSR>0 均为 0**（0/2、0/4）。表面改善是暴露效应，
同币成对比较**符号随种子翻转**。详见
[`P7_REGIME_EVIDENCE.md`](overhaul/P7_REGIME_EVIDENCE.md) 与
[`ALGO_UPGRADE_EVIDENCE.md`](overhaul/ALGO_UPGRADE_EVIDENCE.md) §8。

### 5.3 冠军只交易被评估过的币

冠军 YAML 过去写 `symbols: []`（= 交易自选列表里的**全部**币），而 GA 只评估了 job 的篮子。
现在 `evolve()` 把**被评估的篮子**写进冠军的 `symbols:`，执行路径
（`core/strategy/engine.py` 的 `_on_kline` / `evaluate_all_now`：`strategy.symbols` 非空时跳过表外币，
启动日志 `Strategy '<name>' restricted to symbols: [...]`）因此只能交易它被评估过的币。老 YAML 的
`symbols: []` 仍表示"不限"，保持向后兼容。

### 5.4 `benchmark_mode` — 发布门基准

发布门原来只用**满仓买入持有**（`metrics["buy_hold_pct"]`，同窗口同币种等权）比**总收益**。对一个只在
一小部分时间持仓、回撤 0.11 % 的短周期策略，这个比较**没有做敞口/风险匹配**——它惩罚的是"拿着现金"。

| 取值 | 定义 | 说明 |
|---|---|---|
| `buy_hold`（**代码缺省**） | 满仓等权买入持有（历史行为） | 键不存在 = 该值；与加入该键之前逐位一致 |
| `exposure_matched`（`config/config.yaml` **已启用**，推荐） | 同一篮子**只在策略持仓期间**持有 | 用策略自己的成交（`opened_at`→`closed_at`，裁到窗口）重建每个币的**持仓区间并集**，算该币在这些区间上的买入持有收益（区间收益**复合**，区间之间算现金 0 %），再按策略**实际投入的保证金占比** `wₛ = mean(amount_usdt)/initial_balance` 加权。没交易过的币权重为 0；窗口结束时**未平仓**按窗口末裁剪；**零成交** ⇒ 基准 0、alpha = 策略收益（门仍以 `no_trades`/DSR/净期望拒绝）；区间内**缺 bar** ⇒ 该币剔除，全不可用 ⇒ 基准 `None`、门**跳过**该判据并在 provenance 记 `benchmark_available: false` |
| `risk_matched` | 满仓基准按策略**已实现日波动率**缩放 | `buy_hold_pct × σ_strategy/σ_benchmark`（`σ_benchmark = 0` 时退回原值并记 `risk_scale_fallback`） |
| `none` | 无基准 | **只**关闭基准判据；`dsr/psr` 与净期望仍然门控 |

校验：未知取值在**配置加载**（`Config.load`，`UnknownBenchmarkModeError`）与 **job 加载**
（`scripts/ga_worker.py::job_benchmark_mode`）都立刻失败并点名取值；缺省字段 = 跟
`config.ga_benchmark_mode`。上报（不门控）：策略 vs 基准 Sharpe、信息比率、Jensen 式 alpha/beta、
扣费后每笔净边际、在场时间占比；冠军 provenance 新增 `benchmark` 块。

对照表：`python tools/ga_benchmark_modes_table.py`（只读，逐模式各跑一次真实回测并打印基准收益、
alpha 与门结论）。测试：`tests/test_ga_benchmark_mode.py`。

### 5.5 `keep_checkpoint` — 检查点保留与续跑

`core/ga/evolver.py` 过去在**干净完成**时调用 `clear_checkpoint()`，所以跑完的任务**没有留下任何
可续跑的东西**：检查点是 `<data_dir>/data/ga_checkpoint.pkl`，上一次运行后它并不存在，`resume=True`
只能救崩溃/手动停止的任务。现在：

| 关注点 | 约定 |
|---|---|
| **缺省 = 保留** | job 字段 `keep_checkpoint` 缺失 → `config.ga_keep_checkpoint`（`config/config.yaml` 里 `true`）→ 代码缺省 `True`：**完成的运行保留检查点**，可以稍后在此基础上继续演化 |
| **显式关闭** | `keep_checkpoint: false`（job 字段或配置键）= **旧行为**：干净完成时删除检查点。**停止/崩溃**的运行无论该键为何值都保留检查点 |
| **续跑语义** | `resume: true` 从检查点的第 g 代**继续到第 g+1 代**，上界仍是本次 job 的 `generations`：`generations: 32` 从第 12 代的检查点续跑 = 只评估 13..32 代，`result["generations"] == 32` |
| **窗口守卫** | 检查点记录 `window_key`；续跑请求的窗口与它**不一致**时抛出**具名** `CheckpointWindowMismatchError`（worker 的 result 里 `error_type` 同名），**拒绝**在另一个窗口上静默续跑 |
| **试错计数（DSR 诚实性）** | 检查点保存 `prior_trials` + `trials_this_run`，续跑时把二者之和作为**下限**并入 `prior_trials`：即使 `data/ga_trials.json` 被清掉，第 13 代的 DSR 也仍以"已试过的次数"去膨胀；不会重复计数（取 `max`） |
| **可审计** | 检查点内含 generation / `window_key` / population hash / symbols / `timeframe_pool` / 试错计数；续跑后的冠军 `provenance` 记录 `keep_checkpoint`、`resumed_from_generation`、`checkpoint{...}` 与 `trials{...}` |
| **可见性** | `python scripts/ga_job_status.py`（最新任务或 `--job-id`）打印检查点行：路径、是否存在、**代数、mtime**、窗口、population hash、试错数与"续跑将从第 g+1 代继续"；`--checkpoint-file` 可指向别处。GA 面板新增 **Keep checkpoint** 勾选框（默认勾选），完成的运行显示 "checkpoint kept at generation g" 与 Resume 按钮 |

测试：`tests/test_ga_checkpoint_resume.py`、`tests/test_ga_symbols.py`。

---

## 6. 默认关闭的能力与实验开关

**"打开开关"不等于"能力生效"，更不等于"有优势"。**

| 能力 | 开关（当前值） | 为什么默认关 |
|---|---|---|
| **ML 门控** | `ml.enabled: false`；门在 `core/ml/credibility.py::credibility_gate`（OOS AUC > 0.55 **且** 净成本期望 > 0），由 `core/ml/predictor.py::_gate_config` 消费 | 实测线上模型**差于多数类**（accuracy 0.41–0.47 vs 0.54–0.67，OOS AUC 0.396–0.447）且反校准（预测 0.91 → 实际 0.22），开启是**负贡献** |
| **波动率目标化** | `risk.vol_targeting.enabled: false` | 完全 opt-in；关闭时仓位与止损宽度与固定比例实现**逐位一致**。该块里 `barrier_vol_multiple` / `barrier_min_pct` / `barrier_max_pct` 是 **RESERVED/惰性**键（无生产调用方，设成非默认值只在启动时打 WARNING） |
| **`risk.liquidity`（成交量/冲击）** | `enabled: false`、`impact_k: 0.0` | 关闭时 `PositionSizer` 不调用参与度钩子，回测成本保持审计过的 `fees + spread/2` 逐位不变（`tests/test_liquidity.py` 守护）。`impact_k` 在配置里明确标注 **ILLUSTRATIVE, NOT CALIBRATED** |
| **P4 新能力** | 模块级常量全为 `False`：`META_LABELING_ENABLED`、`PAIRS_ENABLED`、`REGIME_GATING_ENABLED`、`MICROSTRUCTURE_ENABLED` | 已实现且有独立测试，但**未接入实时链路**（默认惰性、关闭时全放行）；在真实 1h 主流币数据上配对与 meta-label 门**拒绝**全部被测一级规则 |
| **上层编排器（P7-S3）** | `ai.orchestrator.enabled: false`；实盘还需 `experimental.regime_orchestrator_live: true` **且**调用方注册 | 已实现、有独立测试，规则**固定不学习**；实测样本外它**降低回撤也降低收益**，两臂 DSR 都是 0 |

### 6.1 `experimental:` 开关（默认全关）

P4 那批能力此前只能改 Python 常量；现在一个能力一个开关，`app/config.py::apply_experimental_flags`
由 `app/main.py` 启动路径显式调用（配置全关时**不导入能力模块、不写任何常量**，信号与仓位与旧行为
**逐位一致**，`tests/test_experimental_switches.py` 用 `git worktree` 逐字节比对）。

| 开关（默认 `false`） | 常量 → 实际改动 |
|---|---|
| `engine_regime_diagnostics` | `P4_REGIME_DIAGNOSTICS_ENABLED`：把 regime 标签附进信号缓存（**只诊断，不改方向/仓位**）；这是唯一"单开即可达"的开关 |
| `engine_meta_filter` / `engine_pairs_signals` | `P4_META_FILTER_ENABLED` / `P4_PAIRS_SIGNALS_ENABLED`：已注册的 MetaLabeler / pairs provider 可过滤或替换信号，但生产链路**尚未注册**（`wire_meta_filter` / `wire_pairs_provider` 无调用者），单开无效 |
| `regime_gating` / `regime_diagnostics` | `REGIME_GATING_ENABLED` 切到因果 HMM 并启用门；`REGIME_DIAGNOSTICS_ENABLED` **无生产读取者** |
| `pairs_enabled` / `meta_labeling_enabled` / `microstructure_enabled` | `PAIRS_ENABLED` / `META_LABELING_ENABLED` / `MICROSTRUCTURE_ENABLED`：**均无生产读取者**（microstructure 连**接线缝隙都没有**），打开不改变任何可达行为 |
| `regime_orchestrator_live` | `REGIME_ORCHESTRATOR_LIVE_ENABLED`（P7-S3）：允许实盘在**发布入场前**询问已注册的上层编排器（只可能**否决入场**）。需要**三道锁**同时成立——本开关 + `ai.orchestrator.enabled: true` + 调用方 `wire_regime_orchestrator(...)` 注册；生产链路没有注册者 |

每个开关只打开**那道缝**，能力自身的验收门仍会拒绝：真实 1h 主流币配对 **0/30** 通过协整检验、
meta 门 **10/10** 拒绝一级规则、regime 只接受因果 HMM 标签。启动时会打一条 WARNING 逐项列出已开启的
开关（`experimental_notices`，与 `inert_barrier_key_warnings` 同型），未知键也会被点名而不是静默忽略。

`core/` 里仍有 GA、AI（DeepSeek 控制器 + 策略生命周期）、新闻情绪、代币启发式筛查
（`core/market_data/screener.py`）在链路上；`experimental/` 是**不在交易链路上**的死代码存档
（见 [`experimental/ml/README.md`](../experimental/ml/README.md)）。

---

## 7. 数据与数据库

| 路径 | 内容 |
|---|---|
| `data/binance_trader.db` | SQLite（WAL，v4 schema）。`--db` / `--data-dir` 可改 |
| `data/market/<SYM>/<tf>.parquet` | K 线缓存（`index = close_time`(UTC)，列 `open, high, low, close, volume`，另有 `quote_volume` / `trade_count`） |
| `data/models/`、`data/backtest/`、`data/ga_jobs/`、`data/ga_strategies/` | ML 模型产物、回测结果、GA 任务与冠军 |
| `config/config.yaml`、`config/risk_params.yaml` | 运行配置与风控阈值（深合并，`risk_params.yaml` 覆盖代码类默认值） |
| `system_config` 表 | **数据库侧**设置：自选列表 `watchlist_symbols`、手续费档位 |

`data/` 整体被 gitignore，一个文件都没跟踪：干净克隆里既没有历史数据也没有模型。

### 7.1 数据刷新

```bash
# 下载历史（--symbols / --intervals 是逗号分隔）
python scripts/download_history.py --symbols BTCUSDT,SOLUSDT --intervals 1h,4h \
    --start 2024-01-01 --end 2024-03-01 [--merge] [--concurrency 4] [--data-host URL]
python scripts/download_history.py --backfill --symbols BTCUSDT   # 补 quote_volume / trade_count 列（可续跑）
python scripts/download_history.py --list-intervals               # 合法周期

# ML 可信度测量（--symbols / --intervals 是 nargs="*"，必须空格分隔）
python scripts/ml_credibility_measure.py --symbols BTCUSDT ETHUSDT --intervals 1h
python tools/p6_volume_bars_experiment.py all --symbols BTCUSDT ETHUSDT SOLUSDT
```

> 两种 `--symbols` 写法**不能混**：`download_history.py` / `check_data_integrity.py` 收逗号分隔的
> 字符串；`ml_credibility_measure.py` / `tools/p6_volume_bars_experiment.py` 是 `nargs="*"`，
> 写成逗号会被当成**一个** symbol，随后 `FileNotFoundError`。

下载落地到 `<data-dir>/market/<SYMBOL>/<interval>.parquet`，每页最多 1000 根，页间隔 0.12s 避免限频。
Web 端等价入口是回测页的「获取数据」按钮（`POST /api/backtest/fetch-data`，trader）。

### 7.2 审计与完整性脚本（全部只读，可放进定时任务）

```bash
python scripts/audit_db.py [path\to\binance_trader.db]        # 账本对账
python scripts/check_data_integrity.py                        # K 线缓存缺口
python scripts/regen_route_baseline.py --check                # 路由基线奇偶校验
python -m compileall -q app core web db scripts alerts        # 语法完整性
```

- **`audit_db.py`**：把库复制到临时目录再以 `mode=ro` 打开（**绝不写生产库**），输出账本恒等式逐项
  计算与 `delta`、`trades` 的 action 分布、每张表行数、索引清单。恒等式是
  `10000 − Σ(open 行: quantity × entry_price) + Σ(close 行: pnl) == system_config.sim_balance`。
  退出码：`0` = 等式成立（`RESULT: IDENTITY HOLDS`）**或**空库/无 `sim_balance` 行
  （`FRESH DATABASE — NO LEDGER YET`）；非 0 = 真有漂移或库文件不存在。
- **`check_data_integrity.py`**：逐个 parquet 报 bar 数、跨度与日历缺口，超过阈值（默认 1.5 × bar 长度）
  即 `GAP`；`--strict` 在有缺口时退出 1。带缺口的序列会被波动率路径的 splice guard 拒绝，修法是按输出
  末尾给出的 `download_history.py ... --merge` 回填。
- **`regen_route_baseline.py`**：用生产同构的 `create_app`（指向一次性临时 DB 与空 config 目录）
  枚举路由并与基线对比；`--check` 只报告不写。

### 7.3 备份 / 恢复 / 清理

| 操作 | 方式 |
|---|---|
| 备份（下载文件） | `GET /api/db/backup`（admin）：先 `shutil.copy2` 到 `<db>_backup_<YYYYmmdd_HHMMSS>.db` 再以附件返回 |
| 恢复 | `POST /api/db/restore`（admin，上传文件）：**校验前 16 字节必须是 `SQLite format 3\0`**，把当前库复制成 `<db>.pre_restore` 后再覆盖 |
| 清理 | `POST /api/db/cleanup`（admin）：删 90 天前的 alerts / ai_suggestions 与 365 天前已实现 PnL 的 close/reduce 成交，然后 `VACUUM` |
| 优化 / 导出 / 删行 | `POST /api/db/optimize`（`VACUUM` + `REINDEX`）、`GET /api/db/export/{table}` → CSV、`DELETE /api/db/row/{table}/{row_id}`（均 admin） |
| **手工备份（推荐）** | **先停服务**，再复制 `data/binance_trader.db` **连同 `-wal` / `-shm`**（WAL 模式） |

`ledger_reconciliation` 表记录每一次**人工账本修复**（修复前/后余额、delta、原因、备份路径、操作者）。

### 7.4 日志与重启

loguru 默认输出到 **stderr**，项目**没有配置文件 sink**；要留档就自己重定向（`run_logs/` 已 gitignore）：

```bash
python -m app.main --mode sim >> run_logs/service.log 2>&1   # PowerShell 用 *>> run_logs\service.log
```

`Ctrl+C` 按逆序关停各组件；持仓与余额本来就在 DB 里，关机时不需要落盘。重启后的状态：持仓从
`positions` 表恢复（含 basis 与止损）、余额以 `system_config.sim_balance` 为权威、挂单从
`pending_orders` 恢复（撮合器每 ~5s 跑一次）、自选列表与手续费档位从 `system_config` 生效、
**会话在内存里所以全部失效需重新登录**（JWT 因密钥落盘仍有效）。优雅重启入口是
`POST /api/settings/restart`（admin）。

---

## 8. 常见故障

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

## 9. 测试

```bash
python -m pytest tests/ --collect-only -q -p no:cacheprovider   # 末行: 1528 tests collected
python -m pytest tests/ -q -p no:cacheprovider                  # 全量
python -m pytest tests/ -q -m "not slow"                        # 跳过慢测
```

`pytest.ini`：`testpaths=tests`、`asyncio_mode=strict`、`markers=slow`。**没有安装
`pytest-timeout`**，所以 `--timeout=` 会直接报参数错误。

唯一容易被机器负载误报的是 `tests/test_ml_credibility.py::test_feature_pipeline_cost_is_bounded`：
它断言特征流水线耗时 `< 3.0 s`，并发压力下会超时失败。看到只有这一条失败时，先单独重跑它再判断。

**干净克隆的隐含前提**：`data/` 全部 gitignore，所以没有任何缓存历史的克隆直接跑全量**不是全绿**——
`tests/test_engine_parity_variants.py` 的 7 个真实数据变体依赖 `2026-05-25..2026-05-31` 的
BTCUSDT + ETHUSDT 1h 缓存（常量就在该文件头部），缺数据时以同一句 `NO_MARKET_DATA_MESSAGE` 失败
（`tests/test_hybrid_equivalence.py` 同类用例会 skip）。CI 或新机器先下这段历史即可：

```bash
python scripts/download_history.py --symbols BTCUSDT,ETHUSDT --intervals 1h --start 2026-05-25 --end 2026-05-31
```

---

## 10. 算法层

公式与研究性描述不在本文：算法总纲见
[`docs/research/CORE_ALGORITHMS.md`](research/CORE_ALGORITHMS.md)；按子系统拆分的 **16** 篇
（`00-ERRATA` + `01`–`15`）见 [`docs/core-algorithms/`](core-algorithms/)。看公式前先读
[`00-ERRATA.md`](core-algorithms/00-ERRATA.md)：它汇总文档与代码的差异。

### 10.1 两套回测引擎，一个评估内核

| | **legacy**（`core/backtest/engine.py`，逐 tick） | **hybrid**（`core/backtest/engine_hybrid.py`，向量化两阶段） |
|---|---|---|
| 结构 | 单循环遍历时间线 | `SignalMatrixBuilder` 预生成信号矩阵 → `EventDrivenExecutor` 回放 |
| ML | 支持 LightGBM / TFT / PatchTST | **不支持**（`_select_engine` 直接抛 `ValueError`） |
| 部分减仓 `reduce_conditions` | 支持 | **不支持** |
| 成交口径 `backtest.fill_convention` | `close` 与 `next_open` 都支持 | **只支持 `close`**；`next_open` 抛具名 `FillConventionUnsupportedError`（它没有这条缝，而不是静默按旧口径成交） |

两者共用 `core/strategy/evaluation_kernel.py` 与 `core/backtest/trade_book.py::close_position`。
`backtest.engine_mode: auto`（默认）下：策略数 ≥ 3 且无 ML、无 `reduce_conditions` → hybrid，否则 legacy；
hybrid 运行期抛异常会记 warning 并**回退 legacy**。

### 10.2 成本模型：两套，故意不合并

| | **sim 成本模型** | **backtest 成本模型** |
|---|---|---|
| 代码 | `app/config.py::sim_cost_quote` | `core/backtest/cost_model.py::apply_trading_costs` |
| 作用点 | **改写成交价本身**（买贵、卖便宜），`pnl` 存入即已扣净 | **平仓时**按往返一次性叠加成本，**不动入场价**（否则止损/止盈价位会级联变化） |
| 价差来源 | `sim.cost_model.spread_pct.<SYM>` → `default` → `0.02` | override → live depth → `default_spread_pct`（`0.03`） |

`spread_pct.<SYM>` 的语义是**完整买卖价差**（%，`BTCUSDT: 0.01` = 1 bp 盘口），成本模型**每边收一半**
（`spread/2`）；sim 与 backtest 用同一约定。

```
edge_pct = 0                                     # 限价单，或 enabled=false
         | spread_pct/2 + slippage_bps/100       # 市价单
买入 fill = price × (1 + edge_pct/100)；卖出 fill = price × (1 − edge_pct/100)
fee = quantity × fill × fee_pct/100              # 档位取 maker/taker，BNB 折扣 ×0.75
slippage_usdt = |fill − price| × quantity；cost_usdt = fee + slippage_usdt
```

回测价差解析顺序固定为 `override → live（公开盘口，缓存 300s，失败结果不缓存）→ default`，并在每次
运行开始时**冻结**本次用到的每个 symbol 的价差，交易循环里不再做 I/O。**两套兜底常量不一致**
（0.03 vs 0.02）是已知的刻意口径分歧。

### 10.3 上层编排器（`ai.orchestrator`，P7-S3）

`core/ai/orchestrator.py::RegimeOrchestrator` 逐 bar 回答"策略 `s` 现在允不允许被启用"，输入 =
**因果**状态标签（`core/strategy/regime_causal.py`；样本内 HMM 标签按名拒绝，抛
`InSampleRegimeLabelError`）+ 波动率状态（`core/ml/volatility.py` 的因果 EWMA）+ **市场广度**
（`core/market_data/breadth.py`；缺失/过期按配置回退）+ 策略自己的**连亏记录**。它**不选币、不定仓、
不下单、不学习**。

| 规则 | 语义 | 配置键（shipped 值） |
|---|---|---|
| **1 状态映射** | `eligible = r(t) ∈ L(s)`；未列出的策略按 `default_action`；`range_unknown`（序列头部 31 根）按 `unknown_label_action`；整个系列缺失按 `missing_regime_action` | `regime.allowed`（`{}`）、`default_action`/`missing_regime_action`/`unknown_label_action`（全 `allow`） |
| **2 连亏熔断** | 同一状态段内连续 `N` 笔 `pnl < loss_threshold` ⇒ 锁死到本段结束；**状态变化**即清零解锁 | `kill_switch.consecutive_losses`（`0` = 关闭）、`loss_threshold`（`0.0`）、`reset_on_regime_change` |
| **3 波动率门** | `v(t) > m · med(t)`，`med` = **t 之前**尾部 `window` 个样本的中位数 | `vol.multiple`（`0.0` = 关闭）、`vol.window`（`200`）、`vol.min_samples`（`30`）、`deny_on_high_vol` |
| **4 广度门** | `up_share < min_up_share` 或 `coverage < min_coverage` ⇒ 拒绝 | `breadth.min_up_share`/`min_coverage`（未设 = 关闭）、`max_staleness_ms`（`1800000`） |

决策顺序 状态 → 熔断 → 波动率 → 广度；`reason` 是第一条命中的规则，`blocked_reasons` 列出全部，
`enabled ≡ (blocked_reasons == ())`——**规则只能拦、不能放行**。配置在**加载时**校验，未知键、未知标签
抛具名 `OrchestratorConfigError`。确定性由 `RegimeOrchestrator.replay` 保证：同规则 + 同事件流 ⇒
逐位相同的时间线。

**阈值是固定规则，不得在评估窗口上调。** S4 的纪律是"只用训练窗选/锁规则，样本外只评一次"
（`--holdout`）；看着样本外改 `ai.orchestrator` 就等于把 S4 的数字变成样本内。

实测（真实缓存，样本外 2026-02-01~06-01，BTC+ETH 1h，30 个变体，DSR 试验数 30）：

| 臂 | 成交 | 收益 % | 最大回撤 % | 在场时间 % | Sharpe | DSR |
|---|---|---|---|---|---|---|
| always-on | 1407 | −1.7600 | 2.5145 | 70.6944 | −2.7402 | 0.0 |
| orchestrated | 1229 | −2.0030 | 2.2615 | 66.7361 | −4.2916 | 0.0 |

被拒成交 178 笔、合计 PnL **+24.30 USDT**（79 胜 / 97 负）。编排器把回撤和在场时间都压低了，
也把收益与 Sharpe 压低了——它拒掉的是净盈利的敞口；两臂 DSR 都是 0.0。逐位一致由
`tests/test_p7_orchestrator.py` 用 `git worktree` 比对。详见
[`P7_REGIME_EVIDENCE.md`](overhaul/P7_REGIME_EVIDENCE.md)。
