# 全币种行情 / 币种信息 / 数据 / 代币检测 — 接口与页面契约（冻结版）

**日期**: 2026-09-29
**前置事实（已实测，务必遵守）**

| 主机 | 结果 | 用途 |
|------|------|------|
| `https://api.binance.com` 及 api1–4 | **连不通（超时）** | 不可用 |
| `https://testnet.binance.vision` | 可达 | 当前下单/账户用（testnet） |
| **`https://data-api.binance.vision`** | **可达**（3716 symbols，**496 个 USDT/TRADING**，1h 历史回溯到 2017-08-17，K 线无缺口） | **行情数据统一走这里**（主网公开行情镜像） |
| **`wss://data-stream.binance.vision/ws/...`** | **可达**（已实测收到 BTCUSDT kline） | **实时流统一走这里** |
| `wss://stream.binance.com` | 连不通 | 不可用 |

**结论**：行情（K 线、24h、盘口、成交、exchangeInfo、历史下载）全部切到 `.vision` 主网公开域名；
下单/账户继续用 testnet 客户端（`binance.testnet`）。这样"看到所有币种 + 与币安数据对齐"才成立。

---

## 0. 全局约束

1. **不得**再用 `AsyncClient.create()` 无参数或硬编码 `api.binance.com`。
2. 行情 REST 走 `config.market_data_host`（默认 `https://data-api.binance.vision`）；
   实时流走 `config.market_stream_host`（默认 `wss://data-stream.binance.vision`）。
3. 读取接口登录即可；写接口（自选列表、下载数据）要求 `_require_trader`。
4. 现有 107 条路由的 method+path 不得改变，只能新增（`docs/overhaul/route-baseline.json`）。
5. 全量测试必须保持全绿（当前 296 项）。
6. 所有新接口返回结构化 JSON，失败给 `{"error": "..."}` + 合理状态码，不抛裸栈。
7. 外部行情请求必须有 **超时 + 缓存**（见各接口），避免页面轮询打爆交易所或卡死线程。

---

## 1. 新增行情接口（`web/routes/market.py`，W1 负责）

| 路由 | 参数 | 响应（要点） |
|------|------|------|
| `GET /api/market/symbols` | `q`(搜索，匹配 symbol/baseAsset), `quote`(默认 USDT), `limit`(默认50, 1..200), `offset`(默认0), `sort`(`volume|change|symbol`, 默认 `volume`) | `{"total":496,"offset":0,"limit":50,"symbols":[{"symbol","baseAsset","quoteAsset","status","last","change_pct","high","low","volume","quote_volume","count","listing_date","tick_size","step_size","min_notional","has_cached_data"}...]}` |
| `GET /api/market/ticker24h` | 无（全市场） | `{"updated_at":...,"tickers":{"BTCUSDT":{"last","change_pct","high","low","volume","quote_volume","count"}...}}`，**缓存 60s**（上游一次 13s，绝不能每次请求都打） |
| `GET /api/market/watchlist` | 无 | `{"symbols":["BTCUSDT",...],"max":30}` |
| `POST /api/market/watchlist` | Form `symbols`(逗号分隔) | 校验全部存在于 universe 且 `status=TRADING`；上限 30；持久化；返回 `{"ok":true,"symbols":[...],"restart_required":true}` |
| `GET /api/coin/{symbol}` | 无 | 见 §2 |
| `GET /api/data/overview` | 无 | 见 §3 |
| `GET /api/kline/{symbol}` | 既有，**改为走 data host** | 不改变响应形状（`time` 为**秒**） |

**缓存**：`ticker24h` 60s；`symbols` 60s（基于 ticker24h 缓存）；`depth/trades` 保留现有 2s TTL；`coin/{symbol}` 30s。

**`has_cached_data`**：`data/market/{symbol}/{interval}.parquet` 是否存在（供回测/策略筛选）。

## 2. `GET /api/coin/{symbol}` — 币种信息（W1）

```json
{
  "symbol": "SOLUSDT", "base_asset": "SOL", "quote_asset": "USDT", "status": "TRADING",
  "listing_date": "2020-08-11",
  "filters": {"tick_size": 0.01, "step_size": 0.001, "min_notional": 5.0},
  "ticker": {"last":120.18,"change_pct":1.2,"high":130.0,"low":110.0,"volume":5362.0,"quote_volume":640000.0,"count":120000},
  "depth_summary": {"spread":0.01,"spread_pct":1.2e-05,"bid_depth_1pct":123456.0,"ask_depth_1pct":234567.0,"bid_total":...,"ask_total":...},
  "performance": {"d1":1.2,"d7":-3.4,"d30":12.0,"d90":-8.0},
  "risk": {
     "volatility_30d_annualized": 0.85, "max_drawdown_90d": 32.1,
     "liquidity_score": 0..100, "spread_score": 0..100, "volume_score": 0..100,
     "overall_score": 0..100, "flags": ["低流动性","价差偏大",...]
  },
  "correlation_btc_30d": 0.72,
  "cached_intervals": ["1m","5m","15m","1h","4h"]
}
```
- 指标口径写在源码注释里；数据不足（如新币历史短）时字段为 `null` 而非报错。
- `risk` 为**启发式指标**（基于交易所公开数据），**不是链上合约审计**，页面上必须标明。

## 3. `GET /api/data/overview` — 全市场数据页（W1）

```json
{
  "updated_at": "...",
  "totals": {"symbols": 496, "quote_volume_usdt": 1.2e10, "up": 300, "down": 180, "flat": 16},
  "top_gainers": [{"symbol","last","change_pct","quote_volume"}...],   // 前 10
  "top_losers":   [...],
  "top_volume":   [...],
  "most_active":  [{"symbol","count","quote_volume"}...],             // 按成交笔数
  "volatility_leaders": [{"symbol","volatility_30d_annualized"}...],   // 采样前 100 名成交额币种即可
  "spread_widest": [{"symbol","spread_pct"}...],
  "btc_eth_share": {"BTCUSDT": 0.42, "ETHUSDT": 0.18}                  // 占统计池成交额比例（近似）
}
```

## 4. 代币检测（W2 负责后端与页面）

| 路由 | 说明 |
|------|------|
| `GET /api/audit/screen` | `limit`(默认50, ≤200), `min_quote_volume`(默认100000), `sort`(`score|volume`), `interval`(默认 `1h`) → `{"updated_at","results":[{"symbol","overall_score","risk_level":"low|medium|high","flags":[...],"metrics":{...}}]}` |
| `GET /api/audit/{symbol}` | 单个币种的完整检测明细（指标 + 每个 flag 的依据 + 原始数值） |

**评分维度（全部由币安公开数据推导，必须在页面显著位置声明"非链上合约审计"）**：
1. **流动性**：24h 计价成交额、盘口 ±1% 深度、价差百分比
2. **波动性**：30d 年化波动率、90d 最大回撤、单根 K 线异常涨跌（>15% 次数）
3. **价格质量**：上下影线占比、连续同向缺口、成交额/市值代理（volume ratio）
4. **上市时间**：由最早 K 线推断，<30 天标记"新上市"
5. **集中度代理**：平均单笔成交额（quote_volume/count）异常大或极小
输出 0–100 分与 `low/medium/high`，并给出**每条 flag 的具体数值依据**。

## 5. 页面（前端各执行者）

| 页面 | 模板 | 要点 |
|------|------|------|
| `/market` 行情/币种信息 | `web/templates/market.html` | 全币种可搜索/排序/分页表格（24h 涨跌、价、量、成交额、笔数、是否有本地数据），支持仅看自选、点击行跳 `/trade?symbol=`、顶部涨跌统计 |
| `/coin/{symbol}` 币种详情 | `web/templates/coin.html` | 基本信息与交易规则、24h、盘口摘要、多周期表现（1d/7d/30d/90d 小型走势图）、风险评分与 flags、相关性与缓存区间、"去交易"/"去检测"按钮 |
| `/data` 数据 | `web/templates/data.html` | 市场总览卡片 + 涨幅榜/跌幅榜/成交额榜/最活跃/波动率榜/价差榜，各榜可点击跳转 |
| `/audit` 代币检测 | `web/templates/audit.html` + 详情抽屉 | 检测列表（分数、等级、flags、关键指标）+ 单币明细；顶部醒目声明"启发式，非链上审计" |
| `/trade` 现货交易 | `web/templates/trade.html`（W4 改造） | K 线修复与增强；交易对下拉改为**全币种搜索选择**；底部新增"策略信号"面板 |
| 回测/策略 | `backtest.html`、`strategies.html`（W5） | 币种多选（全币种搜索）、显示哪些币种有本地数据、一键下载所选币种数据 |

**路由由 Lead 统一加在 `web/routes/pages.py`**（模板名如上，已冻结）：`/market`、`/coin/{symbol}`、`/data`、`/audit`；
`/` 与 `/dashboard` 改为 **302 → `/trade`**（仪表盘职能与现货页重复，只保留现货页）。

## 6. K 线显示问题（W4 负责修复，必须给出前后对比证据）

已知事实：`/api/kline` 返回 `time` 为 **epoch 秒**；主网 K 线连续无缺口；testnet 亦连续。
排查并修复方向：
1. 默认只取 200 根且**无 dataZoom**，视觉上"挤在一起"；
2. 周期只有 1m/5m/15m/1h/4h，缺少 1d/1w；
3. 十字光标/提示框信息量少（无开高低收、涨跌幅、量）；
4. 切换交易对或周期时若请求失败，图表残留旧数据且无错误提示；
5. 时间轴与蜡烛的 `time` 单位必须统一为毫秒后再喂给 ECharts；
6. 成交量柱需与蜡烛共用 x 轴缩放。
修复后需提供：修复前后同一交易对的截图或 DOM/数据对比、以及"数据条数/时间范围/渲染条数"的实测数字。

## 7. 交付与验收

- 每个执行者交付时附：改动文件清单、行数、`python -m pytest tests/ -q` 结果、`compileall` 结果、
  路由基线校验（`python "$env:TEMP\bt_route_verify.py"` 应显示 EXACT MATCH 或仅新增预期路由）、
  以及自己端到端的实测输出（真实主网数据，禁止只跑单测）。
- Lead 最终验收：真实服务 + 临时会话逐页逐接口实测；任何页面引用的接口都必须返回 200。
