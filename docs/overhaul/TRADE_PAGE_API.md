# 交易页面（Binance 风格）接口契约 — 冻结版

**日期**: 2026-09-29
**目的**: 复刻 Binance 现货交易页的信息密度与布局，替换现有过于简单/数据过少的交易界面。

---

## 硬约束（实现者必读）

1. **只用 testnet 可达的数据源**：本机到主网 `api.binance.com` **不通**（10s 超时），到 testnet 正常。
   任何新建的 Binance 客户端必须走 `app/config` 配置（`testnet=config.binance_testnet` + keys），
   禁止 `AsyncClient.create()` 无参数写法。可参考 `web/routes/dashboard_partials.py::_configured_client`。
2. **认证**：所有 `/api/market/*`、`/api/account`、`/api/orders` 读取接口要求已登录（viewer 起）；
   **写接口**（下单、撤单）要求 trader 起，用 `web/deps.py::_require_trader`。
3. **不破坏现有行为**：现有 97 条路由的 method+path 必须保持不变（只能新增）；`tests/` 现有 235 项必须继续通过。
4. **不写 `web/server.py` 以外的人负责的文件**：见下方分工。
5. 所有新接口失败时必须返回结构化 JSON（`{"error": "..."}` + 合理 HTTP 码），不得抛 500 裸栈；
   交易所不可达时前端要能显示原因。
6. 金额/数量统一：`amount_usdt` 为计价货币金额，`quantity` 为基础资产数量；金额保留 2 位，数量保留 6 位。

---

## 一、市场数据（GET，登录即可）

| 路由 | 参数 | 响应 |
|------|------|------|
| `/api/market/ticker` | `symbol` | `{"symbol","last","open","high","low","volume","quote_volume","change","change_pct","time"}`（24h 统计，来自 testnet `get_ticker(symbol=...)`；字段缺失用 `null`） |
| `/api/market/depth` | `symbol`, `limit`(默认20, 1..100) | `{"symbol","lastUpdateId","bids":[[price,qty],...],"asks":[[price,qty],...],"spread","spread_pct","bid_total","ask_total"}`，bids 降序、asks 升序 |
| `/api/market/trades` | `symbol`, `limit`(默认30, 1..100) | `{"symbol","trades":[{"id","price","qty","quote_qty","time","is_buyer_maker"}]}` 最新在前 |
| `/api/market/overview` | 无 | `{"symbols":[{"symbol","last","change_pct","quote_volume"}...], "updated_at"}`，覆盖引擎监控的 5 个交易对（BTCUSDT/ETHUSDT/BNBUSDT/SOLUSDT/XRPUSDT） |
| `/api/kline/{symbol}` | `interval`, `limit` | **已存在，不要改**（图表用） |

> 缓存：`/api/market/*` 允许 1–2 秒级内存缓存（同一 symbol 的并发请求合并），避免每次刷新新建交易所连接。

> **时间单位（已确认，前端已兼容两种）**：
> - `/api/market/ticker.time`、`/api/market/trades[].time` → **epoch 秒**（实现如此，测试已断言）
> - `/api/kline` → **epoch 毫秒**（与 Binance 原始 payload 一致，历史接口保持不动）
> 消费方不要假设两者单位相同；前端用 `toMs()`（值 < 1e11 时 ×1000）统一处理。


## 二、账户与持仓（GET，登录即可）

`/api/account` → 
```json
{
  "balance": 8528.96, "available": 7000.0, "frozen": 0.0,
  "equity": 8528.96, "positions_value": 1528.96, "unrealized_pnl": -3.21,
  "positions": [{
     "symbol": "BTCUSDT", "side": "long", "quantity": 0.00269, "entry_price": 84055.6,
     "current_price": 84242.0, "unrealized_pnl": 0.5, "pnl_pct": 0.22,
     "position_value": 226.0, "amount_usdt": 226.0, "stop_loss": 82374.5,
     "strategy_name": "ga_champion_...", "position_type": "core"
  }],
  "pending_count": 1, "mode": "sim"
}
```
- `available = balance - frozen`；`frozen` = 所有未成交限价单的 `amount_usdt` 之和。
- `equity = balance + positions_value`（`balance` 为现金）。
- 数据来源：`app.state.executor.get_open_positions()`、`app.state.balance`/`load_sim_balance`、
  `app.state.get_price`（引擎价格缓存）；**不要求**实时交易所调用。

## 三、订单（写接口需 trader）

### 下单 `POST /api/order`（Form）
`symbol, side(long|short), type(market|limit), amount_usdt, price(限价必填), position_type(core|satellite), stop_loss_pct`

- **market**：构造 signal → `risk_manager.check_signal()` → 通过后 `event_bus.publish(ORDER_REQUEST)`
  （与现有 `/api/trade` 同路径，复用而不是重写风控）。返回 `{"ok":true,"order":{"symbol","side","type":"market","quantity","price","amount_usdt","status":"filled"}}`
- **limit**：校验（方向、价格>0、金额>0、余额足够）→ 冻结资金（不扣现金，仅计入 `frozen`）→
  写入待成交订单 → 返回 `{"ok":true,"order":{"id","symbol","side","type":"limit","price","quantity","amount_usdt","status":"open"}}`
- 失败：风控拒绝 → `{"ok":false,"error":"风控拒绝: ..."}`（HTTP 200 或 400 均可，需结构化）；
  参数非法 → 400 `{"error":"..."}`。

### 查询 `GET /api/orders`
`status=open|history|all`（默认 all）、`limit`(默认50, 1..200) →
```json
{"orders":[{"id":12,"symbol":"BTCUSDT","side":"long","type":"limit","price":82000.0,
  "quantity":0.0012,"amount_usdt":98.4,"status":"open|filled|cancelled",
  "created_at":"...","filled_at":"...","fill_price":null,"reason":"..."}]}
```

### 撤单 `POST /api/orders/{order_id}/cancel`
→ `{"ok":true,"order":{...status:"cancelled"}}`；不存在 → 404 `{"error":"order not found"}`。

### 成交历史 `GET /api/history/trades`
`limit`(默认50, 1..200) → `{"trades":[{"id","symbol","side","entry_price","exit_price","quantity","pnl","pnl_pct","strategy","exit_reason","opened_at","closed_at"}]}`（读 `trades` 表，`status='closed'`，最新在前）。

## 四、限价单撮合（后端行为）

- 待成交订单持久化到 SQLite 新表 `pending_orders`（新增 `db/database.py` 的 schema 迁移，`schema_version` 递增）。
- 后台循环（建议 5 秒）用引擎价格（`get_price`，回退 `get_historical` 最后一根收盘）判断：
  - 买单：`last <= price` → 成交；卖单（short）：`last >= price` → 成交。
  - 成交时走与市价单**完全相同**的风控 + 下单路径（`RiskManager.check_signal` → `ORDER_REQUEST`），
    并把订单标记 `filled`（含 `fill_price`、`filled_at`）。风控拒绝则标记 `cancelled` 并写 `reason`。
  - 订单生命周期不因重启丢失（从表恢复）。
- 循环任务需可被优雅停止（跟随组件 `stop()`）。

## 五、前端页面（另一个执行者负责）

- 新页面 `GET /trade`（模板 `web/templates/trade.html`），导航栏加入口（布局参考 Binance 现货：顶部行情条 / 左侧订单簿 + 最新成交 / 中间 K 线 + 周期切换 + 指标 / 右侧下单面板 + 账户 / 底部 委托·持仓·成交历史 标签页）。
- 只消费上述契约中的接口；不得新增后端接口。
- 轮询节奏：订单簿/最新成交 2s，账户/委托 3s，K 线 5s；页面隐藏时暂停（`document.hidden`）。
- 风格与现有 UI 一致（Tailwind 深色 + ECharts 5.5 CDN，已在 `base.html` 引入）。

## 五之二、手续费 / 滑点 / 预估成本（2026-09-29 追加冻结）

**目标**：交易页可查看手续费等级、按当前下单输入实时预估手续费与滑点；**模拟盘真实计入手续费与滑点**。

### 配置（`config/config.yaml` 新增 `sim:` 段）
```yaml
sim:
  cost_model:
    enabled: true
    fee_tier: VIP0            # 见 /api/fee/tier 的档位表
    use_bnb_discount: false    # 勾选后按 75 折计入
    slippage_bps: 2            # 市价单每边滑点（基点）；限价单为 0
    spread_pct:                # 每边半分价差（%），缺省用 0.02
      BTCUSDT: 0.01
      ETHUSDT: 0.02
```

### 模拟盘成本模型（`core/executor/executor.py::_execute_sim`）
- **买入**：成交价 = 现价 × (1 + (spread/2 + slippage)/100)，并计 `fee = 名义 × taker%`（BNB 折扣生效时 ×0.75）。
- **卖出/平仓**：成交价 = 现价 × (1 − (spread/2 + slippage)/100)，同样计 fee。
- 成本从余额中真实扣除，使 `pnl` 为**净额**；同时把 `fee`、`slippage`、`fill_price` 落库（迁移 v3 给 `trades` 加列）。
- 我的反例检查：一个来回约 500 USDT 的仓位，taker 0.1% × 2 + 滑点/价差约 2-4bp，单次往返成本约 **1.2–1.5 USDT**。

### 新接口（`web/routes/market.py` 或新 `web/routes/fees.py`）
| 路由 | 参数 | 响应 |
|------|------|------|
| `GET /api/fee/tier` | 无 | `{"current":"VIP0","use_bnb_discount":false,"bnb_discount_pct":25,"tiers":[{"tier":"VIP0","maker_pct":0.1000,"taker_pct":0.1000}, ... VIP1..VIP9],"note":"..."}` |
| `GET /api/fee/estimate` | `symbol,side(long/short),type(market/limit),amount_usdt,price(限价时)` | `{"symbol","type","notional","fee_pct","fee_usdt","slippage_bps","spread_pct","slippage_usdt","total_cost_usdt","effective_price","notes":[...]}` |
| `GET /api/account`（扩展） | — | 增加 `fees_paid_total`、`slippage_paid_total`、`net_pnl_total` |
| `GET /api/history/trades`（扩展） | — | 每条成交在可用时增加 `fee`、`slippage`、`fill_price`、`net_pnl` |

**说明（必须在页面标注）**：手续费等级为**手动选择**（本机无法读取币安账户的 30 天交易量与 BNB 持仓，主网账户接口不可达），档位表为币安现货标准费率。

## 六、验证要求
- 后端：`python -m pytest tests/ -q` 全绿；新增 `tests/test_market_api.py` 覆盖各接口的形状、
  鉴权（viewer 可读 / viewer 不可下单）、限价单挂单→撮合→撤单、风控拒绝路径；
  用临时 DB 与假行情，**不得**依赖网络或真实 `data/binance_trader.db`。
- 前端：用临时会话实测 `GET /trade` 200 且包含订单簿/下单面板/标签页容器；页面内引用的每个接口都能返回 200。
- 两者完成后由 Lead 统一验收（路由表、全量测试、真实 testnet 端到端）。
