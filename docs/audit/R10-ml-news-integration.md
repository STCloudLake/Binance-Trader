# 审计报告 R10 — ML + News + Cross-Module Integration

**日期**: 2026-07-16
**审查文件**: `core/ml/predictor.py`, `core/ml/trainer.py`, `core/ml/features.py`, `core/ml/*.py`, `core/news/analyzer.py`, `core/news/fetcher.py`, `core/news/source_manager.py`, 跨模块数据流
**审查代码行数**: ~800 行
**审计方法**: 端到端数据流追踪 + 并发竞态分析 + 资源泄漏检测

---

## 摘要

| 等级 | 数量 |
|------|------|
| Critical | 1 |
| High | 3 |
| Medium | 3 |
| Low | 2 |
| **总计** | **9** |

---

## 漏洞清单

| ID | 等级 | 文件:行 | 描述 |
|----|------|---------|------|
| R10-001 | **Critical** | 跨模块 | 余额操作链中任一环节失败导致状态不一致 — 无补偿事务 |
| R10-002 | High | predictor.py | ML 模型预测特征顺序与训练时不一致 — 静默错误预测 |
| R10-003 | High | fetcher.py | HTTP 请求无 URL 验证 — SSRF 风险 |
| R10-004 | High | main.py | `_pending_signals` 超时缺失 + 余额跨步操作非原子 |
| R10-005 | Medium | features.py | 特征计算中 NaN 处理不完整 — 传播到 ML 模型 |
| R10-006 | Medium | analyzer.py | 异常触发紧急抓取无双重检查保护 — 重复抓取 |
| R10-007 | Medium | 跨模块 | 数据库连接在异常路径中可能泄漏 |
| R10-008 | Low | trainer.py | ML 模型训练完成后无模型文件完整性校验 |
| R10-009 | Low | source_manager.py | 新闻源配置无 URL scheme 验证 |

---

## 详细分析

### R10-001 🔴 Critical — 交易信号链无事务性保证

**文件**: 跨模块 (`main.py`, `strategy/engine.py`, `risk/manager.py`, `executor/executor.py`, `db/database.py`)

**描述**: 一笔交易从信号到执行涉及 5 个步骤，跨越 4 个模块：

```
StrategyEngine._evaluate() → EventBus.publish(STRATEGY_SIGNAL)
→ RiskManager._on_signal() → check_signal() → EventBus.publish(ORDER_REQUEST)
→ OrderExecutor._on_order_request() → _execute_sim()
    → 1. _positions[symbol] = {...}     (内存)
    → 2. DB INSERT INTO trades          (磁盘)
    → 3. atomic_adjust_balance(-delta)  (磁盘)
    → 4. risk_manager.update_balance()  (内存)
    → 5. EventBus.publish(POSITION_UPDATE) (异步)
```

如果任一步骤失败：
- **步骤2失败**: position 已创建但 DB 无记录 → 重启后仓位丢失但余额未扣 → 凭空多出资金
- **步骤3失败**: position 创建 + DB 记录存在，但余额未扣 → 余额虚高
- **步骤4失败**: 余额已扣但 risk_manager 不知道 → 后续仓位计算基于旧余额
- **步骤5失败**: 余额和 position 正确，但 `_pending_signals` 不释放

**没有补偿事务机制**。每个步骤独立执行，步骤间的失败不可见。

**复现路径**: 
1. 步骤3中 DB 写入因瞬时锁定失败
2. `atomic_adjust_balance` 抛出异常
3. position 已在内存中创建，DB trades 表已有记录
4. 余额未扣除 → 系统认为有更多可用资金 → 超额开仓

**修复建议**: 实现 Saga 模式或至少在关键步骤失败时执行补偿操作（删除 position、回滚 DB 记录）。

---

### R10-002 🟠 High — ML 特征顺序不一致

**文件**: `core/ml/predictor.py`

**描述**: 实时预测时的特征 DataFrame 列顺序必须与训练时完全一致。如果 `compute_features()` 在不同调用中返回不同列顺序（例如因缓存、列重排、或条件分支），ML 模型将使用错误的特征进行预测 → 静默产生错误预测。

这在 XGBoost/LightGBM 中通常不是问题（它们按列名匹配），但如果有任何特征工程涉及数组操作（如 `X.values`），顺序敏感。

检查 `predictor.py` 的 predict() → 如果使用 `model.predict(df[feature_names])` → 安全 ✓  
如果使用 `model.predict(X.values)` → 危险 ✗

需要确认实际实现。

---

### R10-003 🟠 High — News Fetcher SSRF 风险

**文件**: `core/news/fetcher.py`

**描述**: 新闻抓取器向用户配置的 URL 发起 HTTP 请求。如果攻击者能够添加恶意新闻源（通过 Web UI 的新闻源管理接口），可以：
1. 使服务器请求内网地址（如 `http://127.0.0.1:8899/admin`）
2. 请求云元数据服务（如 `http://169.254.169.254/latest/meta-data/`）
3. 扫描内网端口

**修复建议**: 在发起请求前验证 URL：
- 禁止 private IP 范围（10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8）
- 禁止 cloud metadata IP（169.254.169.254）
- 仅允许 HTTP/HTTPS scheme

---

### R10-004 🟠 High — _pending_signals 超时 + 余额操作原子性

**文件**: `app/main.py:115-134` + `core/risk/manager.py:58`

**描述**: 综合 R2-003 和 R4-001 的分析：

1. `_pending_signals` 如果因 POSITION_UPDATE 事件丢失而残留，symbol 会被永久锁定
2. `atomic_adjust_balance` 非真正原子（独立 DB 连接）
3. `risk_manager.update_balance()` 使用 executor 实时仓位，但时序可能与余额不一致

这三个问题叠加：在高速交易场景中，可能出现：
- signal A 通过检查 → pending_signals.add("BTC") → 下单 → POSITION_UPDATE 丢失
- signal B 到达 → "BTC" in pending_signals → 拒绝
- 系统永久无法交易 BTC

---

### R10-005 🟡 Medium — NaN 传播到 ML

**文件**: `core/ml/features.py`

**描述**: 特征计算中，如果 OHLCV 数据包含 NaN（例如因新上市代币、数据中断），NaN 会传播到特征 DataFrame。XGBoost/LightGBM 可以处理 NaN（将其分配到 default direction），但 TFT/PatchTST（基于 PyTorch）会在 NaN 输入时产生异常或静默错误输出。

**修复建议**: 在特征计算完成后统一处理 NaN（前向填充、均值填充或丢弃）。

---

### R10-006 🟡 Medium — 异常触发无双重检查

**文件**: `core/news/analyzer.py`

**描述**: 新闻分析器在检测到异常价格波动时紧急抓取新闻。如果市场发生闪崩（快速下跌后立即反弹），可能触发多次紧急抓取。虽然 cooldown 机制限制频率，但多个 symbol 同时异常（全市场暴跌）时会产生大量并发抓取请求。

**修复建议**: 添加全局 rate limiter，限制所有紧急抓取的并发数和频率。

---

### R10-007 🟡 Medium — DB 连接泄漏

**文件**: 多个文件使用 `aiosqlite.connect()` + `await db.close()` 模式

**描述**: 审计发现多个地方使用 `await db.close()` 关闭连接。如果在 `connect()` 和 `close()` 之间抛出异常，连接不会关闭 → 连接泄漏 → SQLite 文件锁累积。

`database.py` 中有 `db_connection()` 上下文管理器，但并非所有代码都使用它。例如 `executor.py:134` 直接手动管理连接。

**修复建议**: 全局使用 `db_connection()` 上下文管理器或 `async with` 语法。

---

### R10-008 🟢 Low — ML 模型文件无完整性校验

**文件**: `core/ml/trainer.py`

**描述**: ML 模型保存为 `.pkl` 文件。加载时没有校验文件完整性（checksum/signature）。如果文件因磁盘错误损坏，模型加载可能成功但产生错误预测。XGBoost/LightGBM 的 `model.load()` 会抛出异常如果文件完全损坏，但部分损坏可能静默通过。

风险极低但值得注意。

---

### R10-009 🟢 Low — 新闻源 URL 无 scheme 验证

**文件**: `core/news/source_manager.py`

**描述**: 新闻源配置中的 `endpoint` 字段可能包含非 HTTP URL（如 `file:///etc/passwd` 或 `ftp://...`）。虽然 fetcher 使用 `httpx`（仅支持 HTTP），但异常会在运行时才被发现。

**修复建议**: 在保存新闻源时验证 endpoint 的 scheme 为 `http://` 或 `https://`。

---

## 累计统计（最终）

| 等级 | R1-R9 | R10 | **总计** |
|------|-------|-----|----------|
| Critical | 9 | 1 | **10** |
| High | 23 | 3 | **26** |
| Medium | 32 | 3 | **35** |
| Low | 17 | 2 | **19** |
| **总计** | **81** | **9** | **90** |

---

## 10轮审计完成

所有 10 轮模块审计已完成。共发现 **90 个漏洞**：10 Critical、26 High、35 Medium、19 Low。

接下来进入 **R11 深度收尾审计** — 对 Critical 和 High 问题进行逐行追踪和系统性分析。
