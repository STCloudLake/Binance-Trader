# 审计报告 R7 — AI Controller + Strategy Lifecycle

**日期**: 2026-07-16
**审查文件**: `core/ai/deepseek_ctl.py`, `core/ai/prompts.py`, `core/ai/strategy_lifecycle.py`
**审查代码行数**: ~600 行
**审计方法**: API 安全性审查 + JSON 注入分析 + 全自动模式风险边界评估

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
| R7-001 | **Critical** | deepseek_ctl.py:268-276 | AI 在 full_auto 模式可设置 position_size_pct 无上限 — 可能设极大值 |
| R7-002 | High | deepseek_ctl.py:283-298 | API 调用 bare except 吞没所有异常 — rate limit/认证失败静默 |
| R7-003 | High | deepseek_ctl.py:346-349 | AI 返回的 signal_weights 无范围验证直接应用 — 权重异常 |
| R7-004 | High | deepseek_ctl.py:101-118 | breaker 决策 15s 超时 fallback 到 close_all 过于激进 |
| R7-005 | Medium | deepseek_ctl.py:109 | JSON 解析使用 strip().removeprefix() 链 — 脆弱且可绕过 |
| R7-006 | Medium | deepseek_ctl.py:309-336 | `_build_market_context()` 拼接持仓数据到 prompt — 包含不可信字段 |
| R7-007 | Medium | prompts.py | prompt 模板使用 `.format()` 拼接 — 无输入转义 |
| R7-008 | Low | deepseek_ctl.py:83-92 | `_build_breaker_context` 中裸 except 吞价格获取失败 |
| R7-009 | Low | strategy_lifecycle.py | AI 生成的策略在部署前是否经过回测验证？ |

---

## 详细分析

### R7-001 🔴 Critical — AI 可设置无上限仓位

**文件**: `core/ai/deepseek_ctl.py:268-276`

```python
if self.config.ai_mode == "full_auto":
    pct = max(result.get("position_size_pct", 5.0), 1.0)  # floor 1%, NO ceiling!
    sl = max(result.get("stop_loss_pct", 2.0), 0.5)
    lev = max(result.get("leverage", 2), 1)
    self.config.update_soft_params(
        risk_appetite=result.get("risk_appetite", "balanced"),
        position_size_pct=pct,
        stop_loss_pct=sl,
        leverage=lev,
    )
```

**描述**: full_auto 模式下 AI 可以直接修改软风控参数。虽然 `position_size_pct` 有 floor（1%），但**没有 ceiling**。如果 DeepSeek API 返回 `"position_size_pct": 100`（可能由 prompt 注入、模型幻觉、或恶意构造的 API 响应），代码会接受该值。

硬限制 `HardRiskLimits.max_position_size_pct` 在 `PositionSizer.calculate_position_size()` 中会被应用（position_sizer.py:22: `max_risk = account_balance * (self.hard.max_position_size_pct / 100)`），但这是针对**单笔交易**的。AI 如果将 `position_size_pct` 设为极大值，每笔交易都会触及硬上限，可能超过预期的组合风险。

另外，`leverage` 参数有 `max(lev, 1)` 的 floor 但无 `min(lev, max_leverage)` 的 ceiling。虽然 RiskManager 的 check_signal 会截断杠杆（manager.py:128-130），但这种多层防护中只要有一层失效就会出问题。

**复现路径**: AI 返回 `{"position_size_pct": 50, "leverage": 10}` → 参数被直接应用 → 可能违反风险策略。

**修复建议**:
```python
pct = min(max(result.get("position_size_pct", 5.0), 0.5), 20.0)  # ceiling 20%
lev = min(max(result.get("leverage", 2), 1), self.config.hard_limits.max_leverage)
```

---

### R7-002 🟠 High — API 调用异常静默吞没

**文件**: `core/ai/deepseek_ctl.py:283-298`

```python
async def _call_deepseek(self, system_prompt: str, user_prompt: str) -> Optional[str]:
    if not self.client:
        return None
    try:
        response = await self.client.chat.completions.create(...)
        return response.choices[0].message.content
    except Exception:
        return None
```

**描述**: 所有 API 调用异常（网络超时、Rate Limit、认证失败、模型不可用）全部被静默吞没并返回 `None`。调用者无法区分"API 不可用"和"无有效响应"，统一当作"跳过本次决策"处理。

在 full_auto 模式下，这可能导致：
- AI 连续多轮无法更新风险参数 → 参数保持旧值 → 市场环境变化后策略不适应
- AI 无法响应 breaker trip → fallback 到 `close_all`

**修复建议**: 区分可恢复异常（timeout、rate limit）和不可恢复异常（auth failure），分别处理。对可恢复异常实现重试。

---

### R7-003 🟠 High — AI 返回权重无边界验证

**文件**: `core/ai/deepseek_ctl.py:225-228`

```python
if self.config.ai_mode in ("semi_auto", "full_auto"):
    weights = assessment.get("signal_weights", {})
    if weights:
        self.config.update_signal_weights(**weights)
```

**描述**: AI 返回的 `signal_weights` dict 直接传给 `update_signal_weights()`，该方法不进行任何范围验证（config.py:183-186: `setattr(self.signal_weights, k, v)`）。AI 可能返回负权重、>1 的权重、或非数值。

虽然信号融合公式中 `total_weight > 0` 检查提供了部分保护，但异常权重仍会导致信号偏向。

**修复建议**: 添加权重范围验证（0.0 ~ 1.0）并在融合后归一化。

---

### R7-004 🟠 High — Breaker 决策超时 fallback 过于激进

**文件**: `core/ai/deepseek_ctl.py:114-119`

```python
except asyncio.TimeoutError:
    logger.warning("AI breaker decision timed out, fallback to close_all")
except Exception as e:
    logger.warning(f"AI breaker decision failed: {e}, fallback to close_all")
return "close_all"
```

**描述**: 任何异常（超时、JSON解析失败、API不可用）都 fallback 到 `close_all` — 即关闭所有持仓。这在半自动模式 (`semi_auto`) 中尤为危险，因为用户可能期望系统 `block_only`（仅阻止新开仓）。

例如：网络短暂不可用 → 15秒超时 → 所有持仓被强制平仓 → 用户损失。

**修复建议**: 区分模式：
- `semi_auto`: fallback 到配置的 `hard_limits.circuit_breaker_action`（用户预设值）
- `full_auto`: fallback 到 `close_all`（最安全的默认值）

---

### R7-005 🟡 Medium — JSON 解析脆弱

**文件**: `core/ai/deepseek_ctl.py:109`

```python
data = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
```

**描述**: 这个解析链试图处理 AI 返回的 markdown 代码块包裹的 JSON。但它假设：
1. 代码块标记是 ```` ```json ````（可能是 ```` ``` ```` 或 ```` ```JSON ```` ）
2. 代码块结束后没有额外字符
3. 没有嵌套的 markdown

如果 AI 返回：
```json
{"action": "close_all"}
```
解析成功。但如果返回：
```
根据分析，建议执行以下操作：
{"action": "close_all", "rationale": "市场极度恐慌"}
```
则 `removeprefix("```json")` 不匹配 → JSON 中有中文前缀 → `json.loads` 失败。

**修复建议**: 使用正则提取第一个 JSON 对象或数组，而非简单的字符串操作。

---

### R7-006 🟡 Medium — Market Context 包含不可信数据

**文件**: `core/ai/deepseek_ctl.py:309-336`

**描述**: `_build_market_context()` 将持仓数据（symbol、side、quantity、entry_price）拼接到 prompt 中。这些数据来自 executor 的内存状态，如果被外部篡改（理论上），可能构成 prompt 注入。

实际风险较低，因为这些数据来自系统内部。但如果有任何用户输入路径能影响到持仓数据（例如策略名包含特殊字符），可能影响 AI 行为。

---

### R7-007 🟡 Medium — Prompt 模板使用 .format()

**文件**: `core/ai/deepseek_ctl.py:98,134,340-388`

```python
prompt = BREAKER_ACTION_PROMPT.format(context=context)
```

**描述**: 所有 prompt 模板使用 Python 的 `str.format()` 方法。如果 `context` 字符串中包含花括号 `{` 或 `}`，会导致 `KeyError` 或格式化异常。

当前 `_build_market_context()` 和 `_build_breaker_context()` 中构建的字符串不太可能包含花括号。但如果未来添加包含 JSON 的字段（如 AI 建议的内容），可能触发此问题。

---

### R7-008 🟢 Low — Context 构建异常处理过于宽泛

**文件**: `core/ai/deepseek_ctl.py:83-92`

```python
except Exception as e:
    logger.warning(f"Breaker context: failed to read market prices: {e}")
```

每个数据源的访问异常都被裸 except 吞没。如果市场数据获取持续失败（例如 WebSocket 断开），AI 将在不完整上下文中做决策，但不会得到明确告警。

---

### R7-009 🟢 Low — AI 生成策略无强制回测验证

**文件**: `core/ai/strategy_lifecycle.py`

**描述**: `generate_strategy()` 使用 AI 生成新策略。虽然 lifecycle manager 有回测验证步骤，但如果 `skip_backtest` 参数被设为 True（或回测引擎不可用），策略可能未经验证就被部署。

**修复建议**: 在部署前强制要求回测通过（最低 Sharpe、最低胜率等门槛），即使是 AI 生成的策略。

---

## 累计统计

| 等级 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | 累计 |
|------|-----|-----|-----|-----|-----|-----|-----|------|
| Critical | 1 | 1 | 2 | 1 | 0 | 1 | 1 | 7 |
| High | 4 | 3 | 1 | 3 | 2 | 2 | 3 | 18 |
| Medium | 5 | 3 | 2 | 4 | 4 | 4 | 3 | 25 |
| Low | 3 | 2 | 1 | 1 | 2 | 2 | 2 | 13 |
| **总计** | | | | | | | | **63** |
