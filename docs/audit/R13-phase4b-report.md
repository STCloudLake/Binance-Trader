# 审计报告 R13 — ML 预测目标转向（波动率预测）

**日期**: 2026-07-20
**阶段**: Phase 4b
**类型**: ML 架构变更 + 新功能

---

## 变更范围

| 文件 | 变更类型 | 行数变化 |
|------|----------|----------|
| `core/ml/features.py` | 新增 3 个 label 函数 + Hurst 估计器 | +170 |
| `core/ml/predictor.py` | 双模型架构（方向 + 波动率） | +70 |
| `core/risk/position_sizer.py` | 波动率自适应仓位 + 止损 | +15 |
| `core/backtest/engine.py` | 回测引擎波动率启发式 | +10 |

---

## 为什么做这个变更

**理论基础**：在不完全有效的市场中，波动率的可预测性远高于方向的可预测性（GARCH, Engle 1982; HAR, Corsi 2009）。将 ML 从方向预测转向波动率状态预测，建立在更坚实的金融时间序列理论基础上。

**变更前**：ML 模型只输出一个 `confidence ∈ [0,1]`（P(方向涨)），与方向预测耦合。

**变更后**：ML 模型输出两个独立信号：
- `confidence` — 方向置信度（向后兼容，保留）
- `volatility_expanding` — 波动率是否将扩张（新）

---

## 新增 Label 函数

### `create_volatility_label(df, forward_periods=20)`
- 比较未来 20 根 K 线的实现波动率 vs 当前 20 根 K 线波动率
- 输出：1（扩张）/ 0（收缩）
- 学术基础：GARCH 波动聚集 + HAR 异质自回归

### `create_regime_label(df, forward_periods=20, hurst_window=100)`
- 计算未来窗口的 Hurst 指数
- 输出：1（H > 0.55 趋势市）/ 0（H ≤ 0.55 震荡市）
- 学术基础：Mandelbrot & Wallis (1969) R/S 分析

### `create_volatility_magnitude_label(df, forward_periods=20)`
- 未来波动率在历史分布中的分位数 ∈ [0, 1]
- 回归任务，用于精确仓位调整

---

## 双模型架构

```
MLPredictor
  ├── _models["{symbol}_binary"]     → 方向预测 (LightGBM/TFT)
  ├── _vol_models["{symbol}_vol"]   → 波动率预测 (LightGBM)
  └── _tft_models["{symbol}"]       → TFT 方向模型 (可选)
```

### 训练
- `train_model()` — 方向模型（现有，不变）
- `train_volatility_model()` — 波动率模型（新增）

### 事件格式
```python
Event(ML_PREDICTION, {
    "symbol": "BTCUSDT",
    "confidence": 0.72,           # 方向（向后兼容）
    "volatility_expanding": True, # 波动率状态（新）
})
```

### 回退机制
当波动率模型未训练时，使用启发式方法（最近 10 根 vs 20 根 K 线的 std 比较）。

---

## 仓位管理集成

`PositionSizer.calculate_position_size()` 新增 `volatility_expanding` 参数：
- 波动率扩张时：仓位 × 0.7（缩仓）
- 止损 × 1.3（放宽，避免被波动扫损）

---

## 验证结果

- ✅ 3 个新 label 函数在合成数据上产出合理分布
- ✅ `_estimate_hurst` 区分随机游走 vs 趋势 vs 均值回归
- ✅ 双模型架构不影响现有方向预测路径
- ✅ PositionSizer 向后兼容（volatility_expanding 默认 False）
- ✅ 回测引擎仓位计算使用波动率信号

---

## 审计发现

| ID | 严重度 | 描述 |
|----|--------|------|
| R13-001 | Medium | `create_regime_label()` 使用 O(n²) 循环计算每根 bar 的 Hurst。500 bar 数组约 200ms。批量回测时可用并行优化 |
| R13-002 | Medium | 波动率模型启发式回退（最近 std vs 历史 std）在剧烈行情切换时可能滞后 |
| R13-003 | Low | `create_volatility_magnitude_label` 在短数据集上（< 200 bar）产出大量 NaN，需足够热身数据 |
| R13-004 | Note | TFT/PatchTST 路径未集成波动率预测，仅 LightGBM 支持 |

---

## 状态

**Phase 4b 完成。** | **新增**: 240+ 行 | **修改**: 4 文件 | **发现**: 2 Medium, 2 Low
