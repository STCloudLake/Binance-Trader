# 审计报告 R14 — 市场结构特征

**日期**: 2026-07-20
**阶段**: Phase 4c
**类型**: 新指标 + ML 特征扩展 + GA 进化池扩展

---

## 变更范围

| 文件 | 变更内容 | 行数变化 |
|------|----------|----------|
| `core/strategy/indicators.py` | 3 个新指标分支 + 4 个辅助函数 | +160 |
| `core/ml/features.py` | DEFAULT_FEATURES 30→40 维 + 计算逻辑 | +30 |
| `core/ga/genome.py` | 3 个新 INDICATOR_NAMES + 条件模板 + 基因编码 | +35 |

---

## 新增指标

### 1. Hurst Exponent (`hurst`)

**计算**：滚动 R/S (Rescaled Range) 分析
**输出列**：`hurst`（原始 H 值），`hurst_signal`（滚动均值）
**参数**：`lookback`（100，默认），`max_lag`（20，默认）

```
H > 0.55 → 趋势市（persistent）
H ≈ 0.50 → 随机游走
H < 0.45 → 均值回归
```

**学术基础**：Mandelbrot & Wallis (1969), "Robustness of the rescaled range R/S in the measurement of noncyclic long-run statistical dependency"

**已知限制**：经典 R/S 有小样本向上偏差。在滚动窗口中使用时，相对比较（跨资产/跨时间窗口）比绝对值更可靠。

### 2. Swing Points (`swing_points`)

**计算**：滑动窗口极值检测 + 前向填充
**输出列**：`swing_high`，`swing_low`，`dist_to_high_pct`，`dist_to_low_pct`，`swing_range_pct`
**参数**：`lookback`（5，默认）

Swing high：`high[t] > max(high[t-5 : t+6])`（左右各 5 根 K 线中的最高点）

**交易含义**：
- `dist_to_high_pct < 2%` → 价格接近阻力位（做空信号）
- `dist_to_low_pct < 2%` → 价格接近支撑位（做多信号）
- 这些水平有效的原因是**自证预言**：足够多的市场参与者在这些位置有未平仓头寸

### 3. Fractional Differentiation (`frac_diff`)

**计算**：固定宽度窗口权重卷积
**输出列**：`frac_close`
**参数**：`d`（0.4，默认），`threshold`（0.001，默认）

整数差分（returns）破坏所有长期记忆。分数差分 `d ∈ (0, 1)` 在平稳性和记忆保留之间取得平衡。

**学术基础**：Lopez de Prado (2018), *Advances in Financial Machine Learning*, Chapter 5

**参数选择**：加密货币的 `d ≈ 0.3–0.4` 效果最好，基于 Hurst 指数 0.55–0.65 的实证发现。

---

## ML 特征扩展

`DEFAULT_FEATURES` 从 30 维扩展到 40 维：

| 新特征 | 来源 | 含义 |
|--------|------|------|
| `hurst` | hurst 指标 | 当前 Hurst 值 |
| `hurst_signal` | hurst 指标 | 滚动平均 Hurst |
| `roll_hurst_20` | hurst 指标 | 20 bar 滑动平均 |
| `dist_to_swing_high` | swing_points | 距最近阻力位 % |
| `dist_to_swing_low` | swing_points | 距最近支撑位 % |
| `swing_range_pct` | swing_points | 最近波动区间宽度 |
| `swing_reversal_count_50` | swing_points | 50 bar 内反转次数 |
| `frac_ret_5` | frac_diff | 分数差分 5 bar 收益 |
| `frac_ret_10` | frac_diff | 分数差分 10 bar 收益 |
| `frac_vol_10` | frac_diff | 分数差分波动率 |

缺失时全部回退到 0.0（安全默认值）。

---

## GA 进化池扩展

### 新条件模板

```
做多：dist_to_low_pct < 0.02  （接近支撑位）
     swing_range_pct > 0.03    （宽幅波动，突破潜力）
     hurst > 0.55              （趋势市，顺势交易）

做空：dist_to_high_pct < 0.02 （接近阻力位）
     swing_range_pct > 0.03
     hurst < 0.45              （均值回归市，反转交易）
```

### 新可进化参数

| 基因 | 范围 | 步长 | 默认值 |
|------|------|------|--------|
| `hurst_lookback` | 50–200 | 10 | 100 |
| `swing_lookback` | 3–10 | 1 | 5 |
| `frac_diff_d` | 0.10–0.60 | 0.05 | 0.40 |

### 初始概率

```python
"hurst": 0.3, "swing_points": 0.35, "frac_diff": 0.2
```

保持较低的初始概率以避免策略过度复杂化。

---

## 验证结果

- ✅ 3 个新指标在 500 bar 数据上产出合理值（Hurst: 400/500, Swing: 500/500, FracDiff: 446/500）
- ✅ 策略条件 `hurst > 0.5`、`dist_to_low_pct < 0.02` 正确评估
- ✅ 40 维特征矩阵无 NaN/Inf
- ✅ GA 染色体 round-trip（随机 → 编码 → 解码）保持指标配置
- ✅ 向后兼容：现有 3 个策略不需要 `hurst`/`swing_points`/`frac_diff` 即可正常运行

---

## 审计发现

| ID | 严重度 | 描述 |
|----|--------|------|
| R14-001 | Medium | `_compute_hurst_indicator` 使用 O(n²) 滚动循环。300 bar × 100 lookback = 30,000 次 `_rs_hurst` 调用。可用 Numba JIT 加速 10x+ |
| R14-002 | Medium | `_detect_swing_points` 的窗口最大值检查使用 Python 原生 `max()`。改用 `np.max(h_window)` 可加速 |
| R14-003 | Low | FracDiff 权重长度随 d 增大而增大。d=0.5 时约需 140 个权重（threshold=1e-3），导致前 139 根 K 线的 `frac_close` 为 NaN |
| R14-004 | Low | GA 条件池新增的条件模板（hurst/swing）可能在随机初始化时被选中，但对应的 indicator 也可能被 BooleanGene 关闭 → `_sanitize_conditions` 会删除这些条件并注入 fallback |

---

## 状态

**Phase 4c 完成。** | **新增**: 225+ 行 | **修改**: 3 文件 | **发现**: 2 Medium, 2 Low
