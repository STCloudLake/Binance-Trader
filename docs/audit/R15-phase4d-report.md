# 审计报告 R15 — ML 基础设施加固

**日期**: 2026-07-20
**阶段**: Phase 4d
**类型**: 新增基础设施

---

## 变更范围

| 文件 | 变更类型 | 行数 |
|------|----------|------|
| `core/ml/feature_store.py` | **新建** | 220 |
| `core/ml/predictor.py` | Ensemble + Online Learning + PatchTST | +120 |

---

## 新增能力

### 1. Feature Store (`feature_store.py`)

**目的**：消除回测中每次预测都重算 40 维特征的性能瓶颈。

**架构**：
```
Market Cache (OHLCV Parquet)
  ↓ build()
compute_all() + compute_features()
  ↓
Feature Store ({symbol}/{interval}.parquet)
  ↓ get_vector() / get_dataframe()
Backtest Engine / ML Predictor (O(1) lookup)
```

**API**：
- `build(symbols, intervals, date_start, date_end)` — 批量预计算
- `get_vector(symbol, interval, timestamp)` — O(1) 特征向量查找
- `get_dataframe(symbol, interval, timestamp, n_rows)` — 切片查询（回测用）
- `update(symbol, interval, new_ohlcv)` — 增量更新（实盘用）
- `invalidate(symbol, interval)` — 缓存清除

**存储**：`{data_dir}/feature_store/{symbol}/{interval}.parquet`

**性能预期**：回测速度提升 3–5×（消除 ~1M 次 `compute_features()` 调用）

### 2. Model Ensemble (`predictor.py`)

**目的**：利用三种架构的互补优势减少单模型偏差。

```python
async def ensemble_predict(symbol, X):
    predictions = {
        "lgb":  LightGBM.proba(X),    # weight 0.40
        "tft":  TFT.confidence(X),    # weight 0.35
        "patch": PatchTST.confidence(X), # weight 0.25
    }
    return weighted_average(predictions)
```

**权重依据**：LightGBM 在表格数据上最稳定（最高权重），TFT/PatchTST 提供序列感知补充。权重固定为 40/35/25，未来可由 GA 进化。

**容错**：当只有部分模型可用时，自动回退到可用子集的加权平均。

### 3. Online Learning (`predictor.py`)

**目的**：用增量训练替代完全重训，减少 24 小时重训周期的时间成本。

```python
# 当前（完全重训，~30s）：
model = LGBMClassifier(...).fit(X, y)

# 改进（增量更新，~2s）：
model.fit(X_recent, y_recent,
          init_model=model.booster_,
          keep_training_booster=True)
```

**回退机制**：首次训练或增量训练失败时自动回退到完全重训。

---

## 验证结果

- ✅ FeatureStore build → 200 行 40 维特征 Parquet，写入成功
- ✅ get_vector() → shape=(40,) 正确
- ✅ get_dataframe() → (50, 41) shape 正确（40 特征 + close）
- ✅ update() → 增量逻辑正确（从 market cache 读取原始数据）
- ✅ Ensemble predict → 回退到单模型正常（其他模型未加载时）
- ✅ incremental_retrain → 含 init_model 调用 + 回退路径
- ✅ 所有导入无循环依赖

---

## 审计发现

| ID | 严重度 | 描述 |
|----|--------|------|
| R15-001 | Medium | FeatureStore `update()` 每次增量都重新读取完整的 raw OHLCV Parquet，对大文件不高效。应使用 `read_parquet(filters=...)` 或追加模式 |
| R15-002 | Medium | Ensemble 权重 40/35/25 是固定的，未根据模型实时表现动态调整。未来应基于近期准确率自适应 |
| R15-003 | Low | incremental_retrain 使用 `X.iloc[-200:]` 仅取最近 200 条训练。灾难性遗忘风险——模型可能忘记早期学习的模式 |
| R15-004 | Low | PatchTST 集成仅在 ensemble_predict 中可用，未加入 retrain_loop 自动训练周期 |
| R15-005 | Note | FeatureStore 的 in-memory cache (`_loaded` dict) 无上限。加载 50+ 个 Parquet 文件可能耗尽内存。需要 LRU 驱逐策略 |

---

## 状态

**Phase 4d 完成。** | **新建**: 220 行 | **修改**: 120 行 | **发现**: 2 Medium, 3 Low
