# Binance Trader

![Binance Trader](social-preview.png)

基于 Python 的自动化加密货币交易系统，集成多策略引擎、遗传算法优化、ML 预测、AI 决策、三层风控和实时 Web 管理面板。

## 架构

```
binance_trader/
├── app/                          # 应用入口、配置、事件总线
│   ├── main.py                   # 入口：组件初始化、依赖注入、生命周期管理
│   ├── config.py                 # 单例配置加载器 (YAML → Pydantic 模型)
│   └── event_bus.py              # 异步事件总线 (17 种事件类型, 10000 容量)
├── core/
│   ├── strategy/                 # 策略引擎
│   │   ├── engine.py             # 策略求值、信号融合 (指标 + ML + 新闻)
│   │   ├── loader.py             # YAML 策略加载/保存/删除 (Pydantic 模型)
│   │   └── indicators.py         # TA-Lib 技术指标 (RSI, MACD, BB, EMA, ADX 等)
│   ├── ga/                       # 遗传算法策略优化 (NEW)
│   │   ├── evolver.py            # GA 进化引擎：选择/交叉/变异/精英保留
│   │   ├── genome.py             # 策略基因组编码：参数→染色体映射
│   │   └── fitness.py            # 适应度函数：夏普×胜率×稳定性
│   ├── market_data/              # 行情数据
│   │   ├── provider.py           # Binance WebSocket 多流 + REST 轮询混合
│   │   └── ohlcv_cache.py        # Parquet 文件 + 内存双层 OHLCV 缓存
│   ├── risk/                     # 风控系统 (三层)
│   │   ├── manager.py            # 7 步信号审核管线
│   │   ├── circuit_breaker.py    # 熔断器 (回撤/亏损/连续亏损)
│   │   ├── position_guard.py     # PositionGuard: 移动止损 + 紧急止损
│   │   └── position_sizer.py     # 仓位计算、止损/止盈价计算
│   ├── executor/
│   │   └── executor.py           # 订单执行 (模拟 + 实盘 Binance API)
│   ├── ml/                       # 机器学习 (3 种引擎)
│   │   ├── predictor.py          # ML 预测器：LightGBM / TFT / PatchTST
│   │   ├── trainer.py            # LightGBM + XGBoost 训练器
│   │   ├── features.py           # 37 维特征工程 + Triple Barrier 标签
│   │   ├── tft_model.py          # TFT: 变量选择 + LSTM + 多头注意力
│   │   ├── tft_trainer.py        # TFT 训练循环 + 序列窗口
│   │   ├── patchtst_model.py     # PatchTST: Patch嵌入 + Transformer编码器
│   │   └── patchtst_trainer.py   # PatchTST 训练循环
│   ├── ai/                       # AI 决策
│   │   ├── deepseek_ctl.py       # DeepSeek 控制器 (4 个后台循环任务)
│   │   ├── prompts.py            # 5 组 AI 提示词模板
│   │   └── vibe_connector.py     # Vibe-Trading MCP 集成
│   ├── auth/                     # 认证授权
│   │   └── auth.py               # JWT + bcrypt + RBAC (3 角色)
│   └── news/                     # 新闻分析
│       ├── analyzer.py           # 新闻抓取、情感分析、异常监测
│       ├── fetcher.py            # httpx 异步 HTTP 客户端
│       └── source_manager.py     # 新闻源配置管理
├── web/
│   ├── server.py                 # FastAPI 应用 (80+ 路由, WebSocket, HTMX)
│   ├── i18n.py                   # 中英文翻译 (200+ 条目)
│   └── templates/                # Jinja2 模板
├── strategies/                   # YAML 策略定义
├── config/                       # 系统配置、风控参数、告警规则
├── data/                         # 运行时数据 (DB, 模型, 缓存)
├── scripts/                      # 工具脚本 (数据下载等)
└── tests/                        # 测试文件
```

## 核心能力

### 策略引擎
- **声明式策略定义** — 通过 YAML 文件完整描述策略：指标参数、入场/出场条件、减仓规则、时间框架
- **信号融合引擎** — `(indicator×w_ind + ML×w_ml + news×w_news) / total_weight`，支持动态权重调整
- **多时间框架** — 1m/5m/15m/1h/4h 自由组合，短线确认 + 长线过滤
- **策略×交易对矩阵** — 每个策略独立配置交易币种，回测中精确衡量策略-交易对适配度
- **Web UI 全生命周期管理** — 创建、编辑、启用/禁用、删除策略，即时生效无需重启

### 遗传算法策略优化 (GA)
- **自动进化策略** — 初始化种群 → 并行回测评估 → 选择/交叉/变异 → 迭代进化
- **基因组编码** — 指标参数（连续基因）+ 入场/出场条件（结构基因）+ 时间框架
- **适应度函数** — 夏普比率×2 + 胜率 + 盈亏比 - 回撤惩罚 - 过拟合惩罚
- **过拟合防护** — 参数敏感度检测 + 样本外验证 + 复杂度惩罚
- **精英保留 + 随机注入** — 防止早熟收敛，保持种群多样性

### 回测系统
- **全模拟回测** — 完整的入场/出场/减仓/止损/止盈模拟
- **策略×交易对矩阵** — 分别显示每个策略在每个币种上的收益/胜率/交易数
- **ML 引擎可切换** — LightGBM（树模型）/ PatchTST（Transformer）/ TFT（序列模型）
- **AI 权重模拟** — 回测中模拟市场状态驱动的动态权重调整
- **跳过训练选项** — 复用缓存模型加速重复回测
- **指标预计算缓存** — 消除 ~125 万次重复计算，40-50% 加速
- **轮询重训练** — 每步只训练 1 个模型，避免 GPU 突发过载

### 机器学习 (3 种引擎)
- **LightGBM** — 树模型，快速稳定，~3 小时全矩阵回测
- **TFT (Temporal Fusion Transformer)** — 变量选择 + LSTM + 注意力
- **PatchTST** — 2026 年基准排名 #1，Patch 嵌入 + Transformer 编码器
- **Triple Barrier 标签** — 路径感知的上轨/下轨/超时三分类，优于传统二元标签
- **37 维特征** — 价格动量、波动率、成交量、价格位置、趋势指标、微观结构、时序周期
- **扩展窗口归一化** — 保留趋势信息，避免逐窗口标准化破坏信号

### AI 决策
- **4 个后台循环任务** — 市场评估(60min) / 币种选择(240min) / 策略优化(1440min) / 风控调整(1440min)
- **自动模式** — DeepSeek 全自动：评估市场、推荐币种、优化策略、调整风控

### 三层风控
| 层级 | 组件 | 功能 |
|------|------|------|
| 第 1 层 | CircuitBreaker | 日回撤%/日亏损$/连续亏损次数检测 |
| 第 2 层 | RiskManager | 7 步信号审核管线，信号去重 |
| 第 3 层 | PositionGuard | 15s 循环：移动止损 + 紧急强制平仓 |

### 预警系统
- **规则引擎** — 5 条默认规则 + 可自定义，条件匹配 + 冷却时间
- **WebSocket 实时推送** — 无需刷新页面
- **级别分类** — critical / warning / info

### 多用户认证
- **RBAC 3 角色** — admin（全部权限）/ trader（交易+设置）/ viewer（只读）
- **安全特性** — bcrypt 密码哈希 / JWT(HS256) / HttpOnly+SameSite Cookie

## 快速开始

### 1. 安装

```bash
cd binance_trader
pip install -r requirements.txt
```

> **注意**: TA-Lib 需要系统级安装。Windows 用户参见 [ta-lib-python](https://github.com/TA-Lib/ta-lib-python)。

### 2. 配置

```bash
cp config/secrets.yaml.example config/secrets.yaml
```

编辑 `config/secrets.yaml`，填入 API 密钥。

### 3. 下载历史数据

```bash
python scripts/download_history.py
```

### 4. 启动

```bash
python app/main.py
```

访问 `http://127.0.0.1:8899`，默认账户 `admin` / `admin`。
