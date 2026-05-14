
本仓库包含两个项目：**Q1** 为美股日频单 Agent 研究与回测 Demo；**Q2** 为 Agent 驱动的自动化特征工程（方向 A）。运行前请阅读根目录 [`mustRead.md`](mustRead.md)（作者已移除 API Key，需自行配置后再跑）。

---

## 仓库结构

| 路径 | 说明 |
|------|------|
| [`mustRead.md`](mustRead.md) | API 与最低成本复现说明 |
| [`Q1/`](Q1/) | 美股回测：轻量前端 + Python `http.server` 后端 + SQLite |
| [`Q2/`](Q2/) | 特征工程：DeepSeek 工具调用 Agent + 向量化指标 + LightGBM |
| `data.pq`（需自备） | Q2 使用的面板数据，放在**本仓库根目录**（与 `Q1/`、`Q2/` 同级） |

详细设计文档：

- Q1：[`Q1/Q1.md`](Q1/Q1.md)
- Q2：[`Q2/Q2.md`](Q2/Q2.md)

---

## 通用：API Key

作者为隐私已清空密钥。**Q1 最低成本**：仅配置 DeepSeek（或任意 OpenAI 兼容端点）即可启动 [`Q1/backend/server.py`](Q1/backend/server.py)，在浏览器里跑回测（数据集导入结果已缓存于项目中，无需再拉全量外部数据）。**Q2**：同样需要可用的 LLM API（与 Q1 类似，通过环境变量或 `.env` 注入）。

---

## Q1：美股单 Agent 回测

### 做什么

规则层生成候选股与市场状态 → **单 Agent（LLM）** 输出结构化 research card（enter / hold / trim / reject）→ 规则层落仓；按交易日推进，T 日收盘决策、T+1 开盘调仓。前端为静态 HTML/CSS/JS，净值曲线用 SVG 自绘，无重型框架。

更完整的流程、打分公式与 Agent I/O 约束见 [`Q1/Q1.md`](Q1/Q1.md)。

### 环境与依赖

```bash
cd Q1/backend
pip install -r requirements.txt
```

主要依赖：`requests`、`pandas`、`pyarrow`、`yfinance`（以 `requirements.txt` 为准）。

### LLM 配置（`Q1/backend/app/llm_client.py`）

支持从 **`Q1/backend/.env`** 或环境变量读取，优先级含：`DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`，以及 `Q1_OPENAI_*`、`OPENAI_*` 等兼容项。未配置 Key 时无法调用模型做 underwriting。

### 启动服务

```bash
cd Q1/backend
python server.py
```

默认监听 **`http://127.0.0.1:8010`**（可通过环境变量 `Q1_PORT` 修改）。浏览器打开该地址即可使用可视化回测界面。

### 数据与复现数据集

- 默认使用内置数据集 ID `hetero_trade_mas_cache_us_v1`，数据来自本地 SQLite（见启动时 bootstrap 说明）。
- 若需**完整复现**从外部拉数与导入流程，需在 `Q1/backend/.env` 中配置 Alpaca、FMP、Finnhub、EODHD、SEC、FRED 等（详见 [`Q1/backend/app/config.py`](Q1/backend/app/config.py) 中 `Settings` 字段）。
- 运行结果截图可参考 `Q1/` 下作者提供的图片（如 `Q1.png` 等）。

---

## Q2：Agent 驱动的自动化特征工程（方向 A）

### 做什么

**FeatureAgent**（DeepSeek + function calling）通过 `exec_python` 在共享 `state` 上执行代码，配合 `log_decision` 记录决策；流程为 **Diagnose → Clean → Evaluate → Select**，其中大规模 IC / AUC / 相关矩阵由确定性 Python 向量化计算，Agent 侧重诊断、清洗策略与筛选逻辑。最终用 **Top 50 特征 + LightGBM** 在 **70%/30% 按日期** 的训练/测试切分下做二分类验证（标签 **Y7**，去掉中性类 `0`）。

架构图、完整 Prompt、防泄漏与 fallback 说明见 [`Q2/Q2.md`](Q2/Q2.md)。

### 数据

将题目提供的 **`data.pq` 放在仓库根目录**（路径与 `Q2/run.py` 中 `../data.pq` 一致）。

### 环境与依赖

```bash
cd Q2
pip install -r requirements.txt
```

建议 Python **3.10**（与 Notebook 元数据一致）。

### LLM 配置（`Q2/code/agent.py`）

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | 必填（未设置则无法调用） |
| `DEEPSEEK_BASE_URL` | OpenAI 兼容 Base URL（代码内有默认网关，可按需覆盖） |

模型常量默认为 `deepseek/deepseek-chat`。

### 运行方式

**脚本（无头环境友好）：**

```bash
cd Q2
export DEEPSEEK_API_KEY="你的密钥"
python run.py
```

产物在 `Q2/outputs/`（图表与 CSV 等）。

**Notebook（提交/展示）：**

```bash
cd Q2
jupyter notebook Q2.ipynb
```

执行 **Kernel → Restart & Run All**。若需从模板重建 Notebook：`python Q2/create_notebook.py`。

更细的目录说明、输出文件列表与常见问题见 [`Q2/README.md`](Q2/README.md)（与本文互补，侧重 Q2 子目录）。

---

## 许可证与致谢

本项目为笔试/课程提交用途；外部数据与 API 使用须遵守各提供方条款。

如有问题，可先对照 [`mustRead.md`](mustRead.md) 与各子目录下的 `Q1.md` / `Q2.md`。
