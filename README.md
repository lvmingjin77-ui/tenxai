# LLM 增强的量化研究工作流

一套面向研究与原型的 **双模块仓库**：**Q1** 把规则层选股/风控与 **单 Agent（LLM）** 的结构化研究判断接在同一套回测闭包里；**Q2** 把 **工具调用型 Agent** 嵌进可执行的清洗—评估—筛选流水线，并与确定性数值计算、**LightGBM** 验证对齐。适合作为个人实验台或组件参考，按需拆开使用。

使用前请阅读 [`mustRead.md`](mustRead.md)：仓库内未提交任何 API Key，需自行配置环境变量或 `.env` 后再运行。

---

## 仓库结构

| 路径 | 说明 |
|------|------|
| [`mustRead.md`](mustRead.md) | API、最低配置成本与运行提示 |
| [`Q1/`](Q1/) | **Research Workbench**：美股日频、轻量前端 + 标准库 HTTP 服务 + SQLite |
| [`Q2/`](Q2/) | **FeatureAgent Pipeline**：OpenAI 兼容模型 + `exec_python` / `log_decision` + IC/AUC 等向量化指标 + LightGBM |
| `data.pq`（自备） | Q2 使用的面板数据，置于**仓库根目录**（与 `Q1/`、`Q2/` 同级） |

深度设计说明：

- Q1：[`Q1/Q1.md`](Q1/Q1.md)
- Q2：[`Q2/Q2.md`](Q2/Q2.md)

---

## 配置：LLM 与外部 API

出于隐私，本仓库不包含密钥。**Q1** 在仅配置 DeepSeek（或任意 OpenAI 兼容端点）时即可启动 [`Q1/backend/server.py`](Q1/backend/server.py) 并在浏览器中跑通主链路；默认 SQLite 中已带有可读的缓存数据集，不必立刻接入全部行情源。**Q2** 同样需要可用的 LLM，变量名与注入方式见下文各模块。

---

## Q1：美股研究与回测工作台（Research Workbench）

### 能力概览

规则层生成候选股与市场状态 → **单 Agent（LLM）** 输出结构化 research card（`enter` / `hold` / `trim` / `reject`）→ 规则层落仓；按交易日推进，T 日收盘决策、T+1 开盘调仓。前端为静态 HTML/CSS/JS，净值曲线用 SVG 自绘，刻意避免重型框架，便于迁移与阅读。

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

### 数据与数据源

- 默认使用数据集 ID `hetero_trade_mas_cache_us_v1`，由本地 SQLite 提供（启动时 bootstrap 会说明路径与备注）。
- 若要**从外部重新构建**数据导入链路，在 `Q1/backend/.env` 中配置 Alpaca、FMP、Finnhub、EODHD、SEC、FRED 等（字段见 [`Q1/backend/app/config.py`](Q1/backend/app/config.py) 内 `Settings`）。
- 界面效果可参考仓库内 `Q1/` 下的截图（如 `Q1.png`）。

---

## Q2：可执行的特征工程 Agent（Feature Pipeline）

### 能力概览

**FeatureAgent**（DeepSeek + function calling）通过 `exec_python` 在共享 `state` 上执行代码，配合 `log_decision` 记录决策；流程为 **Diagnose → Clean → Evaluate → Select**，其中大规模 IC / AUC / 相关矩阵由确定性 Python 向量化计算，Agent 侧重诊断、清洗策略与筛选逻辑。最终用 **Top 50 特征 + LightGBM** 在 **70%/30% 按日期** 的训练/测试切分下做二分类验证（标签 **Y7**，去掉中性类 `0`）。

架构图、完整 Prompt、防泄漏与 fallback 说明见 [`Q2/Q2.md`](Q2/Q2.md)。

### 数据

将你的 **`data.pq`**（面板格式，与代码中列约定一致）放在**仓库根目录**，使 `Q2/run.py` 中的 `../data.pq` 能正确解析。

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

**Jupyter Notebook（交互分析、可保留 cell 输出）：**

```bash
cd Q2
jupyter notebook Q2.ipynb
```

执行 **Kernel → Restart & Run All**。若需从模板重建 Notebook：`python Q2/create_notebook.py`。

更细的目录说明、输出文件列表与常见问题见 [`Q2/README.md`](Q2/README.md)（与本文互补，侧重 Q2 子目录）。

---

## 许可证与合规

代码以研究与学习用途分享；引用的外部数据与第三方 API 须遵守各自服务条款与授权范围。

起步文档：[`mustRead.md`](mustRead.md)；模块级深度说明：[`Q1/Q1.md`](Q1/Q1.md)、[`Q2/Q2.md`](Q2/Q2.md)。
