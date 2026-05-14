# Q2 方向 A：Agent 驱动的自动化特征工程系统

本目录实现一套**可执行的 Agent 工作流**：大模型通过工具调用生成并运行 Python 代码，在共享 `state` 上完成「诊断 → 清理 → 评估 → 筛选」，再用 **LightGBM** 在严格时间切分下做二分类验证。设计细节、完整 Prompt 与取舍说明见同目录下的 [`Q2.md`](Q2.md)。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| 数据 | 读取仓库根目录的 `../data.pq` 面板数据（约 8 万行 × 321 列，含 `X1`～`X300` 与标签等） |
| 标签 | 使用 **`Y7`**（三分类 `-1/0/+1` 分布较均衡）；建模时丢弃 `Y7==0`，仅 **+1 vs -1** 二分类 |
| Agent | **DeepSeek Chat**（OpenAI 兼容 API）+ **function calling**：`exec_python`、`log_decision` |
| 确定性计算 | 批量截面 **IC**、**IC_IR**、单特征 **AUC**、分组单调性、**Spearman 相关矩阵** 由 `metrics.py` / `run.py` 向量化计算，避免全靠 Agent 写循环 |
| 产出 | 图表与 CSV 写入 `outputs/`；完整交互与说明见 **`Q2.ipynb`**（可用 `create_notebook.py` 重新生成） |

---

## 系统架构（摘要）

```
FeatureAgent（Orchestrator，DeepSeek + tool calling）
    ├── exec_python   → 在共享命名空间执行代码（state / np / pd / stats）
    ├── state         → 各阶段读写诊断表、清洗数据、评估表、Top50 等
    └── log_decision  → 关键决策落日志

四阶段工作流：Diagnose → Clean → Evaluate（Agent 审阅 + 预计算指标）→ Select
        → LightGBM（AUC / Precision / Recall / F1 + ROC + 混淆矩阵）
```

更完整的架构图、各阶段 `state` 键名与 Prompt 全文见 [`Q2.md`](Q2.md)。

---

## 环境与依赖

- **Python**：建议 3.10（与 Notebook metadata 一致）
- **数据**：将赛题/课程提供的 `data.pq` 放在**仓库根目录**（与 `Q1/`、`Q2/` 同级），即 `Q2/run.py` 中使用的路径 `../data.pq` 能解析到该文件

安装依赖（在 `Q2/` 下执行）：

```bash
cd Q2
pip install -r requirements.txt
```

或使用 Conda：

```bash
conda create -n tenxai python=3.10 -y
conda activate tenxai
pip install -r requirements.txt
```

---

## API 与密钥配置

项目为隐私已移除密钥。复现前需配置 **DeepSeek（或兼容端点）** 的访问凭证。

[`code/agent.py`](code/agent.py) 中通过环境变量读取（推荐在 shell 或 `.env` 注入工具里设置）：

| 变量 | 含义 | 默认值（代码内） |
|------|------|-------------------|
| `DEEPSEEK_API_KEY` | API Key | 空字符串（未设置则无法调用模型） |
| `DEEPSEEK_BASE_URL` | OpenAI 兼容 Base URL | `https://openrouter.fans/v1` |
| 模型名 | 常量 `MODEL` | `deepseek/deepseek-chat` |

示例：

```bash
export DEEPSEEK_API_KEY="你的密钥"
# 若使用官方 DeepSeek 或其它兼容网关，可覆盖：
# export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
```

也可直接修改 `agent.py` 里 `os.getenv(..., "默认值")` 的默认值（不推荐提交到版本库）。

根目录 [`mustRead.md`](../mustRead.md) 说明：Q2 复现一般只需配置上述 API 即可。

---

## 运行方式

**工作目录**：请先 `cd` 到本目录 `Q2/`，否则相对路径 `../data.pq`、`outputs/`、`code/` 会错位。

### 方式一：一键脚本（适合快速复现）

```bash
cd Q2
python run.py
```

- 使用**非交互** matplotlib 后端（`Agg`），适合服务器或无显示器环境
- 全流程约 **5～15 分钟**（视机器与 API 延迟而定）
- 各阶段若 Agent 未写入关键 `state` 键，脚本内建有 **fallback**（与 Notebook 思路一致）；特征筛选还有 **QC → 二次 repair prompt → 固定规则兜底** 链，保证可跑完

### 方式二：Jupyter Notebook（提交/报告用）

```bash
cd Q2
jupyter notebook Q2.ipynb
```

在菜单中选择 **Kernel → Restart & Run All**。Notebook 内含「结果索引」Markdown，便于跳转到数据概况、各 Phase、ROC/混淆矩阵、泄漏检测等单元。

### 重新生成 Notebook

若修改了 `create_notebook.py` 中的单元模板，可重新写出 `Q2.ipynb`：

```bash
cd Q2
python create_notebook.py
```

---

## 目录与文件说明

| 路径 | 说明 |
|------|------|
| [`Q2.ipynb`](Q2.ipynb) | 主 Notebook：含完整流程与说明单元（提交物之一） |
| [`Q2.md`](Q2.md) | 系统设计文档：架构、四阶段 Prompt、防泄漏、异常与 fallback |
| [`run.py`](run.py) | 与 Notebook 等价的流水线脚本，输出写入 `outputs/` |
| [`create_notebook.py`](create_notebook.py) | 用 `nbformat` 程序化生成 `Q2.ipynb` |
| [`requirements.txt`](requirements.txt) | Python 依赖 |
| [`code/agent.py`](code/agent.py) | `FeatureAgent`：OpenAI SDK、`exec_python` / `log_decision`、对话与工具循环（默认 `max_iter=25`） |
| [`code/prompts.py`](code/prompts.py) | 四阶段任务字符串：`DIAGNOSE`、`CLEAN`、`EVALUATE`、`SELECT` |
| [`code/metrics.py`](code/metrics.py) | `batch_cs_ic`、`ic_ir`、`feature_auc`、`group_monotonicity` |
| [`outputs/`](outputs/) | 运行产物：概况图、诊断/评估 CSV、特征图、Top50、模型评估图等（随运行更新） |

---

## 流水线与 `state` 约定（与代码对齐）

1. **数据切分**：按 `trade_date` 排序后 **前 70% 日期为训练**、**后 30% 为测试**；`train_dates` / `test_dates` 为 `set`，供 Agent 与清洗逻辑约束「统计量仅用训练集」。
2. **Phase 1 — DIAGNOSE**：Agent 生成代码写入 `state['diagnose_report']`（DataFrame，含缺失、偏度、IQR 异常率等）；缺失时 `run.py` 有简化 fallback。
3. **Phase 2 — CLEAN**：目标为 `state['df_clean']`（`underlying` 上 `ffill`、训练集截面中位数补余、按偏度选择分位或 3σ clip、按日 `rank(pct=True)`）；缺失时执行与 Notebook 一致的标准 fallback。
4. **Phase 3 — 评估**：脚本侧在训练集上计算 `eval_df`、`corr_matrix` 写入 `state`，再 `agent.run(EVALUATE)` 做解读与 `state['eval_summary']`。
5. **Phase 4 — SELECT**：四轮漏斗（弱信号过滤 → 加权 rank 分 Top100 → 高相关去冗余 → Top50）；`run.py` 含质量检查与兜底 `_manual_select`（层次聚类近似「\|r\|>0.8 成组保留最高分」）。
6. **模型**：`df_clean` 上剔除 `Y7==0`，`y = (Y7==1)`，仅用 **Top50** 特征 + **LightGBM** 在测试日期上报告 AUC / P / R / F1，并保存 ROC、混淆矩阵、特征重要度图。
7. **泄漏自检**：打印训练/测试日期是否不交叠、时间先后及训练 vs 测试截面 IC 量级比等（见 `run.py` 末尾）。

---

## `outputs/` 产物说明（`run.py`）

成功执行后常见文件包括：

| 文件 | 内容 |
|------|------|
| `data_overview.png` | 标签分布、缺失率、各标的样本数 |
| `diagnose_report.csv` / `diagnose_report.png` | 特征诊断表与可视化 |
| `clean_comparison.png` | 示例特征清洗前后对比 |
| `eval_report.csv` | 全特征 IC/AUC/单调性等 |
| `feature_eval.png` / `feature_corr.png` | IC/AUC 分布与 Top30 相关热力图 |
| `top50_features.csv` | 最终 50 个特征名 |
| `top50_selection.png` | 漏斗各轮特征数量 |
| `model_evaluation.png` | 测试集 ROC、混淆矩阵、LightGBM 重要度 |

若在 **Notebook** 中跑通带保存逻辑的版本，还可能额外生成 `top50_scores.csv`（含 `feature, ic_mean, ic_ir, auc, score`）。`run.py` 当前以特征表与图表为主；需要该 CSV 时可从 Notebook 单元复制保存逻辑或自行 `state['top50_scores'].to_csv(...)`。

---

## 常见问题

- **提示 401 / 连接失败**：检查 `DEEPSEEK_API_KEY` 与 `DEEPSEEK_BASE_URL` 是否与所用网关一致。
- **找不到 `data.pq`**：确认文件在仓库根目录，且命令在 `Q2/` 下执行。
- **中文图例乱码**：`run.py` 已设置 `Arial Unicode MS`、`PingFang HK` 等 sans-serif；Linux 若无相应字体，可安装中文字体或改 `matplotlib` 字体配置。
- **Agent 与脚本结果不完全一致**：属预期；脚本侧 fallback / QC / 固定规则保证端到端可复现，Agent 负责主要探索与决策日志。

---

## 延伸阅读

- [`Q2.md`](Q2.md)：完整 Prompt 归档、截面 rank 归一化动机、Agent 与确定性计算分层、异常处理与防泄漏条款。
- [`../mustRead.md`](../mustRead.md)：全仓库 API 与复现说明。
