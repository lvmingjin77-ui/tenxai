# Q2 方向A：自动化特征工程系统

## 请在code/agent.py中自行添加deepseek的api，然后进行运行

## 环境安装

```bash
conda create -n tenxai python=3.10 -y
conda activate tenxai
pip install -r requirements.txt
```

## 运行方式

**方式一：直接运行脚本（快速验证）**
```bash
cd Q2
python run.py
```
输出文件保存在 `outputs/` 目录，约 5-10 分钟完成。

**方式二：Jupyter Notebook（提交版本）**
```bash
jupyter notebook Q2.ipynb
# Kernel → Restart & Run All
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `Q2.ipynb` | 主 Notebook，含完整运行输出 |
| `Q2.md` | Agent 系统设计文档（架构/Prompt/工作流） |
| `run.py` | 独立运行脚本 |
| `code/agent.py` | FeatureAgent 核心（DeepSeek + tool calling） |
| `code/metrics.py` | IC/AUC/单调性等特征评估指标 |
| `code/prompts.py` | 四阶段 Agent 任务 Prompt |
| `outputs/` | 所有输出图表和 CSV |

## Agent 说明

- **模型**：DeepSeek V3（`deepseek/deepseek-chat`）via OpenRouter
- **工具**：`exec_python`（生成并执行代码）、`log_decision`（记录决策）
- **流程**：诊断 → 清理 → 评估 → 筛选 → LightGBM 验证
- **目标标签**：Y7（三分类中最均衡：-1/0/+1 各约 34%/32%/34%）
